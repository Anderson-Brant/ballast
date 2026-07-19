"""Tests for factors/regression.py.

Notes
-----
The strongest test synthetic data allows: generate stock returns FROM
known factor returns and exposures, then demand the regression hand the
factor returns back. If recovery works with 40 symbols and small noise,
the estimator is right; everything else here is guard rails. The French
cross-check (real-universe acceptance) runs later, when long history and
the French library are wired up.
"""

import numpy as np
import pandas as pd
import pytest

from ballast.factors.exposures import FACTORS
from ballast.factors.regression import (
    FactorModel,
    RegressionError,
    build_panel,
    fit_cross_section,
    fit_panel,
)


def make_ground_truth(n_symbols=40, n_periods=120, noise_std=0.001, seed=5):
    """Returns (panel, returns, true_factor_returns, true_market)."""
    rng = np.random.default_rng(seed)
    symbols = [f"S{i:02d}" for i in range(n_symbols)]
    dates = pd.bdate_range("2020-01-03", periods=n_periods, freq="W-FRI")

    exposures = pd.DataFrame(
        rng.normal(0.0, 1.0, (n_symbols, len(FACTORS))), index=symbols, columns=list(FACTORS)
    )
    true_f = pd.DataFrame(
        rng.normal(0.0, 0.01, (n_periods, len(FACTORS))), index=dates, columns=list(FACTORS)
    )
    true_market = pd.Series(rng.normal(0.002, 0.015, n_periods), index=dates)

    panel, returns = {}, {}
    for ts in dates:
        noise = rng.normal(0.0, noise_std, n_symbols)
        r = true_market[ts] + exposures.to_numpy() @ true_f.loc[ts].to_numpy() + noise
        panel[ts] = exposures  # constant exposures: simplest identifiable case
        returns[ts] = pd.Series(r, index=symbols)
    return panel, returns, true_f, true_market


# ----------------------------------------------------------------- recovery


def test_recovers_known_factor_returns():
    panel, returns, true_f, true_market = make_ground_truth()
    model = fit_panel(panel, returns)
    assert isinstance(model, FactorModel)
    for factor in FACTORS:
        corr = model.factor_returns[factor].corr(true_f[factor])
        assert corr > 0.99, f"{factor}: corr={corr:.3f}"
    assert model.factor_returns["market"].corr(true_market) > 0.99


def test_r2_near_one_when_noise_is_tiny():
    panel, returns, *_ = make_ground_truth(noise_std=1e-4)
    model = fit_panel(panel, returns)
    assert model.r2.mean() > 0.99


def test_specific_variance_recovers_the_noise():
    noise_std = 0.002
    panel, returns, *_ = make_ground_truth(noise_std=noise_std, n_periods=200)
    model = fit_panel(panel, returns)
    # Median across symbols of the per-symbol residual variance should sit
    # near noise_std^2 (sampling error shrinks with 200 periods).
    assert model.specific_variance.median() == pytest.approx(noise_std**2, rel=0.30)


def test_factor_cov_is_labeled_and_psd():
    panel, returns, *_ = make_ground_truth()
    model = fit_panel(panel, returns)
    cov = model.factor_cov
    assert list(cov.index) == ["market", *FACTORS]
    assert np.linalg.eigvalsh(cov.to_numpy()).min() >= -1e-12
    assert model.periods_per_year == 52  # units contract for the decomposer


# ------------------------------------------------------------------- guards


def test_thin_cross_section_raises():
    panel, returns, *_ = make_ground_truth(n_symbols=6)  # 7 coefs need >= 9
    with pytest.raises(RegressionError, match="too thin|periods were too thin"):
        fit_panel(panel, returns)


def test_nan_exposure_drops_symbol_for_that_period_only():
    panel, returns, *_ = make_ground_truth()
    first = sorted(panel)[0]
    poked = panel[first].copy()
    poked.loc["S00", "value"] = np.nan
    panel[first] = poked
    model = fit_panel(panel, returns)
    assert model.n_obs[first] == 39  # dropped once...
    assert model.n_obs.iloc[1] == 40  # ...not forever


def test_too_many_skipped_periods_fails_loudly():
    panel, returns, *_ = make_ground_truth(n_periods=20)
    # Starve 30% of the periods below the breadth floor.
    for ts in sorted(panel)[:6]:
        panel[ts] = panel[ts].iloc[:4]
        returns[ts] = returns[ts].iloc[:4]
    with pytest.raises(RegressionError, match="too thin to fit"):
        fit_panel(panel, returns)


def test_wls_weights_pull_the_fit():
    # Two clusters with different mean returns and zero exposures: the
    # intercept is a weighted mean, so up-weighting cluster A must pull it
    # toward A. Directional, definitional.
    symbols = [f"S{i}" for i in range(20)]
    # Exactly-zero exposures: the factor columns contribute nothing and
    # lstsq's minimum-norm solution leaves them at 0, so the intercept IS
    # the (weighted) mean return. Injecting tiny noise instead would let
    # lstsq blow the near-zero columns up into huge spurious coefficients.
    exposures = pd.DataFrame(0.0, index=symbols, columns=list(FACTORS))
    returns = pd.Series([0.10] * 10 + [-0.10] * 10, index=symbols)
    heavy_a = pd.Series([100.0] * 10 + [1.0] * 10, index=symbols)

    coefs_ols, _, _ = fit_cross_section(exposures, returns)
    coefs_wls, _, _ = fit_cross_section(exposures, returns, weights=heavy_a)
    assert coefs_ols["market"] == pytest.approx(0.0, abs=1e-3)
    assert coefs_wls["market"] > 0.08  # pulled almost all the way to +0.10


def test_nonpositive_weights_rejected():
    panel, returns, *_ = make_ground_truth(n_periods=3)
    first = sorted(panel)[0]
    bad = pd.Series(0.0, index=panel[first].index)
    with pytest.raises(RegressionError, match="positive"):
        fit_cross_section(panel[first], returns[first], weights=bad)


# -------------------------------------------------------------- build_panel


def test_build_panel_end_to_end(tmp_path):
    # 10 symbols, ~16 months of prices, fundamentals visible early: the
    # panel builder must produce aligned weekly exposures and returns that
    # fit_panel can consume. Numbers are smoke-level; correctness of each
    # stage is pinned by its own module's tests.
    from ballast.data.prices import store_prices
    from tests.test_exposures import put_fundamental
    from tests.test_prices import tidy

    db = tmp_path / "t.duckdb"
    rng = np.random.default_rng(23)
    symbols = [f"S{i}" for i in range(10)]
    for symbol in symbols:
        closes = (100 * np.cumprod(1 + rng.normal(0.0003, 0.012, 340))).tolist()
        store_prices(tidy(closes, symbol=symbol, start="2023-01-02"), db_path=db)
        for field, low, high in [
            ("book_equity", 200, 900),
            ("net_income", 20, 120),
            ("gross_profit", 50, 300),
            ("assets", 800, 2000),
            ("liabilities", 100, 900),
            ("shares_outstanding", 5, 40),
        ]:
            put_fundamental(db, symbol, field, float(rng.uniform(low, high)), filed="2023-03-01")

    exposures_panel, period_returns = build_panel(
        symbols, start="2024-02-01", end="2024-04-30", db_path=db
    )
    assert len(exposures_panel) >= 8  # ~12 weeks minus the last (no next period)
    assert set(exposures_panel) == set(period_returns)

    first = sorted(exposures_panel)[0]
    assert list(exposures_panel[first].columns) == list(FACTORS)

    model = fit_panel(exposures_panel, period_returns, min_specific_obs=5)
    assert np.isfinite(model.factor_returns.to_numpy()).all()
    assert ((model.r2 >= 0) & (model.r2 <= 1)).all()
    assert (model.n_obs == 10).all()
