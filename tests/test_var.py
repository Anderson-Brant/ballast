"""Tests for risk/var.py.

Notes
-----
The normal closed forms are the anchor: on a large Gaussian sample every
method must land near 2.326*sigma at 99% (that's what makes them
comparable). The regime tests are where the methods are SUPPOSED to
disagree -- FHS reacting to current vol while plain historical dilutes it
-- so those assert differences, not equality. Sign convention everywhere:
positive number = loss.
"""

import numpy as np
import pandas as pd
import pytest

from ballast.risk.var import (
    ROLLING_METHODS,
    VaRError,
    _cornish_fisher_z,
    filtered_historical_var,
    historical_var,
    monte_carlo_var,
    parametric_var,
    rolling_var,
)

Z99 = 2.3263  # -norm.ppf(0.01)
ES99_FACTOR = 2.6652  # pdf(z)/(1-c) for the normal at 99%


@pytest.fixture(scope="module")
def gaussian():
    """100k draws, sigma=1%: big enough that sample moments are the truth."""
    rng = np.random.default_rng(5)
    return pd.Series(
        rng.normal(0.0, 0.01, 100_000),
        index=pd.RangeIndex(100_000),
    )


# ----------------------------------------------------- normal closed forms


def test_parametric_matches_closed_form(gaussian):
    est = parametric_var(gaussian, confidence=0.99)
    assert est.var == pytest.approx(Z99 * 0.01, rel=0.02)
    assert est.es == pytest.approx(ES99_FACTOR * 0.01, rel=0.02)
    assert est.es > est.var  # always: ES is deeper in the tail


def test_all_methods_agree_on_gaussian_data(gaussian):
    # On genuinely normal data the full-sample methods must converge --
    # that's what makes disagreement on real data informative.
    p = parametric_var(gaussian).var
    h = historical_var(gaussian).var
    f = filtered_historical_var(gaussian).var
    assert h == pytest.approx(p, rel=0.05)
    # FHS runs ~15-20% HIGH here, structurally: z_t = r_t / sigma_t where
    # sigma_t is an EWMA estimate with ~32 effective observations, so the
    # z's have Student-t-like fat tails (Jensen on 1/sigma) even when r is
    # exactly normal. Known cost of regime responsiveness -- pinned as a
    # band, not hidden with a loose tolerance.
    assert 1.0 < f / p < 1.35


def test_horizon_scales_by_sqrt(gaussian):
    one = parametric_var(gaussian, horizon_days=1)
    month = parametric_var(gaussian, horizon_days=21)
    # mu ~ 0 in this sample, so the sqrt term dominates.
    assert month.var == pytest.approx(one.var * 21**0.5, rel=0.02)


# ----------------------------------------------------------- Cornish-Fisher


def test_cf_expansion_is_identity_at_zero_moments():
    assert _cornish_fisher_z(-2.326, 0.0, 0.0) == pytest.approx(-2.326)


def test_cf_negative_skew_raises_var(gaussian):
    # Bolt a crash tail onto the sample: skew goes negative, and the CF
    # correction must report MORE risk than the plain normal fit.
    crashy = pd.concat([gaussian, pd.Series([-0.08] * 300)], ignore_index=True)
    plain = parametric_var(crashy, cornish_fisher=False)
    cf = parametric_var(crashy, cornish_fisher=True)
    assert cf.var > plain.var


def test_cf_es_reduces_to_normal_es(gaussian):
    # With sample skew/kurt ~ 0 the numerical CF-ES must reproduce the
    # closed-form normal ES.
    plain = parametric_var(gaussian, cornish_fisher=False)
    cf = parametric_var(gaussian, cornish_fisher=True)
    assert cf.es == pytest.approx(plain.es, rel=0.02)


# ---------------------------------------------------------------- historical


def test_historical_hand_example():
    # 100 points: worst -10%, second worst -1%, the rest +1%. At 99%,
    # np.quantile interpolates at position 0.01*99 = 0.99 between the two
    # worst: q = -0.10 + 0.99*(-0.01 - (-0.10)) = -0.0109.
    # ES: the only point at or below q is -0.10 itself.
    values = [-0.10, -0.01] + [0.01] * 98
    est = historical_var(pd.Series(values), confidence=0.99)
    assert est.var == pytest.approx(0.0109)
    assert est.es == pytest.approx(0.10)


# ---------------------------------------------------- filtered historical


def make_regimes(calm_days: int, wild_days: int, calm_first: bool = True) -> pd.Series:
    rng = np.random.default_rng(9)
    calm = rng.normal(0.0, 0.005, calm_days)
    wild = rng.normal(0.0, 0.03, wild_days)
    values = np.concatenate([calm, wild] if calm_first else [wild, calm])
    return pd.Series(values, index=pd.bdate_range("2020-01-01", periods=len(values)))


def test_fhs_reacts_to_current_storm():
    # Calm year, then a violent quarter, measured TODAY (mid-storm):
    # plain historical dilutes the storm with 250 calm days; FHS rescales
    # everything to current vol and must report much more risk.
    series = make_regimes(250, 60, calm_first=True)
    fhs = filtered_historical_var(series, confidence=0.95)
    hist = historical_var(series, confidence=0.95)
    assert fhs.var > hist.var * 1.5


def test_fhs_relaxes_after_the_storm():
    # The mirror image: storm long past, calm now. FHS must report LESS
    # risk than the storm-contaminated historical estimate.
    series = make_regimes(250, 60, calm_first=False)
    fhs = filtered_historical_var(series, confidence=0.95)
    hist = historical_var(series, confidence=0.95)
    assert fhs.var < hist.var * 0.67


# ---------------------------------------------------------------- monte carlo


def test_mc_single_asset_matches_closed_form():
    cov = pd.DataFrame([[0.01**2]], index=["A"], columns=["A"])
    est = monte_carlo_var(pd.Series({"A": 1.0}), cov, confidence=0.99, seed=1)
    assert est.var == pytest.approx(Z99 * 0.01, rel=0.03)
    assert est.es == pytest.approx(ES99_FACTOR * 0.01, rel=0.03)


def test_mc_perfect_hedge_has_no_risk():
    # Two assets, correlation -1, equal weights: the portfolio is flat.
    s = 0.01
    cov = pd.DataFrame([[s**2, -(s**2)], [-(s**2), s**2]], index=["A", "B"], columns=["A", "B"])
    est = monte_carlo_var(pd.Series({"A": 0.5, "B": 0.5}), cov, confidence=0.99, seed=1)
    assert est.var == pytest.approx(0.0, abs=1e-6)


def test_mc_is_deterministic_given_seed():
    cov = pd.DataFrame([[1e-4]], index=["A"], columns=["A"])
    w = pd.Series({"A": 1.0})
    a = monte_carlo_var(w, cov, seed=42)
    b = monte_carlo_var(w, cov, seed=42)
    assert a.var == b.var  # bitwise equal, not approx


# -------------------------------------------------------------- rolling VaR


def test_rolling_var_day_never_sees_itself():
    # Flat 0.1% returns, then one -20% day. That day's VaR was estimated
    # from the calm window BEFORE it, so it must be small -- and breached.
    values = [0.001] * 120 + [-0.20] + [0.001] * 5
    series = pd.Series(values, index=pd.bdate_range("2024-01-01", periods=len(values)))
    out = rolling_var(series, method="historical", confidence=0.95, window=100)
    crash_day = series.index[120]
    assert out.loc[crash_day, "var"] < 0.01  # calm-window estimate
    assert bool(out.loc[crash_day, "breach"])  # and the crash breached it


@pytest.mark.parametrize("method", ROLLING_METHODS)
def test_rolling_breach_rate_is_sane_on_gaussian(method, gaussian):
    # On stationary normal data every method's 95% breach rate should sit
    # near 5%. Wide tolerance: this is a smoke check, not the Kupiec test
    # (that's validate/coverage.py's job).
    series = gaussian.iloc[:3000]
    series.index = pd.bdate_range("2012-01-02", periods=len(series))
    out = rolling_var(series, method=method, confidence=0.95, window=500)
    rate = out["breach"].mean()
    assert 0.02 < rate < 0.09


def test_rolling_rejects_short_history():
    series = pd.Series([0.001] * 200, index=pd.bdate_range("2024-01-01", periods=200))
    with pytest.raises(VaRError, match="window"):
        rolling_var(series, window=750)


# ------------------------------------------------------------------- guards


def test_too_few_observations_for_the_tail():
    series = pd.Series([0.01] * 50)
    with pytest.raises(VaRError, match="at least 100"):
        historical_var(series, confidence=0.99)  # 99% needs 100+ points
    historical_var(series, confidence=0.95)  # 95% needs only 20: fine


def test_bad_confidence_rejected(gaussian):
    for bad in (0.3, 1.0, 1.5):
        with pytest.raises(VaRError, match="confidence"):
            historical_var(gaussian, confidence=bad)


def test_nan_rejected():
    series = pd.Series([0.01] * 99 + [float("nan")])
    with pytest.raises(VaRError, match="NaN"):
        historical_var(series, confidence=0.95)
