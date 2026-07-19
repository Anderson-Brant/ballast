"""Tests for risk/decompose.py and the `ballast decompose` command.

Notes
-----
The hand example is small enough to verify on paper (worked in the
comments) and covers every output field. The invariants -- factor parts
sum to factor variance, position parts sum to total variance -- are the
contract; a decomposition that doesn't reconcile is worse than none.
"""

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from ballast.cli.app import app
from ballast.factors.exposures import FACTORS
from ballast.factors.regression import FactorModel
from ballast.risk.decompose import DecompositionError, decompose_portfolio

runner = CliRunner()


def tiny_model(style: str = "style", weekly: bool = True) -> FactorModel:
    """market + one style factor, diagonal covariance, two symbols."""
    names = ["market", style]
    cov = pd.DataFrame(np.diag([4e-4, 1e-4]), index=names, columns=names)
    return FactorModel(
        factor_returns=pd.DataFrame(columns=names),  # unused by decompose
        factor_cov=cov,
        specific_variance=pd.Series({"A": 5e-5, "B": 5e-5}),
        r2=pd.Series(dtype=float),
        n_obs=pd.Series(dtype=int),
        periods_per_year=52 if weekly else 252,
    )


def tiny_exposures(style: str = "style") -> pd.DataFrame:
    return pd.DataFrame({style: [1.0, -1.0]}, index=["A", "B"])


def test_hand_example_reconciles_everywhere():
    # w = [.5, .5]; exposures cancel: x = [1.0 market, 0.0 style].
    # factor var = 1 * 4e-4 = 4e-4 weekly; specific = 2 * .25 * 5e-5 = 2.5e-5.
    # total = 4.25e-4 weekly -> total vol = sqrt(4.25e-4 * 52) = 14.87%.
    # Sigma w = [4.25e-4, 4.25e-4] (worked in the planning notes), so each
    # position contributes exactly half: shares .5/.5, effective bets = 2.
    result = decompose_portfolio(
        pd.Series({"A": 0.5, "B": 0.5}), tiny_exposures(), tiny_model(), name="hand"
    )
    assert result.total_vol == pytest.approx(np.sqrt(4.25e-4 * 52))
    assert result.factor_vol == pytest.approx(np.sqrt(4e-4 * 52))
    assert result.specific_vol == pytest.approx(np.sqrt(2.5e-5 * 52))
    assert result.factor_share == pytest.approx(4e-4 / 4.25e-4)

    assert result.portfolio_exposures["market"] == pytest.approx(1.0)
    assert result.portfolio_exposures["style"] == pytest.approx(0.0)
    # The style factor contributes nothing when exposures cancel.
    assert result.factor_contributions.loc["style", "variance"] == pytest.approx(0.0)

    shares = result.position_contributions["share"]
    assert shares.loc["A"] == pytest.approx(0.5)
    assert result.effective_bets == pytest.approx(2.0)


def test_partial_investment_shows_in_market_exposure():
    # 60% invested (rest cash): market exposure must read 0.6.
    result = decompose_portfolio(pd.Series({"A": 0.3, "B": 0.3}), tiny_exposures(), tiny_model())
    assert result.portfolio_exposures["market"] == pytest.approx(0.6)


def test_invariants_on_a_random_model():
    # Property test: contributions must reconcile for ANY valid inputs.
    rng = np.random.default_rng(11)
    symbols = [f"S{i}" for i in range(12)]
    names = ["market", *FACTORS]
    a = rng.normal(0, 0.01, (30, len(names)))
    cov = pd.DataFrame(a.T @ a / 30, index=names, columns=names)  # PSD by construction
    model = FactorModel(
        factor_returns=pd.DataFrame(columns=names),
        factor_cov=cov,
        specific_variance=pd.Series(rng.uniform(1e-5, 1e-4, 12), index=symbols),
        r2=pd.Series(dtype=float),
        n_obs=pd.Series(dtype=int),
        periods_per_year=52,
    )
    exposures = pd.DataFrame(
        rng.normal(0, 1, (12, len(FACTORS))), index=symbols, columns=list(FACTORS)
    )
    raw = rng.uniform(0.01, 0.2, 12)
    weights = pd.Series(raw / raw.sum() * 0.9, index=symbols)  # 90% invested

    result = decompose_portfolio(weights, exposures, model)
    # Factor parts sum to factor variance; shares sum to 1.
    assert result.factor_contributions["share"].sum() + result.specific_share == pytest.approx(1.0)
    assert result.position_contributions["share"].sum() == pytest.approx(1.0)
    # total = factor + specific in variance terms.
    assert result.total_vol**2 == pytest.approx(result.factor_vol**2 + result.specific_vol**2)


def test_negative_contribution_is_reported_signed():
    # Correlated factors, opposing exposures: one factor HEDGES. Its
    # contribution must come out negative, not absolute-valued away.
    names = ["market", "style"]
    cov = pd.DataFrame([[4e-4, 1.5e-4], [1.5e-4, 1e-4]], index=names, columns=names)
    model = FactorModel(
        factor_returns=pd.DataFrame(columns=names),
        factor_cov=cov,
        specific_variance=pd.Series({"A": 1e-5, "B": 1e-5}),
        r2=pd.Series(dtype=float),
        n_obs=pd.Series(dtype=int),
        periods_per_year=52,
    )
    exposures = pd.DataFrame({"style": [-1.0, -1.0]}, index=["A", "B"])  # short style
    result = decompose_portfolio(pd.Series({"A": 0.5, "B": 0.5}), exposures, model)
    # x = [1, -1]: style part = x_s * (F x)_s = -1 * (1.5e-4 - 1e-4) < 0.
    assert result.factor_contributions.loc["style", "variance"] < 0
    assert result.factor_contributions.loc["style", "vol"] < 0


def test_short_positions_disable_effective_bets():
    result = decompose_portfolio(pd.Series({"A": 1.3, "B": -0.3}), tiny_exposures(), tiny_model())
    # A short position with negative contribution makes the Herfindahl
    # measure meaningless; None is the honest answer.
    if (result.position_contributions["share"] < 0).any():
        assert result.effective_bets is None


def test_missing_exposure_symbol_refused():
    with pytest.raises(DecompositionError, match=r"\['C'\]"):
        decompose_portfolio(pd.Series({"A": 0.5, "C": 0.5}), tiny_exposures(), tiny_model())


def test_nan_exposure_refused():
    exposures = tiny_exposures()
    exposures.loc["B", "style"] = np.nan
    with pytest.raises(DecompositionError, match=r"NaN.*\['B'\]"):
        decompose_portfolio(pd.Series({"A": 0.5, "B": 0.5}), exposures, tiny_model())


def test_missing_specific_variance_refused():
    model = tiny_model()
    model.specific_variance.loc["B"] = np.nan
    with pytest.raises(DecompositionError, match="specific variance.*\\['B'\\]"):
        decompose_portfolio(pd.Series({"A": 0.5, "B": 0.5}), tiny_exposures(), model)


# ------------------------------------------------------------------- CLI


def test_cli_decompose_end_to_end(tmp_path):
    # Reuse the regression test's seeded 10-symbol database recipe.
    import yaml

    from ballast.data.prices import store_prices
    from tests.test_exposures import put_fundamental
    from tests.test_prices import tidy

    db = tmp_path / "t.duckdb"
    rng = np.random.default_rng(29)
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

    spec = tmp_path / "p.yaml"
    spec.write_text(
        yaml.safe_dump(
            {
                "name": "core",
                "positions": [
                    {"symbol": "S0", "shares": 10},
                    {"symbol": "S1", "shares": 5},
                    {"symbol": "S2", "weight": 0.2},
                ],
                "cash": 500,
            }
        )
    )
    result = runner.invoke(
        app,
        ["decompose", str(spec), "--db", str(db), "--start", "2024-02-01", "--end", "2024-04-30"],
    )
    assert result.exit_code == 0, result.output
    for expected in ("Total risk", "Factor contributions", "market", "momentum", "S0"):
        assert expected in result.output


def test_cli_decompose_thin_universe_fails_loudly(tmp_path):
    import yaml

    from ballast.data.prices import store_prices
    from tests.test_prices import tidy

    db = tmp_path / "t.duckdb"
    rng = np.random.default_rng(31)
    for symbol in ("AAA", "BBB"):  # 2 symbols cannot support 6 coefficients
        closes = (100 * np.cumprod(1 + rng.normal(0.0003, 0.012, 340))).tolist()
        store_prices(tidy(closes, symbol=symbol, start="2023-01-02"), db_path=db)
    spec = tmp_path / "p.yaml"
    spec.write_text(yaml.safe_dump({"name": "p", "positions": [{"symbol": "AAA", "weight": 1.0}]}))
    result = runner.invoke(
        app,
        ["decompose", str(spec), "--db", str(db), "--start", "2024-02-01", "--end", "2024-04-30"],
    )
    assert result.exit_code == 1
    assert "error:" in result.output
