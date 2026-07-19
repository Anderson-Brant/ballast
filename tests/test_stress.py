"""Tests for stress/scenarios.py and the `ballast stress` command.

Notes
-----
The replay hand case pins prices exactly (flat 100 through the window
start, flat 80 from before the window end) so the -20% return is exact,
not approximate. Coverage strictness -- a symbol that starts trading
mid-crisis must be an error, not a hole -- is the test that matters most.
"""

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from ballast.cli.app import app
from ballast.data.prices import store_prices
from ballast.stress.scenarios import SCENARIOS, StressError, factor_shock, replay_scenario
from tests.test_prices import tidy

runner = CliRunner()


def two_phase_closes(start: str, days: int, before: float, after: float, switch: str):
    """Prices pinned at `before` until switch date, `after` from then on."""
    dates = pd.bdate_range(start, periods=days)
    return [before if d < pd.Timestamp(switch) else after for d in dates], dates


def seed_covid_db(tmp_path):
    db = tmp_path / "t.duckdb"
    # DOWN: 100 flat until 2020-03-02, then 80 flat -> exactly -20% over
    # the covid2020 window (first stored bar >= 02-19 is 100, last <= 03-23
    # is 80). UP: 50 -> 55, +10% by the same construction.
    for symbol, before, after in (("DOWN", 100.0, 80.0), ("UP", 50.0, 55.0)):
        closes, _ = two_phase_closes("2020-02-03", 40, before, after, "2020-03-02")
        store_prices(tidy(closes, symbol=symbol, start="2020-02-03"), db_path=db)
    return db


# ------------------------------------------------------------------ replay


def test_replay_hand_example(tmp_path):
    db = seed_covid_db(tmp_path)
    weights = pd.Series({"DOWN": 0.5, "UP": 0.3})  # 20% cash
    result = replay_scenario(weights, "covid2020", db_path=db)
    assert result.position_returns["DOWN"] == pytest.approx(-0.20)
    assert result.position_returns["UP"] == pytest.approx(0.10)
    # P&L: .5*(-.2) + .3*(.1) = -0.07; cash sat out.
    assert result.portfolio_return == pytest.approx(-0.07)
    assert result.invested_fraction == pytest.approx(0.8)
    # Worst driver first.
    assert result.contributions.index[0] == "DOWN"


def test_replay_refuses_uncovered_symbols(tmp_path):
    db = seed_covid_db(tmp_path)
    # LATE starts trading mid-crisis: raw replay must refuse, by name.
    closes = [10.0] * 15
    store_prices(tidy(closes, symbol="LATE", start="2020-03-05"), db_path=db)
    weights = pd.Series({"DOWN": 0.5, "LATE": 0.5})
    with pytest.raises(StressError, match=r"\['LATE'\]"):
        replay_scenario(weights, "covid2020", db_path=db)


def test_unknown_scenario_lists_the_menu(tmp_path):
    db = seed_covid_db(tmp_path)
    with pytest.raises(StressError, match="gfc2008"):
        replay_scenario(pd.Series({"DOWN": 1.0}), "dotcom2000", db_path=db)
    assert set(SCENARIOS) == {"gfc2008", "covid2020", "rates2022"}


# ------------------------------------------------------------------- shocks


def test_factor_shock_hand_example():
    # x: market = invested fraction 0.6; momentum = .3*1 + .3*(-1) = 0;
    # value = .3*2 + .3*2 = 1.2. Market -20% alone: 0.6 * -0.2 = -12%.
    weights = pd.Series({"A": 0.3, "B": 0.3})
    exposures = pd.DataFrame({"momentum": [1.0, -1.0], "value": [2.0, 2.0]}, index=["A", "B"])
    result = factor_shock(weights, exposures, {"market": -0.20})
    assert result.exposures["market"] == pytest.approx(0.6)
    assert result.exposures["momentum"] == pytest.approx(0.0)
    assert result.exposures["value"] == pytest.approx(1.2)
    assert result.portfolio_return == pytest.approx(-0.12)

    # A momentum crash does nothing to a momentum-neutral book...
    neutral = factor_shock(weights, exposures, {"momentum": -0.10})
    assert neutral.portfolio_return == pytest.approx(0.0)
    # ...and a value shock scales by the 1.2 exposure.
    value_hit = factor_shock(weights, exposures, {"value": -0.05})
    assert value_hit.portfolio_return == pytest.approx(-0.06)


def test_factor_shock_guards():
    weights = pd.Series({"A": 0.5})
    exposures = pd.DataFrame({"momentum": [1.0]}, index=["A"])
    with pytest.raises(StressError, match="unknown factor"):
        factor_shock(weights, exposures, {"vibes": -0.1})
    with pytest.raises(StressError, match="no shocks"):
        factor_shock(weights, exposures, {})
    holey = exposures.copy()
    holey.loc["A", "momentum"] = np.nan
    with pytest.raises(StressError, match=r"NaN.*\['A'\]"):
        factor_shock(weights, holey, {"momentum": -0.1})


# ------------------------------------------------------------------- CLI


def write_spec(tmp_path, positions, cash=0):
    spec = tmp_path / "p.yaml"
    spec.write_text(yaml.safe_dump({"name": "p", "positions": positions, "cash": cash}))
    return spec


def test_cli_stress_scenario(tmp_path):
    db = seed_covid_db(tmp_path)
    spec = write_spec(
        tmp_path, [{"symbol": "DOWN", "weight": 0.5}, {"symbol": "UP", "weight": 0.3}]
    )
    result = runner.invoke(app, ["stress", str(spec), "--scenario", "covid2020", "--db", str(db)])
    assert result.exit_code == 0, result.output
    for expected in ("covid2020", "Estimated P&L", "DOWN"):
        assert expected in result.output


def test_cli_stress_requires_exactly_one_mode(tmp_path):
    db = seed_covid_db(tmp_path)
    spec = write_spec(tmp_path, [{"symbol": "DOWN", "weight": 1.0}])
    neither = runner.invoke(app, ["stress", str(spec), "--db", str(db)])
    assert neither.exit_code == 1
    both = runner.invoke(
        app,
        ["stress", str(spec), "--scenario", "covid2020", "--shock", "market=-0.2", "--db", str(db)],
    )
    assert both.exit_code == 1


def test_cli_stress_shock(tmp_path):
    # Shock mode needs exposures: reuse the seeded-fundamentals recipe.
    from tests.test_exposures import put_fundamental

    db = tmp_path / "t.duckdb"
    rng = np.random.default_rng(47)
    for symbol in [f"S{i}" for i in range(6)]:
        closes = (100 * np.cumprod(1 + rng.normal(0.0003, 0.012, 320))).tolist()
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

    spec = write_spec(tmp_path, [{"symbol": "S0", "weight": 0.4}, {"symbol": "S1", "weight": 0.4}])
    result = runner.invoke(
        app,
        [
            "stress",
            str(spec),
            "--shock",
            "market=-0.20",
            "--shock",
            "momentum=-0.10",
            "--db",
            str(db),
        ],
    )
    assert result.exit_code == 0, result.output
    for expected in ("Hypothetical", "market", "momentum"):
        assert expected in result.output


def test_cli_stress_bad_shock_syntax(tmp_path):
    db = seed_covid_db(tmp_path)
    spec = write_spec(tmp_path, [{"symbol": "DOWN", "weight": 1.0}])
    result = runner.invoke(app, ["stress", str(spec), "--shock", "market:0.2", "--db", str(db)])
    assert result.exit_code == 1
    assert "factor=value" in result.output
