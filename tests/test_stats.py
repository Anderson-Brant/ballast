"""Tests for portfolio/stats.py and the v0.1.0 CLI commands.

Notes
-----
Math tests use hand-computed values (worked in the comments). CLI tests run
the real command stack -- spec file on disk, seeded DuckDB, Typer runner --
with only the network fetch monkeypatched.
"""

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from ballast.cli.app import app
from ballast.portfolio.spec import Portfolio, Position, ResolvedPortfolio, resolve_weights
from ballast.portfolio.stats import (
    StatsError,
    annualized_vol,
    beta,
    blended_returns,
    cagr,
    compute_stats,
    max_drawdown,
)
from tests.test_prices import make_yf_frame, seed

runner = CliRunner()


def series(values: list[float], start: str = "2024-01-01") -> pd.Series:
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)))


# ---------------------------------------------------------------- the math


def test_annualized_vol_hand_example():
    # values +/-0.01, mean 0: sum of squares = 4e-4, ddof=1 -> /3,
    # sqrt = 0.01154701, x sqrt(252) = 0.18330
    v = annualized_vol(series([0.01, -0.01, 0.01, -0.01]))
    assert v == pytest.approx(0.18330, abs=1e-4)


def test_cagr_definitional():
    # 252 days of exactly 1%: total growth 1.01^252, window is exactly one
    # year, so CAGR is just the total growth minus 1.
    c = cagr(series([0.01] * 252))
    assert c == pytest.approx(1.01**252 - 1)


def test_max_drawdown_hand_example():
    # equity: 1.10 -> 0.55 -> 0.66; peak 1.10, trough 0.55 -> -50%
    dd = max_drawdown(series([0.10, -0.50, 0.20]))
    assert dd == pytest.approx(-0.50)


def test_max_drawdown_is_zero_when_only_up():
    assert max_drawdown(series([0.01, 0.02, 0.03])) == pytest.approx(0.0)


def test_beta_of_a_leveraged_copy_is_two():
    b = series([0.01, -0.02, 0.015, 0.005])
    p = b * 2  # exactly twice the benchmark, so beta must be exactly 2
    assert beta(p, b) == pytest.approx(2.0)


def test_beta_uses_shared_dates_only():
    b = series([0.01, -0.02, 0.015, 0.005])
    p = (b * 2).iloc[1:]  # portfolio missing the first day
    assert beta(p, b) == pytest.approx(2.0)  # alignment, then the same answer


def test_beta_flat_benchmark_is_none():
    # var(benchmark) = 0 -> beta undefined, and None is the honest answer.
    assert beta(series([0.01, 0.02]), series([0.0, 0.0])) is None


def test_beta_no_overlap_is_none():
    p = series([0.01, 0.02], start="2020-01-01")
    b = series([0.01, 0.02], start="2024-01-01")
    assert beta(p, b) is None


def test_blended_returns_hand_example():
    returns = pd.DataFrame(
        {"AAA": [0.10, 0.00], "BBB": [-0.10, 0.05]},
        index=pd.bdate_range("2024-01-01", periods=2),
    )
    resolved = ResolvedPortfolio(
        name="p", weights={"AAA": 0.6, "BBB": 0.2}, cash_weight=0.2, nav=1000.0
    )
    blended = blended_returns(returns, resolved)
    # day 1: 0.6*0.10 + 0.2*(-0.10) + 0.2*0 = 0.04 -- cash drag included
    assert blended.iloc[0] == pytest.approx(0.04)
    assert blended.iloc[1] == pytest.approx(0.2 * 0.05)


def test_blended_cash_only_raises():
    resolved = ResolvedPortfolio(name="p", weights={}, cash_weight=1.0, nav=500.0)
    with pytest.raises(StatsError, match="only cash"):
        blended_returns(pd.DataFrame(), resolved)


def test_blended_missing_column_raises():
    returns = pd.DataFrame({"AAA": [0.01, 0.02]}, index=pd.bdate_range("2024-01-01", periods=2))
    resolved = ResolvedPortfolio(
        name="p", weights={"AAA": 0.5, "GONE": 0.5}, cash_weight=0.0, nav=None
    )
    with pytest.raises(StatsError, match=r"\['GONE'\]"):
        blended_returns(returns, resolved)


def test_compute_stats_assembles_everything():
    returns = pd.DataFrame(
        {"AAA": [0.01, -0.01, 0.02]}, index=pd.bdate_range("2024-01-01", periods=3)
    )
    portfolio = Portfolio(name="one", positions=(Position("AAA", weight=1.0),))
    resolved = resolve_weights(portfolio, {})
    result = compute_stats(resolved, returns, None, "SPY")
    assert result.n_days == 3
    assert result.beta is None  # no benchmark supplied
    assert result.nav is None  # weights-only spec is scale-free
    assert result.weights == {"AAA": 1.0}


# ------------------------------------------------------------------ the CLI


def write_spec(tmp_path, data: dict):
    path = tmp_path / "portfolio.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def seeded_db(tmp_path):
    """A DB with two portfolio symbols and SPY, five business days each."""
    db = tmp_path / "t.duckdb"
    seed(db, "AAA", [100.0, 101.0, 102.0, 101.5, 103.0])
    seed(db, "BBB", [50.0, 50.5, 51.0, 50.8, 51.5])
    seed(db, "SPY", [470.0, 471.0, 473.0, 472.0, 475.0])
    return db


def test_cli_stats_end_to_end(tmp_path):
    db = seeded_db(tmp_path)
    spec = write_spec(
        tmp_path,
        {
            "name": "core",
            "positions": [{"symbol": "AAA", "shares": 10}, {"symbol": "BBB", "weight": 0.2}],
            "cash": 100,
        },
    )
    result = runner.invoke(app, ["stats", str(spec), "--db", str(db)])
    assert result.exit_code == 0, result.output
    for expected in ("CAGR", "Ann. vol", "Max drawdown", "Beta vs SPY", "AAA", "cash"):
        assert expected in result.output


def test_cli_stats_missing_benchmark_degrades(tmp_path):
    db = seeded_db(tmp_path)
    spec = write_spec(tmp_path, {"name": "p", "positions": [{"symbol": "AAA", "weight": 1.0}]})
    result = runner.invoke(app, ["stats", str(spec), "--db", str(db), "--benchmark", "ZZZ"])
    assert result.exit_code == 0, result.output  # stats still print...
    assert "no ZZZ data" in result.output  # ...with beta reported as absent


def test_cli_stats_bad_spec_exits_one(tmp_path):
    db = seeded_db(tmp_path)
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: p\npositions:\n  - {symbol: AAA, shares: 1, weight: 0.5}\n")
    result = runner.invoke(app, ["stats", str(bad), "--db", str(db)])
    assert result.exit_code == 1
    assert "error:" in result.output


def test_cli_stats_cash_only_exits_one(tmp_path):
    db = seeded_db(tmp_path)
    spec = write_spec(tmp_path, {"name": "p", "positions": [], "cash": 500})
    result = runner.invoke(app, ["stats", str(spec), "--db", str(db)])
    assert result.exit_code == 1
    assert "only cash" in result.output


def test_cli_ingest(tmp_path, monkeypatch):
    def fake_fetch(symbol, start):
        return make_yf_frame([100.0, 101.0, 102.0], symbol=symbol)

    monkeypatch.setattr("ballast.data.prices._fetch_yfinance", fake_fetch)
    db = tmp_path / "t.duckdb"
    result = runner.invoke(app, ["ingest", "spy", "--db", str(db)])
    assert result.exit_code == 0, result.output
    assert "wrote 3 rows" in result.output


def test_cli_import_sentinel(tmp_path):
    from tests.test_sentinel_import import make_sentinel_db

    src = make_sentinel_db(tmp_path / "sentinel.duckdb")
    db = tmp_path / "ballast.duckdb"
    result = runner.invoke(app, ["import-sentinel", str(src), "--db", str(db)])
    assert result.exit_code == 0, result.output
    assert "imported 6 new rows" in result.output


def test_cli_import_sentinel_missing_file_exits_one(tmp_path):
    result = runner.invoke(
        app, ["import-sentinel", str(tmp_path / "nope.duckdb"), "--db", str(tmp_path / "b.duckdb")]
    )
    assert result.exit_code == 1
    assert "error:" in result.output
