"""Tests for optimize/compare.py and the `ballast optimize compare` command.

Notes
-----
Structure over outcomes: on synthetic random walks nobody should reliably
beat anybody, so the tests pin the RACE's mechanics -- everyone present,
sorted by Sharpe, skips visible with reasons -- rather than pretending to
know which optimizer wins on noise.
"""

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from ballast.cli.app import app
from ballast.optimize.compare import DEFAULT_STRATEGIES, compare_optimizers
from tests.test_prices import tidy

cvxpy = pytest.importorskip("cvxpy")

runner = CliRunner()


def random_walk_returns(n_symbols=4, days=620, seed=41) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = rng.normal(0.0003, 0.012, (days, n_symbols))
    return pd.DataFrame(
        values,
        index=pd.bdate_range("2022-01-03", periods=days),
        columns=[f"S{i}" for i in range(n_symbols)],
    )


def test_full_field_runs_with_a_long_window():
    # window=252 gives CVaR ~12.6 tail scenarios at 95%: everyone races.
    returns = random_walk_returns()
    result = compare_optimizers(returns, window=252, step=63)
    names = {row.name for row in result.rows}
    assert names == set(DEFAULT_STRATEGIES)
    assert result.skipped == ()
    sharpes = [row.sharpe for row in result.rows]
    assert sharpes == sorted(sharpes, reverse=True)  # best first
    assert result.n_windows == len(range(252, 620, 63))


def test_short_window_skips_cvar_visibly():
    # window=120 -> 6 tail scenarios at 95%: CVaR must refuse, and the
    # race must record why instead of hiding the no-show.
    returns = random_walk_returns(days=400)
    result = compare_optimizers(returns, window=120, step=60)
    skipped_names = {name for name, _ in result.skipped}
    assert "min_cvar" in skipped_names
    reason = dict(result.skipped)["min_cvar"]
    assert "tail" in reason
    assert {row.name for row in result.rows} == set(DEFAULT_STRATEGIES) - skipped_names


def test_equal_weight_is_always_in_the_field():
    # The baseline can't fail (no estimation, no solver): if it's missing,
    # the race is meaningless.
    returns = random_walk_returns(days=400)
    result = compare_optimizers(returns, window=120, step=60)
    assert any("1/N" in row.name for row in result.rows)


def test_cli_optimize_compare(tmp_path):
    from ballast.data.prices import store_prices

    db = tmp_path / "t.duckdb"
    rng = np.random.default_rng(43)
    for symbol in ("AAA", "BBB", "CCC"):
        closes = (100 * np.cumprod(1 + rng.normal(0.0003, 0.012, 420))).tolist()
        store_prices(tidy(closes, symbol=symbol), db_path=db)

    result = runner.invoke(
        app,
        ["optimize", "compare", "--db", str(db), "--window", "252", "--step", "63"],
    )
    assert result.exit_code == 0, result.output
    for expected in ("equal_weight", "min_variance", "hrp", "Sharpe"):
        assert expected in result.output


def test_cli_optimize_compare_empty_db_exits_one(tmp_path):
    result = runner.invoke(app, ["optimize", "compare", "--db", str(tmp_path / "empty.duckdb")])
    assert result.exit_code == 1
    assert "at least 2 symbols" in result.output
