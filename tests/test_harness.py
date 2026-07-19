"""Tests for covariance/harness.py and the `ballast cov compare` command.

Notes
-----
The ground-truth check matters most: on synthetic data drawn from a known
covariance (conftest fixture), the min-variance portfolio built from ANY
decent estimator must realize lower vol than 1/N -- that's the entire
premise of the harness. If that ever fails on the fixed seed, the engine
or an estimator broke.
"""

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from ballast.cli.app import app
from ballast.covariance.harness import (
    DEFAULT_ESTIMATORS,
    compare_estimators,
    min_variance_weights,
)
from tests.test_prices import tidy

runner = CliRunner()


# ------------------------------------------------------ min-variance math


def test_min_variance_hand_example():
    # Diagonal covariance diag(1, 4): w ~ S^-1 1 = [1, .25] -> [0.8, 0.2].
    cov = pd.DataFrame([[1.0, 0.0], [0.0, 4.0]], index=["A", "B"], columns=["A", "B"])
    w = min_variance_weights(cov)
    assert w["A"] == pytest.approx(0.8)
    assert w["B"] == pytest.approx(0.2)
    assert w.sum() == pytest.approx(1.0)


def test_min_variance_survives_singular_matrix():
    # Rank-1 matrix (perfectly correlated assets): solve() would blow up;
    # the pinv fallback must return finite unit-sum weights instead.
    cov = pd.DataFrame([[1.0, 2.0], [2.0, 4.0]], index=["A", "B"], columns=["A", "B"])
    w = min_variance_weights(cov)
    assert np.isfinite(w.to_numpy()).all()
    assert w.sum() == pytest.approx(1.0)


# ------------------------------------------------------------- the race


@pytest.fixture
def race(synthetic_returns):
    # 1500 days, quarterly-ish rebalance: ~20 windows, fast but meaningful.
    return compare_estimators(synthetic_returns.iloc[:1500], window=252, step=63, cost_bps=2.0)


def test_race_includes_all_estimators_plus_baseline(race):
    names = [row.name for row in race.rows]
    for expected in DEFAULT_ESTIMATORS:
        assert expected in names
    assert "equal_weight (1/N)" in names
    assert len(race.rows) == len(DEFAULT_ESTIMATORS) + 1


def test_race_rows_sorted_by_realized_vol(race):
    vols = [row.ann_vol for row in race.rows]
    assert vols == sorted(vols)


def test_min_variance_beats_one_over_n_on_ground_truth(race):
    # The premise check. With a known covariance generating the data, the
    # GMV portfolio (heavy in the 1%-vol asset) must be calmer than 1/N.
    by_name = {row.name: row for row in race.rows}
    baseline = by_name["equal_weight (1/N)"]
    assert by_name["sample"].ann_vol < baseline.ann_vol
    assert by_name["ledoit_wolf"].ann_vol < baseline.ann_vol


def test_diagnostics_present_where_they_belong(race):
    by_name = {row.name: row for row in race.rows}
    assert by_name["equal_weight (1/N)"].avg_condition is None  # no matrix
    for name in DEFAULT_ESTIMATORS:
        assert by_name[name].avg_condition >= 1.0  # cond >= 1 by definition


def test_window_count_matches_the_calendar(synthetic_returns):
    data = synthetic_returns.iloc[:600]
    result = compare_estimators(data, window=252, step=63)
    assert result.n_windows == len(range(252, 600, 63))


# ------------------------------------------------------------------- CLI


def seed_random_walks(tmp_path, symbols=("AAA", "BBB", "CCC"), days=40):
    """Store independent random walks so the covariance is well-behaved."""
    db = tmp_path / "t.duckdb"
    rng = np.random.default_rng(11)
    for symbol in symbols:
        closes = (100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, days))).tolist()
        from ballast.data.prices import store_prices

        store_prices(tidy(closes, symbol=symbol), db_path=db)
    return db


def test_cli_cov_compare(tmp_path):
    db = seed_random_walks(tmp_path)
    result = runner.invoke(
        app,
        ["cov", "compare", "AAA", "BBB", "CCC", "--db", str(db), "--window", "10", "--step", "5"],
    )
    assert result.exit_code == 0, result.output
    assert "equal_weight" in result.output
    assert "Realized vol" in result.output


def test_cli_cov_compare_defaults_to_all_symbols(tmp_path):
    db = seed_random_walks(tmp_path)
    result = runner.invoke(
        app, ["cov", "compare", "--db", str(db), "--window", "10", "--step", "5"]
    )
    assert result.exit_code == 0, result.output
    assert "3 symbols" in result.output


def test_cli_cov_compare_empty_db_exits_one(tmp_path):
    result = runner.invoke(app, ["cov", "compare", "--db", str(tmp_path / "empty.duckdb")])
    assert result.exit_code == 1
    assert "at least 2 symbols" in result.output
