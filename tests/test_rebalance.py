"""Tests for portfolio/rebalance.py and the `ballast rebalance` command.

Notes
-----
The hand plan uses the decompose demo's numbers (NAV $12,500, 64/20/16)
so the arithmetic is checkable on paper. Band and min-trade tests verify
RESTRAINT -- the feature of this module is the trades it refuses to make.
"""

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from ballast.cli.app import app
from ballast.portfolio.rebalance import RebalanceError, plan_rebalance
from ballast.portfolio.spec import ResolvedPortfolio

runner = CliRunner()


def book(weights: dict, cash_weight: float, nav: float | None = 12_500.0) -> ResolvedPortfolio:
    return ResolvedPortfolio(name="p", weights=weights, cash_weight=cash_weight, nav=nav)


CURRENT = book({"A": 0.64, "B": 0.20}, cash_weight=0.16)


# --------------------------------------------------------------- hand plan


def test_hand_plan():
    # Targets 40/40: A drifts -0.24 (sell $3,000), B +0.20 (buy $2,500).
    # At 10 bps per side: costs $3.00 and $2.50. Cash after: 20%.
    plan = plan_rebalance(CURRENT, pd.Series({"A": 0.4, "B": 0.4}), band=0.05, cost_bps=10.0)
    assert len(plan.trades) == 2
    first, second = plan.trades  # sorted by |dollars|: the sell is larger
    assert first.symbol == "A" and first.dollars == pytest.approx(-3000.0)
    assert second.symbol == "B" and second.dollars == pytest.approx(2500.0)
    assert first.est_cost == pytest.approx(3.0)
    assert plan.total_traded == pytest.approx(5500.0)
    assert plan.total_cost == pytest.approx(5.5)
    assert plan.cash_weight_after == pytest.approx(0.20)


def test_band_restraint_is_visible():
    # A is 2 points off target: inside the 5-point band, left alone -- and
    # REPORTED as left alone, drift included.
    plan = plan_rebalance(CURRENT, pd.Series({"A": 0.66, "B": 0.20}), band=0.05)
    assert plan.trades == ()
    assert ("A", pytest.approx(0.02)) in [(s, d) for s, d in plan.skipped]
    assert plan.cash_weight_after == pytest.approx(0.16)  # book unchanged


def test_min_trade_filters_silly_dollars():
    # 6-point drift clears the band but $7.50 of trading doesn't clear a
    # $100 minimum on a tiny account.
    small = book({"A": 0.64, "B": 0.20}, cash_weight=0.16, nav=125.0)
    plan = plan_rebalance(
        small, pd.Series({"A": 0.58, "B": 0.20}), band=0.05, min_trade_dollars=100
    )
    assert plan.trades == ()
    assert plan.skipped[0][0] == "A"


def test_exits_and_entries_use_the_union_universe():
    current = book({"A": 0.5}, cash_weight=0.5, nav=1000.0)
    plan = plan_rebalance(current, pd.Series({"B": 0.5}), band=0.05)
    by_symbol = {t.symbol: t for t in plan.trades}
    assert by_symbol["A"].dollars == pytest.approx(-500.0)  # full exit
    assert by_symbol["B"].dollars == pytest.approx(500.0)  # new position


def test_scale_free_book_refused():
    scale_free = book({"A": 0.6, "B": 0.4}, cash_weight=0.0, nav=None)
    with pytest.raises(RebalanceError, match="scale-free"):
        plan_rebalance(scale_free, pd.Series({"A": 0.5, "B": 0.5}))


def test_bad_targets_named():
    with pytest.raises(RebalanceError, match="sum"):
        plan_rebalance(CURRENT, pd.Series({"A": 0.7, "B": 0.7}))
    with pytest.raises(RebalanceError, match="long-only"):
        plan_rebalance(CURRENT, pd.Series({"A": -0.2, "B": 0.5}))
    with pytest.raises(RebalanceError, match="band"):
        plan_rebalance(CURRENT, pd.Series({"A": 0.5}), band=1.5)


# ------------------------------------------------------------------- CLI


def test_cli_rebalance_end_to_end(tmp_path):
    from ballast.data.prices import store_prices
    from tests.test_prices import tidy

    db = tmp_path / "t.duckdb"
    store_prices(tidy([100.0, 100.0], symbol="AAA"), db_path=db)
    store_prices(tidy([50.0, 50.0], symbol="BBB"), db_path=db)

    current = tmp_path / "current.yaml"
    current.write_text(
        yaml.safe_dump(
            {
                "name": "book",
                "positions": [{"symbol": "AAA", "shares": 80}],  # $8,000
                "cash": 2000,
            }
        )
    )
    target = tmp_path / "target.yaml"
    target.write_text(
        yaml.safe_dump(
            {
                "name": "target",
                "positions": [
                    {"symbol": "AAA", "weight": 0.5},
                    {"symbol": "BBB", "weight": 0.3},
                ],
            }
        )
    )
    result = runner.invoke(
        app, ["rebalance", str(current), "--target", str(target), "--db", str(db)]
    )
    assert result.exit_code == 0, result.output
    # AAA is 80% -> 50%: SELL $3,000. BBB 0% -> 30%: BUY $3,000.
    assert "SELL" in result.output and "BUY" in result.output
    assert "3,000" in result.output


def test_cli_rebalance_scale_free_exits_one(tmp_path):
    from ballast.data.prices import store_prices
    from tests.test_prices import tidy

    db = tmp_path / "t.duckdb"
    store_prices(tidy([100.0, 100.0], symbol="AAA"), db_path=db)
    current = tmp_path / "current.yaml"
    current.write_text(
        yaml.safe_dump({"name": "p", "positions": [{"symbol": "AAA", "weight": 1.0}]})
    )
    result = runner.invoke(
        app, ["rebalance", str(current), "--target", str(current), "--db", str(db)]
    )
    assert result.exit_code == 1
    assert "scale-free" in result.output
