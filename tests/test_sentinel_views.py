"""Tests for the Sentinel views bridge: scores file -> views -> target book.

Notes
-----
The direction test is the contract: the top-scored symbol's view sits
ABOVE its prior and the bottom-scored below, scaled by IC and vol. The
CLI test closes the whole loop -- scores.csv in, target.yaml out, and the
target feeds `ballast rebalance` without complaint. That round trip is
the two-project story working end to end.
"""

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from ballast.cli.app import app
from ballast.data.sentinel_views import ScoresError, load_scores
from ballast.optimize.black_litterman import implied_returns, views_from_scores
from ballast.optimize.mvo import OptimizationError
from tests.test_prices import tidy

cvxpy = pytest.importorskip("cvxpy")

runner = CliRunner()


def cov_frame(matrix, symbols) -> pd.DataFrame:
    return pd.DataFrame(matrix, index=symbols, columns=symbols)


# ------------------------------------------------------------- scores file


def test_load_scores_happy_path(tmp_path):
    path = tmp_path / "scores.csv"
    path.write_text("symbol,score\nnvda,9.1\nV,8.9\nADBE,8.4\n")
    scores = load_scores(path)
    assert list(scores.index) == ["NVDA", "V", "ADBE"]  # upper-cased
    assert scores["NVDA"] == pytest.approx(9.1)


def test_load_scores_guards(tmp_path):
    missing = tmp_path / "nope.csv"
    with pytest.raises(ScoresError, match="not found"):
        load_scores(missing)

    bad_header = tmp_path / "bad.csv"
    bad_header.write_text("ticker,rank\nA,1\nB,2\n")
    with pytest.raises(ScoresError, match="symbol,score"):
        load_scores(bad_header)

    dupes = tmp_path / "dupes.csv"
    dupes.write_text("symbol,score\nA,1\nA,2\n")
    with pytest.raises(ScoresError, match="duplicate"):
        load_scores(dupes)

    flat = tmp_path / "flat.csv"
    flat.write_text("symbol,score\nA,5\nB,5\n")
    with pytest.raises(ScoresError, match="identical"):
        load_scores(flat)


# ---------------------------------------------------------- score -> views


def test_views_direction_and_scaling():
    # Equal vols and market weights: the prior is symmetric, so the views
    # differ ONLY through the scores. z = [+1, -1] for two symbols, so the
    # views must straddle the prior by exactly ic * vol.
    symbols = ["HIGH", "LOW"]
    cov = cov_frame(np.diag([0.04, 0.04]).tolist(), symbols)  # vol 20% each
    market = pd.Series({"HIGH": 0.5, "LOW": 0.5})
    prior = implied_returns(cov, market)

    views = views_from_scores(
        pd.Series({"HIGH": 9.0, "LOW": 7.0}), cov, market, ic=0.05, confidence=0.4
    )
    by_symbol = {next(iter(v.assets)): v for v in views}
    assert by_symbol["HIGH"].value == pytest.approx(prior["HIGH"] + 0.05 * 0.2)
    assert by_symbol["LOW"].value == pytest.approx(prior["LOW"] - 0.05 * 0.2)
    assert all(v.confidence == 0.4 for v in views)


def test_unscored_symbols_get_no_view():
    symbols = ["A", "B", "C"]
    cov = cov_frame(np.diag([0.04, 0.04, 0.04]).tolist(), symbols)
    market = pd.Series(1 / 3, index=symbols)
    views = views_from_scores(pd.Series({"A": 2.0, "B": 1.0}), cov, market)
    viewed = {next(iter(v.assets)) for v in views}
    assert viewed == {"A", "B"}  # C stays at the prior, by design


def test_views_guards():
    cov = cov_frame(np.diag([0.04, 0.04]).tolist(), ["A", "B"])
    market = pd.Series({"A": 0.5, "B": 0.5})
    with pytest.raises(OptimizationError, match=r"\['Z'\]"):
        views_from_scores(pd.Series({"A": 1.0, "Z": 2.0}), cov, market)
    with pytest.raises(OptimizationError, match="identical"):
        views_from_scores(pd.Series({"A": 1.0, "B": 1.0}), cov, market)
    with pytest.raises(OptimizationError, match="ic"):
        views_from_scores(pd.Series({"A": 1.0, "B": 2.0}), cov, market, ic=0.0)


# --------------------------------------------------------- the full circle


def seed_price_db(tmp_path, symbols=("AAA", "BBB", "CCC", "DDD"), days=300):
    db = tmp_path / "t.duckdb"
    rng = np.random.default_rng(53)
    for symbol in symbols:
        closes = (100 * np.cumprod(1 + rng.normal(0.0003, 0.012, days))).tolist()
        # start dates in 2025 so the default 1-year window covers them
        from ballast.data.prices import store_prices

        store_prices(tidy(closes, symbol=symbol, start="2025-06-02"), db_path=db)
    return db


def test_cli_views_to_target_to_rebalance_round_trip(tmp_path):
    # THE two-project story: scores in, target out, rebalance consumes it.
    db = seed_price_db(tmp_path)
    scores = tmp_path / "scores.csv"
    scores.write_text("symbol,score\nAAA,9.0\nBBB,8.0\nCCC,7.0\nDDD,6.0\n")
    target = tmp_path / "target.yaml"

    result = runner.invoke(
        app,
        ["optimize", "views", str(scores), "--db", str(db), "--output", str(target)],
    )
    assert result.exit_code == 0, result.output
    assert "Black-Litterman target" in result.output
    assert "equal-weight prior" in result.output  # no fundamentals: stated
    assert target.is_file()

    # The written target is a valid spec on its own terms...
    from ballast.portfolio.spec import load_portfolio

    book = load_portfolio(target)
    assert sum(p.weight for p in book.positions) == pytest.approx(1.0, abs=1e-6)

    # ...and rebalance accepts it against a current book.
    current = tmp_path / "current.yaml"
    current.write_text(
        yaml.safe_dump(
            {"name": "book", "positions": [{"symbol": "AAA", "shares": 10}], "cash": 500}
        )
    )
    rebalance = runner.invoke(
        app, ["rebalance", str(current), "--target", str(target), "--db", str(db)]
    )
    assert rebalance.exit_code == 0, rebalance.output


def test_cli_views_tilts_toward_the_top_score(tmp_path):
    db = seed_price_db(tmp_path)
    scores = tmp_path / "scores.csv"
    scores.write_text("symbol,score\nAAA,9.0\nBBB,8.0\nCCC,7.0\nDDD,6.0\n")
    target = tmp_path / "target.yaml"
    result = runner.invoke(
        app,
        ["optimize", "views", str(scores), "--db", str(db), "--output", str(target), "--ic", "0.2"],
    )
    assert result.exit_code == 0, result.output
    weights = {p["symbol"]: p["weight"] for p in yaml.safe_load(target.read_text())["positions"]}
    # Top score out-weighs bottom score (equal-weight prior, so the tilt
    # is pure view; DDD may be squeezed out entirely by long-only).
    assert weights["AAA"] > weights.get("DDD", 0.0)


def test_cli_views_missing_prices_exits_one(tmp_path):
    db = seed_price_db(tmp_path, symbols=("AAA",), days=300)
    scores = tmp_path / "scores.csv"
    scores.write_text("symbol,score\nAAA,9.0\nZZZ,8.0\n")
    result = runner.invoke(app, ["optimize", "views", str(scores), "--db", str(db)])
    assert result.exit_code == 1
    assert "ZZZ" in result.output
