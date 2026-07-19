"""Tests for optimize/black_litterman.py.

Notes
-----
BL's defining behaviors, each hand-checkable: no views returns the prior
exactly; a CERTAIN view is hit exactly; views propagate through
correlations (the spillover test -- the reason BL beats naive forecast
plugging); confidence interpolates monotonically between prior and view.
"""

import numpy as np
import pandas as pd
import pytest

from ballast.optimize.black_litterman import (
    absolute_view,
    black_litterman_returns,
    implied_returns,
    relative_view,
)
from ballast.optimize.mvo import OptimizationError, mvo_weights

cvxpy = pytest.importorskip("cvxpy")  # only the round-trip test needs it


def cov_frame(matrix, symbols) -> pd.DataFrame:
    return pd.DataFrame(matrix, index=symbols, columns=symbols)


DIAG = cov_frame([[0.04, 0.0], [0.0, 0.01]], ["A", "B"])
EQUAL_W = pd.Series({"A": 0.5, "B": 0.5})


# ---------------------------------------------------------------- the prior


def test_implied_returns_hand_example():
    # pi = delta * Sigma * w: diag(0.04, 0.01), w=[.5,.5], delta=2
    # -> pi = [2*0.04*0.5, 2*0.01*0.5] = [0.04, 0.01]. Riskier asset must
    # promise more return to justify its market weight.
    pi = implied_returns(DIAG, EQUAL_W, risk_aversion=2.0)
    assert pi["A"] == pytest.approx(0.04)
    assert pi["B"] == pytest.approx(0.01)


def test_no_views_returns_the_prior_exactly():
    prior = implied_returns(DIAG, EQUAL_W)
    posterior = black_litterman_returns(DIAG, EQUAL_W, views=[])
    assert posterior.equals(prior)


def test_prior_round_trips_through_mvo_to_market_weights():
    # The self-consistency that defines reverse optimization: optimize on
    # the implied returns and the market portfolio comes back out.
    rng = np.random.default_rng(31)
    a = rng.normal(0, 0.01, (120, 4))
    symbols = ["A", "B", "C", "D"]
    cov = cov_frame(a.T @ a / 120, symbols)
    market = pd.Series([0.4, 0.3, 0.2, 0.1], index=symbols)
    pi = implied_returns(cov, market, risk_aversion=3.0)
    w = mvo_weights(cov, expected_returns=pi, risk_aversion=3.0, long_only=False)
    for s in symbols:
        assert w[s] == pytest.approx(market[s], abs=1e-5)


# ------------------------------------------------------------------- views


def test_certain_view_is_hit_exactly():
    # confidence=1 -> omega=0 -> the posterior must satisfy the view.
    posterior = black_litterman_returns(
        DIAG, EQUAL_W, views=[absolute_view("A", 0.10, confidence=1.0)]
    )
    assert posterior["A"] == pytest.approx(0.10)
    # Uncorrelated B is untouched: no channel for the view to travel.
    assert posterior["B"] == pytest.approx(implied_returns(DIAG, EQUAL_W)["B"])


def test_views_propagate_through_correlations():
    # THE reason BL exists. Bullish view on A only; B is 60%-correlated.
    # The posterior for B must rise too -- "tech is cheap" expressed through
    # one name lifts the neighborhood instead of shorting it.
    s = 0.04
    cov = cov_frame([[s, 0.6 * s], [0.6 * s, s]], ["A", "B"])
    prior = implied_returns(cov, EQUAL_W)
    posterior = black_litterman_returns(
        cov, EQUAL_W, views=[absolute_view("A", prior["A"] + 0.05, confidence=0.8)]
    )
    assert posterior["A"] > prior["A"]
    assert posterior["B"] > prior["B"]  # the spillover


def test_confidence_interpolates_monotonically():
    prior = implied_returns(DIAG, EQUAL_W)
    target = prior["A"] + 0.06
    gaps = []
    for confidence in (0.2, 0.5, 0.9):
        posterior = black_litterman_returns(
            DIAG, EQUAL_W, views=[absolute_view("A", target, confidence)]
        )
        gaps.append(abs(target - posterior["A"]))
    assert gaps[0] > gaps[1] > gaps[2]  # more confidence, closer to the view


def test_relative_view_widens_the_spread():
    prior = implied_returns(DIAG, EQUAL_W)
    prior_spread = prior["A"] - prior["B"]
    posterior = black_litterman_returns(
        DIAG, EQUAL_W, views=[relative_view("A", "B", prior_spread + 0.04, confidence=0.9)]
    )
    assert (posterior["A"] - posterior["B"]) > prior_spread


# ------------------------------------------------------------------- guards


def test_bad_inputs_named():
    with pytest.raises(OptimizationError, match="strictly positive"):
        implied_returns(DIAG, pd.Series({"A": 1.0, "B": -0.2}))
    with pytest.raises(OptimizationError, match=r"\['B'\]"):
        implied_returns(DIAG, pd.Series({"A": 1.0}))
    with pytest.raises(OptimizationError, match="unknown symbol"):
        black_litterman_returns(DIAG, EQUAL_W, views=[absolute_view("Z", 0.1, 0.5)])
    with pytest.raises(OptimizationError, match="confidence"):
        black_litterman_returns(DIAG, EQUAL_W, views=[absolute_view("A", 0.1, 0.0)])
    with pytest.raises(OptimizationError, match="tau"):
        black_litterman_returns(DIAG, EQUAL_W, views=[absolute_view("A", 0.1, 0.5)], tau=0)
