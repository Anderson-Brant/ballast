"""Tests for optimize/mvo.py.

Notes
-----
The anchor test: with constraints off, the convex program must reproduce
the closed-form GMV from covariance/harness.py -- two completely different
code paths (cvxpy vs linear algebra) agreeing to 1e-6 is strong evidence
both are right. Constraint tests use hand-computed cases where the
constraint visibly binds.
"""

import numpy as np
import pandas as pd
import pytest

from ballast.covariance.harness import min_variance_weights
from ballast.optimize.mvo import OptimizationError, mvo_weights

cvxpy = pytest.importorskip("cvxpy")  # [opt] extra; skip cleanly where absent


def cov_frame(matrix: list[list[float]], symbols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(matrix, index=symbols, columns=symbols)


DIAG = cov_frame([[1.0, 0.0], [0.0, 4.0]], ["A", "B"])


# ------------------------------------------------------------ min variance


def test_unconstrained_matches_closed_form_gmv():
    # diag(1, 4): GMV = [0.8, 0.2] by hand, and by harness closed form.
    w = mvo_weights(DIAG, long_only=False)
    closed_form = min_variance_weights(DIAG)
    assert w["A"] == pytest.approx(closed_form["A"], abs=1e-6)
    assert w["B"] == pytest.approx(closed_form["B"], abs=1e-6)
    assert w.sum() == pytest.approx(1.0)


def test_unconstrained_matches_closed_form_on_a_random_matrix():
    rng = np.random.default_rng(13)
    a = rng.normal(0, 0.01, (60, 5))
    symbols = [f"S{i}" for i in range(5)]
    cov = cov_frame((a.T @ a / 60).tolist(), symbols)
    convex = mvo_weights(cov, long_only=False)
    closed = min_variance_weights(cov)
    assert np.allclose(convex.to_numpy(), closed.to_numpy(), atol=1e-5)


def test_long_only_forces_the_short_to_zero():
    # sigma1=0.1, sigma2=0.2, rho=0.9: unconstrained GMV shorts asset B
    # (w_B = (0.01 - 0.018) / 0.014 < 0, worked by hand). Long-only must
    # land on the boundary: everything in the low-vol asset.
    cov = cov_frame([[0.01, 0.018], [0.018, 0.04]], ["A", "B"])
    unconstrained = mvo_weights(cov, long_only=False)
    assert unconstrained["B"] < 0  # confirms the short is genuinely wanted
    constrained = mvo_weights(cov, long_only=True)
    assert constrained["A"] == pytest.approx(1.0, abs=1e-6)
    assert constrained["B"] == pytest.approx(0.0, abs=1e-6)


def test_max_weight_binds():
    symbols = ["A", "B", "C"]
    cov = cov_frame(np.diag([1.0, 1.0, 1.0]).tolist(), symbols)
    w = mvo_weights(cov, max_weight=0.5)
    assert (w <= 0.5 + 1e-9).all()
    assert w.sum() == pytest.approx(1.0)


def test_sector_cap_binds():
    symbols = ["T1", "T2", "F1", "F2"]
    # Tech assets are much calmer: unconstrained min-variance would load
    # them heavily. The 30% tech cap must bind.
    cov = cov_frame(np.diag([0.01, 0.01, 0.09, 0.09]).tolist(), symbols)
    sectors = {"T1": "tech", "T2": "tech", "F1": "fin", "F2": "fin"}
    uncapped = mvo_weights(cov)
    assert uncapped[["T1", "T2"]].sum() > 0.5  # tech dominates without the cap
    capped = mvo_weights(cov, sectors=sectors, sector_caps={"tech": 0.3})
    assert capped[["T1", "T2"]].sum() == pytest.approx(0.3, abs=1e-6)
    assert capped.sum() == pytest.approx(1.0)


# ------------------------------------------------------------ mean-variance


def test_expected_returns_tilt_and_risk_aversion_pulls_back():
    symbols = ["A", "B"]
    cov = cov_frame([[0.04, 0.0], [0.0, 0.04]], symbols)  # identical risk
    mu = pd.Series({"A": 0.10, "B": 0.02})
    greedy = mvo_weights(cov, expected_returns=mu, risk_aversion=1.0)
    cautious = mvo_weights(cov, expected_returns=mu, risk_aversion=100.0)
    # The forecasted winner gets more weight...
    assert greedy["A"] > cautious["A"] >= 0.5
    # ...and infinite risk aversion converges on min-variance (50/50 here).
    assert cautious["A"] == pytest.approx(0.5, abs=0.02)


def test_min_variance_ignores_returns_entirely():
    # Same covariance, no mu: the answer must not depend on any forecast.
    w = mvo_weights(DIAG)
    assert w["A"] == pytest.approx(0.8, abs=1e-6)


# ------------------------------------------------------------------- guards


def test_infeasible_max_weight_caught_by_arithmetic():
    with pytest.raises(OptimizationError, match="cannot sum to 1"):
        mvo_weights(DIAG, max_weight=0.4)  # 2 x 0.4 < 1


def test_contradictory_constraints_raise_not_fallback():
    symbols = ["A", "B"]
    cov = cov_frame(np.diag([1.0, 1.0]).tolist(), symbols)
    sectors = {"A": "x", "B": "x"}
    with pytest.raises(OptimizationError, match="status|contradict"):
        # Everything is sector x, capped at 0.5, but weights must sum to 1.
        mvo_weights(cov, sectors=sectors, sector_caps={"x": 0.5})


def test_partial_investment_lets_cash_absorb():
    # Same contradiction, but fully_invested=False: sum(w) <= 1 makes the
    # 0.5 sector cap satisfiable with 50% cash.
    symbols = ["A", "B"]
    cov = cov_frame(np.diag([1.0, 1.0]).tolist(), symbols)
    sectors = {"A": "x", "B": "x"}
    w = mvo_weights(cov, fully_invested=False, sectors=sectors, sector_caps={"x": 0.5})
    assert w.sum() <= 0.5 + 1e-6  # min-variance with optional cash: hold cash


def test_bad_inputs_named():
    with pytest.raises(OptimizationError, match="NaN"):
        mvo_weights(cov_frame([[float("nan"), 0], [0, 1]], ["A", "B"]))
    with pytest.raises(OptimizationError, match=r"\['B'\]"):
        mvo_weights(DIAG, expected_returns=pd.Series({"A": 0.1}))
    with pytest.raises(OptimizationError, match="risk_aversion"):
        mvo_weights(DIAG, expected_returns=pd.Series({"A": 0.1, "B": 0.1}), risk_aversion=0)
    with pytest.raises(OptimizationError, match="no symbols"):
        mvo_weights(DIAG, sectors={"A": "x", "B": "x"}, sector_caps={"nope": 0.5})
    with pytest.raises(OptimizationError, match="cash interpretation"):
        mvo_weights(DIAG, long_only=False, fully_invested=False)
