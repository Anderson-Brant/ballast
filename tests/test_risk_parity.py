"""Tests for optimize/risk_parity.py.

Notes
-----
The definitional test is the one that matters: on any valid covariance,
every asset's risk contribution share must equal its budget. The diagonal
hand case doubles as the three-optimizer contrast -- same matrix, three
different philosophies, three different answers, all hand-computable.
"""

import numpy as np
import pandas as pd
import pytest

from ballast.optimize.mvo import OptimizationError
from ballast.optimize.risk_parity import risk_parity_weights


def cov_frame(matrix, symbols) -> pd.DataFrame:
    return pd.DataFrame(matrix, index=symbols, columns=symbols)


def contribution_shares(w: pd.Series, cov: pd.DataFrame) -> np.ndarray:
    """The shared definition: RC_i = w_i (Sigma w)_i, as shares of total."""
    sigma = cov.to_numpy()
    contributions = w.to_numpy() * (sigma @ w.to_numpy())
    return contributions / contributions.sum()


# -------------------------------------------------------------- hand cases


def test_diagonal_case_is_inverse_vol():
    # diag(1, 4): vols [1, 2] -> w proportional to [1, 1/2] -> [2/3, 1/3].
    # Contributions: w1^2*1 = 4/9 and w2^2*4 = 4/9 -- equal, as promised.
    # Contrast on the SAME matrix: GMV/HRP give [0.8, 0.2], 1/N [0.5, 0.5].
    w = risk_parity_weights(cov_frame([[1.0, 0.0], [0.0, 4.0]], ["A", "B"]))
    assert w["A"] == pytest.approx(2 / 3, abs=1e-8)
    assert w["B"] == pytest.approx(1 / 3, abs=1e-8)


def test_equal_iid_assets_recover_one_over_n():
    symbols = ["A", "B", "C", "D"]
    w = risk_parity_weights(cov_frame(np.eye(4) * 0.04, symbols))
    for s in symbols:
        assert w[s] == pytest.approx(0.25, abs=1e-8)


def test_budgeted_diagonal_hand_case():
    # diag(1, 1), budgets [0.8, 0.2]: w_i ~ sqrt(b_i) -> ratio 2:1.
    # Check: shares w_i^2 / sum = 0.8 / 1.0 = 0.8. Worked by hand.
    w = risk_parity_weights(
        cov_frame(np.eye(2).tolist(), ["A", "B"]),
        budgets=pd.Series({"A": 0.8, "B": 0.2}),
    )
    assert w["A"] / w["B"] == pytest.approx(2.0, abs=1e-6)


# ----------------------------------------------------------- the definition


def test_contributions_equal_on_random_matrices():
    # The property that IS the optimizer. Several seeds, correlations and
    # all: every asset's contribution share must equal 1/n.
    for seed in (5, 11, 42):
        rng = np.random.default_rng(seed)
        a = rng.normal(0, 0.01, (120, 8))
        symbols = [f"S{i}" for i in range(8)]
        cov = cov_frame(a.T @ a / 120, symbols)
        w = risk_parity_weights(cov)
        shares = contribution_shares(w, cov)
        assert np.abs(shares - 1.0 / 8).max() < 1e-7, f"seed {seed}"
        assert (w > 0).to_numpy().all()  # long-only by construction
        assert w.sum() == pytest.approx(1.0)


def test_budgets_are_hit_on_a_correlated_matrix():
    rng = np.random.default_rng(3)
    a = rng.normal(0, 0.01, (150, 5))
    symbols = [f"S{i}" for i in range(5)]
    cov = cov_frame(a.T @ a / 150, symbols)
    budgets = pd.Series([0.4, 0.3, 0.15, 0.1, 0.05], index=symbols)
    w = risk_parity_weights(cov, budgets=budgets)
    shares = contribution_shares(w, cov)
    assert np.abs(shares - budgets.to_numpy()).max() < 1e-7


def test_negative_correlation_still_converges():
    # c_i goes negative here; the positive-root update must stay defined.
    cov = cov_frame([[0.04, -0.018], [-0.018, 0.01]], ["A", "B"])
    w = risk_parity_weights(cov)
    shares = contribution_shares(w, cov)
    assert np.abs(shares - 0.5).max() < 1e-7


def test_ordering_beats_vol_ordering_not_dollar_ordering():
    # Calm asset gets MORE weight, wild asset less -- but never as extreme
    # as GMV. vols 1%, 2%, 4% -> weights strictly decreasing.
    cov = cov_frame(np.diag([0.0001, 0.0004, 0.0016]).tolist(), ["CALM", "MID", "WILD"])
    w = risk_parity_weights(cov)
    assert w["CALM"] > w["MID"] > w["WILD"] > 0


# ------------------------------------------------------------------- guards


def test_no_convergence_raises_no_partial_answer():
    cov = cov_frame([[1.0, 0.0], [0.0, 4.0]], ["A", "B"])
    with pytest.raises(OptimizationError, match="did not converge"):
        risk_parity_weights(cov, max_iter=1, tol=1e-16)


def test_bad_inputs_named():
    with pytest.raises(OptimizationError, match="NaN"):
        risk_parity_weights(cov_frame([[float("nan"), 0], [0, 1]], ["A", "B"]))
    with pytest.raises(OptimizationError, match=r"\['B'\]"):
        risk_parity_weights(cov_frame([[1.0, 0.0], [0.0, 0.0]], ["A", "B"]))
    with pytest.raises(OptimizationError, match="budgets missing"):
        risk_parity_weights(
            cov_frame(np.eye(2).tolist(), ["A", "B"]), budgets=pd.Series({"A": 1.0})
        )
    with pytest.raises(OptimizationError, match="strictly positive"):
        risk_parity_weights(
            cov_frame(np.eye(2).tolist(), ["A", "B"]),
            budgets=pd.Series({"A": 1.0, "B": -0.5}),
        )


def test_single_asset_degenerates_gracefully():
    w = risk_parity_weights(cov_frame([[0.04]], ["A"]))
    assert w["A"] == 1.0
