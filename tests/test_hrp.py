"""Tests for optimize/hrp.py.

Notes
-----
The diagonal hand case doubles as a cross-check against MVO: with
independent assets, HRP's inverse-variance splits and the GMV formula
give the same answer, so three code paths (HRP, cvxpy MVO, closed form)
all meet at [0.8, 0.2]. The singular-covariance test is the selling
point: the input that destabilizes MVO is routine for HRP.
"""

import numpy as np
import pandas as pd
import pytest

from ballast.optimize.hrp import hrp_weights
from ballast.optimize.mvo import OptimizationError


def cov_frame(matrix, symbols) -> pd.DataFrame:
    return pd.DataFrame(matrix, index=symbols, columns=symbols)


# ------------------------------------------------------------- hand cases


def test_two_independent_assets_match_inverse_variance():
    # diag(1, 4): one split, singleton clusters, cluster variance = own
    # variance. alpha = 1 - 1/(1+4) = 0.8 -> [0.8, 0.2]. Same as GMV.
    w = hrp_weights(cov_frame([[1.0, 0.0], [0.0, 4.0]], ["A", "B"]))
    assert w["A"] == pytest.approx(0.8)
    assert w["B"] == pytest.approx(0.2)


def test_equal_iid_assets_recover_one_over_n():
    # Four identical independent assets: every split is 50/50 -> 1/N.
    symbols = ["A", "B", "C", "D"]
    w = hrp_weights(cov_frame(np.eye(4) * 0.04, symbols))
    for s in symbols:
        assert w[s] == pytest.approx(0.25)


def test_correlated_pair_shares_one_budget():
    # A and B move together (rho=0.95); C is independent. The tree pairs
    # A,B into one branch, so they SPLIT a branch budget while C gets its
    # own -- the diversifying asset must out-weigh each twin.
    s = 0.04
    matrix = [
        [s, 0.95 * s, 0.0],
        [0.95 * s, s, 0.0],
        [0.0, 0.0, s],
    ]
    w = hrp_weights(cov_frame(matrix, ["A", "B", "C"]))
    assert w["C"] > w["A"]
    assert w["A"] == pytest.approx(w["B"], rel=1e-9)  # symmetric twins
    assert w.sum() == pytest.approx(1.0)


# -------------------------------------------------------------- properties


def test_long_only_and_unit_sum_always(synthetic_returns):
    from ballast.covariance.estimators import sample_cov

    w = hrp_weights(sample_cov(synthetic_returns))
    assert (w >= 0).all()  # structural: budgets are only ever split
    assert w.sum() == pytest.approx(1.0)


def test_deterministic_but_not_permutation_invariant():
    # A finding worth pinning: HRP is DETERMINISTIC (same input, same
    # output) but NOT permutation invariant -- the dendrogram's left/right
    # orientation depends on input order, and bisection splits the leaf
    # list by position. This is a known property of Lopez de Prado's
    # original algorithm, not a bug; the module notes say so.
    rng = np.random.default_rng(7)
    a = rng.normal(0, 0.01, (100, 6))
    symbols = [f"S{i}" for i in range(6)]
    cov = cov_frame(a.T @ a / 100, symbols)

    assert hrp_weights(cov).equals(hrp_weights(cov))  # deterministic

    shuffled_order = ["S3", "S0", "S5", "S1", "S4", "S2"]
    permuted = hrp_weights(cov.loc[shuffled_order, shuffled_order])
    # Still a valid portfolio either way -- the contract that DOES hold.
    assert permuted.sum() == pytest.approx(1.0)
    assert (permuted >= 0).all()


def test_singular_covariance_is_business_as_usual():
    # The selling point. 10 assets, 6 observations: the sample matrix is
    # singular -- MVO territory where inversion explodes. HRP reads only
    # variances and correlations, so it just... works.
    rng = np.random.default_rng(19)
    a = rng.normal(0, 0.01, (6, 10))
    symbols = [f"S{i}" for i in range(10)]
    cov = cov_frame(a.T @ a / 6, symbols)
    assert np.linalg.matrix_rank(cov.to_numpy()) < 10  # genuinely singular
    w = hrp_weights(cov)
    assert np.isfinite(w.to_numpy()).all()
    assert w.sum() == pytest.approx(1.0)
    assert (w >= 0).all()


def test_linkage_method_changes_the_tree_but_not_the_contract():
    rng = np.random.default_rng(23)
    a = rng.normal(0, 0.01, (80, 8))
    cov = cov_frame(a.T @ a / 80, [f"S{i}" for i in range(8)])
    for method in ("single", "ward", "average", "complete"):
        w = hrp_weights(cov, linkage_method=method)
        assert w.sum() == pytest.approx(1.0)
        assert (w >= 0).all()


def test_single_asset_degenerates_gracefully():
    w = hrp_weights(cov_frame([[0.04]], ["A"]))
    assert w["A"] == 1.0


# ------------------------------------------------------------------- guards


def test_bad_inputs_named():
    with pytest.raises(OptimizationError, match="NaN"):
        hrp_weights(cov_frame([[float("nan"), 0], [0, 1]], ["A", "B"]))
    with pytest.raises(OptimizationError, match=r"\['B'\]"):
        hrp_weights(cov_frame([[1.0, 0.0], [0.0, 0.0]], ["A", "B"]))  # zero variance
    with pytest.raises(OptimizationError, match="linkage_method"):
        hrp_weights(cov_frame([[1.0, 0.0], [0.0, 1.0]], ["A", "B"]), linkage_method="magic")
