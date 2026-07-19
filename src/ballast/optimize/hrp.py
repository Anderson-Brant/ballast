"""Hierarchical Risk Parity (Lopez de Prado 2016). v0.5.0 -- implemented.

Notes
-----
What this file does: portfolio weights without inverting the covariance
matrix and without expected returns -- the two most fragile steps of MVO
removed by construction rather than by care.

The three stages, exactly as the paper lays them out:
1. Tree clustering. Correlation becomes a distance,
       d_ij = sqrt((1 - rho_ij) / 2)
   (0 for perfectly correlated assets, 1 for perfectly anti-correlated),
   and hierarchical linkage builds the asset family tree from it.
2. Quasi-diagonalization. Reorder assets in dendrogram-leaf order, so
   similar assets sit next to each other and the covariance matrix's mass
   hugs the diagonal.
3. Recursive bisection. Walk down the ordered list splitting it in half;
   each half's risk is measured with inverse-variance weights inside it,
   and the halves get budget in inverse proportion to their risk:
       alpha_left = 1 - var_left / (var_left + var_right)
   Repeat inside each half until singletons.

Why this design survives what kills MVO: inverting a noisy covariance
matrix amplifies its smallest (least reliable) eigenvalues into enormous
long-short bets. HRP only ever READS variances and correlations -- a
singular matrix, which crashes or destabilizes MVO, is business as usual
here. Weights are also long-only and sum to 1 by construction: every
split hands out fractions of a budget that starts at 1.

The cost, stated plainly: HRP ignores everything the tree doesn't
capture. Two assets in different branches get no credit for hedging each
other. Whether the robustness is worth that is exactly what the v0.5.0
comparison decides -- and if HRP can't beat 1/N after costs either, that
result goes in the table.

Design rule: the linkage method is a parameter ("single" is the paper's
choice; "ward" is the popular variant), because it changes the tree and
therefore the weights. Callers record it with results.

Known quirk, kept deliberately: HRP is deterministic but NOT permutation
invariant. The dendrogram's left/right orientation depends on input
order, and bisection splits the leaf list by position -- shuffle the
columns and the weights shift somewhat. That's Lopez de Prado's original
algorithm faithfully reproduced (seriation-based fixes exist but change
the published method). Practical consequence: keep symbol order stable
(sorted) across rebalances, which load_returns already does.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from ballast.optimize.mvo import OptimizationError

__all__ = ["hrp_weights"]

_LINKAGE_METHODS = ("single", "complete", "average", "ward")


def _cluster_variance(cov: np.ndarray, members: list[int]) -> float:
    """Variance of a cluster under inverse-variance weighting of its members.

    This is the paper's proxy for "how risky is this branch": cheap, needs
    only the diagonal for the weights, and never inverts the matrix.
    """
    sub = cov[np.ix_(members, members)]
    inverse_variance = 1.0 / np.diag(sub)
    inverse_variance /= inverse_variance.sum()
    return float(inverse_variance @ sub @ inverse_variance)


def hrp_weights(cov: pd.DataFrame, linkage_method: str = "single") -> pd.Series:
    """HRP weights from a covariance matrix. Long-only, sum to 1, no inversion."""
    symbols = list(cov.columns)
    n = len(symbols)
    if n == 0 or list(cov.index) != symbols:
        raise OptimizationError("cov must be a square symbol-labeled DataFrame")
    if linkage_method not in _LINKAGE_METHODS:
        raise OptimizationError(f"linkage_method must be one of {_LINKAGE_METHODS}")
    sigma = cov.to_numpy(dtype=float)
    if not np.isfinite(sigma).all():
        raise OptimizationError("covariance contains NaN or inf")
    variances = np.diag(sigma)
    if (variances <= 0).any():
        bad = [symbols[i] for i in np.where(variances <= 0)[0]]
        raise OptimizationError(f"non-positive variance for symbol(s) {bad}")
    if n == 1:
        return pd.Series([1.0], index=symbols)  # nothing to allocate between

    # ---- stage 1: correlation -> distance -> tree ------------------------
    vols = np.sqrt(variances)
    corr = sigma / np.outer(vols, vols)
    corr = np.clip(corr, -1.0, 1.0)  # float dust can push |rho| past 1
    # (1 - rho)/2 can dip a hair below 0 at rho ~ 1; clamp before sqrt.
    distance = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, None))
    np.fill_diagonal(distance, 0.0)
    # squareform wants a perfectly symmetric matrix; enforce against dust.
    condensed = squareform((distance + distance.T) / 2.0, checks=False)
    tree = linkage(condensed, method=linkage_method)

    # ---- stage 2: quasi-diagonalization ----------------------------------
    order = leaves_list(tree).tolist()  # dendrogram leaf order

    # ---- stage 3: recursive bisection ------------------------------------
    weights = np.ones(n)
    stack: list[list[int]] = [order]
    while stack:
        cluster = stack.pop()
        if len(cluster) <= 1:
            continue
        half = len(cluster) // 2
        left, right = cluster[:half], cluster[half:]
        var_left = _cluster_variance(sigma, left)
        var_right = _cluster_variance(sigma, right)
        total = var_left + var_right
        # Two zero-variance branches can't happen (validated above), but a
        # denominator guard costs nothing and documents the intent.
        alpha = 0.5 if total <= 0 else 1.0 - var_left / total
        weights[left] *= alpha
        weights[right] *= 1.0 - alpha
        stack.extend([left, right])

    # Budget conservation is structural: every split distributes exactly
    # what came in. The assert documents the invariant, tests pin it.
    assert np.isclose(weights.sum(), 1.0, atol=1e-12)
    return pd.Series(weights, index=symbols)
