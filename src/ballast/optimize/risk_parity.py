"""Equal risk contribution weights. v0.5.0 -- implemented.

Notes
-----
What this file does: finds weights where every asset contributes the same
share of total portfolio risk -- "diversify the risk, not the dollars."
1/N equalizes capital; this equalizes risk, which is the relevant
quantity when one holding is 3x as volatile as another.

Where it sits among its siblings, on the diag(1, 4) two-asset example:
- GMV / HRP:      [0.80, 0.20]   (inverse VARIANCE -- risk minimizers)
- risk parity:    [2/3,  1/3 ]   (inverse VOL -- risk equalizer)
- equal weight:   [0.50, 0.50]   (inverse nothing)
ERC deliberately lands between minimum variance and 1/N: it refuses to
concentrate as hard as GMV, but unlike 1/N it actually looks at risk.

Definition: RC_i = w_i * (Sigma w)_i is asset i's contribution to
portfolio variance (the same formula risk/decompose.py allocates
position risk with -- one definition, two files, pinned by test). ERC
solves RC_i equal for all i; the risk-budgeted variant solves
RC_i / total = b_i for a target budget vector b.

Implementation: no closed form exists beyond the diagonal case, so this
uses Spinu's convex formulation -- minimize (1/2) w'Sigma w minus
sum(b_i ln w_i) -- solved by cyclical coordinate descent. Each coordinate
update is the positive root of a quadratic:

    w_i = ( -c_i + sqrt(c_i^2 + 4 sigma_ii b_i) ) / (2 sigma_ii),
    c_i = (Sigma w)_i - sigma_ii w_i    (covariance with everyone else)

b_i > 0 keeps the discriminant positive, so the update is always defined,
including under negative correlations. The unnormalized fixed point is
scaled to sum to 1 at the end (the scaling doesn't change contribution
SHARES). Long-only by construction: positive roots only.

Honesty guards: failure to converge RAISES (no partial answer), and the
converged weights are re-checked against the budgets with the
contribution formula itself -- if a semi-definite corner case slips
through the iteration, the self-check refuses to return it.
"""

import numpy as np
import pandas as pd

from ballast.optimize.mvo import OptimizationError

__all__ = ["risk_parity_weights"]


def risk_parity_weights(
    cov: pd.DataFrame,
    budgets: pd.Series | None = None,
    tol: float = 1e-10,
    max_iter: int = 10_000,
) -> pd.Series:
    """ERC weights (or risk-budgeted weights when `budgets` is given).

    budgets: target share of total risk per symbol, positive; normalized
    to sum to 1. None means equal shares -- classic risk parity.
    """
    symbols = list(cov.columns)
    n = len(symbols)
    if n == 0 or list(cov.index) != symbols:
        raise OptimizationError("cov must be a square symbol-labeled DataFrame")
    sigma = cov.to_numpy(dtype=float)
    if not np.isfinite(sigma).all():
        raise OptimizationError("covariance contains NaN or inf")
    variances = np.diag(sigma)
    if (variances <= 0).any():
        bad = [symbols[i] for i in np.where(variances <= 0)[0]]
        raise OptimizationError(f"non-positive variance for symbol(s) {bad}")

    if budgets is None:
        b = np.full(n, 1.0 / n)
    else:
        missing = sorted(set(symbols) - set(budgets.index))
        if missing:
            raise OptimizationError(f"budgets missing symbol(s) {missing}")
        b = budgets.reindex(symbols).to_numpy(dtype=float)
        if not np.isfinite(b).all() or (b <= 0).any():
            raise OptimizationError("budgets must be finite and strictly positive")
        b = b / b.sum()  # shares of risk: only proportions are meaningful

    if n == 1:
        return pd.Series([1.0], index=symbols)

    # Inverse-vol start: exact for diagonal matrices, close everywhere else.
    w = 1.0 / np.sqrt(variances)
    w /= w.sum()

    converged = False
    for _ in range(max_iter):
        max_change = 0.0
        for i in range(n):
            # Covariance of asset i with the REST of the current portfolio.
            c = float(sigma[i] @ w) - variances[i] * w[i]
            new = (-c + np.sqrt(c * c + 4.0 * variances[i] * b[i])) / (2.0 * variances[i])
            max_change = max(max_change, abs(new - w[i]))
            w[i] = new  # Gauss-Seidel style: later coordinates see it now
        if max_change < tol:
            converged = True
            break
    if not converged:
        raise OptimizationError(
            f"risk parity did not converge in {max_iter} iterations "
            "(covariance may be badly conditioned); no partial answer returned"
        )

    w = w / w.sum()  # scale-free contributions: normalizing is safe

    # Self-check with the SAME contribution formula decompose uses:
    # converged-but-wrong must be impossible to return.
    contributions = w * (sigma @ w)
    shares = contributions / contributions.sum()
    if np.abs(shares - b).max() > 1e-6:
        raise OptimizationError(
            "converged weights do not meet the risk budgets "
            f"(worst gap {np.abs(shares - b).max():.2e}); refusing to return them"
        )
    return pd.Series(w, index=symbols)
