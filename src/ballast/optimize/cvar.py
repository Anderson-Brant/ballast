"""CVaR (expected shortfall) optimization. v0.5.0 -- implemented.

Notes
-----
What this file does: minimizes expected loss in the worst (1-c) fraction
of scenarios, rather than variance. Variance punishes upside and downside
symmetrically; CVaR looks only at the tail, which is the part an investor
actually fears. This is the optimizer that can tell apart two assets with
IDENTICAL variance where one earns it smoothly and the other earns it
with occasional crashes -- MVO literally cannot see that difference.

The Rockafellar-Uryasev trick (2000), which makes this tractable: CVaR of
a portfolio over T historical scenarios becomes a LINEAR program by
adding an auxiliary threshold zeta (which converges to the VaR) and one
slack per scenario for losses beyond it:

    minimize   zeta + 1/((1-c) T) * sum_t u_t
    subject to u_t >= -(r_t . w) - zeta      (loss beyond the threshold)
               u_t >= 0
               sum w = 1,  [w >= 0],  [w <= max_weight]

No distribution is assumed anywhere: the scenarios ARE the model. That is
both the appeal (real tails, real skew) and the cost -- the optimizer can
only fear crashes it has seen, so the scenario window must actually
contain some. The guard below refuses to run with fewer than ~10 tail
scenarios; averaging three points and calling it expected shortfall would
be numerology.

Sanity anchor (pinned by test): for elliptical (e.g. normal) returns,
CVaR is a monotone function of variance, so min-CVaR and min-variance
choose the SAME portfolio. On Gaussian scenarios this LP must
approximately reproduce the GMV weights; where the two disagree on real
data, the disagreement IS the skew information.

Same conventions as the rest of optimize/: solver failures raise, never a
silent fallback; cvxpy imported lazily via the shared helper.
"""

import math

import numpy as np
import pandas as pd

from ballast.optimize.mvo import _ACCEPTABLE, OptimizationError, _require_cvxpy

__all__ = ["cvar_weights"]

_MIN_TAIL_SCENARIOS = 10


def cvar_weights(
    returns: pd.DataFrame,
    confidence: float = 0.95,
    long_only: bool = True,
    max_weight: float | None = None,
) -> pd.Series:
    """Minimum-CVaR weights over historical scenarios. Fully invested.

    returns: scenarios x symbols (e.g. a load_returns window) -- each row
    is one joint outcome, tails, skew, correlations and all.
    confidence: the tail definition; 0.95 averages the worst 5% of rows.
    """
    cp = _require_cvxpy()

    symbols = list(returns.columns)
    n = len(symbols)
    if n == 0:
        raise OptimizationError("returns frame has no columns")
    scenarios = returns.to_numpy(dtype=float)
    if not np.isfinite(scenarios).all():
        raise OptimizationError("returns contain NaN or inf; fix alignment upstream")
    if not 0.5 < confidence < 1.0:
        raise OptimizationError(f"confidence must be in (0.5, 1), got {confidence}")

    t = len(scenarios)
    tail_count = t * (1.0 - confidence)
    if tail_count < _MIN_TAIL_SCENARIOS:
        needed = math.ceil(_MIN_TAIL_SCENARIOS / (1.0 - confidence))
        raise OptimizationError(
            f"{t} scenarios give only ~{tail_count:.1f} tail observations at "
            f"{confidence:.0%}; need at least {needed} rows to see the tail"
        )
    if max_weight is not None:
        if not 0 < max_weight <= 1:
            raise OptimizationError(f"max_weight must be in (0, 1], got {max_weight}")
        if max_weight * n < 1.0 - 1e-12:
            raise OptimizationError(
                f"max_weight={max_weight} x {n} positions cannot sum to 1; infeasible"
            )

    # ---- the Rockafellar-Uryasev LP --------------------------------------
    w = cp.Variable(n)
    zeta = cp.Variable()  # converges to the portfolio's VaR at optimum
    excess = cp.Variable(t)  # per-scenario loss beyond zeta

    portfolio_losses = -(scenarios @ w)  # loss = negative return, per row
    objective = zeta + cp.sum(excess) / ((1.0 - confidence) * t)
    constraints = [
        excess >= portfolio_losses - zeta,
        excess >= 0,
        cp.sum(w) == 1,
    ]
    if long_only:
        constraints.append(w >= 0)
    if max_weight is not None:
        constraints.append(w <= max_weight)

    problem = cp.Problem(cp.Minimize(objective), constraints)
    try:
        problem.solve()
    except cp.error.SolverError as exc:
        raise OptimizationError(f"solver failed: {exc}") from exc
    if problem.status not in _ACCEPTABLE:
        raise OptimizationError(f"no solution: status={problem.status!r}")

    values = np.asarray(w.value, dtype=float).ravel()
    values[np.abs(values) < 1e-10] = 0.0  # solver dust
    return pd.Series(values, index=symbols)
