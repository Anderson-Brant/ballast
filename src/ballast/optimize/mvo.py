"""Mean-variance optimization with constraints. v0.5.0 -- implemented.

Notes
-----
What this file does: the classical Markowitz optimizer as a convex program
(cvxpy), so real-world constraints are first-class citizens instead of
afterthoughts bolted onto a closed-form formula.

Two modes, chosen by whether expected returns are supplied:
- minimum variance (expected_returns=None): minimize w' S w. The honest
  default -- expected returns are the least reliable input in finance, and
  a portfolio that doesn't need them can't be poisoned by them.
- mean-variance: minimize (risk_aversion/2) w' S w - mu' w. Higher
  risk_aversion pulls the solution toward the min-variance portfolio;
  lower lets the return forecast dominate. mu typically comes from
  Black-Litterman (which is where Sentinel's scores will enter).

Constraints, all arguments rather than hardcoded:
- fully_invested: sum(w) == 1 (default), or sum(w) <= 1 with the remainder
  read as cash (long_only required for that reading to make sense).
- long_only: w >= 0 (default True; the harness's unconstrained GMV is the
  shorts-allowed benchmark, this is the implementable version).
- max_weight: per-position cap. Infeasible caps (n x cap < 1) are caught
  with arithmetic before the solver ever runs.
- sector caps: sum of weights within a sector <= cap, given a sector map.

The unconstrained sanity anchor: with long_only=False and no caps, the
solution must equal the closed-form GMV w = S^-1 1 / (1' S^-1 1) that
covariance/harness.py computes -- a test pins the two against each other.

Design rules:
- The optimizer never sees raw returns: it takes a covariance (and
  optionally expected returns), so any estimator plugs in.
- Solver failures RAISE; they never silently fall back to equal weight.
  A fallback would turn every infeasibility bug into a quiet 1/N.
- cvxpy is imported lazily: it's the [opt] extra, and importing this
  module (e.g. the smoke test) must not require it.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd

__all__ = ["OptimizationError", "mvo_weights"]

_ACCEPTABLE = ("optimal", "optimal_inaccurate")


class OptimizationError(RuntimeError):
    """Raised for bad inputs, infeasible constraints, or solver failures."""


def _require_cvxpy():
    try:
        import cvxpy
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise OptimizationError(
            "cvxpy is not installed; the optimizers need it: pip install 'ballast[opt]'"
        ) from exc
    return cvxpy


def mvo_weights(
    cov: pd.DataFrame,
    expected_returns: pd.Series | None = None,
    risk_aversion: float = 2.0,
    long_only: bool = True,
    fully_invested: bool = True,
    max_weight: float | None = None,
    sectors: Mapping[str, str] | None = None,
    sector_caps: Mapping[str, float] | None = None,
) -> pd.Series:
    """Solve the constrained Markowitz problem. Returns weights by symbol.

    See module notes for the two modes and each constraint's meaning.
    """
    cp = _require_cvxpy()

    # ---- input validation: fail with sentences, not solver stack traces --
    symbols = list(cov.columns)
    n = len(symbols)
    if n == 0 or list(cov.index) != symbols:
        raise OptimizationError("cov must be a square symbol-labeled DataFrame")
    sigma = cov.to_numpy(dtype=float)
    if not np.isfinite(sigma).all():
        raise OptimizationError("covariance contains NaN or inf")

    mu = None
    if expected_returns is not None:
        missing = sorted(set(symbols) - set(expected_returns.index))
        if missing:
            raise OptimizationError(f"expected_returns missing symbol(s) {missing}")
        mu = expected_returns.reindex(symbols).to_numpy(dtype=float)
        if not np.isfinite(mu).all():
            raise OptimizationError("expected_returns contain NaN or inf")
        if risk_aversion <= 0:
            raise OptimizationError(f"risk_aversion must be > 0, got {risk_aversion}")

    if max_weight is not None:
        if not 0 < max_weight <= 1:
            raise OptimizationError(f"max_weight must be in (0, 1], got {max_weight}")
        if fully_invested and max_weight * n < 1.0 - 1e-12:
            # Cheap arithmetic beats an opaque INFEASIBLE status.
            raise OptimizationError(
                f"max_weight={max_weight} x {n} positions cannot sum to 1; infeasible"
            )
    if not fully_invested and not long_only:
        raise OptimizationError(
            "sum(w) <= 1 with shorts allowed has no cash interpretation; "
            "use fully_invested=True for long/short portfolios"
        )

    # ---- the convex program ---------------------------------------------
    w = cp.Variable(n)
    # psd_wrap: the estimators guarantee PSD up to float dust; without the
    # wrap, cvxpy's strict PSD check can reject a matrix over a -1e-17
    # eigenvalue.
    risk = cp.quad_form(w, cp.psd_wrap(sigma))
    objective = risk if mu is None else 0.5 * risk_aversion * risk - mu @ w

    constraints = [cp.sum(w) == 1] if fully_invested else [cp.sum(w) <= 1]
    if long_only:
        constraints.append(w >= 0)
    if max_weight is not None:
        constraints.append(w <= max_weight)
    if sector_caps:
        if sectors is None:
            raise OptimizationError("sector_caps given without a sectors map")
        for sector, cap in sector_caps.items():
            members = [i for i, s in enumerate(symbols) if sectors.get(s) == sector]
            if not members:
                raise OptimizationError(f"sector cap for {sector!r} matches no symbols")
            constraints.append(cp.sum(w[members]) <= cap)

    problem = cp.Problem(cp.Minimize(objective), constraints)
    try:
        problem.solve()
    except cp.error.SolverError as exc:
        raise OptimizationError(f"solver failed: {exc}") from exc
    if problem.status not in _ACCEPTABLE:
        raise OptimizationError(
            f"no solution: status={problem.status!r} "
            "(check the constraints -- they likely contradict each other)"
        )

    values = np.asarray(w.value, dtype=float).ravel()
    # Solver dust: -3e-12 weights are zeros wearing float clothing.
    values[np.abs(values) < 1e-10] = 0.0
    return pd.Series(values, index=symbols)
