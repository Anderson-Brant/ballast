"""The covariance horse race. v0.2.0 -- implemented.

Notes
-----
What this file does: settles which estimator Ballast trusts, with an
out-of-sample experiment instead of a literature citation.

The experiment: walk forward. At each rebalance, each estimator builds its
covariance from the trailing window, the minimum-variance portfolio is
formed from that matrix, and the portfolio is held (drifting) until the
next rebalance. An estimator is good exactly to the degree that its
min-variance portfolios turn out to be low-variance OUT OF SAMPLE --
realized vol is the score, lowest wins, and the winner becomes the config
default. Equal weight (1/N) rides along as the baseline: it uses no
covariance at all, so any estimator that can't beat it isn't earning its
complexity.

Why minimum-variance is the right test: the GMV portfolio w = S^-1 1 /
(1' S^-1 1) depends on NOTHING but the covariance -- no expected returns
to argue about. It is also maximally sensitive to covariance error,
because inverting a noisy matrix amplifies the noise into the weights.
Weights are unconstrained (shorts allowed): constraints would mask
exactly the instability the experiment is trying to measure.

Diagnostics recorded per estimator: the average condition number of its
matrices (max/min eigenvalue). An ill-conditioned matrix produces wild
weights even when the vol score looks fine, and it predicts trouble for
the v0.5.0 optimizers.

Design rules:
- Reuses backtest/engine.py for the loop; no second loop here.
- One command reproduces the whole table: `ballast cov compare`.
- The result (on the real universe) gets written into docs/methodology.md
  with the run date.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ballast.backtest.engine import run_backtest
from ballast.covariance.estimators import (
    CovarianceError,
    ewma_cov,
    ledoit_wolf_cov,
    oas_cov,
    sample_cov,
)

__all__ = [
    "DEFAULT_ESTIMATORS",
    "EstimatorRow",
    "ComparisonResult",
    "min_variance_weights",
    "compare_estimators",
]

DEFAULT_ESTIMATORS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "sample": sample_cov,
    "ewma": ewma_cov,
    "ledoit_wolf": ledoit_wolf_cov,
    "oas": oas_cov,
}


@dataclass(frozen=True, slots=True)
class EstimatorRow:
    """One line of the comparison table."""

    name: str
    ann_vol: float  # the score: realized OOS vol of the GMV portfolio
    sharpe: float | None
    max_drawdown: float
    avg_turnover: float
    avg_condition: float | None  # None for 1/N (it has no matrix)


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """The full experiment: rows sorted by the score, plus its parameters."""

    rows: tuple[EstimatorRow, ...]  # ascending ann_vol: rows[0] is the winner
    window: int
    step: int
    cost_bps: float
    n_windows: int
    n_symbols: int
    start: object  # first/last simulated dates
    end: object


def min_variance_weights(cov: pd.DataFrame) -> pd.Series:
    """Global minimum variance portfolio: w = S^-1 1 / (1' S^-1 1).

    solve() when the matrix is invertible; pseudo-inverse when it isn't
    (a singular SAMPLE matrix is a legitimate competitor here -- pinv gives
    the minimum-norm solution instead of crashing the whole race).
    """
    matrix = cov.to_numpy(dtype=float)
    ones = np.ones(len(matrix))
    try:
        raw = np.linalg.solve(matrix, ones)
    except np.linalg.LinAlgError:
        raw = np.linalg.pinv(matrix) @ ones
    total = raw.sum()
    if not np.isfinite(total) or total == 0.0:
        raise CovarianceError("min-variance weights are degenerate (bad covariance matrix)")
    return pd.Series(raw / total, index=cov.columns)


def compare_estimators(
    returns: pd.DataFrame,
    *,
    window: int = 252,
    step: int = 21,
    cost_bps: float = 2.0,
    estimators: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] | None = None,
) -> ComparisonResult:
    """Run the horse race on a returns matrix. Same loop, same days, per estimator."""
    chosen = DEFAULT_ESTIMATORS if estimators is None else estimators
    rows: list[EstimatorRow] = []
    n_windows = 0

    for est_name, estimator in chosen.items():
        conditions: list[float] = []

        # Default-arg binding (est=estimator, rec=conditions) freezes the
        # loop variables into the closure -- without it every closure would
        # see the LAST estimator (the classic late-binding bug).
        def weights_fn(window_df: pd.DataFrame, est=estimator, rec=conditions) -> pd.Series:
            cov = est(window_df)
            eigenvalues = np.linalg.eigvalsh(cov.to_numpy())
            # Condition number: how much inversion amplifies noise.
            rec.append(float(eigenvalues[-1] / max(eigenvalues[0], 1e-18)))
            return min_variance_weights(cov)

        result = run_backtest(
            returns, weights_fn, window=window, step=step, cost_bps=cost_bps, name=est_name
        )
        n_windows = result.n_rebalances
        rows.append(
            EstimatorRow(
                name=est_name,
                ann_vol=result.ann_vol,
                sharpe=result.sharpe,
                max_drawdown=result.max_drawdown,
                avg_turnover=result.avg_turnover,
                avg_condition=float(np.mean(conditions)),
            )
        )

    # The baseline: no covariance, no estimation error, almost no turnover.
    def equal_weight(window_df: pd.DataFrame) -> pd.Series:
        n = window_df.shape[1]
        return pd.Series(1.0 / n, index=window_df.columns)

    baseline = run_backtest(
        returns, equal_weight, window=window, step=step, cost_bps=cost_bps, name="equal_weight"
    )
    rows.append(
        EstimatorRow(
            name="equal_weight (1/N)",
            ann_vol=baseline.ann_vol,
            sharpe=baseline.sharpe,
            max_drawdown=baseline.max_drawdown,
            avg_turnover=baseline.avg_turnover,
            avg_condition=None,
        )
    )

    simulated = returns.sort_index().index[window:]
    return ComparisonResult(
        rows=tuple(sorted(rows, key=lambda r: r.ann_vol)),  # lowest vol first
        window=window,
        step=step,
        cost_bps=cost_bps,
        n_windows=n_windows,
        n_symbols=returns.shape[1],
        start=simulated[0],
        end=simulated[-1],
    )
