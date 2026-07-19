"""The optimizer race: everything vs 1/N, walk-forward, after costs. v0.5.0.

Notes
-----
What this file does: settles which optimizer earns its complexity, with
the same out-of-sample discipline as the covariance horse race. Every
strategy is a weights function; the shared backtest engine walks them all
over the SAME days with the SAME costs; the table sorts by realized
Sharpe. The bar is equal weight: DeMiguel et al. (2009) found almost
nothing beats 1/N after costs, and this table either reproduces that
embarrassment or earns the right to disagree.

Fairness rules, deliberately strict:
- Every covariance-consuming optimizer gets the SAME estimator
  (Ledoit-Wolf, for conditioning) -- differences in the table are then
  differences between OPTIMIZERS, not between covariance estimates.
- All strategies are long-only and fully invested: comparable turnover,
  comparable risk, engine-compatible.
- A strategy that cannot run (CVaR needs enough tail scenarios for the
  window) is SKIPPED with its reason recorded and displayed -- not
  silently dropped, not allowed to kill the race.

Black-Litterman is absent by design: it needs views and market caps,
which makes it a different experiment (and the v0.6.0 Sentinel bridge's
job), not a fair entrant in a views-free race.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ballast.backtest.engine import BacktestError, run_backtest
from ballast.covariance.estimators import CovarianceError, ledoit_wolf_cov
from ballast.optimize.cvar import cvar_weights
from ballast.optimize.hrp import hrp_weights
from ballast.optimize.mvo import OptimizationError, mvo_weights
from ballast.optimize.risk_parity import risk_parity_weights

__all__ = ["OptimizerRow", "OptimizerComparisonResult", "compare_optimizers"]


def _equal_weight(window: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0 / window.shape[1], index=window.columns)


def _min_variance(window: pd.DataFrame) -> pd.Series:
    return mvo_weights(ledoit_wolf_cov(window))  # long-only by default


def _hrp(window: pd.DataFrame) -> pd.Series:
    return hrp_weights(ledoit_wolf_cov(window))


def _risk_parity(window: pd.DataFrame) -> pd.Series:
    return risk_parity_weights(ledoit_wolf_cov(window))


def _min_cvar(window: pd.DataFrame) -> pd.Series:
    return cvar_weights(window, confidence=0.95)


DEFAULT_STRATEGIES: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "equal_weight (1/N)": _equal_weight,  # the bar
    "min_variance": _min_variance,
    "hrp": _hrp,
    "risk_parity": _risk_parity,
    "min_cvar": _min_cvar,
}


@dataclass(frozen=True, slots=True)
class OptimizerRow:
    """One line of the comparison table."""

    name: str
    ann_return: float
    ann_vol: float
    sharpe: float | None
    max_drawdown: float
    avg_turnover: float
    cost_drag: float


@dataclass(frozen=True, slots=True)
class OptimizerComparisonResult:
    """Rows sorted by Sharpe (best first), plus what couldn't run and why."""

    rows: tuple[OptimizerRow, ...]
    skipped: tuple[tuple[str, str], ...]  # (strategy, reason)
    window: int
    step: int
    cost_bps: float
    n_windows: int
    n_symbols: int
    start: Any
    end: Any


def compare_optimizers(
    returns: pd.DataFrame,
    window: int = 252,
    step: int = 21,
    cost_bps: float = 2.0,
    strategies: dict[str, Callable[[pd.DataFrame], pd.Series]] | None = None,
) -> OptimizerComparisonResult:
    """Race every strategy over the same walk-forward calendar."""
    chosen = DEFAULT_STRATEGIES if strategies is None else strategies
    rows: list[OptimizerRow] = []
    skipped: list[tuple[str, str]] = []
    n_windows = 0

    for name, weights_fn in chosen.items():
        try:
            result = run_backtest(
                returns, weights_fn, window=window, step=step, cost_bps=cost_bps, name=name
            )
        except (OptimizationError, CovarianceError, BacktestError) as exc:
            # Recorded and displayed, never silently dropped: a race with
            # invisible no-shows would misrepresent the field.
            skipped.append((name, str(exc)))
            continue
        n_windows = result.n_rebalances
        rows.append(
            OptimizerRow(
                name=name,
                ann_return=result.ann_return,
                ann_vol=result.ann_vol,
                sharpe=result.sharpe,
                max_drawdown=result.max_drawdown,
                avg_turnover=result.avg_turnover,
                cost_drag=result.cost_drag,
            )
        )

    if not rows:
        first = skipped[0]
        raise OptimizationError(f"every strategy failed; first reason ({first[0]}): {first[1]}")

    simulated = returns.sort_index().index[window:]
    return OptimizerComparisonResult(
        rows=tuple(sorted(rows, key=lambda r: (r.sharpe is None, -(r.sharpe or 0.0)))),
        skipped=tuple(skipped),
        window=window,
        step=step,
        cost_bps=cost_bps,
        n_windows=n_windows,
        n_symbols=returns.shape[1],
        start=simulated[0],
        end=simulated[-1],
    )
