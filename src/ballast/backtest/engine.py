"""Walk-forward portfolio simulation. Shared infrastructure -- implemented.

Notes
-----
What this file does: the one walk-forward loop in the codebase. The
covariance horse race (v0.2.0) and the optimizer comparison (v0.5.0) are
thin wrappers around run_backtest; neither implements its own loop.

The contract: caller supplies a weights function (window of returns ->
target weights, indexed by symbol). The engine walks the calendar, calls
it at each rebalance, applies costs on turnover, and compounds the equity
curve. Everything a caller might want to report comes back in a
BacktestResult; the engine never prints.

Mechanics that matter (each one is a classic source of fake backtests):
- The 1-bar shift. The window passed to the weights function ends at day
  t-1; the weights it returns first earn day t's return. A strategy can
  never see the day it is about to be scored on.
- Weights DRIFT between rebalances. Holding 50/50 without trading means
  the winner grows past 50%: w -> w * (1 + r) / (1 + r_portfolio) each
  day. Turnover at the next rebalance is measured against the DRIFTED
  weights -- that's what you'd actually have to trade.
- Costs are charged on total traded notional: cost_bps / 10000 x
  sum(|target - current|), subtracted from the rebalance day's return.
  The very first position build pays costs too (going from cash to
  invested is a real trade). Costs are never optional.
- Simulation starts at the first rebalance (day index `window`); earlier
  days exist only as training data.

Conventions: weights must sum to 1 (fully invested; shorts allowed, cash
is not modeled here -- this is returns-space simulation). Turnover is
reported one-sided (sum |trades| / 2), the industry convention.

Failure style: BacktestError. A weights function returning NaN, the wrong
symbols, or a non-unit sum is a bug in the caller and fails loudly.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ballast.portfolio.stats import annualized_vol, cagr, max_drawdown

__all__ = ["BacktestError", "BacktestResult", "run_backtest"]

TRADING_DAYS = 252

# window of returns (dates x symbols) -> target weights (indexed by symbol)
WeightsFn = Callable[[pd.DataFrame], pd.Series]


class BacktestError(RuntimeError):
    """Raised for unusable inputs or a strategy that wipes out."""


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Everything downstream reporting needs; nothing it must recompute."""

    name: str
    equity: pd.Series  # compounded net equity curve, starts near 1.0
    returns: pd.Series  # daily NET returns (after costs)
    ann_return: float  # geometric (CAGR) on net returns
    ann_vol: float
    sharpe: float | None  # mean/std * sqrt(252); None if vol is zero
    max_drawdown: float
    avg_turnover: float  # mean one-sided turnover per rebalance
    cost_drag: float  # gross CAGR minus net CAGR: what trading cost you
    n_rebalances: int
    window: int
    step: int


def _validate_weights(weights: pd.Series, columns: list[str], name: str) -> np.ndarray:
    """A weights function's output is caller code -- check it like input."""
    if not isinstance(weights, pd.Series):
        raise BacktestError(
            f"{name}: weights function must return a Series, got {type(weights).__name__}"
        )
    missing = sorted(set(columns) - set(weights.index))
    if missing:
        raise BacktestError(f"{name}: weights are missing symbol(s) {missing}")
    aligned = weights.reindex(columns).to_numpy(dtype=float)
    if not np.isfinite(aligned).all():
        raise BacktestError(f"{name}: weights contain NaN or inf")
    total = aligned.sum()
    if abs(total - 1.0) > 1e-6:
        raise BacktestError(f"{name}: weights must sum to 1 (fully invested), got {total:.6f}")
    return aligned


def run_backtest(
    returns: pd.DataFrame,
    weights_fn: WeightsFn,
    *,
    window: int = 252,
    step: int = 21,
    cost_bps: float = 2.0,
    name: str = "strategy",
) -> BacktestResult:
    """Walk `returns` forward, rebalancing via weights_fn every `step` days."""
    if not isinstance(returns, pd.DataFrame):
        raise BacktestError(f"returns must be a DataFrame, got {type(returns).__name__}")
    if window < 2 or step < 1:
        raise BacktestError(f"window must be >= 2 and step >= 1, got {window}/{step}")
    clean = returns.sort_index()
    values = clean.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise BacktestError("returns contain NaN or inf; fix alignment upstream")
    n_rows, n_cols = values.shape
    if n_rows < window + 1:
        raise BacktestError(
            f"need at least window+1={window + 1} rows to have one out-of-sample day, got {n_rows}"
        )
    columns = list(clean.columns)

    rebalance_days = set(range(window, n_rows, step))
    current = np.zeros(n_cols)  # pre-first-rebalance: all cash
    net_returns: list[float] = []
    gross_returns: list[float] = []
    turnovers: list[float] = []

    for t in range(window, n_rows):
        cost = 0.0
        if t in rebalance_days:
            # iloc[t - window : t] ends at row t-1: the function sees
            # everything up to yesterday and nothing from today. This line
            # IS the no-lookahead rule; treat any edit to it as suspect.
            window_df = clean.iloc[t - window : t]
            target = _validate_weights(weights_fn(window_df), columns, name)
            traded = float(np.abs(target - current).sum())
            turnovers.append(traded / 2.0)  # one-sided convention
            cost = traded * cost_bps / 10_000.0
            current = target

        day = values[t]
        gross = float(current @ day)
        if 1.0 + gross <= 0.0:
            # -100% in a day: equity hits zero and log/compound math below
            # would produce garbage. A real account would be closed too.
            raise BacktestError(
                f"{name}: wiped out on {clean.index[t].date()} (return {gross:.1%})"
            )
        net_returns.append(gross - cost)
        gross_returns.append(gross)
        # Drift: each holding grows by its own return, then re-express as
        # weights of the (grown) portfolio.
        current = current * (1.0 + day) / (1.0 + gross)

    index = clean.index[window:]
    net = pd.Series(net_returns, index=index)
    gross_series = pd.Series(gross_returns, index=index)

    std = float(net.std(ddof=1))
    sharpe = None if std == 0.0 else float(net.mean()) / std * TRADING_DAYS**0.5

    return BacktestResult(
        name=name,
        equity=(1.0 + net).cumprod(),
        returns=net,
        ann_return=cagr(net),
        ann_vol=annualized_vol(net),
        sharpe=sharpe,
        max_drawdown=max_drawdown(net),
        avg_turnover=float(np.mean(turnovers)),
        cost_drag=cagr(gross_series) - cagr(net),
        n_rebalances=len(turnovers),
        window=window,
        step=step,
    )
