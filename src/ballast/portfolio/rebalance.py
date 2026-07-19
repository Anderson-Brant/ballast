"""Drift-band rebalancing: current book -> target book, as a trade plan. v0.6.0.

Notes
-----
What this file does: compares the portfolio you HOLD against the weights
you WANT and produces the minimal honest trade list -- which positions to
touch, by how many dollars, at what estimated cost. It plans; it never
executes (Ballast produces information, not orders).

The drift band is the whole idea: trading back to target on every
one-point wobble burns cost for nothing, so a position only trades when
|current - target| exceeds `band` (in absolute weight points, e.g. 0.05
= five points of NAV). Positions inside the band are reported as skipped
WITH their drift -- visible restraint, not silence. A minimum dollar
trade size filters the economically silly remainder ("sell $37 of AAPL").

Semantics pinned down so nobody argues later:
- Trades are computed on the CURRENT NAV; a scale-free portfolio
  (weights-only spec, no dollar anchor) cannot be planned and says so.
- The symbol universe is the UNION of current and target: a holding
  absent from the target is an exit (target 0); a target absent from the
  book is a new position (current 0).
- Targets are long-only weights summing to <= 1; the remainder is cash.
  Costs are cost_bps per side on traded notional -- same convention as
  the backtest engine, one definition across the codebase.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ballast.portfolio.spec import PortfolioSpecError, ResolvedPortfolio

__all__ = ["RebalanceError", "Trade", "RebalancePlan", "plan_rebalance"]

_TOL = 1e-9


class RebalanceError(ValueError):
    """Raised when a trade plan cannot be produced honestly."""


@dataclass(frozen=True, slots=True)
class Trade:
    """One planned trade. Positive dollars = buy, negative = sell."""

    symbol: str
    current_weight: float
    target_weight: float
    drift: float  # target - current, in weight points
    dollars: float
    est_cost: float


@dataclass(frozen=True, slots=True)
class RebalancePlan:
    """The full plan: trades, visible restraint, and the bill."""

    nav: float
    band: float
    trades: tuple[Trade, ...]  # largest absolute dollars first
    skipped: tuple[tuple[str, float], ...]  # (symbol, drift) inside the band
    total_traded: float  # sum of |dollars|
    total_cost: float
    cash_weight_after: float


def plan_rebalance(
    current: ResolvedPortfolio,
    targets: pd.Series,
    band: float = 0.05,
    cost_bps: float = 2.0,
    min_trade_dollars: float = 0.0,
) -> RebalancePlan:
    """Plan the trades that move `current` to `targets`, band-gated."""
    if current.nav is None:
        raise RebalanceError(
            "portfolio is scale-free (weights-only spec, no cash): dollar "
            "trades need a NAV -- give the spec shares or cash"
        )
    if not 0.0 <= band < 1.0:
        raise RebalanceError(f"band must be in [0, 1), got {band}")
    if cost_bps < 0 or min_trade_dollars < 0:
        raise RebalanceError("cost_bps and min_trade_dollars must be >= 0")

    values = targets.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise RebalanceError("targets must be finite, long-only weights")
    if values.sum() > 1.0 + _TOL:
        raise RebalanceError(f"target weights sum to {values.sum():.6f}, must be <= 1")
    if targets.index.has_duplicates:
        raise RebalanceError("targets contain duplicate symbols")

    # Union universe: exits (in book, no target) and entries (target, no book).
    current_w = pd.Series(current.weights)
    universe = sorted(set(current_w.index) | set(targets.index))
    cur = current_w.reindex(universe).fillna(0.0)
    tgt = targets.reindex(universe).fillna(0.0)

    trades: list[Trade] = []
    skipped: list[tuple[str, float]] = []
    for symbol in universe:
        drift = float(tgt[symbol] - cur[symbol])
        dollars = drift * current.nav
        if abs(drift) <= band or abs(dollars) < min_trade_dollars:
            if abs(drift) > _TOL:  # perfectly-on-target rows are just noise
                skipped.append((symbol, drift))
            continue
        trades.append(
            Trade(
                symbol=symbol,
                current_weight=float(cur[symbol]),
                target_weight=float(tgt[symbol]),
                drift=drift,
                dollars=dollars,
                est_cost=abs(dollars) * cost_bps / 10_000.0,
            )
        )

    trades.sort(key=lambda t: -abs(t.dollars))
    traded_symbols = {t.symbol for t in trades}
    # Post-trade book: traded positions land on target, the rest stay put.
    after = pd.Series(
        {s: (tgt[s] if s in traded_symbols else cur[s]) for s in universe}, dtype=float
    )
    return RebalancePlan(
        nav=current.nav,
        band=band,
        trades=tuple(trades),
        skipped=tuple(sorted(skipped, key=lambda item: -abs(item[1]))),
        total_traded=float(sum(abs(t.dollars) for t in trades)),
        total_cost=float(sum(t.est_cost for t in trades)),
        cash_weight_after=float(1.0 - after.sum()),
    )


def targets_from_resolved(resolved: ResolvedPortfolio) -> pd.Series:
    """Convenience: a resolved portfolio spec reused as a target book."""
    if not resolved.weights:
        raise PortfolioSpecError("target portfolio holds only cash")
    return pd.Series(resolved.weights)
