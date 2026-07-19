"""Portfolio-level descriptive stats: the math behind `ballast stats`. v0.1.0.

Notes
-----
What this file does: turns a resolved portfolio plus a returns matrix into
the four headline numbers -- CAGR, annualized vol, max drawdown, beta --
packaged in a PortfolioStats dataclass for the renderer. Pure math: no
database, no network, no printing.

The one modeling assumption, stated up front: the portfolio's history is
computed with CURRENT weights held constant over the whole window (which
implies daily rebalancing back to those weights). That answers "how has
this mix behaved?", not "what did my account actually earn?" -- the latter
needs transaction history, which the spec deliberately doesn't carry.
Cash earns 0% and drags on both return and vol exactly as it should.

Definitions used (all standard):
- CAGR: total compounded growth, annualized as x^(252/n) - 1.
- annualized vol: sample std (ddof=1) of daily returns x sqrt(252).
- max drawdown: worst peak-to-trough fall of the compounded equity curve;
  reported as a negative number.
- beta: cov(portfolio, benchmark) / var(benchmark) over their SHARED dates.
  None when there's no usable benchmark -- absence is honest, 0.0 is a lie.

Failure style: StatsError for degenerate inputs (cash-only portfolio,
window too short, weights that don't match the returns columns).
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ballast.portfolio.spec import ResolvedPortfolio

__all__ = [
    "StatsError",
    "PortfolioStats",
    "blended_returns",
    "cagr",
    "annualized_vol",
    "max_drawdown",
    "beta",
    "compute_stats",
]

TRADING_DAYS = 252  # the standard annualization convention for daily bars


class StatsError(ValueError):
    """Raised for inputs that have no meaningful statistics."""


@dataclass(frozen=True, slots=True)
class PortfolioStats:
    """Everything the stats renderer needs; nothing it has to compute."""

    name: str
    start: Any  # first/last dates of the window (pandas Timestamps)
    end: Any
    n_days: int
    cagr: float
    ann_vol: float
    max_drawdown: float
    beta: float | None  # None = no benchmark data, deliberately not 0.0
    benchmark: str
    nav: float | None  # None for scale-free portfolios (weights only)
    weights: dict[str, float]
    cash_weight: float


def blended_returns(returns: pd.DataFrame, resolved: ResolvedPortfolio) -> pd.Series:
    """Daily portfolio returns: weighted sum of asset returns, cash at 0%.

    Cash needs no term of its own: weights + cash_weight sum to 1, and the
    cash leg contributes cash_weight * 0. The drag shows up because the
    asset weights sum to LESS than 1.
    """
    if not resolved.weights:
        raise StatsError(
            f"portfolio {resolved.name!r} holds only cash; there is nothing to measure"
        )
    missing = sorted(set(resolved.weights) - set(returns.columns))
    if missing:
        raise StatsError(f"returns matrix is missing column(s) {missing}")

    symbols = list(resolved.weights)
    weights = pd.Series(resolved.weights)
    # Row-by-row dot product: r_p(t) = sum_i w_i * r_i(t).
    return (returns[symbols] * weights).sum(axis=1)


def cagr(daily: pd.Series) -> float:
    """Compound annual growth rate of a daily return series."""
    _require_window(daily)
    total = float((1.0 + daily).prod())  # total growth factor over the window
    return total ** (TRADING_DAYS / len(daily)) - 1.0


def annualized_vol(daily: pd.Series) -> float:
    """Sample standard deviation of daily returns, annualized by sqrt(252)."""
    _require_window(daily)
    return float(daily.std(ddof=1)) * TRADING_DAYS**0.5


def max_drawdown(daily: pd.Series) -> float:
    """Worst peak-to-trough decline of the compounded equity curve (<= 0)."""
    _require_window(daily)
    equity = (1.0 + daily).cumprod()
    # cummax carries the running peak forward; equity/peak - 1 is how far
    # below the peak each day sits. The minimum of that is the max drawdown.
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def beta(portfolio: pd.Series, benchmark: pd.Series) -> float | None:
    """cov(p, b) / var(b) over shared dates; None if it can't be estimated."""
    # Inner join on dates: beta only makes sense where both series exist.
    joined = pd.concat([portfolio, benchmark], axis=1, join="inner").dropna()
    if len(joined) < 2:
        return None
    p, b = joined.iloc[:, 0], joined.iloc[:, 1]
    var_b = float(b.var(ddof=1))
    if var_b == 0.0:  # flat benchmark -> division by zero, not a real beta
        return None
    return float(p.cov(b)) / var_b


def compute_stats(
    resolved: ResolvedPortfolio,
    returns: pd.DataFrame,
    benchmark_returns: pd.Series | None,
    benchmark_name: str,
) -> PortfolioStats:
    """Assemble the full PortfolioStats from pre-loaded data."""
    daily = blended_returns(returns, resolved)
    return PortfolioStats(
        name=resolved.name,
        start=daily.index[0],
        end=daily.index[-1],
        n_days=len(daily),
        cagr=cagr(daily),
        ann_vol=annualized_vol(daily),
        max_drawdown=max_drawdown(daily),
        beta=beta(daily, benchmark_returns) if benchmark_returns is not None else None,
        benchmark=benchmark_name,
        nav=resolved.nav,
        weights=dict(resolved.weights),
        cash_weight=resolved.cash_weight,
    )


def _require_window(daily: pd.Series) -> None:
    if len(daily) < 2:
        raise StatsError(f"need at least 2 daily returns, got {len(daily)}")
