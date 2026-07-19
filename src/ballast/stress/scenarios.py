"""Scenario replay and hypothetical shocks. v0.6.0 -- implemented.

Notes
-----
What this file does: answers two different questions with two different
tools, both about the CURRENT portfolio.

1. "What would these exact holdings have done through <crisis>?"
   replay_scenario(): buy-and-hold the current weights through a named
   historical window using stored prices. Each position's window return
   is compounded from actual bars; the portfolio return is the weighted
   sum (cash contributes zero); contributions name the drivers.

   Coverage is STRICT: a symbol whose price history doesn't span the
   window (a 2024 IPO in gfc2008) is an error naming the symbol, never a
   silent hole. The factor-based alternative -- mapping the scenario to
   factor returns and pushing them through current exposures, which would
   cover every holding -- needs stored factor-return history through the
   scenario itself; it is deferred until the panel covers those years,
   and this docstring says so instead of pretending.

2. "What if <factor> moves by X, right now?"
   factor_shock(): pure arithmetic on the portfolio's current factor
   exposures: estimated return = sum over shocked factors of
   x_f * shock_f, with x = B'w (market exposure = invested fraction).
   No history needed, covers every holding, answers questions like
   "what does a -20% market with a momentum crash do to me?" The
   linearity is the model's, and its limit: specific risk and convexity
   are outside the estimate.

Scenarios are DATA (dates + description), not code branches: adding one
means adding a constant to SCENARIOS, nothing else.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ballast.data.prices import load_prices

__all__ = [
    "StressError",
    "Scenario",
    "SCENARIOS",
    "StressResult",
    "ShockResult",
    "replay_scenario",
    "factor_shock",
]


class StressError(ValueError):
    """Raised for unknown scenarios, missing coverage, or bad shocks."""


@dataclass(frozen=True, slots=True)
class Scenario:
    """A named historical window."""

    name: str
    start: str  # ISO dates, inclusive
    end: str
    description: str


SCENARIOS: dict[str, Scenario] = {
    "gfc2008": Scenario(
        "gfc2008", "2008-09-01", "2009-03-09", "Lehman weekend to the March 2009 bottom"
    ),
    "covid2020": Scenario(
        "covid2020", "2020-02-19", "2020-03-23", "COVID crash, S&P peak to trough"
    ),
    "rates2022": Scenario(
        "rates2022", "2022-01-03", "2022-10-14", "the 2022 rate-hike bear market"
    ),
}

# A symbol's history must reach within this many calendar days of both
# window edges to count as covering the scenario.
_EDGE_TOLERANCE_DAYS = 7


@dataclass(frozen=True, slots=True)
class StressResult:
    """One replay: what the current book would have done."""

    scenario: Scenario
    portfolio_return: float  # weighted sum; cash at 0
    position_returns: pd.Series  # each holding's own window return
    contributions: pd.Series  # weight x return, sorted worst first
    invested_fraction: float


@dataclass(frozen=True, slots=True)
class ShockResult:
    """One hypothetical: linear factor-model estimate."""

    shocks: dict[str, float]
    portfolio_return: float
    factor_contributions: pd.Series  # x_f * shock_f per shocked factor
    exposures: pd.Series  # the x vector used


def replay_scenario(
    weights: pd.Series,
    scenario: str | Scenario,
    db_path: Any = None,
) -> StressResult:
    """Buy-and-hold the current weights through a named historical window."""
    if isinstance(scenario, str):
        if scenario not in SCENARIOS:
            raise StressError(f"unknown scenario {scenario!r}; available: {sorted(SCENARIOS)}")
        scenario = SCENARIOS[scenario]
    if weights.empty:
        raise StressError("no positions to stress")

    symbols = list(weights.index)
    prices = load_prices(symbols, start=scenario.start, end=scenario.end, db_path=db_path)

    start_ts = pd.Timestamp(scenario.start)
    end_ts = pd.Timestamp(scenario.end)
    tolerance = pd.Timedelta(days=_EDGE_TOLERANCE_DAYS)
    position_returns = {}
    uncovered: list[str] = []
    for symbol in symbols:
        series = prices[symbol].dropna()
        # Strict coverage: history must reach both edges of the window.
        if (
            series.empty
            or series.index[0] > start_ts + tolerance
            or series.index[-1] < end_ts - tolerance
        ):
            uncovered.append(symbol)
            continue
        position_returns[symbol] = float(series.iloc[-1] / series.iloc[0] - 1.0)
    if uncovered:
        raise StressError(
            f"no {scenario.name} price coverage for symbol(s) {sorted(uncovered)}; "
            "raw replay cannot include assets that didn't trade through the window"
        )

    returns = pd.Series(position_returns).reindex(symbols)
    contributions = (weights * returns).sort_values()  # worst drivers first
    return StressResult(
        scenario=scenario,
        portfolio_return=float(contributions.sum()),
        position_returns=returns,
        contributions=contributions,
        invested_fraction=float(weights.sum()),
    )


def factor_shock(
    weights: pd.Series,
    exposures: pd.DataFrame,
    shocks: dict[str, float],
) -> ShockResult:
    """Linear estimate of a hypothetical factor move, through current exposures.

    shocks: factor -> assumed return, e.g. {"market": -0.20, "momentum": -0.10}.
    "market" is always available (exposure = invested fraction); style
    factors must exist in the exposures frame.
    """
    if not shocks:
        raise StressError("no shocks given")
    if weights.empty:
        raise StressError("no positions to stress")
    missing_rows = sorted(set(weights.index) - set(exposures.index))
    if missing_rows:
        raise StressError(f"no exposures for symbol(s) {missing_rows}")
    b_style = exposures.loc[list(weights.index)]
    holes = sorted(b_style.index[b_style.isna().any(axis=1)])
    if holes:
        raise StressError(f"exposures contain NaN for symbol(s) {holes}")

    # x = B'w with the implicit market column of ones.
    x = pd.Series(
        np.concatenate([[weights.sum()], b_style.to_numpy().T @ weights.to_numpy()]),
        index=["market", *b_style.columns],
    )

    unknown = sorted(set(shocks) - set(x.index))
    if unknown:
        raise StressError(f"unknown factor(s) {unknown}; available: {list(x.index)}")
    for factor, value in shocks.items():
        if not np.isfinite(value):
            raise StressError(f"shock for {factor!r} must be finite")

    contributions = pd.Series({f: x[f] * v for f, v in shocks.items()}).sort_values()
    return ShockResult(
        shocks=dict(shocks),
        portfolio_return=float(contributions.sum()),
        factor_contributions=contributions,
        exposures=x,
    )
