"""Cross-sectional regressions -> factor returns -> factor covariance. v0.4.0 -- implemented.

Notes
-----
What this file does: the estimation half of the factor model. For each
period (weekly by convention), regress that period's stock returns on the
exposures known at the period's START:

    r_{t -> t+1} = market + B_t f + e        (one regression per period)

The coefficient series ARE the factor returns: f_hat("value") for a week
is the return you'd have earned that week per unit of value exposure,
holding everything else constant. The intercept is the market factor --
with z-scored (mean-zero) exposures it's the return of the average stock.

From the panel of fits:
- factor_returns: periods x (market + style factors)
- factor_cov: EWMA covariance of the factor returns (lambda=0.97, the
  RiskMetrics convention for WEEKLY data; 0.94 is the daily one)
- specific_variance: per-symbol variance of that symbol's residuals --
  the risk the factors don't explain. NaN below min_specific_obs periods.
- r2 and n_obs per period: the diagnostics, stored not discarded. A
  factor whose contribution never shows up belongs in methodology.md as
  a finding, not in the model as decoration.

Units discipline: everything here is in PER-PERIOD units (weekly returns,
weekly variance). periods_per_year rides along in the FactorModel so the
decomposition layer annualizes exactly once, at the end.

WLS: pass per-symbol weights (convention: sqrt of market cap) so megacaps
don't drown in a sea of small caps -- or rather so small caps don't drown
the megacaps, since equal weighting lets the many small names dominate
the fit. Default is equal weights (OLS); the choice is the caller's and
recorded by them.

Skip policy: a period whose cross-section is too thin to fit (fewer
symbols than coefficients + 2) is SKIPPED and counted; more than 20%
skips fails the whole fit. Silent gaps lie, occasional gaps happen.

The acceptance test this module must eventually pass (the reason
data/french.py exists): correlation of home-built factor returns against
the published French series, momentum vs UMD >= 0.8. That needs long
real-universe history; the offline tests here use the stronger check
available to synthetic data -- exact recovery of KNOWN factor returns.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ballast.covariance.estimators import ewma_cov
from ballast.data.prices import load_prices
from ballast.factors.exposures import compute_exposures

__all__ = [
    "RegressionError",
    "FactorModel",
    "fit_cross_section",
    "fit_panel",
    "build_panel",
]

WEEKLY_LAMBDA = 0.97  # RiskMetrics decay for weekly observations
PERIODS_PER_YEAR = 52


class RegressionError(ValueError):
    """Raised when a cross-section or panel cannot support a fit."""


@dataclass(frozen=True, slots=True)
class FactorModel:
    """The fitted model: everything risk/decompose.py needs, in period units."""

    factor_returns: pd.DataFrame  # periods x (market + style factors)
    factor_cov: pd.DataFrame  # EWMA, per-period units
    specific_variance: pd.Series  # per symbol, per-period units; NaN if thin
    r2: pd.Series  # per period
    n_obs: pd.Series  # symbols used per period
    periods_per_year: int  # 52 for the weekly convention


def fit_cross_section(
    exposures: pd.DataFrame,
    realized: pd.Series,
    weights: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series, float]:
    """One period's regression. Returns (coefficients, residuals, r2).

    exposures: symbols x factors, as known at the period start.
    realized:  each symbol's return OVER the period.
    weights:   optional WLS weights (e.g. sqrt mcap); positive where given.

    Symbols missing any exposure, the return, or the weight are dropped for
    this period -- the NaN-never-guess policy arriving at its destination.
    """
    frame = exposures.copy()
    frame["_r"] = realized
    if weights is not None:
        frame["_w"] = weights
    frame = frame.dropna()

    n_obs = len(frame)
    n_coefs = exposures.shape[1] + 1  # +1: the intercept / market factor
    if n_obs < n_coefs + 2:
        raise RegressionError(f"cross-section too thin: {n_obs} symbols for {n_coefs} coefficients")

    x = np.column_stack([np.ones(n_obs), frame[list(exposures.columns)].to_numpy(dtype=float)])
    y = frame["_r"].to_numpy(dtype=float)
    w = frame["_w"].to_numpy(dtype=float) if weights is not None else np.ones(n_obs)
    if not np.isfinite(w).all() or (w <= 0).any():
        raise RegressionError("WLS weights must be finite and positive")

    # WLS objective sum(w_i e_i^2) == OLS on rows scaled by sqrt(w_i).
    scale = np.sqrt(w)
    coefs, *_ = np.linalg.lstsq(x * scale[:, None], y * scale, rcond=None)

    fitted = x @ coefs
    residuals = y - fitted
    ss_res = float((residuals**2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    names = ["market", *exposures.columns]
    return (
        pd.Series(coefs, index=names),
        pd.Series(residuals, index=frame.index),
        r2,
    )


def fit_panel(
    exposures_panel: Mapping[Any, pd.DataFrame],
    period_returns: Mapping[Any, pd.Series],
    weights_panel: Mapping[Any, pd.Series] | None = None,
    lam: float = WEEKLY_LAMBDA,
    periods_per_year: int = PERIODS_PER_YEAR,
    min_specific_obs: int = 20,
    max_skip_fraction: float = 0.2,
) -> FactorModel:
    """Fit every period, assemble the model. Pure computation.

    Keys of exposures_panel and period_returns must align: for date t the
    exposures are as-of t and the return covers (t, next period]. The
    caller (build_panel) guarantees that alignment; this function only
    intersects the keys.
    """
    dates = sorted(set(exposures_panel) & set(period_returns))
    if not dates:
        raise RegressionError("no overlapping dates between exposures and returns")

    coef_rows: dict[Any, pd.Series] = {}
    resid_rows: dict[Any, pd.Series] = {}
    r2_by_date: dict[Any, float] = {}
    nobs_by_date: dict[Any, int] = {}
    skipped = 0

    for ts in dates:
        w = weights_panel.get(ts) if weights_panel is not None else None
        try:
            coefs, residuals, r2 = fit_cross_section(exposures_panel[ts], period_returns[ts], w)
        except RegressionError:
            skipped += 1  # thin week: counted, not hidden
            continue
        coef_rows[ts] = coefs
        resid_rows[ts] = residuals
        r2_by_date[ts] = r2
        nobs_by_date[ts] = len(residuals)

    if skipped / len(dates) > max_skip_fraction:
        raise RegressionError(
            f"{skipped} of {len(dates)} periods were too thin to fit; "
            "the universe is too small for this factor set"
        )
    if not coef_rows:
        raise RegressionError("every period failed to fit")

    factor_returns = pd.DataFrame(coef_rows).T.sort_index()
    residuals = pd.DataFrame(resid_rows).T.sort_index()  # periods x symbols, NaN gaps

    # Specific variance needs enough residuals to mean anything; below the
    # floor it's NaN and the decomposition will say so rather than guess.
    counts = residuals.notna().sum()
    specific = residuals.var(ddof=1).where(counts >= min_specific_obs)

    return FactorModel(
        factor_returns=factor_returns,
        factor_cov=ewma_cov(factor_returns, lam=lam),
        specific_variance=specific,
        r2=pd.Series(r2_by_date).sort_index(),
        n_obs=pd.Series(nobs_by_date).sort_index(),
        periods_per_year=periods_per_year,
    )


def build_panel(
    symbols: list[str],
    start: str | date,
    end: str | date,
    db_path: Path | str | None = None,
    sectors: Mapping[str, str] | None = None,
) -> tuple[dict, dict]:
    """DB -> (exposures_panel, period_returns), weekly, aligned for fit_panel.

    Panel dates are each week's LAST trading day within [start, end].
    Exposures at date t use compute_exposures(as_of=t) -- so fundamentals
    by filing date, prices through t. The return for key t covers t to the
    next panel date: strictly after the exposures' information set, which
    is the entire point.
    """
    if not symbols:
        raise RegressionError("build_panel: no symbols given")
    start_iso = start.isoformat() if isinstance(start, date) else start
    end_iso = end.isoformat() if isinstance(end, date) else end

    # Full adjusted history through `end` (momentum needs ~13 months of
    # runway before `start`, so no start filter on the price load).
    prices = load_prices(symbols, end=end_iso, db_path=db_path)

    in_window = prices.index[(prices.index >= start_iso) & (prices.index <= end_iso)]
    if len(in_window) < 10:
        raise RegressionError(
            f"only {len(in_window)} trading days between {start_iso} and {end_iso}"
        )
    # Last trading day of each ISO week: robust to holidays, no calendar math.
    week_ends = in_window.to_series().groupby(in_window.to_period("W")).max().tolist()

    exposures_panel: dict = {}
    period_returns: dict = {}
    for current, nxt in zip(week_ends[:-1], week_ends[1:], strict=False):
        exposures_panel[current] = compute_exposures(
            symbols, as_of=current.date(), db_path=db_path, sectors=sectors
        )
        period_returns[current] = prices.loc[nxt] / prices.loc[current] - 1.0

    if not exposures_panel:
        raise RegressionError("not enough weeks in the window to build a panel")
    return exposures_panel, period_returns
