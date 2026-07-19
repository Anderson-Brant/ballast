"""Style factor exposures per stock per date. v0.4.0 -- implemented.

Notes
-----
What this file does: builds one cross-section -- symbol x factor exposures
as of a given date -- from prices and point-in-time fundamentals. The
regression module calls this repeatedly (weekly) to build the panel.

The five factors, their ingredients, and their orientation (positive
exposure always means MORE of the named thing):
- value: mean of z(book/price) and z(earnings/price). Book equity, annual
  net income, and shares outstanding come from EDGAR as-of the date;
  price is the raw close the market saw that day.
- momentum: the 12-1 return -- from 252 through 21 trading days ago,
  skipping the most recent month because short-horizon returns REVERSE
  (buying last week's winner is a different, losing strategy).
- size: z(log market cap). Positive = large-cap. (The academic factor SMB
  is small-minus-big, i.e. the other sign; the French cross-check in the
  regression module handles the flip.)
- quality: mean of z(gross profit / assets) and z(-liabilities / assets)
  -- Novy-Marx profitability plus low leverage.
- low_vol: -z(trailing 252-day realized vol). Positive = calm stock.

The cleaning pipeline, in this order, applied per cross-section:
1. winsorize each raw characteristic at the 1st/99th percentile (a single
   meme-stock B/P must not own the whole z-scale)
2. z-score cross-sectionally (mean 0, std 1)
3. build the two-ingredient composites (value, quality), re-standardized
4. sector-neutralize by demeaning within sector, when a sector map is
   supplied -- exposures then say "cheap FOR a bank", not "cheap because
   banks look cheap". Without a map this step is skipped, stated loudly
   in the docstring rather than silently approximated. (Sector data
   source lands with the universe work; the pipeline is ready for it.)

Missing data policy: a symbol lacking an ingredient gets NaN for the
factors needing it and real values for the rest. NaN, never a guess --
the regression drops NaN rows per date. A symbol with under ~13 months of
prices has no momentum, by definition rather than by bug.

The no-leakage rule lives in the loaders this module calls:
latest_fundamentals(as_of=...) and load_latest_prices(as_of=...) can only
serve what was public by that date.
"""

import math
from collections.abc import Mapping
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ballast.data.edgar import latest_fundamentals
from ballast.data.prices import load_latest_prices, load_prices

__all__ = [
    "ExposureError",
    "FACTORS",
    "raw_characteristics",
    "clean_cross_section",
    "compute_exposures",
]

FACTORS = ("value", "momentum", "size", "quality", "low_vol")

MOMENTUM_LONG = 252  # 12 months back...
MOMENTUM_SKIP = 21  # ...to 1 month back (the "12-1" convention)
VOL_WINDOW = 252
MIN_VOL_OBS = 60  # fewer daily returns than this -> vol is NaN, not noise


class ExposureError(ValueError):
    """Raised when a cross-section cannot be built at all."""


def _winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Clip to cross-sectional quantiles; NaNs pass through untouched."""
    if series.dropna().empty:
        return series
    lo, hi = series.quantile(lower), series.quantile(upper)
    return series.clip(lo, hi)


def _zscore(series: pd.Series) -> pd.Series:
    """Cross-sectional z-score; a degenerate (constant) column becomes 0."""
    std = series.std(ddof=0)
    if not np.isfinite(std) or std == 0.0:
        return series * 0.0  # keeps NaNs as NaN, constants as 0
    return (series - series.mean()) / std


def raw_characteristics(
    adj_prices: pd.DataFrame,
    close_as_of: Mapping[str, float],
    fundamentals: pd.DataFrame,
) -> pd.DataFrame:
    """Raw (un-normalized) characteristics per symbol. Pure function.

    adj_prices: wide adjusted price history through the as-of date, one
    column per symbol (its own history; NaN where it didn't trade).
    close_as_of: raw close per symbol at the as-of date (for market cap).
    fundamentals: the latest_fundamentals frame (symbols x fields).
    """
    rows = {}
    for symbol in adj_prices.columns:
        prices = adj_prices[symbol].dropna()
        f = fundamentals.loc[symbol] if symbol in fundamentals.index else pd.Series(dtype=float)

        # Market cap: raw close x shares outstanding, both as-of.
        close = close_as_of.get(symbol, np.nan)
        shares = f.get("shares_outstanding", np.nan)
        mcap = close * shares if np.isfinite(close) and np.isfinite(shares) else np.nan
        if not np.isfinite(mcap) or mcap <= 0:
            mcap = np.nan  # a nonpositive mcap poisons every ratio below

        # Momentum 12-1: price 21 days ago over price 252 days ago.
        momentum = np.nan
        if len(prices) >= MOMENTUM_LONG + 1:
            momentum = prices.iloc[-1 - MOMENTUM_SKIP] / prices.iloc[-1 - MOMENTUM_LONG] - 1.0

        # Trailing realized vol (daily units; z-scoring makes scale moot).
        returns = prices.pct_change(fill_method=None).dropna().iloc[-VOL_WINDOW:]
        vol = float(returns.std(ddof=1)) if len(returns) >= MIN_VOL_OBS else np.nan

        book = f.get("book_equity", np.nan)
        income = f.get("net_income", np.nan)
        gross = f.get("gross_profit", np.nan)
        assets = f.get("assets", np.nan)
        liabilities = f.get("liabilities", np.nan)

        rows[symbol] = {
            "bp": book / mcap if np.isfinite(book) and np.isfinite(mcap) else np.nan,
            "ep": income / mcap if np.isfinite(income) and np.isfinite(mcap) else np.nan,
            "log_mcap": math.log(mcap) if np.isfinite(mcap) else np.nan,
            "gp_assets": gross / assets
            if np.isfinite(gross) and np.isfinite(assets) and assets > 0
            else np.nan,
            "leverage": liabilities / assets
            if np.isfinite(liabilities) and np.isfinite(assets) and assets > 0
            else np.nan,
            "momentum": momentum,
            "vol": vol,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def clean_cross_section(
    raw: pd.DataFrame, sectors: Mapping[str, str] | None = None
) -> pd.DataFrame:
    """winsorize -> z-score -> composites -> optional sector demeaning.

    The order is the contract (see module notes); don't rearrange it.
    """
    if raw.empty:
        raise ExposureError("empty cross-section: no symbols with any data")

    z = raw.apply(_winsorize).apply(_zscore)

    out = pd.DataFrame(index=raw.index)
    # Two-ingredient composites: mean of available z's (one missing
    # ingredient -> use the other; both missing -> NaN), re-standardized
    # so every factor column is on the same scale.
    out["value"] = _zscore(pd.concat([z["bp"], z["ep"]], axis=1).mean(axis=1))
    out["momentum"] = z["momentum"]
    out["size"] = z["log_mcap"]
    out["quality"] = _zscore(pd.concat([z["gp_assets"], -z["leverage"]], axis=1).mean(axis=1))
    out["low_vol"] = -z["vol"]

    if sectors is not None:
        known = out.index.to_series().map(dict(sectors))
        for factor in FACTORS:
            # Demean within sector: "cheap for a bank", not "banks are
            # cheap". Symbols without a sector keep their global z.
            group_means = out[factor].groupby(known).transform("mean")
            out[factor] = out[factor].where(known.isna(), out[factor] - group_means)

    return out[list(FACTORS)]


def compute_exposures(
    symbols: list[str],
    as_of: str | date,
    db_path: Path | str | None = None,
    sectors: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """One cross-section from the database: symbols x FACTORS at as_of.

    Every input honors the as-of date: fundamentals by filing date, prices
    by bar date. The output is what a researcher standing at that date's
    close could actually have computed.
    """
    if not symbols:
        raise ExposureError("compute_exposures: no symbols given")
    requested = list(dict.fromkeys(s.strip().upper() for s in symbols if s.strip()))
    as_of_iso = (
        as_of.isoformat() if isinstance(as_of, date) else date.fromisoformat(as_of).isoformat()
    )

    adj_prices = load_prices(requested, end=as_of_iso, db_path=db_path)
    close_as_of = load_latest_prices(requested, db_path=db_path, as_of=as_of_iso)
    fundamentals = latest_fundamentals(requested, as_of=as_of_iso, db_path=db_path)

    raw = raw_characteristics(adj_prices, close_as_of, fundamentals)
    return clean_cross_section(raw, sectors)
