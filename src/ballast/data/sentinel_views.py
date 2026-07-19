"""Sentinel screen scores -> Ballast. The bridge's data half. v0.6.0.

Notes
-----
What this file does: loads a scores file -- the interchange format between
the two projects. Sentinel's screen ranks tickers; Ballast turns rankings
into portfolios. The contract is deliberately minimal so it survives both
projects evolving:

    symbol,score
    NVDA,9.1
    V,8.9
    ...

A plain CSV with exactly those two columns (Sentinel's screen milestone
exports it; until then, write one by hand or from any ranking you trust).
Scores are UNITLESS rankings -- only their cross-sectional ordering and
spread matter, because views_from_scores z-scores them before mapping to
expected returns. A score of 9.1 means "top of this list", not "9.1%".

Strictness, as everywhere: duplicate symbols, NaN scores, or a constant
column (no ranking information) are errors, not warnings.
"""

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["ScoresError", "load_scores"]


class ScoresError(ValueError):
    """Raised when a scores file is missing, malformed, or uninformative."""


def load_scores(path: Path | str) -> pd.Series:
    """Read the symbol,score CSV into a Series (symbols upper-cased)."""
    path = Path(path)
    if not path.is_file():
        raise ScoresError(f"scores file not found: {path}")
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise ScoresError(f"could not parse {path} as CSV: {exc}") from exc

    columns = [c.strip().lower() for c in frame.columns]
    if columns[:2] != ["symbol", "score"]:
        raise ScoresError(f"{path}: expected columns 'symbol,score', found {list(frame.columns)}")
    frame.columns = columns

    symbols = frame["symbol"].astype(str).str.strip().str.upper()
    if symbols.duplicated().any():
        dupes = sorted(symbols[symbols.duplicated()].unique())
        raise ScoresError(f"{path}: duplicate symbol(s) {dupes}")

    scores = pd.to_numeric(frame["score"], errors="coerce")
    if scores.isna().any():
        bad = sorted(symbols[scores.isna()])
        raise ScoresError(f"{path}: non-numeric score(s) for {bad}")
    values = scores.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ScoresError(f"{path}: scores must be finite")
    if len(values) < 2:
        raise ScoresError(f"{path}: need at least 2 scored symbols to rank anything")
    if float(np.std(values)) == 0.0:
        raise ScoresError(f"{path}: all scores identical; there is no ranking here")

    return pd.Series(values, index=list(symbols))
