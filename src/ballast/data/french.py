"""Ken French data library download + the factor cross-check. v0.4.0 -- implemented.

Notes
-----
What this file does: fetches the published daily factor return series
(Mkt-RF, SMB, HML from the 3-factor file; Mom from the momentum file) and
correlates Ballast's home-built factor returns against them. This is the
acceptance test for the whole factor pipeline: if the numbers disagree,
the bug is in our construction, not in fifty years of published data.

The pairings and their EXPECTED signs:
- market   vs Mkt-RF   (+): our intercept is the average stock's return;
  Mkt-RF is the cap-weighted market minus the risk-free rate. Not
  identical, but they must move together.
- value    vs HML      (+)
- momentum vs Mom/UMD  (+): the headline check -- >= 0.8 or the
  construction is wrong somewhere upstream.
- size     vs SMB      (-): deliberately negative. Ballast's size
  exposure is log market cap (positive = LARGE), SMB is small-minus-big.
  A positive correlation here would mean a sign bug.
Quality and low_vol have no counterpart in these two files (RMW/CMA live
in the 5-factor file; a later addition if wanted) and are simply not
cross-checked.

Alignment: home factor returns are WEEKLY, keyed by period START, covering
(start, next period]. French dailies are compounded into exactly those
intervals before correlating -- same convention as fit_panel, or the
correlation is meaningless by construction.

Format defensiveness (the files are quirky and pinned by test fixtures):
zipped CSVs; several description lines before the header; dates as
YYYYMMDD; values in PERCENT, not decimals; -99.99 / -999 are missing-data
markers; monthly files (not used here, but people grab the wrong URL)
append annual tables after the daily block -- parsing stops at the first
junk line after data begins.

Design rules:
- Never used as a model input. Ballast's risk numbers come from its own
  factor model; this file is the answer key, not the textbook.
- One network seam (_download), mockable, same pattern as everywhere else.
"""

import io
import urllib.request
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "FrenchDataError",
    "CROSS_CHECK_PAIRS",
    "fetch_french_daily",
    "cross_check_factors",
]

_FACTORS_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)
_MOMENTUM_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"
)

# (ballast factor, french column, expected sign of the correlation)
CROSS_CHECK_PAIRS: tuple[tuple[str, str, int], ...] = (
    ("market", "mkt_rf", +1),
    ("value", "hml", +1),
    ("momentum", "mom", +1),
    ("size", "smb", -1),  # ours is "bigness", SMB is small-minus-big
)

ACCEPTANCE_CORR = 0.8  # the roadmap's bar, applied after the sign flip


class FrenchDataError(RuntimeError):
    """Raised when the French library is unreachable or unparseable."""


@dataclass(frozen=True, slots=True)
class CrossCheckRow:
    """One line of the cross-check verdict table."""

    ballast_factor: str
    french_factor: str
    expected_sign: int
    corr: float
    n_periods: int
    passes: bool  # corr * expected_sign >= ACCEPTANCE_CORR


def _download(url: str) -> bytes:
    """The network seam: one place that touches the internet, mockable."""
    request = urllib.request.Request(url, headers={"User-Agent": "ballast-research"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()
    except Exception as exc:
        raise FrenchDataError(f"French library download failed for {url}: {exc}") from exc


def _parse_french_csv(raw_zip: bytes) -> pd.DataFrame:
    """Zipped French CSV -> DataFrame of DECIMAL daily returns.

    Header detection: the last multi-field line before the first data row
    (a row whose first field is an 8-digit date). Parsing stops at the
    first non-data line after data begins -- that's where monthly files
    hide their annual tables.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(raw_zip)) as archive:
            text = archive.read(archive.namelist()[0]).decode("latin-1")
    except Exception as exc:
        raise FrenchDataError(f"not a readable zip archive: {exc}") from exc

    header: list[str] | None = None
    candidate: list[str] | None = None
    dates: list[str] = []
    rows: list[list[float]] = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts[0]) == 8 and parts[0].isdigit():
            if header is None:
                header = candidate  # the line just above the first data row
            values = []
            for field in parts[1:]:
                try:
                    value = float(field)
                except ValueError:
                    value = np.nan
                # -99.99 / -999 are the library's missing-data markers.
                values.append(np.nan if value <= -99.0 else value)
            dates.append(parts[0])
            rows.append(values)
        elif rows:
            break  # junk after the daily block: annual tables, copyright
        elif len(parts) > 1:
            candidate = parts

    if header is None or not rows:
        raise FrenchDataError("unrecognized file format: no header/data rows found")

    # Header like ['', 'Mkt-RF', 'SMB', 'HML', 'RF'] -> mkt_rf, smb, hml, rf
    names = [c.strip().lower().replace("-", "_") for c in header[1:]]
    frame = pd.DataFrame(rows, columns=names[: len(rows[0])])
    frame.index = pd.to_datetime(dates, format="%Y%m%d")
    return frame / 100.0  # percent -> decimal, once, here


def fetch_french_daily(start: str | None = None) -> pd.DataFrame:
    """Daily mkt_rf/smb/hml/rf/mom, merged from the two library files."""
    factors = _parse_french_csv(_download(_FACTORS_URL))
    momentum = _parse_french_csv(_download(_MOMENTUM_URL))
    merged = factors.join(momentum, how="inner")
    if start is not None:
        merged = merged.loc[merged.index >= start]
    if merged.empty:
        raise FrenchDataError("no overlapping French data in the requested window")
    return merged


def _compound_to_periods(daily: pd.DataFrame, period_starts: pd.DatetimeIndex) -> pd.DataFrame:
    """Compound daily returns into (start_i, start_{i+1}] intervals.

    Output is keyed by the LEFT edge -- the same convention as fit_panel,
    where the return keyed t covers t to the next period.
    """
    bins = pd.cut(daily.index, bins=period_starts, right=True, labels=False)
    mask = ~np.isnan(bins)
    if not mask.any():
        raise FrenchDataError("French data does not overlap the model's periods")
    grouped = (1.0 + daily[mask]).groupby(bins[mask].astype(int)).prod() - 1.0
    grouped.index = period_starts[grouped.index.astype(int)]
    return grouped


def cross_check_factors(
    factor_returns: pd.DataFrame, french_daily: pd.DataFrame
) -> list[CrossCheckRow]:
    """Correlate home-built factor returns against the published series.

    factor_returns: the FactorModel's periods x factors frame (weekly).
    french_daily:   fetch_french_daily()'s output (or a test fixture).
    """
    if len(factor_returns) < 8:
        raise FrenchDataError(
            f"only {len(factor_returns)} fitted periods; the cross-check needs more history"
        )
    period_starts = pd.DatetimeIndex(sorted(factor_returns.index))
    weekly_french = _compound_to_periods(french_daily, period_starts)

    rows: list[CrossCheckRow] = []
    for ballast_name, french_name, sign in CROSS_CHECK_PAIRS:
        if ballast_name not in factor_returns.columns:
            continue  # a reduced factor set is legal; check what exists
        if french_name not in weekly_french.columns:
            raise FrenchDataError(f"French data is missing the {french_name!r} column")
        joined = pd.concat(
            [factor_returns[ballast_name], weekly_french[french_name]], axis=1, join="inner"
        ).dropna()
        if len(joined) < 8:
            raise FrenchDataError(
                f"only {len(joined)} overlapping periods for {ballast_name} vs {french_name}"
            )
        corr = float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))
        rows.append(
            CrossCheckRow(
                ballast_factor=ballast_name,
                french_factor=french_name,
                expected_sign=sign,
                corr=corr,
                n_periods=len(joined),
                passes=corr * sign >= ACCEPTANCE_CORR,
            )
        )
    return rows
