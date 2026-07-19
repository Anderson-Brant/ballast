"""Point-in-time fundamentals from SEC EDGAR. v0.4.0 -- implemented.

Notes
-----
What this file does: pulls the fundamental fields factor construction
needs (book equity, net income, revenue, gross profit, assets,
liabilities, shares outstanding) from EDGAR's company-facts API and
stores every reported value twice-dated: the fiscal PERIOD it describes
and the day it was FILED.

Why the filing date is sacred: a 10-K for fiscal 2025 might not be public
until March 2026. If exposures use it before the filing date, the factor
model is trained on information that didn't exist yet -- the same leakage
rule as everywhere else in Ballast, enforced here structurally:
latest_fundamentals() takes an as_of date and only ever serves rows with
filed <= as_of. Downstream code cannot leak by accident.

EDGAR mechanics worth knowing:
- No API key, but requests MUST carry a User-Agent identifying you with
  contact info (set BALLAST_EDGAR_USER_AGENT), and the fair-use limit is
  ~10 requests/second. A polite delay is built into the fetch seam.
- Companies tag the same concept differently across years and industries,
  so each canonical field maps to a LIST of candidate XBRL tags, tried in
  order. Banks have no GrossProfit; a missing field is tolerated, a
  symbol yielding nothing at all is an error.
- Amended filings (10-K/A) appear as a second row for the same fiscal
  period with a later filed date -- both are kept. As-of queries pick the
  freshest row VISIBLE at that date, which is exactly right: before the
  amendment you'd have used the original number.

Design rules:
- Store raw reported values; derived ratios (B/P, leverage, gross
  profitability) belong to factors/exposures.py, not here.
- The network lives in one function (_fetch_json) so tests run offline
  against fixture JSON, same pattern as prices.py.
"""

import json
import time
import urllib.request
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from ballast.config import get_settings
from ballast.data.prices import connect

__all__ = [
    "EdgarError",
    "FIELDS",
    "load_cik_map",
    "ingest_fundamentals",
    "latest_fundamentals",
]

# Canonical field -> candidate us-gaap tags, first hit wins per filing.
# Order matters: put the most standard tag first.
FIELDS: dict[str, list[str]] = {
    "book_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "net_income": ["NetIncomeLoss"],
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
    ],
    "gross_profit": ["GrossProfit"],
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
}
# Shares outstanding lives in the "dei" namespace with unit "shares".
_SHARES_TAG = "EntityCommonStockSharesOutstanding"

# Duration (income statement) fields are only comparable at the same span:
# a 10-Q reports THREE months of net income, a 10-K reports twelve. Mixing
# them makes E/P jump 4x on filing days. Until proper TTM aggregation lands,
# these fields serve only their ANNUAL (fp='FY') values -- up to a year
# stale, but consistent, and staleness is visible while inconsistency isn't.
_ANNUAL_ONLY_FIELDS = ("net_income", "revenue", "gross_profit")

_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
_ACCEPTED_FORMS = {"10-K", "10-Q", "10-K/A", "10-Q/A"}
_REQUEST_DELAY_S = 0.15  # ~6 req/s, comfortably under EDGAR's 10/s limit

_FUNDAMENTALS_SCHEMA = """
CREATE TABLE IF NOT EXISTS fundamentals (
    symbol         VARCHAR NOT NULL,
    field          VARCHAR NOT NULL,
    value          DOUBLE,
    period_end     DATE    NOT NULL,
    filed          DATE    NOT NULL,
    form           VARCHAR,
    fiscal_year    INTEGER,
    fiscal_period  VARCHAR,
    PRIMARY KEY (symbol, field, period_end, filed)
);
"""


class EdgarError(RuntimeError):
    """Raised when EDGAR data is missing, unreachable, or malformed."""


def _fetch_json(url: str) -> dict:
    """The network seam: one place that touches the internet, mockable.

    Sends the required User-Agent and sleeps first so any loop calling
    this stays under EDGAR's rate limit without each caller remembering to.
    """
    time.sleep(_REQUEST_DELAY_S)
    request = urllib.request.Request(url, headers={"User-Agent": get_settings().edgar_user_agent})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # URLError, HTTPError, timeout, bad JSON
        raise EdgarError(f"EDGAR request failed for {url}: {exc}") from exc


def load_cik_map() -> dict[str, int]:
    """Ticker -> CIK from the SEC's official mapping file.

    Fetched fresh per ingest run (one request); deliberately NOT cached at
    module level so tests stay hermetic and long-lived processes don't
    hold a stale map.
    """
    raw = _fetch_json(_TICKER_MAP_URL)
    # Shape: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": ...}, ...}
    return {entry["ticker"].upper(): int(entry["cik_str"]) for entry in raw.values()}


def _extract_rows(facts: dict, symbol: str) -> list[tuple]:
    """companyfacts JSON -> rows matching the fundamentals schema."""
    rows: list[tuple] = []

    def collect(items: list[dict], field: str) -> None:
        for item in items:
            value = item.get("val")
            filed = item.get("filed")
            end = item.get("end")
            form = item.get("form")
            if value is None or filed is None or end is None:
                continue
            if form not in _ACCEPTED_FORMS:
                continue  # ignore 8-Ks, S-1s, etc.: not periodic reports
            rows.append(
                (symbol, field, float(value), end, filed, form, item.get("fy"), item.get("fp"))
            )

    gaap = facts.get("facts", {}).get("us-gaap", {})
    for field, candidate_tags in FIELDS.items():
        for tag in candidate_tags:
            units = gaap.get(tag, {}).get("units", {})
            if "USD" in units:
                collect(units["USD"], field)
                break  # first candidate tag that exists wins; don't mix

    dei = facts.get("facts", {}).get("dei", {})
    shares_units = dei.get(_SHARES_TAG, {}).get("units", {})
    if "shares" in shares_units:
        collect(shares_units["shares"], "shares_outstanding")

    return rows


def _fundamental_rows(con: duckdb.DuckDBPyConnection) -> int:
    """count(*) always yields a row; the None guard narrows fetchone's Optional."""
    row = con.execute("SELECT count(*) FROM fundamentals").fetchone()
    if row is None:
        raise EdgarError("count(*) on fundamentals returned no row")
    return int(row[0])


def ingest_fundamentals(
    symbols: list[str], db_path: Path | str | None = None, cik_map: dict[str, int] | None = None
) -> int:
    """Fetch and store point-in-time fundamentals. Returns rows written.

    Idempotent (INSERT OR IGNORE on the full point-in-time key): re-running
    after a new quarter adds only the new filings.
    """
    if not symbols:
        raise EdgarError("ingest_fundamentals: no symbols given")
    cleaned = list(dict.fromkeys(s.strip().upper() for s in symbols if s.strip()))
    if not cleaned:
        raise EdgarError("ingest_fundamentals: no symbols given")

    ciks = cik_map if cik_map is not None else load_cik_map()
    unknown = sorted(s for s in cleaned if s not in ciks)
    if unknown:
        raise EdgarError(f"no CIK found for symbol(s) {unknown}; not SEC filers?")

    con = connect(db_path)
    try:
        con.execute(_FUNDAMENTALS_SCHEMA)
        before = _fundamental_rows(con)
        for symbol in cleaned:
            facts = _fetch_json(_FACTS_URL.format(cik=ciks[symbol]))
            rows = _extract_rows(facts, symbol)
            if not rows:
                raise EdgarError(
                    f"{symbol}: EDGAR returned no usable 10-K/10-Q facts "
                    "(foreign filer, fund, or very new listing?)"
                )
            con.executemany(
                "INSERT OR IGNORE INTO fundamentals VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows
            )
        after = _fundamental_rows(con)
        return after - before
    finally:
        con.close()


def latest_fundamentals(
    symbols: list[str],
    as_of: str | date,
    db_path: Path | str | None = None,
) -> pd.DataFrame:
    """The point-in-time read: freshest value per (symbol, field) VISIBLE at as_of.

    "Visible" means filed <= as_of -- the no-leakage contract. Among visible
    rows the one describing the most recent fiscal period wins, with the
    latest filing (amendments) breaking ties. Returns a symbols x fields
    frame; a company that hasn't reported a field shows NaN, never a guess.
    """
    if not symbols:
        raise EdgarError("latest_fundamentals: no symbols given")
    requested = list(dict.fromkeys(s.strip().upper() for s in symbols if s.strip()))
    as_of_iso = (
        as_of.isoformat() if isinstance(as_of, date) else date.fromisoformat(as_of).isoformat()
    )

    placeholders = ", ".join("?" for _ in requested)
    con = connect(db_path)
    try:
        con.execute(_FUNDAMENTALS_SCHEMA)
        # QUALIFY keeps, per (symbol, field), only the top row of the
        # ordering: newest fiscal period first, newest filing breaking ties.
        annual_placeholders = ", ".join("?" for _ in _ANNUAL_ONLY_FIELDS)
        frame = con.execute(
            f"""
            SELECT symbol, field, value
            FROM fundamentals
            WHERE symbol IN ({placeholders}) AND filed <= ?
              AND (field NOT IN ({annual_placeholders}) OR fiscal_period = 'FY')
            QUALIFY row_number() OVER (
                PARTITION BY symbol, field
                ORDER BY period_end DESC, filed DESC
            ) = 1
            """,
            [*requested, as_of_iso, *_ANNUAL_ONLY_FIELDS],
        ).df()
    finally:
        con.close()

    if frame.empty:
        raise EdgarError(
            f"no fundamentals visible at {as_of_iso} for {requested}; "
            "run `ingest_fundamentals` (or the as_of predates every filing)"
        )
    wide = frame.pivot(index="symbol", columns="field", values="value")
    # Guarantee the full column set: a field nobody has filed (banks have
    # no gross_profit) must exist as a NaN column, not vanish -- callers
    # index columns positionally and by name.
    all_fields = [*FIELDS.keys(), "shares_outstanding"]
    return wide.reindex(index=requested, columns=all_fields)
