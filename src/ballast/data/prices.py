"""Daily OHLCV via yfinance into DuckDB. v0.1.0 -- implemented.

Notes
-----
What this file does: owns the `prices` table. Fetches daily bars from
yfinance (ingest_prices), stores them idempotently, and serves them back
as an analysis-ready returns matrix (load_returns). Every other layer --
covariance, factors, backtest -- consumes that matrix; nothing else in the
codebase re-derives returns.

Schema parity: the table definition below is copied verbatim from
Sentinel's duckdb_store.py, so sentinel_import.py can move rows across
with a plain INSERT ... SELECT. Don't "improve" it unilaterally; the two
projects change it together or not at all.

Decisions made here, once, so callers can't get them wrong:
- Returns use COALESCE(adj_close, close). Adjusted closes fold dividends
  and splits back into the series; raw closes around a 4:1 split would
  show a fake -75% return. close is the fallback for imported rows that
  lack an adjusted series.
- Missing-data policy is INNER alignment: load_returns keeps only dates
  where EVERY requested symbol has a price. Forward-filling would
  fabricate prices, and fabricated prices become fabricated returns.
  The cost (a few dropped dates) is visible; the alternative is invisible.
- Writes are INSERT OR REPLACE keyed on (symbol, date): re-ingesting a
  range is a refresh, not a duplicate. REPLACE (not IGNORE) because
  adjusted history legitimately changes after every dividend.
- yfinance is imported lazily inside the fetch function. It drags in a
  tree of dependencies and does network work; importing this module (for
  load_returns, say, or in tests) shouldn't require any of that.

Failure style: PriceDataError naming the symbol(s). Partial progress
persists -- ingesting [SPY, TYPO] stores SPY, then raises about TYPO;
re-running after the fix is cheap because writes are idempotent.
"""

from collections.abc import Sequence
from datetime import date as date_type
from pathlib import Path

import duckdb
import pandas as pd

from ballast.config import get_settings

__all__ = [
    "PriceDataError",
    "connect",
    "ingest_prices",
    "store_prices",
    "load_returns",
    "load_latest_prices",
    "load_prices",
    "list_symbols",
]

# Copied verbatim from Sentinel (storage/duckdb_store.py). See module notes.
_PRICES_SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    symbol     VARCHAR NOT NULL,
    date       DATE    NOT NULL,
    open       DOUBLE,
    high       DOUBLE,
    low        DOUBLE,
    close      DOUBLE,
    adj_close  DOUBLE,
    volume     DOUBLE,
    PRIMARY KEY (symbol, date)
);
"""

# Column order used everywhere a full row is read or written.
_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"]


class PriceDataError(RuntimeError):
    """Raised when price data is missing, empty, or malformed."""


def connect(db_path: Path | str | None = None) -> duckdb.DuckDBPyConnection:
    """Open the Ballast DuckDB and make sure the prices table exists.

    db_path=None uses the configured location; tests pass a tmp_path file.
    Caller owns the connection and must close it (every function in this
    module does so via try/finally).
    """
    path = Path(db_path) if db_path is not None else get_settings().db_path
    if str(path) != ":memory:":
        # First run on a fresh checkout: data/ may not exist yet.
        path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    # CREATE TABLE IF NOT EXISTS is a no-op when the table is already there,
    # so running this on every connect is cheap and removes a whole class of
    # "did anyone create the schema yet?" bugs.
    con.execute(_PRICES_SCHEMA)
    return con


def _fetch_yfinance(symbol: str, start: str | None) -> pd.DataFrame:
    """Network call, isolated so tests can monkeypatch it out.

    Everything below this function is pure logic and runs offline.
    """
    import yfinance as yf  # lazy: see module notes

    # Gotcha worth knowing: yf.download without start defaults to ONE MONTH
    # of data (period="1mo"), not full history. period="max" is explicit.
    if start is None:
        return yf.download(symbol, period="max", auto_adjust=False, progress=False)
    return yf.download(symbol, start=start, auto_adjust=False, progress=False)


def _normalize(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """yfinance's frame -> tidy rows matching the prices schema.

    Handles the three shapes yfinance actually returns: flat columns,
    MultiIndex (field, ticker) columns (the default since ~0.2.40 even for
    a single ticker), and auto-adjusted frames with no 'Adj Close' column.
    """
    if raw is None or raw.empty:
        raise PriceDataError(f"no price data returned for {symbol!r} (typo, or delisted?)")
    df = raw.copy()

    if isinstance(df.columns, pd.MultiIndex):
        # ('Close', 'SPY') -> 'Close'. Single-ticker download, so level 1
        # carries no information.
        df.columns = df.columns.get_level_values(0)

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    if "close" not in df.columns:
        raise PriceDataError(f"{symbol!r}: yfinance frame has no Close column: {list(df.columns)}")
    if "adj_close" not in df.columns:
        # auto_adjust=True sources fold adjustments into Close itself, so
        # Close IS the adjusted series. Store it in both columns.
        df["adj_close"] = df["close"]
    for col in ("open", "high", "low", "volume"):
        if col not in df.columns:
            df[col] = float("nan")  # tolerated as NULL by the schema

    # Index -> plain dates. Daily bars are sometimes tz-aware depending on
    # source; the DATE column doesn't care about timezones.
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df["date"] = idx.date
    df["symbol"] = symbol

    # A bar with no close is no bar. (Half-empty rows appear around IPO
    # dates and exchange holidays.)
    df = df.dropna(subset=["close"])
    if df.empty:
        raise PriceDataError(f"{symbol!r}: every returned row was empty")

    # yfinance occasionally repeats the most recent bar; last write wins.
    df = df.drop_duplicates(subset=["date"], keep="last")

    return df[_COLUMNS].reset_index(drop=True)


def store_prices(tidy: pd.DataFrame, db_path: Path | str | None = None) -> int:
    """Write normalized rows. Idempotent: (symbol, date) collisions replace.

    Returns the number of rows written.
    """
    missing = [c for c in _COLUMNS if c not in tidy.columns]
    if missing:
        raise PriceDataError(f"store_prices: frame is missing column(s) {missing}")

    con = connect(db_path)
    try:
        # register() exposes the DataFrame to SQL as a view named 'incoming';
        # DuckDB reads it zero-copy. Explicit column list so a reordered
        # frame can't silently write open prices into the close column.
        con.register("incoming", tidy)
        con.execute(
            "INSERT OR REPLACE INTO prices "
            "SELECT symbol, date, open, high, low, close, adj_close, volume FROM incoming"
        )
        return len(tidy)
    finally:
        con.close()


def ingest_prices(
    symbols: Sequence[str],
    start: str | None = None,
    db_path: Path | str | None = None,
) -> int:
    """Fetch and store daily bars for each symbol. Returns total rows written.

    Symbols are normalized (upper-cased, de-duplicated, order kept) to match
    the spec loader. Fails fast on the first bad symbol -- everything stored
    before the failure stays stored, and re-runs are cheap (idempotent).
    """
    if not symbols:
        raise PriceDataError("ingest_prices: no symbols given")

    # dict.fromkeys: the standard order-preserving de-dupe.
    cleaned = list(dict.fromkeys(s.strip().upper() for s in symbols if s.strip()))
    if not cleaned:
        raise PriceDataError("ingest_prices: no symbols given")

    total = 0
    for symbol in cleaned:
        raw = _fetch_yfinance(symbol, start)
        tidy = _normalize(raw, symbol)
        total += store_prices(tidy, db_path)
    return total


def load_returns(
    symbols: Sequence[str],
    start: str | date_type | None = None,
    end: str | date_type | None = None,
    db_path: Path | str | None = None,
) -> pd.DataFrame:
    """Daily simple returns, dates x symbols, from stored prices.

    Columns follow the requested symbol order. Dates are the INTERSECTION of
    all symbols' histories (see module notes for why there is no
    forward-fill). Raises if any symbol has no stored rows at all -- a typo'd
    ticker must fail loudly, not come back as an empty column.
    """
    if not symbols:
        raise PriceDataError("load_returns: no symbols given")
    requested = list(dict.fromkeys(s.strip().upper() for s in symbols if s.strip()))
    if not requested:
        raise PriceDataError("load_returns: no symbols given")

    # Date filters accept '2020-01-01' or datetime.date; reject junk here
    # with a clear message instead of letting it leak into SQL.
    def _as_iso(value: str | date_type | None, label: str) -> str | None:
        if value is None:
            return None
        if isinstance(value, date_type):
            return value.isoformat()
        try:
            return date_type.fromisoformat(value).isoformat()
        except (TypeError, ValueError):
            raise PriceDataError(f"{label} must be a date or 'YYYY-MM-DD', got {value!r}") from None

    start_iso = _as_iso(start, "start")
    end_iso = _as_iso(end, "end")

    # One '?' placeholder per symbol; never build SQL by string-formatting
    # values in (that habit ends in injection bugs, even in local tools).
    placeholders = ", ".join("?" for _ in requested)
    query = (
        "SELECT date, symbol, COALESCE(adj_close, close) AS px "  # see notes
        f"FROM prices WHERE symbol IN ({placeholders})"
    )
    params: list[object] = list(requested)
    if start_iso:
        query += " AND date >= ?"
        params.append(start_iso)
    if end_iso:
        query += " AND date <= ?"
        params.append(end_iso)
    query += " ORDER BY date"

    con = connect(db_path)
    try:
        long_df = con.execute(query, params).df()
    finally:
        con.close()

    # Loud failure for symbols with zero rows in the window.
    found = set(long_df["symbol"].unique())
    missing = sorted(set(requested) - found)
    if missing:
        raise PriceDataError(
            f"no stored prices for symbol(s) {missing}; run `ballast ingest` first"
        )

    # Long -> wide: one row per date, one column per symbol.
    prices = long_df.pivot(index="date", columns="symbol", values="px")
    prices = prices[requested]  # caller's column order, not alphabetical
    prices = prices.dropna()  # INNER alignment -- the module-notes policy
    if len(prices) < 2:
        raise PriceDataError(
            f"fewer than 2 overlapping dates across {requested}; not enough to compute a return"
        )

    # Simple returns: p_t / p_{t-1} - 1. fill_method=None stops pandas from
    # quietly forward-filling gaps before differencing (deprecated behavior,
    # and exactly the fabrication the alignment policy forbids).
    returns = prices.pct_change(fill_method=None).iloc[1:]  # row 0 has no prior day
    returns.columns.name = None  # cosmetic: drop the 'symbol' axis label
    return returns


def load_prices(
    symbols: Sequence[str],
    start: str | date_type | None = None,
    end: str | date_type | None = None,
    db_path: Path | str | None = None,
    adjusted: bool = True,
) -> pd.DataFrame:
    """Wide price history: dates x symbols, one column per symbol.

    adjusted=True serves COALESCE(adj_close, close) -- the series for
    momentum and vol calculations. adjusted=False serves raw closes.
    Unlike load_returns there is NO inner alignment: each column keeps its
    own history (NaN where a symbol didn't trade), because per-symbol
    characteristics (momentum, trailing vol) use per-symbol histories.
    """
    if not symbols:
        raise PriceDataError("load_prices: no symbols given")
    requested = list(dict.fromkeys(s.strip().upper() for s in symbols if s.strip()))
    if not requested:
        raise PriceDataError("load_prices: no symbols given")

    column = "COALESCE(adj_close, close)" if adjusted else "close"
    placeholders = ", ".join("?" for _ in requested)
    query = f"SELECT date, symbol, {column} AS px FROM prices WHERE symbol IN ({placeholders})"
    params: list[object] = list(requested)
    for label, value in (("date >= ?", start), ("date <= ?", end)):
        if value is not None:
            query += f" AND {label}"
            params.append(value.isoformat() if isinstance(value, date_type) else value)
    query += " ORDER BY date"

    con = connect(db_path)
    try:
        long_df = con.execute(query, params).df()
    finally:
        con.close()
    if long_df.empty:
        raise PriceDataError(f"no stored prices for {requested} in the requested window")

    wide = long_df.pivot(index="date", columns="symbol", values="px")
    wide.columns.name = None
    # Missing symbols come back as NaN columns, present and honest.
    return wide.reindex(columns=requested)


def list_symbols(db_path: Path | str | None = None) -> list[str]:
    """Every symbol with stored prices, sorted. Empty DB -> empty list."""
    con = connect(db_path)
    try:
        rows = con.execute("SELECT DISTINCT symbol FROM prices ORDER BY symbol").fetchall()
    finally:
        con.close()
    return [row[0] for row in rows]


def load_latest_prices(
    symbols: Sequence[str],
    db_path: Path | str | None = None,
    as_of: str | date_type | None = None,
) -> dict[str, float]:
    """Most recent stored RAW close per symbol, for valuing share positions.

    Raw close, not adj_close: converting shares to dollars needs the actual
    market price. (Adjustments rewrite history; the latest bar is the same
    either way, but raw close is the semantically correct column.)

    as_of restricts to bars dated on or before that day -- "the latest
    price the market had seen at as_of", which is what point-in-time
    factor construction needs.
    """
    if not symbols:
        raise PriceDataError("load_latest_prices: no symbols given")
    requested = list(dict.fromkeys(s.strip().upper() for s in symbols if s.strip()))
    if not requested:
        raise PriceDataError("load_latest_prices: no symbols given")

    placeholders = ", ".join("?" for _ in requested)
    query = f"SELECT symbol, arg_max(close, date) FROM prices WHERE symbol IN ({placeholders})"
    params: list[object] = list(requested)
    if as_of is not None:
        as_of_iso = as_of.isoformat() if isinstance(as_of, date_type) else as_of
        query += " AND date <= ?"
        params.append(as_of_iso)
    query += " GROUP BY symbol"

    con = connect(db_path)
    try:
        # arg_max(close, date): the close value on each symbol's latest date.
        # One aggregate does what a self-join or window function would.
        rows = con.execute(query, params).fetchall()
    finally:
        con.close()

    latest = {sym: px for sym, px in rows}
    missing = sorted(set(requested) - set(latest))
    if missing:
        raise PriceDataError(
            f"no stored prices for symbol(s) {missing}; run `ballast ingest` first"
        )
    return latest
