"""Tests for data/prices.py. Entirely offline.

Notes
-----
The network call lives in one function (_fetch_yfinance) precisely so these
tests can replace it with synthetic frames (monkeypatch) and exercise every
other line for real: normalization, storage, idempotency, and the returns
math. The DB is a throwaway file under tmp_path per test.

make_yf_frame() fabricates what yfinance actually returns, including its
quirks (MultiIndex columns, missing Adj Close), so the normalizer is tested
against the shapes seen in the wild, not an idealized input.
"""

import pandas as pd
import pytest

from ballast.data.prices import (
    PriceDataError,
    _normalize,
    connect,
    ingest_prices,
    load_returns,
    store_prices,
)


def make_yf_frame(
    closes: list[float],
    start: str = "2024-01-01",
    multiindex: bool = False,
    with_adj: bool = True,
    symbol: str = "TEST",
) -> pd.DataFrame:
    """Build a frame shaped like a real yf.download() result."""
    idx = pd.bdate_range(start, periods=len(closes))  # business days, like real bars
    data = {
        "Open": closes,
        "High": [c * 1.01 for c in closes],
        "Low": [c * 0.99 for c in closes],
        "Close": closes,
        "Volume": [1_000_000.0] * len(closes),
    }
    if with_adj:
        data["Adj Close"] = [c * 0.98 for c in closes]  # offset so tests can tell them apart
    df = pd.DataFrame(data, index=idx)
    if multiindex:
        # yfinance >= ~0.2.40 returns (field, ticker) columns even for one ticker
        df.columns = pd.MultiIndex.from_product([df.columns, [symbol]])
    return df


def tidy(closes: list[float], symbol: str = "TEST", **kwargs) -> pd.DataFrame:
    """Shorthand: synthetic yfinance frame -> normalized rows."""
    return _normalize(make_yf_frame(closes, **kwargs), symbol)


# ------------------------------------------------------------ normalization


def test_normalize_flat_columns():
    out = tidy([100.0, 101.0])
    assert list(out.columns) == [
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]
    assert len(out) == 2
    assert out["symbol"].unique().tolist() == ["TEST"]


def test_normalize_multiindex_columns():
    # The post-0.2.40 shape must produce the same result as the flat shape.
    out = tidy([100.0, 101.0], multiindex=True)
    assert out["close"].tolist() == [100.0, 101.0]


def test_normalize_missing_adj_close_falls_back_to_close():
    out = tidy([100.0, 101.0], with_adj=False)
    assert out["adj_close"].tolist() == out["close"].tolist()


def test_normalize_empty_frame_raises():
    with pytest.raises(PriceDataError, match="no price data"):
        _normalize(pd.DataFrame(), "TYPO")


def test_normalize_drops_rows_without_close():
    frame = make_yf_frame([100.0, 101.0, 102.0])
    frame.iloc[1, frame.columns.get_loc("Close")] = float("nan")
    out = _normalize(frame, "TEST")
    assert len(out) == 2  # the NaN-close row is gone
    assert out["close"].tolist() == [100.0, 102.0]


def test_normalize_tz_aware_index():
    frame = make_yf_frame([100.0, 101.0])
    frame.index = frame.index.tz_localize("America/New_York")
    out = _normalize(frame, "TEST")
    assert len(out) == 2  # didn't blow up; dates extracted cleanly


# ------------------------------------------------------------------ storage


def test_store_is_idempotent(tmp_path):
    db = tmp_path / "t.duckdb"
    rows = tidy([100.0, 101.0, 102.0])
    store_prices(rows, db_path=db)
    store_prices(rows, db_path=db)  # second write must not duplicate
    con = connect(db)
    try:
        count = con.execute("SELECT count(*) FROM prices").fetchone()[0]
    finally:
        con.close()
    assert count == 3


def test_store_replaces_on_conflict(tmp_path):
    # REPLACE semantics: re-ingesting refreshed (adjusted) data wins.
    db = tmp_path / "t.duckdb"
    store_prices(tidy([100.0, 101.0]), db_path=db)
    updated = tidy([100.0, 999.0])  # same dates, revised close
    store_prices(updated, db_path=db)
    con = connect(db)
    try:
        closes = [r[0] for r in con.execute("SELECT close FROM prices ORDER BY date").fetchall()]
    finally:
        con.close()
    assert closes == [100.0, 999.0]


def test_store_rejects_missing_columns(tmp_path):
    with pytest.raises(PriceDataError, match="missing column"):
        store_prices(pd.DataFrame({"symbol": ["A"]}), db_path=tmp_path / "t.duckdb")


# ---------------------------------------------------------------- ingestion


def test_ingest_normalizes_and_counts(tmp_path, monkeypatch):
    db = tmp_path / "t.duckdb"

    def fake_fetch(symbol: str, start: str | None) -> pd.DataFrame:
        return make_yf_frame([100.0, 101.0, 102.0], multiindex=True, symbol=symbol)

    # Patch the one network function; everything else runs for real.
    monkeypatch.setattr("ballast.data.prices._fetch_yfinance", fake_fetch)

    written = ingest_prices(["spy", "SPY", " qqq "], db_path=db)  # dupes + junk spacing
    assert written == 6  # 2 unique symbols x 3 rows; "spy"/"SPY" de-duped

    con = connect(db)
    try:
        symbols = {r[0] for r in con.execute("SELECT DISTINCT symbol FROM prices").fetchall()}
    finally:
        con.close()
    assert symbols == {"SPY", "QQQ"}  # upper-cased on the way in


def test_ingest_empty_symbol_list_raises(tmp_path):
    with pytest.raises(PriceDataError, match="no symbols"):
        ingest_prices([], db_path=tmp_path / "t.duckdb")


def test_ingest_bad_symbol_raises_with_name(tmp_path, monkeypatch):
    def fake_fetch(symbol: str, start: str | None) -> pd.DataFrame:
        return pd.DataFrame()  # what yfinance returns for nonsense tickers

    monkeypatch.setattr("ballast.data.prices._fetch_yfinance", fake_fetch)
    with pytest.raises(PriceDataError, match="TYPO"):
        ingest_prices(["TYPO"], db_path=tmp_path / "t.duckdb")


# ------------------------------------------------------------------ returns


def seed(db, symbol: str, closes: list[float], start: str = "2024-01-01") -> None:
    """Store a synthetic series for one symbol."""
    store_prices(tidy(closes, symbol=symbol, start=start), db_path=db)


def test_load_returns_hand_example(tmp_path):
    # adj_close is 0.98 * close everywhere (see make_yf_frame), so the
    # constant cancels in the division: returns equal close-based returns.
    # 100 -> 110 is +10%; 110 -> 99 is -10%.
    db = tmp_path / "t.duckdb"
    seed(db, "AAA", [100.0, 110.0, 99.0])
    r = load_returns(["AAA"], db_path=db)
    assert len(r) == 2
    assert r["AAA"].iloc[0] == pytest.approx(0.10)
    assert r["AAA"].iloc[1] == pytest.approx(-0.10)


def test_load_returns_column_order_follows_request(tmp_path):
    db = tmp_path / "t.duckdb"
    seed(db, "AAA", [100.0, 101.0])
    seed(db, "BBB", [50.0, 51.0])
    r = load_returns(["BBB", "AAA"], db_path=db)
    assert list(r.columns) == ["BBB", "AAA"]  # not alphabetical


def test_load_returns_inner_alignment(tmp_path):
    # AAA has 4 days, BBB only the last 3: the matrix keeps the overlap only.
    db = tmp_path / "t.duckdb"
    seed(db, "AAA", [100.0, 101.0, 102.0, 103.0], start="2024-01-01")
    seed(db, "BBB", [50.0, 51.0, 52.0], start="2024-01-02")
    r = load_returns(["AAA", "BBB"], db_path=db)
    # 3 overlapping price dates -> 2 return rows, no NaNs anywhere.
    assert len(r) == 2
    assert not r.isna().any().any()


def test_load_returns_unknown_symbol_raises(tmp_path):
    db = tmp_path / "t.duckdb"
    seed(db, "AAA", [100.0, 101.0])
    with pytest.raises(PriceDataError, match=r"\['NOPE'\]"):
        load_returns(["AAA", "NOPE"], db_path=db)


def test_load_returns_date_filters(tmp_path):
    db = tmp_path / "t.duckdb"
    seed(db, "AAA", [100.0, 101.0, 102.0, 103.0, 104.0], start="2024-01-01")
    full = load_returns(["AAA"], db_path=db)
    windowed = load_returns(["AAA"], start="2024-01-02", end="2024-01-04", db_path=db)
    assert len(windowed) < len(full)


def test_load_returns_bad_date_rejected(tmp_path):
    db = tmp_path / "t.duckdb"
    seed(db, "AAA", [100.0, 101.0])
    with pytest.raises(PriceDataError, match="YYYY-MM-DD"):
        load_returns(["AAA"], start="last tuesday", db_path=db)


def test_load_returns_needs_two_dates(tmp_path):
    db = tmp_path / "t.duckdb"
    seed(db, "AAA", [100.0])  # a single price can't make a return
    with pytest.raises(PriceDataError, match="fewer than 2"):
        load_returns(["AAA"], db_path=db)


def test_load_returns_coalesce_falls_back_to_close(tmp_path):
    # A row whose adj_close is NULL (e.g. imported from elsewhere) must use
    # close instead of dropping the date.
    db = tmp_path / "t.duckdb"
    rows = tidy([100.0, 110.0])
    rows.loc[1, "adj_close"] = None
    store_prices(rows, db_path=db)
    r = load_returns(["TEST"], db_path=db)
    # day 1: adj 98 -> raw close 110 is a mixed-basis return; the point here
    # is only that the row survives and produces a finite number.
    assert len(r) == 1
    assert pd.notna(r["TEST"].iloc[0])
