"""Tests for data/sentinel_import.py. Entirely offline.

Notes
-----
make_sentinel_db() fabricates a real DuckDB file with Sentinel's exact
schema, so the import runs against the genuine article -- ATTACH, DESCRIBE,
INSERT OR IGNORE and all -- just pointed at throwaway files under tmp_path.
"""

from pathlib import Path

import duckdb
import pytest

from ballast.data.prices import connect, load_returns, store_prices
from ballast.data.sentinel_import import SentinelImportError, import_sentinel
from tests.test_prices import tidy

# Sentinel's DDL, verbatim -- the thing the import validates against.
_SENTINEL_DDL = """
CREATE TABLE prices (
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

_ROWS = [
    ("SPY", "2024-01-02", 470.0, 472.0, 469.0, 471.0, 465.0, 1e6),
    ("SPY", "2024-01-03", 471.0, 473.0, 470.0, 472.0, 466.0, 1e6),
    ("SPY", "2024-01-04", 472.0, 474.0, 471.0, 473.0, 467.0, 1e6),
    ("AAPL", "2024-01-02", 185.0, 186.0, 184.0, 185.5, 184.0, 2e6),
    ("AAPL", "2024-01-03", 185.5, 187.0, 185.0, 186.5, 185.0, 2e6),
    ("AAPL", "2024-01-04", 186.5, 188.0, 186.0, 187.5, 186.0, 2e6),
]


def make_sentinel_db(path: Path, ddl: str = _SENTINEL_DDL, rows=None) -> Path:
    """Create a throwaway DuckDB file that looks like Sentinel's."""
    con = duckdb.connect(str(path))
    try:
        con.execute(ddl)
        for row in _ROWS if rows is None else rows:
            con.execute("INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?)", row)
    finally:
        con.close()
    return path


def test_import_happy_path(tmp_path):
    src = make_sentinel_db(tmp_path / "sentinel.duckdb")
    dst = tmp_path / "ballast.duckdb"

    imported = import_sentinel(src, db_path=dst)
    assert imported == len(_ROWS)

    # The imported rows are immediately usable by the rest of the pipeline.
    r = load_returns(["SPY", "AAPL"], db_path=dst)
    assert list(r.columns) == ["SPY", "AAPL"]
    assert len(r) == 2  # 3 shared dates -> 2 returns


def test_import_is_idempotent(tmp_path):
    src = make_sentinel_db(tmp_path / "sentinel.duckdb")
    dst = tmp_path / "ballast.duckdb"
    assert import_sentinel(src, db_path=dst) == len(_ROWS)
    assert import_sentinel(src, db_path=dst) == 0  # second run: nothing new


def test_existing_ballast_rows_win(tmp_path):
    # Ballast already ingested SPY 2024-01-02 itself (close=999). The import
    # must IGNORE Sentinel's older row for that key, not replace it.
    src = make_sentinel_db(tmp_path / "sentinel.duckdb")
    dst = tmp_path / "ballast.duckdb"

    mine = tidy([999.0], symbol="SPY", start="2024-01-02")
    store_prices(mine, db_path=dst)

    imported = import_sentinel(src, db_path=dst)
    assert imported == len(_ROWS) - 1  # one key collided, five were new

    con = connect(dst)
    try:
        close = con.execute(
            "SELECT close FROM prices WHERE symbol='SPY' AND date='2024-01-02'"
        ).fetchone()[0]
    finally:
        con.close()
    assert close == 999.0  # Ballast's row survived


def test_missing_file_raises(tmp_path):
    with pytest.raises(SentinelImportError, match="not found"):
        import_sentinel(tmp_path / "nope.duckdb", db_path=tmp_path / "b.duckdb")


def test_missing_prices_table_raises(tmp_path):
    src = tmp_path / "empty.duckdb"
    duckdb.connect(str(src)).close()  # valid DuckDB file, but no tables
    with pytest.raises(SentinelImportError, match="no `prices` table"):
        import_sentinel(src, db_path=tmp_path / "b.duckdb")


def test_schema_mismatch_raises(tmp_path):
    # A prices table missing adj_close: plausible drift, must be refused.
    bad_ddl = """
    CREATE TABLE prices (
        symbol VARCHAR NOT NULL,
        date   DATE    NOT NULL,
        close  DOUBLE,
        PRIMARY KEY (symbol, date)
    );
    """
    src = make_sentinel_db(tmp_path / "drifted.duckdb", ddl=bad_ddl, rows=[])
    with pytest.raises(SentinelImportError, match="does not match"):
        import_sentinel(src, db_path=tmp_path / "b.duckdb")


def test_not_a_duckdb_file_raises(tmp_path):
    fake = tmp_path / "fake.duckdb"
    fake.write_text("this is a text file wearing a duckdb extension")
    with pytest.raises(SentinelImportError, match="could not open"):
        import_sentinel(fake, db_path=tmp_path / "b.duckdb")


def test_source_is_never_modified(tmp_path):
    # READ_ONLY attach: after an import, the source contents are untouched.
    src = make_sentinel_db(tmp_path / "sentinel.duckdb")
    import_sentinel(src, db_path=tmp_path / "b.duckdb")

    con = duckdb.connect(str(src))
    try:
        count = con.execute("SELECT count(*) FROM prices").fetchone()[0]
    finally:
        con.close()
    assert count == len(_ROWS)
