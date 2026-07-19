"""Copy the prices table out of an existing Sentinel DuckDB file. v0.1.0 -- implemented.

Notes
-----
What this file does: ATTACHes a Sentinel database read-only, checks that
its `prices` table has exactly the schema Ballast expects, copies the rows
across with INSERT OR IGNORE, detaches, and reports how many rows were new.
Ten years of already-ingested S&P history becomes available on day one
without re-downloading anything.

Design decisions, and why:
- INSERT OR IGNORE, not REPLACE: if Ballast has already ingested a
  (symbol, date) itself, Ballast's row is fresher (adjusted closes drift
  after every dividend) and wins. The import is a bootstrap, not an
  authority.
- Idempotent by construction: the second run finds every row already
  present, ignores them all, and returns 0.
- Schema check is exact -- column names AND types, in order. A Sentinel
  fork that renamed or retyped a column must be rejected, because a wrong
  import silently poisons every number downstream. The check runs BEFORE
  any row is copied.
- READ_ONLY attach: this module cannot write to the Sentinel file even by
  bug. Sentinel's database is someone else's production data.
- Row delta (count after minus count before) is the return value: "rows
  imported" means rows Ballast didn't have, which is the number the caller
  actually wants to see.

Failure style: SentinelImportError with enough context to fix the problem
(path, missing table, or a found-vs-expected schema diff).
"""

from pathlib import Path

import duckdb

from ballast.data.prices import connect

__all__ = ["SentinelImportError", "import_sentinel"]

# What `DESCRIBE sentinel.prices` must return, in order: (name, type).
# Mirrors _PRICES_SCHEMA in prices.py, which mirrors Sentinel itself.
_EXPECTED_SCHEMA: list[tuple[str, str]] = [
    ("symbol", "VARCHAR"),
    ("date", "DATE"),
    ("open", "DOUBLE"),
    ("high", "DOUBLE"),
    ("low", "DOUBLE"),
    ("close", "DOUBLE"),
    ("adj_close", "DOUBLE"),
    ("volume", "DOUBLE"),
]


class SentinelImportError(RuntimeError):
    """Raised when the Sentinel database is missing, unreadable, or mismatched."""


def import_sentinel(source_db: Path | str, db_path: Path | str | None = None) -> int:
    """Copy Sentinel's prices into Ballast's DB. Returns count of NEW rows.

    source_db: path to Sentinel's .duckdb file (e.g. ~/.../sentinel/data/sentinel.duckdb).
    db_path:   Ballast's DB; None means the configured default.
    """
    source = Path(source_db).expanduser()
    if not source.is_file():
        raise SentinelImportError(f"Sentinel database not found: {source}")

    # connect() also creates Ballast's prices table if this is a fresh DB.
    con = connect(db_path)
    try:
        # ATTACH has no '?' placeholder support, so the path is inlined.
        # Doubling single quotes is SQL's escape rule; without it a path
        # like /Users/o'brien/... would break the statement.
        quoted = str(source).replace("'", "''")
        try:
            con.execute(f"ATTACH '{quoted}' AS sentinel (READ_ONLY)")
        except duckdb.Error as exc:
            # Typical causes: not a DuckDB file at all, or written by a
            # newer DuckDB storage format than this library can read.
            raise SentinelImportError(
                f"could not open {source} as a DuckDB database "
                f"(corrupt file, or DuckDB version mismatch?): {exc}"
            ) from exc

        try:
            # DESCRIBE fails if the table doesn't exist -- the clearest
            # available "is this actually a Sentinel database?" probe.
            try:
                described = con.execute("DESCRIBE sentinel.prices").fetchall()
            except duckdb.Error as exc:
                raise SentinelImportError(
                    f"{source} has no `prices` table; is this really a Sentinel database?"
                ) from exc

            # DESCRIBE rows are (name, type, null, key, default, extra);
            # only name and type matter for compatibility.
            found = [(row[0], row[1]) for row in described]
            if found != _EXPECTED_SCHEMA:
                raise SentinelImportError(
                    "Sentinel prices schema does not match Ballast's.\n"
                    f"  expected: {_EXPECTED_SCHEMA}\n"
                    f"  found:    {found}\n"
                    "Refusing to import; fix the schema drift first."
                )

            # Delta counting: cheaper and more honest than trying to make
            # INSERT OR IGNORE report how many rows it skipped.
            before = con.execute("SELECT count(*) FROM prices").fetchone()[0]
            con.execute(
                "INSERT OR IGNORE INTO prices "
                "SELECT symbol, date, open, high, low, close, adj_close, volume "
                "FROM sentinel.prices"
            )
            after = con.execute("SELECT count(*) FROM prices").fetchone()[0]
            return after - before
        finally:
            # Always detach, even on error, so the connection is reusable
            # and the Sentinel file's handle is released promptly.
            con.execute("DETACH sentinel")
    finally:
        con.close()
