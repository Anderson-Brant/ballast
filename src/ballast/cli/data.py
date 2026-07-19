"""`ballast ingest` and `ballast import-sentinel` -- data commands.

Notes
-----
Thin wrappers over data/prices.py and data/sentinel_import.py: parse
arguments, call one function, print one line, map failures to exit code 1.
All the behavior worth testing lives in the data modules.
"""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

console = Console()


def ingest(
    symbols: Annotated[list[str], typer.Argument(help="Ticker symbols, e.g. SPY AAPL MSFT.")],
    start: Annotated[str | None, typer.Option(help="History start, YYYY-MM-DD.")] = None,
    db: Annotated[Path | None, typer.Option(help="DuckDB path (default: configured).")] = None,
) -> None:
    """Fetch daily bars from yfinance into the prices table."""
    from ballast.data.prices import PriceDataError, ingest_prices

    try:
        written = ingest_prices(symbols, start=start, db_path=db)
    except PriceDataError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from None
    console.print(f"wrote {written} rows for {len(set(s.upper() for s in symbols))} symbol(s)")


def import_sentinel(
    source: Annotated[Path, typer.Argument(help="Path to Sentinel's .duckdb file.")],
    db: Annotated[
        Path | None, typer.Option(help="Ballast DuckDB path (default: configured).")
    ] = None,
) -> None:
    """Copy Sentinel's prices table into Ballast (read-only attach; new rows only)."""
    from ballast.data.sentinel_import import SentinelImportError
    from ballast.data.sentinel_import import import_sentinel as run_import

    try:
        imported = run_import(source, db_path=db)
    except SentinelImportError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from None
    console.print(f"imported {imported} new rows from {source}")
