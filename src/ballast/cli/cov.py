"""`ballast cov` -- covariance lab commands (v0.2.0).

Notes
-----
Orchestration only: resolve the symbol list, load returns, hand off to
covariance/harness.py, render. The experiment itself lives in the harness;
this file must stay boring.

`compare` with no symbols runs on every symbol in the database -- the
common case after an import-sentinel bootstrap.
"""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

console = Console()

cov_app = typer.Typer(help="Covariance estimator lab.", no_args_is_help=True)


@cov_app.command("compare")
def compare(
    symbols: Annotated[
        list[str] | None,
        typer.Argument(help="Symbols to include (default: every symbol in the DB)."),
    ] = None,
    start: Annotated[str | None, typer.Option(help="History start, YYYY-MM-DD.")] = None,
    window: Annotated[int, typer.Option(help="Training window, trading days.")] = 252,
    step: Annotated[int, typer.Option(help="Rebalance cadence, trading days.")] = 21,
    cost_bps: Annotated[float, typer.Option(help="Per-side cost in basis points.")] = 2.0,
    db: Annotated[Path | None, typer.Option(help="DuckDB path (default: configured).")] = None,
) -> None:
    """The horse race: which estimator's min-variance portfolio is calmest OOS."""
    from ballast.backtest.engine import BacktestError
    from ballast.covariance.estimators import CovarianceError
    from ballast.covariance.harness import compare_estimators
    from ballast.data.prices import PriceDataError, list_symbols, load_returns
    from ballast.reporting.tables import render_cov_comparison

    try:
        chosen = [s.strip().upper() for s in symbols] if symbols else list_symbols(db_path=db)
        if len(chosen) < 2:
            raise PriceDataError(
                "the comparison needs at least 2 symbols with stored prices; "
                "run `ballast ingest` or `ballast import-sentinel` first"
            )
        returns = load_returns(chosen, start=start, db_path=db)
        result = compare_estimators(returns, window=window, step=step, cost_bps=cost_bps)
    except (PriceDataError, CovarianceError, BacktestError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from None

    console.print(render_cov_comparison(result))
