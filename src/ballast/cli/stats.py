"""`ballast stats` -- the v0.1.0 headline command.

Notes
-----
Orchestration only, per the CLI design rule: load spec -> value share
positions at latest stored closes -> resolve weights -> load returns ->
hand everything to portfolio/stats.py for math and reporting/tables.py
for rendering. If a number is wrong, the bug is in those modules, never
here.

Heavy imports (pandas, duckdb via the data layer) happen inside the
command body so `ballast version` stays instant.
"""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

console = Console()


def stats(
    portfolio_file: Annotated[Path, typer.Argument(help="Path to a portfolio.yaml spec.")],
    start: Annotated[str | None, typer.Option(help="Window start, YYYY-MM-DD.")] = None,
    end: Annotated[str | None, typer.Option(help="Window end, YYYY-MM-DD.")] = None,
    benchmark: Annotated[str, typer.Option(help="Beta benchmark symbol.")] = "SPY",
    db: Annotated[Path | None, typer.Option(help="DuckDB path (default: configured).")] = None,
) -> None:
    """Vol, beta, max drawdown, and CAGR for a portfolio spec."""
    from ballast.data.prices import PriceDataError, load_latest_prices, load_returns
    from ballast.portfolio.spec import PortfolioSpecError, load_portfolio, resolve_weights
    from ballast.portfolio.stats import StatsError, compute_stats
    from ballast.reporting.tables import render_stats

    try:
        portfolio = load_portfolio(portfolio_file)

        # Prices are only needed to value SHARE positions; a weights-only
        # spec resolves without touching the database.
        share_symbols = [p.symbol for p in portfolio.positions if p.shares is not None]
        latest = load_latest_prices(share_symbols, db_path=db) if share_symbols else {}
        resolved = resolve_weights(portfolio, latest)

        if not resolved.weights:
            raise StatsError("portfolio holds only cash; there is nothing to measure")
        returns = load_returns(list(resolved.weights), start=start, end=end, db_path=db)

        # The benchmark is loaded SEPARATELY: putting it in the same
        # load_returns call would shrink the portfolio window to dates the
        # benchmark also has (inner alignment). Missing benchmark data
        # degrades to beta=None instead of failing the whole command.
        benchmark_upper = benchmark.strip().upper()
        try:
            bench = load_returns([benchmark_upper], start=start, end=end, db_path=db)
            bench_series = bench[benchmark_upper]
        except PriceDataError:
            bench_series = None

        result = compute_stats(resolved, returns, bench_series, benchmark_upper)
    except (PortfolioSpecError, PriceDataError, StatsError) as exc:
        # Expected, user-fixable failures: one red line, exit code 1.
        # Anything else is a bug and should traceback loudly.
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from None

    console.print(render_stats(result))
