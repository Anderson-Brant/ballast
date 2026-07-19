"""`ballast rebalance` -- drift-band trade planning (v0.6.0).

Notes
-----
Current book from one spec, target book from another (typically a
weights-only spec produced by an optimizer or by hand), both resolved at
the same latest prices; portfolio/rebalance.py does the math. Plans only
-- Ballast never executes trades.
"""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from ballast.cli.stress import _resolve_book

console = Console()


def rebalance(
    portfolio_file: Annotated[Path, typer.Argument(help="Current book: portfolio.yaml.")],
    target: Annotated[Path, typer.Option(help="Target book: another portfolio.yaml.")],
    band: Annotated[float, typer.Option(help="Drift band in weight points.")] = 0.05,
    cost_bps: Annotated[float, typer.Option(help="Per-side cost in basis points.")] = 2.0,
    min_trade: Annotated[float, typer.Option(help="Minimum trade size in dollars.")] = 0.0,
    db: Annotated[Path | None, typer.Option(help="DuckDB path (default: configured).")] = None,
) -> None:
    """The trade list that moves the current book to the target, band-gated."""
    from ballast.data.prices import PriceDataError
    from ballast.portfolio.rebalance import (
        RebalanceError,
        plan_rebalance,
        targets_from_resolved,
    )
    from ballast.portfolio.spec import PortfolioSpecError
    from ballast.reporting.tables import render_rebalance

    try:
        current = _resolve_book(portfolio_file, db)
        target_book = _resolve_book(target, db)
        plan = plan_rebalance(
            current,
            targets_from_resolved(target_book),
            band=band,
            cost_bps=cost_bps,
            min_trade_dollars=min_trade,
        )
    except (PortfolioSpecError, PriceDataError, RebalanceError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from None

    console.print(render_rebalance(plan))
