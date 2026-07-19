"""`ballast stress` -- scenario replay and hypothetical shocks (v0.6.0).

Notes
-----
Two modes, mutually exclusive:
- --scenario covid2020: buy-and-hold the current book through the named
  window using stored prices (stress/scenarios.py does the math).
- --shock market=-0.20 --shock momentum=-0.10: linear factor-model
  estimate through current exposures (computed across the whole DB
  universe for a meaningful cross-section, then subset to the book).

Orchestration only, as always.
"""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

console = Console()


def _resolve_book(portfolio_file: Path, db):
    from ballast.data.prices import load_latest_prices
    from ballast.portfolio.spec import load_portfolio, resolve_weights

    portfolio = load_portfolio(portfolio_file)
    share_symbols = [p.symbol for p in portfolio.positions if p.shares is not None]
    latest = load_latest_prices(share_symbols, db_path=db) if share_symbols else {}
    return resolve_weights(portfolio, latest)


def stress(
    portfolio_file: Annotated[Path, typer.Argument(help="Path to a portfolio.yaml spec.")],
    scenario: Annotated[
        str | None, typer.Option(help="Named window: gfc2008 | covid2020 | rates2022.")
    ] = None,
    shock: Annotated[
        list[str] | None,
        typer.Option(help="factor=value, repeatable (e.g. --shock market=-0.2)."),
    ] = None,
    db: Annotated[Path | None, typer.Option(help="DuckDB path (default: configured).")] = None,
) -> None:
    """Replay a historical crisis, or apply a hypothetical factor shock."""
    import pandas as pd

    from ballast.data.edgar import EdgarError
    from ballast.data.prices import PriceDataError, list_symbols
    from ballast.factors.exposures import ExposureError
    from ballast.portfolio.spec import PortfolioSpecError
    from ballast.reporting.tables import render_shock, render_stress
    from ballast.stress.scenarios import SCENARIOS, StressError, factor_shock, replay_scenario

    if (scenario is None) == (shock is None):
        console.print(
            f"[red]error:[/red] give exactly one of --scenario or --shock "
            f"(scenarios: {sorted(SCENARIOS)})"
        )
        raise typer.Exit(1)

    try:
        resolved = _resolve_book(portfolio_file, db)
        if not resolved.weights:
            raise StressError("portfolio holds only cash; there is nothing to stress")
        weights = pd.Series(resolved.weights)

        if scenario is not None:
            result = replay_scenario(weights, scenario, db_path=db)
            console.print(render_stress(result))
            return

        shocks: dict[str, float] = {}
        for item in shock or []:
            factor, _, value = item.partition("=")
            try:
                shocks[factor.strip()] = float(value)
            except ValueError:
                raise StressError(f"bad shock {item!r}; expected factor=value") from None

        # Exposures need a cross-section to z-score against: use the whole
        # DB universe (same reasoning as decompose), then shock the book.
        from datetime import date

        from ballast.factors.exposures import compute_exposures

        universe = list_symbols(db_path=db)
        exposures = compute_exposures(universe, as_of=date.today(), db_path=db)
        console.print(render_shock(factor_shock(weights, exposures, shocks)))
    except (PortfolioSpecError, PriceDataError, EdgarError, ExposureError, StressError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from None
