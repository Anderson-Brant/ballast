"""`ballast decompose` -- the v0.4.0 headline command.

Notes
-----
The full pipeline in one command: load spec -> resolve weights -> build
the weekly exposure/return panel over the estimation window -> fit the
factor model -> compute exposures at the as-of date -> decompose ->
render. Orchestration only; each stage's math lives in its module.

The model universe defaults to EVERY symbol in the database, not just the
portfolio's holdings -- factor returns estimated from a broader
cross-section are better estimates, and the regression needs at least
factors + 3 symbols to fit at all. A five-symbol database cannot support
a five-factor model; the error says so instead of fitting garbage.
"""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

console = Console()


def decompose(
    portfolio_file: Annotated[Path, typer.Argument(help="Path to a portfolio.yaml spec.")],
    start: Annotated[
        str | None, typer.Option(help="Estimation window start (default: 3 years back).")
    ] = None,
    end: Annotated[str | None, typer.Option(help="As-of date (default: latest data).")] = None,
    db: Annotated[Path | None, typer.Option(help="DuckDB path (default: configured).")] = None,
) -> None:
    """Factor vs specific risk, per factor and per position."""
    from datetime import date, timedelta

    import pandas as pd

    from ballast.data.edgar import EdgarError
    from ballast.data.prices import PriceDataError, list_symbols, load_latest_prices
    from ballast.factors.exposures import ExposureError
    from ballast.factors.regression import RegressionError, build_panel, fit_panel
    from ballast.portfolio.spec import PortfolioSpecError, load_portfolio, resolve_weights
    from ballast.reporting.tables import render_decomposition
    from ballast.risk.decompose import DecompositionError, decompose_portfolio

    try:
        portfolio = load_portfolio(portfolio_file)
        share_symbols = [p.symbol for p in portfolio.positions if p.shares is not None]
        latest = load_latest_prices(share_symbols, db_path=db) if share_symbols else {}
        resolved = resolve_weights(portfolio, latest)
        if not resolved.weights:
            raise DecompositionError("portfolio holds only cash; there is nothing to decompose")

        universe = list_symbols(db_path=db)
        missing = sorted(set(resolved.weights) - set(universe))
        if missing:
            raise PriceDataError(f"no stored prices for portfolio symbol(s) {missing}")

        end_iso = end or date.today().isoformat()
        start_iso = start or (date.fromisoformat(end_iso) - timedelta(days=3 * 365)).isoformat()

        # Fit the model on the broad universe, weekly.
        exposures_panel, period_returns = build_panel(
            universe, start=start_iso, end=end_iso, db_path=db
        )
        # Specific variance wants 20+ weekly residuals; a short --start/--end
        # window can't supply that, so the floor adapts down to 8. The cost
        # is noisier specific risk -- a consequence of the window the user
        # chose, not hidden: it's their tradeoff to make.
        min_obs = min(20, max(8, len(exposures_panel) - 2))
        model = fit_panel(exposures_panel, period_returns, min_specific_obs=min_obs)

        # Decompose at the LAST panel date: the exposures the model would
        # use for next week, which is what "current risk" means here.
        as_of = sorted(exposures_panel)[-1]
        weights = pd.Series(resolved.weights)
        result = decompose_portfolio(
            weights, exposures_panel[as_of], model, name=resolved.name, as_of=as_of
        )
    except (
        PortfolioSpecError,
        PriceDataError,
        EdgarError,
        ExposureError,
        RegressionError,
        DecompositionError,
    ) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from None

    console.print(render_decomposition(result))
