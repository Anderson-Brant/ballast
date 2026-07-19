"""`ballast var` and `ballast validate var` -- v0.3.0 commands.

Notes
-----
Orchestration only. Both commands share one pipeline: spec -> resolved
weights -> blended daily returns (the same chain `stats` uses), then hand
off to risk/var.py and validate/coverage.py.

`var` runs every method independently and degrades per-method: a spec
with too little history for filtered-historical still gets parametric and
historical rows, with the failure reason shown inline. Monte Carlo builds
its covariance from the trailing year (Ledoit-Wolf -- the v0.2.0 harness
winner among shrinkage estimators for conditioning) of the SAME symbols.

`validate var` backtests one method's rolling VaR and prints the
Kupiec / Christoffersen / traffic-light table.
"""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

console = Console()

validate_app = typer.Typer(help="Statistical validation of risk estimates.", no_args_is_help=True)


def _blended_returns(portfolio_file: Path, start, end, db):
    """spec -> (resolved, blended daily return Series). Shared pipeline."""
    from ballast.data.prices import load_latest_prices, load_returns
    from ballast.portfolio.spec import load_portfolio, resolve_weights
    from ballast.portfolio.stats import blended_returns

    portfolio = load_portfolio(portfolio_file)
    share_symbols = [p.symbol for p in portfolio.positions if p.shares is not None]
    latest = load_latest_prices(share_symbols, db_path=db) if share_symbols else {}
    resolved = resolve_weights(portfolio, latest)
    returns = load_returns(list(resolved.weights), start=start, end=end, db_path=db)
    return resolved, blended_returns(returns, resolved), returns


def var(
    portfolio_file: Annotated[Path, typer.Argument(help="Path to a portfolio.yaml spec.")],
    confidence: Annotated[float, typer.Option(help="Confidence level, e.g. 0.99.")] = 0.99,
    horizon: Annotated[int, typer.Option(help="Horizon in trading days.")] = 1,
    start: Annotated[str | None, typer.Option(help="History start, YYYY-MM-DD.")] = None,
    end: Annotated[str | None, typer.Option(help="History end, YYYY-MM-DD.")] = None,
    db: Annotated[Path | None, typer.Option(help="DuckDB path (default: configured).")] = None,
) -> None:
    """VaR and expected shortfall by every method, side by side."""
    from ballast.covariance.estimators import CovarianceError, ledoit_wolf_cov
    from ballast.data.prices import PriceDataError
    from ballast.portfolio.spec import PortfolioSpecError
    from ballast.portfolio.stats import StatsError
    from ballast.reporting.tables import render_var_estimates
    from ballast.risk.var import (
        VaRError,
        filtered_historical_var,
        historical_var,
        monte_carlo_var,
        parametric_var,
    )

    try:
        resolved, blended, asset_returns = _blended_returns(portfolio_file, start, end, db)
    except (PortfolioSpecError, PriceDataError, StatsError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from None

    # Each method runs independently; one failing (usually: short history)
    # is reported in its row instead of killing the command.
    estimates: list = []
    single_series = [
        ("parametric", lambda: parametric_var(blended, confidence, horizon)),
        (
            "cornish_fisher",
            lambda: parametric_var(blended, confidence, horizon, cornish_fisher=True),
        ),
        ("historical", lambda: historical_var(blended, confidence, horizon)),
        ("filtered_historical", lambda: filtered_historical_var(blended, confidence, horizon)),
    ]
    for method_name, compute in single_series:
        try:
            estimates.append(compute())
        except VaRError as exc:
            estimates.append((method_name, str(exc)))
    try:
        trailing = asset_returns.iloc[-252:]  # MC scale comes from the trailing year
        weights = trailing.columns.to_series().map(resolved.weights)
        estimates.append(
            monte_carlo_var(weights, ledoit_wolf_cov(trailing), confidence, horizon, seed=0)
        )
    except (VaRError, CovarianceError) as exc:
        estimates.append(("monte_carlo", str(exc)))

    console.print(render_var_estimates(estimates, resolved.name, resolved.nav))


@validate_app.command("var")
def validate_var_cmd(
    portfolio_file: Annotated[Path, typer.Argument(help="Path to a portfolio.yaml spec.")],
    method: Annotated[
        str, typer.Option(help="parametric | historical | filtered_historical.")
    ] = "historical",
    confidence: Annotated[float, typer.Option(help="Confidence level, e.g. 0.99.")] = 0.99,
    window: Annotated[int, typer.Option(help="Rolling estimation window, days.")] = 750,
    start: Annotated[str | None, typer.Option(help="History start, YYYY-MM-DD.")] = None,
    db: Annotated[Path | None, typer.Option(help="DuckDB path (default: configured).")] = None,
) -> None:
    """Backtest a VaR method: Kupiec, Christoffersen, Basel traffic light."""
    from ballast.data.prices import PriceDataError
    from ballast.portfolio.spec import PortfolioSpecError
    from ballast.portfolio.stats import StatsError
    from ballast.reporting.tables import render_validation
    from ballast.risk.var import VaRError
    from ballast.validate.coverage import validate_var

    try:
        _, blended, _ = _blended_returns(portfolio_file, start, None, db)
        result = validate_var(blended, method=method, confidence=confidence, window=window)
    except (PortfolioSpecError, PriceDataError, StatsError, VaRError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from None

    console.print(render_validation(result))


@validate_app.command("factors")
def validate_factors(
    start: Annotated[
        str | None, typer.Option(help="Estimation window start (default: 3 years back).")
    ] = None,
    end: Annotated[str | None, typer.Option(help="Window end (default: today).")] = None,
    db: Annotated[Path | None, typer.Option(help="DuckDB path (default: configured).")] = None,
) -> None:
    """Cross-check home-built factor returns against the Ken French library."""
    from datetime import date, timedelta

    from ballast.data.edgar import EdgarError
    from ballast.data.french import FrenchDataError, cross_check_factors, fetch_french_daily
    from ballast.data.prices import PriceDataError, list_symbols
    from ballast.factors.exposures import ExposureError
    from ballast.factors.regression import RegressionError, build_panel, fit_panel
    from ballast.reporting.tables import render_factor_check

    try:
        universe = list_symbols(db_path=db)
        end_iso = end or date.today().isoformat()
        start_iso = start or (date.fromisoformat(end_iso) - timedelta(days=3 * 365)).isoformat()
        exposures_panel, period_returns = build_panel(
            universe, start=start_iso, end=end_iso, db_path=db
        )
        model = fit_panel(
            exposures_panel,
            period_returns,
            min_specific_obs=min(20, max(8, len(exposures_panel) - 2)),
        )
        french = fetch_french_daily(start=start_iso)
        rows = cross_check_factors(model.factor_returns, french)
    except (
        PriceDataError,
        EdgarError,
        ExposureError,
        RegressionError,
        FrenchDataError,
    ) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from None

    console.print(render_factor_check(rows))
