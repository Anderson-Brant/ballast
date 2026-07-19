"""`ballast optimize` -- optimizer commands (v0.5.0).

Notes
-----
Orchestration only, same shape as `ballast cov compare`: resolve symbols,
load returns, hand off to optimize/compare.py, render. No symbols means
the whole database -- the post-import-sentinel default.
"""

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console

console = Console()

optimize_app = typer.Typer(help="Portfolio optimizer lab.", no_args_is_help=True)


@optimize_app.command("compare")
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
    """The optimizer race: every strategy vs 1/N, walk-forward, after costs."""
    from ballast.data.prices import PriceDataError, list_symbols, load_returns
    from ballast.optimize.compare import compare_optimizers
    from ballast.optimize.mvo import OptimizationError
    from ballast.reporting.tables import render_optimizer_comparison

    try:
        chosen = [s.strip().upper() for s in symbols] if symbols else list_symbols(db_path=db)
        if len(chosen) < 2:
            raise PriceDataError(
                "the race needs at least 2 symbols with stored prices; "
                "run `ballast ingest` or `ballast import-sentinel` first"
            )
        returns = load_returns(chosen, start=start, db_path=db)
        result = compare_optimizers(returns, window=window, step=step, cost_bps=cost_bps)
    except (PriceDataError, OptimizationError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from None

    console.print(render_optimizer_comparison(result))


@optimize_app.command("views")
def views(
    scores_file: Annotated[Path, typer.Argument(help="CSV of symbol,score (Sentinel screen).")],
    ic: Annotated[
        float, typer.Option(help="Information coefficient: how much to trust the ranking.")
    ] = 0.05,
    confidence: Annotated[float, typer.Option(help="View confidence in (0, 1].")] = 0.3,
    start: Annotated[
        str | None, typer.Option(help="Covariance window start (default: 1 year back).")
    ] = None,
    max_weight: Annotated[float | None, typer.Option(help="Per-position cap.")] = None,
    output: Annotated[
        Path | None, typer.Option(help="Write the target book as a portfolio.yaml here.")
    ] = None,
    db: Annotated[Path | None, typer.Option(help="DuckDB path (default: configured).")] = None,
) -> None:
    """Sentinel screen scores -> Black-Litterman views -> target portfolio.

    The bridge: rankings become humble return views, views tilt the
    equilibrium prior, MVO turns the posterior into long-only weights.
    Feed --output into `ballast rebalance --target` to close the loop.
    """
    from datetime import date, timedelta

    import yaml

    from ballast.covariance.estimators import CovarianceError, annualize, ledoit_wolf_cov
    from ballast.data.edgar import EdgarError, latest_fundamentals
    from ballast.data.prices import PriceDataError, load_latest_prices, load_returns
    from ballast.data.sentinel_views import ScoresError, load_scores
    from ballast.optimize.black_litterman import black_litterman_returns, views_from_scores
    from ballast.optimize.mvo import OptimizationError, mvo_weights
    from ballast.reporting.tables import render_target_weights

    try:
        scores = load_scores(scores_file)
        symbols = list(scores.index)
        start_iso = start or (date.today() - timedelta(days=365)).isoformat()
        returns = load_returns(symbols, start=start_iso, db_path=db)
        cov = annualize(ledoit_wolf_cov(returns))  # BL wants annual units

        # Market weights for the equilibrium prior: real market caps when
        # fundamentals are ingested; otherwise fall back to equal weights
        # for ALL symbols (mixing real and guessed caps would skew the
        # prior worse than either alone) -- stated, not silent.
        market_note = ""
        try:
            shares = latest_fundamentals(symbols, as_of=date.today(), db_path=db)[
                "shares_outstanding"
            ]
            closes = pd.Series(load_latest_prices(symbols, db_path=db))
            mcaps = shares * closes
            if mcaps.isna().any() or (mcaps <= 0).any():
                raise EdgarError("incomplete shares outstanding")
            market_weights = mcaps / mcaps.sum()
        except EdgarError:
            market_weights = pd.Series(1.0 / len(symbols), index=symbols)
            market_note = "equal-weight prior (no market caps: ingest fundamentals to fix)"

        bl_views = views_from_scores(scores, cov, market_weights, ic=ic, confidence=confidence)
        posterior = black_litterman_returns(cov, market_weights, bl_views)
        weights = mvo_weights(cov, expected_returns=posterior, max_weight=max_weight)
    except (ScoresError, PriceDataError, CovarianceError, OptimizationError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from None

    console.print(render_target_weights(weights, market_weights, posterior, scores, market_note))

    if output is not None:
        # Round for the YAML, then push the rounding error into the largest
        # position so the file still passes load_portfolio's sum check.
        rounded = {s: round(float(w), 6) for s, w in weights.items() if w > 1e-6}
        drift = sum(rounded.values()) - 1.0
        if drift > 0:
            biggest = max(rounded, key=rounded.get)  # type: ignore[arg-type]
            rounded[biggest] = round(rounded[biggest] - drift, 6)
        spec = {
            "name": "bl-target",
            "positions": [{"symbol": s, "weight": w} for s, w in rounded.items()],
        }
        output.write_text(yaml.safe_dump(spec, sort_keys=False))
        console.print(f"target written to {output} (feed it to `ballast rebalance --target`)")
