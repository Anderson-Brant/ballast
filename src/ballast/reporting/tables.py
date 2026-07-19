"""Rich terminal renderers.

Notes
-----
What this file does: turns result dataclasses (PortfolioStats now;
DecompositionResult, ValidationResult, ... at later milestones) into Rich
renderables. The scorecards sketched in _notes/IDEAS.md under "target
output" are specified by these renderers.

Design rules (same as Sentinel's reporting layer):
- Zero computation. If a renderer needs a number, the producing module
  puts it in the dataclass. Renderers format, color, and align -- nothing
  else. This keeps every number testable without parsing terminal output.
- Dense and terse: numbers and test statistics, no adjectives.
- One renderer per result type; CLI modules call exactly one renderer.
"""

from typing import TYPE_CHECKING

from rich.console import Group
from rich.table import Table

if TYPE_CHECKING:  # imports for type hints only; keeps module import light
    from ballast.covariance.harness import ComparisonResult
    from ballast.data.french import CrossCheckRow
    from ballast.optimize.compare import OptimizerComparisonResult
    from ballast.portfolio.rebalance import RebalancePlan
    from ballast.portfolio.stats import PortfolioStats
    from ballast.risk.decompose import DecompositionResult
    from ballast.risk.var import VaREstimate
    from ballast.stress.scenarios import ShockResult, StressResult
    from ballast.validate.coverage import ValidationResult

__all__ = [
    "render_stats",
    "render_cov_comparison",
    "render_var_estimates",
    "render_validation",
    "render_decomposition",
    "render_factor_check",
    "render_optimizer_comparison",
    "render_stress",
    "render_shock",
    "render_rebalance",
    "render_target_weights",
]


def _pct(value: float) -> str:
    return f"{value:+.2%}" if value < 0 else f"{value:.2%}"


def render_stats(stats: "PortfolioStats") -> Group:
    """PortfolioStats -> the `ballast stats` output: headline table + weights."""
    title = (
        f"{stats.name} · {stats.start:%Y-%m-%d} → {stats.end:%Y-%m-%d}"
        f" · {stats.n_days} trading days"
    )

    head = Table(title=title, show_header=False, title_justify="left")
    head.add_column(style="bold")
    head.add_column(justify="right")
    head.add_row("CAGR", _pct(stats.cagr))
    head.add_row("Ann. vol", _pct(stats.ann_vol))
    head.add_row("Max drawdown", f"[red]{stats.max_drawdown:.2%}[/red]")
    if stats.beta is None:
        # Absence, stated as absence -- never rendered as 0.0.
        head.add_row(f"Beta vs {stats.benchmark}", f"— (no {stats.benchmark} data stored)")
    else:
        head.add_row(f"Beta vs {stats.benchmark}", f"{stats.beta:.2f}")
    if stats.nav is None:
        head.add_row("NAV", "— (weights-only spec, scale-free)")
    else:
        head.add_row("NAV", f"${stats.nav:,.2f}")

    weights = Table(title="Weights", title_justify="left")
    weights.add_column("Symbol")
    weights.add_column("Weight", justify="right")
    for symbol, w in stats.weights.items():
        weights.add_row(symbol, f"{w:.1%}")
    weights.add_row("[dim]cash[/dim]", f"[dim]{stats.cash_weight:.1%}[/dim]")

    return Group(head, weights)


def render_cov_comparison(result: "ComparisonResult") -> Table:
    """ComparisonResult -> the `ballast cov compare` table.

    Rows arrive already sorted by the producing module (lowest realized
    vol first); this function only formats them. Winner is bolded.
    """
    title = (
        f"Covariance horse race · {result.n_symbols} symbols · "
        f"window={result.window} step={result.step} cost={result.cost_bps}bps · "
        f"{result.n_windows} rebalances · {result.start:%Y-%m-%d} → {result.end:%Y-%m-%d}"
    )
    table = Table(
        title=title,
        title_justify="left",
        caption="score = realized out-of-sample vol of the min-variance portfolio (lower wins)",
        caption_justify="left",
    )
    table.add_column("Estimator")
    table.add_column("Realized vol", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Max DD", justify="right")
    table.add_column("Turnover", justify="right")
    table.add_column("Cond. number", justify="right")

    for i, row in enumerate(result.rows):
        style = "bold green" if i == 0 else ""
        table.add_row(
            row.name,
            f"{row.ann_vol:.2%}",
            "—" if row.sharpe is None else f"{row.sharpe:.2f}",
            f"{row.max_drawdown:.2%}",
            f"{row.avg_turnover:.2f}",
            "—" if row.avg_condition is None else f"{row.avg_condition:,.0f}",
            style=style,
        )
    return table


def render_var_estimates(
    estimates: list["VaREstimate | tuple[str, str]"],
    portfolio_name: str,
    nav: float | None,
) -> Table:
    """VaR/ES per method. A (method, reason) tuple renders as a degraded row --
    one method lacking data must not blank the whole table."""
    title = f"{portfolio_name} · VaR / expected shortfall"
    table = Table(title=title, title_justify="left")
    table.add_column("Method")
    table.add_column("Confidence", justify="right")
    table.add_column("Horizon", justify="right")
    table.add_column("VaR", justify="right")
    table.add_column("ES", justify="right")
    if nav is not None:
        table.add_column("VaR ($)", justify="right")

    for item in estimates:
        if isinstance(item, tuple):  # (method_name, why it couldn't run)
            method, reason = item
            row = [method, "—", "—", f"[dim]{reason}[/dim]", "—"]
            table.add_row(*(row + ["—"] if nav is not None else row))
            continue
        row = [
            item.method,
            f"{item.confidence:.0%}",
            f"{item.horizon_days}d",
            f"{item.var:.2%}",
            f"{item.es:.2%}",
        ]
        if nav is not None:
            row.append(f"${item.var * nav:,.0f}")
        table.add_row(*row)
    return table


def _pass_fail(p: float | None, passed: bool | None) -> str:
    if p is None:
        return "— (too few breaches to assess)"
    verdict = "[green]pass[/green]" if passed else "[red]FAIL[/red]"
    return f"p={p:.3f}   {verdict}"


def render_validation(result: "ValidationResult") -> Table:
    """ValidationResult -> the centerpiece table (see IDEAS.md target output)."""
    title = (
        f"{result.confidence:.0%} 1-day VaR · {result.method} · "
        f"{result.window}-day window · {result.start:%Y-%m-%d} → {result.end:%Y-%m-%d}"
    )
    zone_style = {"green": "green", "yellow": "yellow", "red": "red"}[result.zone]
    table = Table(title=title, title_justify="left", show_header=False)
    table.add_column(style="bold")
    table.add_column()
    table.add_row(
        "breaches",
        f"{result.n_breaches} observed vs {result.expected_breaches:.1f} expected "
        f"({result.n_obs} days)",
    )
    table.add_row("Kupiec POF", _pass_fail(result.kupiec_p, result.kupiec_pass))
    table.add_row(
        "Christoffersen ind.", _pass_fail(result.independence_p, result.independence_pass)
    )
    table.add_row("conditional coverage", _pass_fail(result.cc_p, result.cc_pass))
    if result.worst_date is not None:
        table.add_row(
            "worst breach",
            f"{result.worst_date:%Y-%m-%d}, loss {result.worst_loss:.2%} "
            f"({result.worst_ratio:.1f}x VaR)",
        )
    else:
        table.add_row("worst breach", "none observed")
    table.add_row(
        "verdict", f"[{zone_style}]{result.zone} zone[/{zone_style}] (Basel traffic light)"
    )
    return table


def render_decomposition(result: "DecompositionResult") -> Group:
    """DecompositionResult -> the `ballast decompose` output."""
    as_of = f" · as of {result.as_of:%Y-%m-%d}" if result.as_of is not None else ""
    title = f"{result.name} · {result.n_positions} positions{as_of}"

    head = Table(title=title, show_header=False, title_justify="left")
    head.add_column(style="bold")
    head.add_column(justify="right")
    head.add_row("Total risk", f"{result.total_vol:.2%} ann.")
    head.add_row("  Factor", f"{result.factor_vol:.2%}  ({result.factor_share:.0%} of variance)")
    head.add_row("  Specific", f"{result.specific_vol:.2%}  ({result.specific_share:.0%})")

    factors = Table(
        title="Factor contributions",
        title_justify="left",
        caption="signed vols: sign*sqrt|variance contribution|; variances add, vols don't",
        caption_justify="left",
    )
    factors.add_column("Factor")
    factors.add_column("Exposure", justify="right")
    factors.add_column("Contribution", justify="right")
    factors.add_column("Share", justify="right")
    for factor, row in result.factor_contributions.iterrows():
        exposure = result.portfolio_exposures[factor]
        vol_str = f"[red]{row['vol']:.2%}[/red]" if row["vol"] < 0 else f"{row['vol']:.2%}"
        factors.add_row(str(factor), f"{exposure:+.2f}", vol_str, f"{row['share']:.0%}")

    positions = Table(title="Position contributions", title_justify="left")
    positions.add_column("Symbol")
    positions.add_column("Weight", justify="right")
    positions.add_column("Share of risk", justify="right")
    positions.add_column("")
    for symbol, row in result.position_contributions.iterrows():
        # Concentration flag: contributing far more risk than weight.
        flag = "⚠ concentration" if row["share"] > 2 * row["weight"] > 0 else ""
        positions.add_row(str(symbol), f"{row['weight']:.1%}", f"{row['share']:.1%}", flag)

    bets = (
        "effective number of bets: — (short hedges present)"
        if result.effective_bets is None
        else f"effective number of bets: {result.effective_bets:.1f} ({result.n_positions} names)"
    )
    tail = Table(show_header=False, box=None)
    tail.add_column()
    tail.add_row(bets)

    return Group(head, factors, positions, tail)


def render_target_weights(weights, market_weights, posterior, scores, note: str = "") -> Table:
    """The `ballast optimize views` output: scores -> views -> target book."""
    table = Table(
        title="Black-Litterman target (Sentinel views)",
        title_justify="left",
        caption=note or None,
        caption_justify="left",
    )
    table.add_column("Symbol")
    table.add_column("Score", justify="right")
    table.add_column("Posterior μ", justify="right")
    table.add_column("Market prior", justify="right")
    table.add_column("Target", justify="right")
    table.add_column("Tilt", justify="right")
    for symbol in weights.index:
        tilt = weights[symbol] - market_weights[symbol]
        tilt_str = f"[green]{tilt:+.1%}[/green]" if tilt > 0 else f"[red]{tilt:+.1%}[/red]"
        table.add_row(
            str(symbol),
            f"{scores.get(symbol, float('nan')):.2f}",
            f"{posterior[symbol]:.2%}",
            f"{market_weights[symbol]:.1%}",
            f"{weights[symbol]:.1%}",
            tilt_str,
        )
    return table


def render_stress(result: "StressResult") -> Group:
    """StressResult -> the `ballast stress --scenario` output."""
    s = result.scenario
    title = f"{s.name} · {s.description} · {s.start} → {s.end}"
    head = Table(title=title, show_header=False, title_justify="left")
    head.add_column(style="bold")
    head.add_column(justify="right")
    color = "red" if result.portfolio_return < 0 else "green"
    head.add_row("Estimated P&L", f"[{color}]{result.portfolio_return:.2%}[/{color}]")
    head.add_row("Invested fraction", f"{result.invested_fraction:.1%} (cash sat out)")

    drivers = Table(title="Drivers (worst first)", title_justify="left")
    drivers.add_column("Symbol")
    drivers.add_column("Own return", justify="right")
    drivers.add_column("Contribution", justify="right")
    for symbol, contribution in result.contributions.items():
        drivers.add_row(
            str(symbol),
            f"{result.position_returns[symbol]:.2%}",
            f"{contribution:.2%}",
        )
    return Group(head, drivers)


def render_shock(result: "ShockResult") -> Group:
    """ShockResult -> the `ballast stress --shock` output."""
    head = Table(
        title="Hypothetical factor shock",
        show_header=False,
        title_justify="left",
        caption="linear factor-model estimate; specific risk and convexity excluded",
        caption_justify="left",
    )
    head.add_column(style="bold")
    head.add_column(justify="right")
    color = "red" if result.portfolio_return < 0 else "green"
    head.add_row("Estimated P&L", f"[{color}]{result.portfolio_return:.2%}[/{color}]")

    parts = Table(title="Per factor", title_justify="left")
    parts.add_column("Factor")
    parts.add_column("Exposure", justify="right")
    parts.add_column("Shock", justify="right")
    parts.add_column("Contribution", justify="right")
    for factor, contribution in result.factor_contributions.items():
        parts.add_row(
            str(factor),
            f"{result.exposures[factor]:+.2f}",
            f"{result.shocks[factor]:+.1%}",
            f"{contribution:.2%}",
        )
    return Group(head, parts)


def render_rebalance(plan: "RebalancePlan") -> Group:
    """RebalancePlan -> the `ballast rebalance` output."""
    head = Table(show_header=False, title_justify="left")
    head.add_column(style="bold")
    head.add_column(justify="right")
    head.add_row("NAV", f"${plan.nav:,.2f}")
    head.add_row("Band", f"{plan.band:.1%} weight points")
    head.add_row("Total to trade", f"${plan.total_traded:,.2f}")
    head.add_row("Estimated cost", f"${plan.total_cost:,.2f}")
    head.add_row("Cash after", f"{plan.cash_weight_after:.1%}")

    trades = Table(title="Trades (largest first)", title_justify="left")
    trades.add_column("Symbol")
    trades.add_column("Action")
    trades.add_column("Amount", justify="right")
    trades.add_column("Weight", justify="right")
    for trade in plan.trades:
        action = "[green]BUY[/green]" if trade.dollars > 0 else "[red]SELL[/red]"
        trades.add_row(
            trade.symbol,
            action,
            f"${abs(trade.dollars):,.2f}",
            f"{trade.current_weight:.1%} → {trade.target_weight:.1%}",
        )
    if not plan.trades:
        trades.add_row("[dim]nothing to trade: everything inside the band[/dim]", "", "", "")
    if plan.skipped:
        inside = ", ".join(f"{s} ({d:+.1%})" for s, d in plan.skipped)
        trades.caption = f"inside the band, left alone: {inside}"
        trades.caption_justify = "left"
    return Group(head, trades)


def render_optimizer_comparison(result: "OptimizerComparisonResult") -> Table:
    """OptimizerComparisonResult -> the `ballast optimize compare` table."""
    title = (
        f"Optimizer race · {result.n_symbols} symbols · "
        f"window={result.window} step={result.step} cost={result.cost_bps}bps · "
        f"{result.n_windows} rebalances · {result.start:%Y-%m-%d} → {result.end:%Y-%m-%d}"
    )
    table = Table(
        title=title,
        title_justify="left",
        caption="the bar is equal weight: beat 1/N after costs or don't earn the complexity",
        caption_justify="left",
    )
    table.add_column("Strategy")
    table.add_column("CAGR", justify="right")
    table.add_column("Vol", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Max DD", justify="right")
    table.add_column("Turnover", justify="right")
    table.add_column("Cost drag", justify="right")

    for i, row in enumerate(result.rows):
        style = "bold green" if i == 0 else ""
        table.add_row(
            row.name,
            f"{row.ann_return:.2%}",
            f"{row.ann_vol:.2%}",
            "—" if row.sharpe is None else f"{row.sharpe:.2f}",
            f"{row.max_drawdown:.2%}",
            f"{row.avg_turnover:.2f}",
            f"{row.cost_drag:.2%}",
            style=style,
        )
    for name, reason in result.skipped:
        table.add_row(name, f"[dim]skipped: {reason}[/dim]", "", "", "", "", "")
    return table


def render_factor_check(rows: list["CrossCheckRow"]) -> Table:
    """Cross-check rows -> the `ballast validate factors` table."""
    table = Table(
        title="Factor construction vs Ken French library",
        title_justify="left",
        caption="pass = correlation x expected sign >= 0.8 (the acceptance bar)",
        caption_justify="left",
    )
    table.add_column("Ballast factor")
    table.add_column("French series")
    table.add_column("Expected sign", justify="right")
    table.add_column("Correlation", justify="right")
    table.add_column("Periods", justify="right")
    table.add_column("Verdict", justify="right")
    for row in rows:
        verdict = "[green]pass[/green]" if row.passes else "[red]FAIL[/red]"
        table.add_row(
            row.ballast_factor,
            row.french_factor,
            "+" if row.expected_sign > 0 else "−",
            f"{row.corr:+.3f}",
            str(row.n_periods),
            verdict,
        )
    return table
