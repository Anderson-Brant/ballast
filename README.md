# Ballast

**Portfolio construction and risk engine.** Companion to [Sentinel](https://github.com/Anderson-Brant/sentinel): Sentinel ranks stocks, Ballast turns a list into a portfolio and tells you what you're actually exposed to.

> **Status: roadmap complete (v0.1.0 - v0.6.0).** Every command below works end to end. Direction and milestones live in `_notes/IDEAS.md` (local).

```bash
ballast import-sentinel ~/path/to/sentinel.duckdb   # bootstrap from Sentinel's data
ballast ingest SPY AAPL MSFT --start 2015-01-01     # or fetch fresh via yfinance
```

## The commands

```bash
ballast stats core.yaml            # vol, beta, max drawdown, CAGR
ballast cov compare                # which covariance estimator wins OOS
ballast var core.yaml              # VaR / ES, five methods side by side
ballast validate var core.yaml     # Kupiec, Christoffersen, Basel traffic light
ballast validate factors           # home-built factors vs the French library
ballast decompose core.yaml        # factor vs specific risk, per position
ballast optimize compare           # MVO / HRP / risk parity / CVaR vs 1/N
ballast optimize views scores.csv  # Sentinel scores -> BL views -> target book
ballast stress core.yaml --scenario covid2020   # replay 2008 / 2020 / 2022
ballast stress core.yaml --shock market=-0.2    # hypothetical factor moves
ballast rebalance core.yaml --target target.yaml  # drift-band trade plan
```

The two-project loop: `sentinel screen` ranks the universe → `ballast optimize
views` turns the ranking into a Black-Litterman target → `ballast rebalance`
prints the trades → `ballast decompose` and `ballast stress` tell you what
you'd actually own.

Ground rules carried over from Sentinel: walk-forward evaluation only, naive baselines first (1/N is the bar), free data only (yfinance, SEC EDGAR, Ken French library), and every risk number ships with the statistical test that could falsify it.

## Layout

```
src/ballast/
├── cli/               Typer CLI, one module per domain
├── config.py          Pydantic settings
├── data/              yfinance, Sentinel import, EDGAR, French library
├── portfolio/         portfolio.yaml spec → resolved positions
├── covariance/        estimators + out-of-sample comparison harness
├── factors/           exposures, cross-sectional regression, factor cov
├── risk/              VaR/ES, decomposition
├── validate/          Kupiec, Christoffersen, traffic light
├── optimize/          MVO, Black-Litterman, HRP, risk parity, CVaR
├── backtest/          walk-forward simulation, costs, turnover
├── stress/            scenario replay, factor shocks
└── reporting/         Rich tables
```

## Dev setup

```bash
pip install -e ".[dev]"
ballast version
make check        # ruff + mypy + pytest
```

## License

MIT. See [LICENSE](LICENSE).
