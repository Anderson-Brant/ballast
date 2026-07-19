# Changelog

## Unreleased

- The Sentinel bridge (v0.6.0 -- ROADMAP COMPLETE): `data/sentinel_views.py`
  (the symbol,score CSV interchange contract), `views_from_scores` in
  black_litterman.py (Grinold: alpha = IC x vol x z-scored rank; unscored
  symbols honestly stay at the prior), and `ballast optimize views` producing a
  target portfolio.yaml that feeds straight into `ballast rebalance --target`.
  Market-cap prior when fundamentals exist, stated equal-weight fallback when
  not. Full circle proven on real data: scores -> BL target -> trade plan.
  (8 tests)
- `stress/scenarios.py` + `ballast stress` implemented (v0.6.0 begins): named
  historical windows (gfc2008, covid2020, rates2022) replayed buy-and-hold on
  the current book from stored prices with STRICT coverage (a symbol that
  didn't trade through the window is a named error, not a hole), plus
  hypothetical factor shocks through current exposures (--shock market=-0.2).
  Real-data replay: the demo book loses -26.2% through covid2020 and -20.8%
  through rates2022, drivers itemized. (10 tests)
- `portfolio/rebalance.py` + `ballast rebalance` implemented: drift-band trade
  planning -- trades only when |current - target| exceeds the band, minimum
  dollar filter, per-side costs, exits/entries via the union universe, and the
  skipped positions reported WITH their drift (visible restraint). Plans only;
  never executes. (8 tests)
- `optimize/black_litterman.py` implemented: reverse optimization (equilibrium
  prior; MVO on it round-trips to market weights, pinned by test), View dataclass
  with absolute/relative helpers, confidence-scaled Omega (c=1 hits the view
  exactly), and the correlation-spillover property proven by test. The Sentinel
  views bridge consumes this at v0.6.0. (9 tests)
- `optimize/compare.py` + `ballast optimize compare` implemented (v0.5.0 core
  complete): all optimizers on the same Ledoit-Wolf covariance vs 1/N, walk-
  forward with costs, per-strategy skips recorded and displayed. First real-data
  race (5 Sentinel symbols, 124 rebalances, 2016-2026): 1/N wins Sharpe (1.08),
  risk parity nearly ties (1.07) with lower vol and drawdown, min-variance and
  min-CVaR deliver the lowest risk but trail on Sharpe -- the DeMiguel (2009)
  finding, reproduced on our own data as the roadmap predicted. (5 tests)
- `optimize/cvar.py` implemented: minimum-CVaR weights via the Rockafellar-
  Uryasev linear program over historical scenarios -- distribution-free, with a
  guard refusing fewer than ~10 tail observations. Anchor: reproduces GMV on
  Gaussian scenarios (elliptical equivalence). The point, pinned by test: on two
  matched-variance assets where one crashes, MVO splits 50/50 and CVaR walks
  away from the crash asset. (8 tests)
- `optimize/risk_parity.py` implemented: equal risk contribution (and general
  risk budgets) via Spinu's convex formulation solved by cyclical coordinate
  descent -- closed-form positive-root updates, safe under negative correlation,
  inverse-vol warm start. Non-convergence raises (no partial answer), and a
  self-check re-verifies contributions against budgets with the same formula
  risk/decompose.py uses. Hand case: diag(1,4) gives [2/3, 1/3] (inverse vol)
  vs GMV/HRP's [0.8, 0.2] and 1/N's [0.5, 0.5]. (11 tests)
- `optimize/hrp.py` implemented: Hierarchical Risk Parity per Lopez de Prado --
  correlation-distance clustering, quasi-diagonalization, recursive bisection
  with inverse-variance branch budgets. No matrix inversion, no solver; handles
  singular covariances that destabilize MVO (pinned by test). Long-only and
  unit-sum by construction. Documented quirk: deterministic but not permutation
  invariant (dendrogram orientation), faithful to the published algorithm.
  (10 tests)
- `optimize/mvo.py` implemented (v0.5.0 begins): constrained Markowitz via cvxpy
  -- min-variance and mean-variance modes; long-only, max-weight, sector-cap,
  and fully-invested-or-cash constraints; lazy cvxpy import ([opt] extra, now
  also in dev for CI); infeasibility caught by arithmetic where cheap, named
  solver status otherwise, never a silent 1/N fallback. Anchor test: the convex
  program reproduces the closed-form GMV to 1e-6 on random matrices. (11 tests)
- `data/french.py` + `ballast validate factors` implemented (v0.4.0 complete):
  Ken French library download with a defensive parser (percent scaling, -99
  missing markers, junk-tail cutoff -- format pinned by fixture tests), daily
  series compounded onto the model's exact weekly (start, next] intervals, and
  the correlation cross-check with expected signs (size vs SMB is deliberately
  negative). Acceptance bar: corr x sign >= 0.8. Exact-construction tests give
  corr = +/-1 by design, so any deviation is an alignment bug. (11 tests)
- `risk/decompose.py` + `ballast decompose` implemented (v0.4.0 scope complete):
  w'(BFB'+D)w split into signed per-factor and per-position contributions (sums
  reconcile exactly, enforced by invariant + property test), market exposure =
  invested fraction, effective number of bets (None when short hedges break it),
  strict refusal on any symbol the model can't cover. CLI fits the model on the
  full DB universe with an adaptive specific-variance floor for short windows.
  (12 tests)
- `factors/regression.py` implemented: per-period cross-sectional WLS with the
  intercept as the market factor, panel fitter with visible skip bookkeeping
  (>20% thin periods fails the fit), EWMA factor covariance (weekly lambda 0.97),
  per-symbol specific variance with a minimum-observations floor, and the weekly
  DB-backed panel builder. Ground-truth test: returns simulated from KNOWN factor
  returns are recovered at corr > 0.99 per factor. (10 tests)
- `factors/exposures.py` implemented: value (B/P + E/P), momentum (12-1), size
  (log mcap), quality (gross profitability + low leverage), low_vol -- built as
  winsorize -> z-score -> composites -> optional sector demeaning, with every
  orientation and the pipeline order pinned by tests. NaN-never-guess missing-data
  policy; the as-of gate proven end to end (a symbol whose 10-K wasn't public yet
  gets NaN fundamentals but real price factors). Data layer grew `load_prices`
  (wide, per-symbol histories), an `as_of` parameter on `load_latest_prices`, and
  an annual-only rule for income-statement fields (no 3-month/12-month mixing).
  (13 tests)
- `data/edgar.py` implemented (v0.4.0 begins): point-in-time fundamentals from
  SEC EDGAR company-facts. Every value stored twice-dated (fiscal period end +
  filing date); `latest_fundamentals(as_of=...)` serves only rows filed on or
  before as_of -- the no-leakage contract, enforced structurally and pinned by
  tests (a filing is invisible the day before it was filed; amendments win only
  after their own filing date). Candidate-tag fallback per field, dei namespace
  for shares outstanding, 8-K/S-1 forms dropped, UA header + rate-limit delay
  built into the single network seam. New `BALLAST_EDGAR_USER_AGENT` setting.
  (12 offline tests; live smoke blocked by sandbox proxy -- run locally.)

- Project scaffold: package layout, CLI entry point (`ballast version`), CI, packaging.
- `portfolio/spec.py` implemented: YAML loading with strict validation, immutable
  dataclasses, and shares/weights resolution at as-of prices (41 tests).
- `data/prices.py` implemented: yfinance ingestion into DuckDB (schema identical to
  Sentinel's `prices` table), idempotent INSERT OR REPLACE writes, and `load_returns`
  with inner-alignment policy and adj_close-based returns (20 offline tests).
- `data/sentinel_import.py` implemented: read-only ATTACH of a Sentinel DuckDB,
  exact schema validation before any copy, INSERT OR IGNORE so Ballast's own rows
  win, returns the count of new rows (8 tests; verified against a real Sentinel DB).
- v0.1.0 scope complete: `portfolio/stats.py` (blended returns, CAGR, ann. vol,
  max drawdown, beta), `load_latest_prices` for NAV valuation, `render_stats`
  Rich output, and CLI commands `stats`, `ingest`, `import-sentinel` (19 tests).
  Verified end to end against real Sentinel data.
- `covariance/estimators.py` implemented (v0.2.0 begins): sample, EWMA
  (RiskMetrics, zero-mean, normalized weights), and from-scratch Ledoit-Wolf and
  OAS shrinkage; PSD guarantee on every output; `annualize` helper. Ground-truth
  recovery fixture in conftest; LW verified equal to scikit-learn's to 1e-8
  (19 tests).
- `backtest/engine.py` implemented: the shared walk-forward loop (1-bar shift,
  drifting weights, turnover costs including the first position build, wipeout
  guard). Leakage boundary pinned by test. (12 tests)
- `covariance/harness.py` + `ballast cov compare` implemented: unconstrained GMV
  per estimator, 1/N baseline, condition-number diagnostics, Rich table sorted by
  realized OOS vol; `list_symbols` added to the data layer. First real-data run
  (5 Sentinel symbols, 124 rebalances): every estimator ~17% realized vol vs 26.4%
  for 1/N; sample wins narrowly at N=5, as theory predicts. (10 tests)
- `validate/coverage.py` + `ballast var` / `ballast validate var` implemented
  (v0.3.0 scope complete): Kupiec POF, Christoffersen independence, conditional
  coverage (xlogy-safe likelihoods, honest None when unassessable), Basel traffic
  light generalized via binomial CDF (reproduces the official 250-day/99% table
  exactly, pinned by test). Real-data run on a 10-year blended portfolio:
  parametric = red zone (53 breaches vs 21 expected), historical = right count
  but clustered (fails independence, p=0.005), filtered historical = passes all
  three. Textbook behavior, observed live. (23 tests)
- `risk/var.py` implemented (v0.3.0 begins): parametric normal, Cornish-Fisher
  (numerical tail-averaged ES), historical, filtered historical (EWMA-standardized,
  burn-in seeded, no full-sample vol leakage), and seeded Monte Carlo -- ES always
  alongside VaR, positive-loss convention, sqrt-horizon scaling documented as an
  approximation. Plus `rolling_var`: the no-lookahead day-by-day series the
  validation suite consumes. Known FHS fat-tail inflation on i.i.d. data pinned
  as a band in tests, not hidden. (20 tests)
