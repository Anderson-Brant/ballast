# Methodology

The rules below held from day one; the findings below them are what the
rules produced. Update the findings with a run date whenever an experiment
is rerun — numbers without dates rot silently.

## Ground rules

1. Walk-forward evaluation only (252-day window, 21-day step). No full-sample fits presented as results.
2. Baselines first: 1/N for optimizers, sample covariance for estimators, parametric-normal for VaR. Everything is reported relative to these.
3. No leakage. Fundamentals enter at filing date, never fiscal period end. Backtest weights decided with data through t-1 first earn day t (the 1-bar shift). Rolling VaR never lets a day participate in its own estimate.
4. Universe is current S&P 500 constituents. Survivorship bias is acceptable for risk modeling; it would not be for alpha claims, so none are made.
5. Every VaR method ships with the tests that could reject it (Kupiec, Christoffersen). A model that never fails a test anywhere means the tests are too weak.
6. Costs are never optional in backtests (2 bps per side on traded notional, including the initial position build).

## Recorded findings

> **Caveat that applies to every number below:** the universe is 5 mega-caps
> (SPY, AAPL, MSFT, AMZN, TSLA) imported from Sentinel, 2015–2026. That is a
> smoke-scale sample with survivorship baked in. Directionally informative,
> not quotable. Rerun at N ≥ 50 before treating any spread as real.

### Covariance horse race — run 2026-07-13

`ballast cov compare` · window=252 step=21 cost=2bps · 124 rebalances · 2016-01-05 → 2026-04-17

| Estimator | Realized vol | Sharpe | Max DD | Turnover | Cond. |
|---|---|---|---|---|---|
| sample | 16.89% | 0.61 | -35.55% | 0.09 | 72 |
| oas | 16.95% | 0.72 | -32.70% | 0.07 | 52 |
| ewma | 17.41% | 0.84 | -34.09% | 0.36 | 108 |
| ledoit_wolf | 17.43% | 0.81 | -29.51% | 0.07 | 37 |
| equal_weight (1/N) | 26.44% | 1.08 | -42.69% | 0.03 | — |

Reading: minimum-variance works (vol cut from 26% to 17% regardless of estimator). The estimators are statistically indistinguishable at N=5 — with T=252 >> N, the sample matrix isn't noisy enough for shrinkage to earn its keep. The interesting comparison starts at N ≥ 50. EWMA's adaptivity costs ~5x the turnover of the shrinkage estimators.

### VaR validation — run 2026-07-13

`ballast validate var` · 99% 1-day · 750-day window · SPY/AAPL/MSFT blend · 2017-12-26 → 2026-04-17

| Method | Breaches (exp. ~21) | Kupiec | Christoffersen | Zone | Worst breach |
|---|---|---|---|---|---|
| parametric | 53 | FAIL (p=0.000) | FAIL (p=0.000) | red | 2020-03-16, 4.3x VaR |
| historical | 28 | pass (p=0.137) | FAIL (p=0.005) | green | 2020-03-16, 3.2x VaR |
| filtered_historical | 22 | pass (p=0.756) | pass (p=0.234) | green | 2018-10-10, 2.5x VaR |

Reading: the textbook, observed live. Normal tails understate equity risk (parametric rejected outright). Historical gets the long-run count right but clusters its failures in COVID week — the exact failure mode Christoffersen exists to catch, and why Kupiec alone is insufficient. FHS passes all three because it rescales to current vol; note its worst breach is October 2018, not March 2020 — by mid-March it had already adapted. Rule 5 satisfied: methods fail where they should.

### Optimizer race — run 2026-07-13

`ballast optimize compare` · window=252 step=21 cost=2bps · 124 rebalances · same universe

| Strategy | CAGR | Vol | Sharpe | Max DD | Turnover |
|---|---|---|---|---|---|
| equal_weight (1/N) | 28.59% | 26.44% | 1.08 | -42.69% | 0.03 |
| risk_parity | 25.29% | 23.71% | 1.07 | -37.46% | 0.03 |
| hrp | 22.28% | 22.48% | 1.01 | -35.38% | 0.05 |
| min_variance | 15.54% | 18.31% | 0.88 | -30.13% | 0.03 |
| min_cvar | 15.59% | 19.64% | 0.84 | -33.33% | 0.05 |

Reading: DeMiguel et al. (2009) reproduced on our own pipeline — nothing beat 1/N on Sharpe after costs. The risk optimizers did exactly what they promise (lowest vol and drawdown); Sharpe simply rewards return-chasing in a mega-cap bull decade. All covariance-consuming strategies used the same Ledoit-Wolf matrix, so rows differ by optimizer, not estimator. No significance testing yet — the Sharpe gaps at N=5 are likely noise.

### Crisis replays — run 2026-07-13

`ballast stress` on a demo book (SPY/AAPL/MSFT, ~82% invested): **covid2020 −26.2%** (SPY the largest contributor, by weight), **rates2022 −20.8%** (MSFT the worst single name at −31%). Raw price replay; cash sat out.

### Pending experiments

- `validate factors` on real breadth: the momentum-vs-UMD ≥ 0.8 acceptance bar (needs 50+ symbols with fundamentals and 2+ years of weekly panel).
- Reruns of everything above at N ≥ 50.
- Significance tests on both races (bootstrap the Sharpe/vol gaps).
