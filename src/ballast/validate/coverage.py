"""Backtests of the VaR itself. The centerpiece. v0.3.0 -- implemented.

Notes
-----
What this file does: treats every VaR method as a hypothesis and tests it
against what subsequently happened. A 99% 1-day VaR is a claim: losses
exceed it on 1% of days, scattered, not clustered. This file checks both
halves of that claim.

The tests:
- kupiec_pof: likelihood-ratio test on the breach COUNT. Too many breaches
  means risk is understated; too few means capital is wasted -- both fail.
  With x breaches in n days at expected rate p = 1-c:
      LR = -2 ln[ L(p) / L(x/n) ]  ~  chi2(1) under H0
- christoffersen_independence: are breaches CLUSTERED? Counts the four
  transition types (calm->calm, calm->breach, breach->calm,
  breach->breach) and asks whether a breach today makes one tomorrow more
  likely. A model can pass Kupiec on the decade average while missing
  every single crisis week; clustering is how that failure shows up.
  Returns None when there are fewer than 2 breaches -- independence of
  events that barely happened is unknowable, and None is honest where a
  fabricated p-value is not.
- conditional_coverage: Christoffersen's joint test, LR_pof + LR_ind ~
  chi2(2). The one-number summary: right rate AND right timing.
- traffic_light: Basel's zones, generalized. Instead of hardcoding the
  250-day/99% table, the zone comes from the binomial CDF of the breach
  count: green while P(X <= x) < 0.95, yellow while < 0.9999, red beyond.
  On (250, 99%) this reproduces the official Basel table exactly (green
  through 4, yellow 5-9, red at 10) -- a test pins that.

validate_var() is the runner: rolling_var() from risk/var.py produces the
no-lookahead breach history, this module scores it, and everything lands
in a ValidationResult for the renderer. Histories are computed on demand;
a persistent var_runs ledger for LIVE runs is deferred to the watch/
monitoring milestone (v0.6+).

Numerical care: likelihoods use scipy's xlogy (x*log(y) with xlogy(0,0)=0)
so zero counts -- no breaches, no breach-pairs -- produce exact zeros
instead of NaN from 0*log(0).
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import binom, chi2

from ballast.risk.var import rolling_var

__all__ = [
    "kupiec_pof",
    "christoffersen_independence",
    "conditional_coverage",
    "traffic_light",
    "ValidationResult",
    "validate_var",
]

# Significance level used for the pass/fail verdicts in ValidationResult.
ALPHA = 0.05


def _kupiec_lr(breaches: int, observations: int, confidence: float) -> float:
    """The Kupiec likelihood ratio statistic (shared with conditional_coverage)."""
    if observations <= 0:
        raise ValueError(f"observations must be positive, got {observations}")
    if not 0 <= breaches <= observations:
        raise ValueError(f"breaches must be in [0, {observations}], got {breaches}")
    if not 0.5 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0.5, 1), got {confidence}")
    p = 1.0 - confidence  # expected breach probability under H0
    pi_hat = breaches / observations  # observed breach probability
    log_h0 = xlogy(observations - breaches, 1.0 - p) + xlogy(breaches, p)
    log_h1 = xlogy(observations - breaches, 1.0 - pi_hat) + xlogy(breaches, pi_hat)
    return -2.0 * (log_h0 - log_h1)


def kupiec_pof(breaches: int, observations: int, confidence: float) -> float:
    """Kupiec proportion-of-failures test. Returns the p-value.

    Small p = the breach COUNT is inconsistent with the claimed confidence
    (in either direction: a 99% VaR with zero breaches in 4 years is
    over-conservative, and this test says so).
    """
    return float(chi2.sf(_kupiec_lr(breaches, observations, confidence), df=1))


def _independence_lr(breach_series: pd.Series | np.ndarray) -> float | None:
    """Christoffersen independence LR statistic; None if unassessable."""
    b = np.asarray(breach_series, dtype=bool).astype(int)
    if len(b) < 3:
        return None
    if b.sum() < 2:
        # One breach has no pair to cluster with; a p-value here would be
        # theater. The caller renders None as "too few breaches to assess".
        return None
    prev, curr = b[:-1], b[1:]
    n00 = int(((prev == 0) & (curr == 0)).sum())
    n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum())
    n11 = int(((prev == 1) & (curr == 1)).sum())

    # H0: breach probability is the same after calm and after breach days.
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    # H1: separate probabilities depending on yesterday's state.
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0

    log_h0 = xlogy(n01 + n11, pi) + xlogy(n00 + n10, 1.0 - pi)
    log_h1 = xlogy(n00, 1.0 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1.0 - pi11) + xlogy(n11, pi11)
    return -2.0 * (log_h0 - log_h1)


def christoffersen_independence(breach_series: pd.Series | np.ndarray) -> float | None:
    """Are breaches clustered in time? Returns the p-value, or None.

    Small p = a breach today predicts a breach tomorrow: the model is slow
    to react to regime changes even if its long-run count looks fine.
    """
    lr = _independence_lr(breach_series)
    return None if lr is None else float(chi2.sf(lr, df=1))


def conditional_coverage(breach_series: pd.Series | np.ndarray, confidence: float) -> float | None:
    """Christoffersen's joint test: right breach rate AND independent timing.

    LR_cc = LR_pof + LR_ind ~ chi2(2). This is the single number that
    catches the model that averages out right but fails every crisis.
    """
    b = np.asarray(breach_series, dtype=bool)
    lr_ind = _independence_lr(b)
    if lr_ind is None:
        return None
    lr_pof = _kupiec_lr(int(b.sum()), len(b), confidence)
    return float(chi2.sf(lr_pof + lr_ind, df=2))


def traffic_light(breaches: int, observations: int, confidence: float) -> str:
    """Basel-style zone from the binomial CDF of the breach count.

    green:  P(X <= x) <  0.95    (count consistent with the model)
    yellow: P(X <= x) <  0.9999  (questionable; regulators add capital)
    red:    otherwise            (model rejected)

    On the canonical (250 days, 99%) setup these cutoffs reproduce the
    official Basel table exactly: green through 4 breaches, yellow 5-9,
    red at 10+.
    """
    if observations <= 0:
        raise ValueError(f"observations must be positive, got {observations}")
    cumulative = float(binom.cdf(breaches, observations, 1.0 - confidence))
    if cumulative < 0.95:
        return "green"
    if cumulative < 0.9999:
        return "yellow"
    return "red"


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Everything the validation renderer needs."""

    method: str
    confidence: float
    window: int
    start: Any  # first/last dates of the scored period
    end: Any
    n_obs: int
    n_breaches: int
    expected_breaches: float
    kupiec_p: float
    kupiec_pass: bool
    independence_p: float | None  # None = too few breaches to assess
    independence_pass: bool | None
    cc_p: float | None
    cc_pass: bool | None
    zone: str  # green / yellow / red
    worst_date: Any | None  # None when there were no breaches
    worst_loss: float | None
    worst_ratio: float | None  # loss / VaR on the worst breach day


def validate_var(
    returns: pd.Series,
    method: str = "historical",
    confidence: float = 0.99,
    window: int = 750,
    lam: float = 0.94,
) -> ValidationResult:
    """Backtest one VaR method on one return series. Pure computation.

    rolling_var supplies the no-lookahead breach history; the coverage
    tests above score it. Verdicts use ALPHA=0.05.
    """
    rolls = rolling_var(returns, method=method, confidence=confidence, window=window, lam=lam)
    breaches = rolls["breach"]
    n_obs = len(rolls)
    n_breaches = int(breaches.sum())

    kupiec_p = kupiec_pof(n_breaches, n_obs, confidence)
    independence_p = christoffersen_independence(breaches)
    cc_p = conditional_coverage(breaches, confidence)

    worst_date = worst_loss = worst_ratio = None
    if n_breaches > 0:
        breached = rolls[breaches]
        # "Worst" = deepest relative to what the model promised that day.
        ratios = -breached["realized"] / breached["var"]
        worst_date = ratios.idxmax()
        worst_loss = float(-breached.loc[worst_date, "realized"])
        worst_ratio = float(ratios.loc[worst_date])

    return ValidationResult(
        method=method,
        confidence=confidence,
        window=window,
        start=rolls.index[0],
        end=rolls.index[-1],
        n_obs=n_obs,
        n_breaches=n_breaches,
        expected_breaches=n_obs * (1.0 - confidence),
        kupiec_p=kupiec_p,
        kupiec_pass=kupiec_p >= ALPHA,
        independence_p=independence_p,
        independence_pass=None if independence_p is None else independence_p >= ALPHA,
        cc_p=cc_p,
        cc_pass=None if cc_p is None else cc_p >= ALPHA,
        zone=traffic_light(n_breaches, n_obs, confidence),
        worst_date=worst_date,
        worst_loss=worst_loss,
        worst_ratio=worst_ratio,
    )
