"""Black-Litterman: equilibrium prior + views -> posterior returns. v0.5.0 -- implemented.

Notes
-----
What this file does: solves the garbage-in problem of MVO. Feeding raw
historical mean returns into an optimizer produces violent, unstable
portfolios because sample means are the noisiest estimate in finance.
Black-Litterman (1992) starts instead from the returns IMPLIED by market
weights -- the forecast the whole market is already making -- and tilts
away from it only where you hold explicit views, by an amount scaled to
your stated confidence.

The two halves:
- Reverse optimization: if the market portfolio w_mkt were mean-variance
  optimal with risk aversion delta, expected returns must be
      pi = delta * Sigma * w_mkt
  That's the equilibrium prior: hold no views, and MVO on pi hands back
  the market portfolio -- a beautifully self-consistent default.
- Views: each is a pick vector over assets, a target value, and a
  confidence in (0, 1]. "AAPL returns 10%" is {AAPL: 1} -> 0.10;
  "AAPL beats MSFT by 2%" is {AAPL: 1, MSFT: -1} -> 0.02. The posterior
      mu = pi + tau Sigma P' (P tau Sigma P' + Omega)^-1 (q - P pi)
  blends prior and views by their relative precisions. Omega (view
  uncertainty) comes from confidence: omega_k = (1/c - 1)(P tau Sigma P')_kk,
  so c=1 means certainty (the posterior HITS the view) and c=0.5 weighs
  the view equal to the prior.

The property that makes BL worth the formulas: views PROPAGATE through
correlations. A bullish view on AAPL raises the posterior for MSFT too,
in proportion to their covariance -- expressing "tech is cheap" through
one name doesn't accidentally make the optimizer short the rest of tech.
A test pins this spillover.

Units follow the covariance: feed an annualized Sigma, read annualized
returns. Posterior mu goes straight into mvo_weights(expected_returns=).

The two-project story: `ballast import sentinel-views` (v0.6.0) maps
Sentinel screen scores into this view vector. This file is where the two
projects actually meet.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ballast.optimize.mvo import OptimizationError

__all__ = [
    "View",
    "absolute_view",
    "relative_view",
    "implied_returns",
    "black_litterman_returns",
    "views_from_scores",
]


@dataclass(frozen=True, slots=True)
class View:
    """One opinion: a pick vector, its expected value, and a confidence."""

    assets: dict[str, float]  # symbol -> pick weight (e.g. {A: 1, B: -1})
    value: float  # expected return of the pick, in Sigma's units
    confidence: float  # (0, 1]; 1 = certain, 0.5 = equal to the prior


def absolute_view(symbol: str, expected_return: float, confidence: float) -> View:
    """ "<symbol> will return <x>": the simplest view."""
    return View({symbol: 1.0}, expected_return, confidence)


def relative_view(long: str, short: str, spread: float, confidence: float) -> View:
    """ "<long> beats <short> by <spread>": the classic relative view."""
    return View({long: 1.0, short: -1.0}, spread, confidence)


def _validate_market_weights(cov: pd.DataFrame, market_weights: pd.Series) -> np.ndarray:
    symbols = list(cov.columns)
    missing = sorted(set(symbols) - set(market_weights.index))
    if missing:
        raise OptimizationError(f"market_weights missing symbol(s) {missing}")
    w = market_weights.reindex(symbols).to_numpy(dtype=float)
    if not np.isfinite(w).all() or (w <= 0).any():
        raise OptimizationError("market_weights must be finite and strictly positive")
    return w / w.sum()  # only proportions matter for the equilibrium


def implied_returns(
    cov: pd.DataFrame, market_weights: pd.Series, risk_aversion: float = 2.5
) -> pd.Series:
    """Reverse optimization: the returns that make the market portfolio optimal."""
    if risk_aversion <= 0:
        raise OptimizationError(f"risk_aversion must be > 0, got {risk_aversion}")
    sigma = cov.to_numpy(dtype=float)
    if not np.isfinite(sigma).all():
        raise OptimizationError("covariance contains NaN or inf")
    w = _validate_market_weights(cov, market_weights)
    return pd.Series(risk_aversion * (sigma @ w), index=cov.columns)


def black_litterman_returns(
    cov: pd.DataFrame,
    market_weights: pd.Series,
    views: list[View],
    tau: float = 0.05,
    risk_aversion: float = 2.5,
) -> pd.Series:
    """Posterior expected returns: equilibrium prior tilted by the views.

    No views returns the prior unchanged -- BL degrades gracefully into
    "just hold the market", which is exactly the right default.
    """
    if tau <= 0:
        raise OptimizationError(f"tau must be > 0, got {tau}")
    symbols = list(cov.columns)
    prior = implied_returns(cov, market_weights, risk_aversion)
    if not views:
        return prior

    # Build P (picks) and q (targets) from the views, validating each.
    n, k = len(symbols), len(views)
    picks = np.zeros((k, n))
    targets = np.zeros(k)
    confidences = np.zeros(k)
    index_of = {s: i for i, s in enumerate(symbols)}
    for row, view in enumerate(views):
        if not view.assets:
            raise OptimizationError(f"view {row}: empty pick vector")
        unknown = sorted(set(view.assets) - set(symbols))
        if unknown:
            raise OptimizationError(f"view {row}: unknown symbol(s) {unknown}")
        if not np.isfinite(view.value):
            raise OptimizationError(f"view {row}: value must be finite")
        if not 0.0 < view.confidence <= 1.0:
            raise OptimizationError(f"view {row}: confidence must be in (0, 1]")
        for symbol, pick in view.assets.items():
            picks[row, index_of[symbol]] = pick
        targets[row] = view.value
        confidences[row] = view.confidence

    sigma = cov.to_numpy(dtype=float)
    tau_sigma = tau * sigma
    p_tau_p = picks @ tau_sigma @ picks.T  # k x k: prior uncertainty of the picks

    # Confidence -> Omega: certainty (c=1) zeroes the view's variance, and
    # c=0.5 makes it exactly as uncertain as the prior thinks the pick is.
    omega = np.diag((1.0 / confidences - 1.0) * np.diag(p_tau_p))

    # mu = pi + tau Sigma P' (P tau Sigma P' + Omega)^-1 (q - P pi)
    # solve() on the k x k system; k is small (a handful of views).
    surprise = targets - picks @ prior.to_numpy()
    adjustment = tau_sigma @ picks.T @ np.linalg.solve(p_tau_p + omega, surprise)
    return pd.Series(prior.to_numpy() + adjustment, index=symbols)


def views_from_scores(
    scores: pd.Series,
    cov: pd.DataFrame,
    market_weights: pd.Series,
    ic: float = 0.05,
    confidence: float = 0.3,
    risk_aversion: float = 2.5,
) -> list[View]:
    """Sentinel screen scores -> Black-Litterman views. The bridge's math half.

    The mapping is Grinold's rule, the standard translation of a ranking
    into expected returns:

        alpha_i = IC x sigma_i x z_i

    z_i is the score's cross-sectional z-score (only the RANKING carries
    information; the raw scale is discarded), sigma_i the asset's vol from
    the covariance, and IC the information coefficient -- the assumed
    correlation between the ranking and future returns. The default 0.05
    is deliberately humble: even good screens barely clear it, and the
    honest way to use this bridge is to treat IC as "how much do I trust
    Sentinel?", not as a dial to crank.

    Each scored symbol becomes an absolute view: prior + alpha, at the
    given confidence. Symbols in the covariance WITHOUT scores simply get
    no view -- Black-Litterman holds them at the prior, which is exactly
    the right treatment for "the screen said nothing about this one".

    Units follow the covariance: feed an annualized Sigma, get annual
    alphas -- same contract as the rest of this module.
    """
    if not 0.0 < ic <= 1.0:
        raise OptimizationError(f"ic must be in (0, 1], got {ic}")
    values = scores.to_numpy(dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise OptimizationError("scores must be >= 2 finite values")
    spread = float(values.std(ddof=0))
    if spread == 0.0:
        raise OptimizationError("all scores identical; there is no ranking to express")
    unknown = sorted(set(scores.index) - set(cov.columns))
    if unknown:
        raise OptimizationError(f"scored symbol(s) {unknown} not in the covariance")

    z = (values - values.mean()) / spread
    prior = implied_returns(cov, market_weights, risk_aversion)
    vols = np.sqrt(np.diag(cov.to_numpy(dtype=float)))
    vol_of = dict(zip(cov.columns, vols, strict=True))

    return [
        absolute_view(symbol, float(prior[symbol] + ic * vol_of[symbol] * z_i), confidence)
        for symbol, z_i in zip(scores.index, z, strict=True)
    ]
