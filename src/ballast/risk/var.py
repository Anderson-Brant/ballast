"""VaR and expected shortfall, four ways. v0.3.0 -- implemented.

Notes
-----
What this file does: answers "how much can this portfolio lose at horizon
h with confidence c" by four methods sharing one output type (VaREstimate),
so the validation suite can compare them fairly. Also provides
rolling_var(): the day-by-day, no-lookahead VaR series that
validate/coverage.py backtests.

Conventions, fixed here so nothing downstream argues about signs:
- VaR and ES are reported as POSITIVE loss fractions: var=0.021 means "a
  loss worse than 2.1% happens with probability 1-c". A negative VaR is
  possible (it means even the tail quantile is a gain) and is not clipped.
- ES (expected shortfall) is the average loss GIVEN you're beyond VaR.
  It is always computed alongside VaR: when tails are fat, VaR is just the
  doorway and ES is the room.
- Multi-day horizons scale by sqrt(h) on the vol/quantile term (and
  linearly on the mean). For the non-parametric methods this is an
  approximation that assumes i.i.d. returns; it is the standard shortcut
  and it is documented here rather than hidden.

The four methods and what each believes:
- parametric normal: returns are Gaussian. Fast, smooth, and famously
  understates tails -- included partly to fail validation honestly.
- Cornish-Fisher: normal, corrected for the sample's skew and excess
  kurtosis by expanding the quantile (z -> z_cf). A middle ground.
- historical: the empirical quantile of the window. No distribution
  assumed, but treats a calm 2017 day and a violent 2020 day as equally
  representative.
- filtered historical (FHS): standardize each past return by ITS day's
  EWMA vol (z_t = r_t / sigma_t, sigma known before the day), then
  rescale those z's by TODAY's vol. History supplies the tail shape,
  today supplies the scale. sigma_t is seeded from the first `burn`
  observations, and those observations are excluded from the quantile --
  no full-sample seeding, which would leak future vol into the past.
- monte carlo: multivariate normal draws from a chosen covariance matrix
  and portfolio weights. The only method here that sees the covariance
  structure directly; seeded, so runs are reproducible.

Guards: estimating a 99% quantile needs at least 1/(1-c) observations to
have ever SEEN the tail; fewer is an error, not a warning.

Validation computes breach histories on demand from rolling_var; a
persistent var_runs ledger (for accumulating LIVE runs over time) is
deferred to the watch/monitoring milestone (v0.6+). This module stays
pure math.
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

__all__ = [
    "VaRError",
    "VaREstimate",
    "parametric_var",
    "historical_var",
    "filtered_historical_var",
    "monte_carlo_var",
    "rolling_var",
    "ROLLING_METHODS",
]


class VaRError(ValueError):
    """Raised for inputs no VaR method can answer meaningfully."""


@dataclass(frozen=True, slots=True)
class VaREstimate:
    """One method's answer. var/es are positive loss fractions."""

    method: str
    confidence: float
    horizon_days: int
    var: float
    es: float


def _check_common(returns: pd.Series, confidence: float, horizon_days: int) -> np.ndarray:
    if not isinstance(returns, pd.Series):
        raise VaRError(f"returns must be a Series, got {type(returns).__name__}")
    if not 0.5 < confidence < 1.0:
        raise VaRError(f"confidence must be in (0.5, 1), got {confidence}")
    if horizon_days < 1:
        raise VaRError(f"horizon_days must be >= 1, got {horizon_days}")
    values = returns.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise VaRError("returns contain NaN or inf")
    # To estimate the (1-c) tail you must have been able to observe it:
    # 99% needs 100+ points, 95% needs 20+. Below that the "estimate" is
    # an extrapolation wearing a quantile's clothes.
    needed = math.ceil(1.0 / (1.0 - confidence))
    if len(values) < needed:
        raise VaRError(
            f"need at least {needed} observations for {confidence:.0%} VaR, got {len(values)}"
        )
    return values


def _cornish_fisher_z(z: float, skew: float, excess_kurtosis: float) -> float:
    """The Cornish-Fisher quantile expansion. z_cf(z, 0, 0) == z exactly."""
    return (
        z
        + (z**2 - 1.0) * skew / 6.0
        + (z**3 - 3.0 * z) * excess_kurtosis / 24.0
        - (2.0 * z**3 - 5.0 * z) * skew**2 / 36.0
    )


def parametric_var(
    returns: pd.Series,
    confidence: float = 0.99,
    horizon_days: int = 1,
    cornish_fisher: bool = False,
) -> VaREstimate:
    """Gaussian VaR/ES from sample mean and std; optional CF tail correction."""
    values = _check_common(returns, confidence, horizon_days)
    mu = float(values.mean())
    sigma = float(values.std(ddof=1))
    scale = math.sqrt(horizon_days)
    z = float(norm.ppf(1.0 - confidence))  # negative, e.g. -2.326 at 99%

    if not cornish_fisher:
        # Closed forms. ES for a normal: sigma * pdf(z) / (1 - c) - mu.
        var = -(mu * horizon_days + z * sigma * scale)
        es = sigma * scale * float(norm.pdf(z)) / (1.0 - confidence) - mu * horizon_days
        return VaREstimate("parametric", confidence, horizon_days, var, es)

    skew = float(pd.Series(values).skew())
    kurt = float(pd.Series(values).kurt())  # pandas kurt() is EXCESS kurtosis
    z_cf = _cornish_fisher_z(z, skew, kurt)
    var = -(mu * horizon_days + z_cf * sigma * scale)
    # No clean closed form for CF-ES: average the CF quantile over the tail
    # numerically (midpoint rule on the tail probabilities). With skew and
    # kurtosis at 0 this reproduces the normal ES -- a test pins that.
    tail_probs = (1.0 - confidence) * (np.arange(200) + 0.5) / 200.0
    tail_z = np.array([_cornish_fisher_z(float(norm.ppf(p)), skew, kurt) for p in tail_probs])
    es = -(mu * horizon_days + tail_z.mean() * sigma * scale)
    return VaREstimate("cornish_fisher", confidence, horizon_days, var, es)


def historical_var(
    returns: pd.Series, confidence: float = 0.99, horizon_days: int = 1
) -> VaREstimate:
    """Empirical quantile of the observed returns; ES is the tail average."""
    values = _check_common(returns, confidence, horizon_days)
    scale = math.sqrt(horizon_days)
    q = float(np.quantile(values, 1.0 - confidence))
    tail = values[values <= q]
    return VaREstimate(
        "historical", confidence, horizon_days, var=-q * scale, es=-float(tail.mean()) * scale
    )


def _ewma_sigma(values: np.ndarray, lam: float, burn: int) -> tuple[np.ndarray, float]:
    """Per-day EWMA vol, aligned so sigma[t] uses data through t-1 only.

    Seeded from the first `burn` observations; entries before `burn` are
    NaN and must not be used. Callers exclude the burn-in from quantiles.
    """
    n = len(values)
    sigma2 = np.full(n, np.nan)
    seed = float(np.mean(values[:burn] ** 2))  # zero-mean convention, like ewma_cov
    if seed == 0.0:
        raise VaRError("burn-in window has zero variance; cannot seed EWMA vol")
    running = seed
    for t in range(burn, n):
        sigma2[t] = running  # uses returns 0..t-1 only
        running = lam * running + (1.0 - lam) * values[t] ** 2
    return np.sqrt(sigma2), math.sqrt(running)  # per-day sigma, next-day forecast


def filtered_historical_var(
    returns: pd.Series,
    confidence: float = 0.99,
    horizon_days: int = 1,
    lam: float = 0.94,
    burn: int = 30,
) -> VaREstimate:
    """FHS: devolatized history, revolatized to today. See module notes."""
    if not 0.0 < lam < 1.0:
        raise VaRError(f"lambda must be in (0, 1), got {lam}")
    values = _check_common(returns, confidence, horizon_days)
    if len(values) <= burn + math.ceil(1.0 / (1.0 - confidence)):
        raise VaRError(
            f"need more than burn+{math.ceil(1 / (1 - confidence))} observations after "
            f"the {burn}-day EWMA seed"
        )
    sigma, sigma_next = _ewma_sigma(values, lam, burn)
    z = values[burn:] / sigma[burn:]  # each return in units of ITS OWN day's vol
    scale = math.sqrt(horizon_days)
    q = float(np.quantile(z, 1.0 - confidence))
    tail = z[z <= q]
    return VaREstimate(
        "filtered_historical",
        confidence,
        horizon_days,
        var=-q * sigma_next * scale,
        es=-float(tail.mean()) * sigma_next * scale,
    )


def monte_carlo_var(
    weights: pd.Series,
    cov: pd.DataFrame,
    confidence: float = 0.99,
    horizon_days: int = 1,
    n_sims: int = 100_000,
    seed: int = 0,
) -> VaREstimate:
    """Simulate portfolio returns from a covariance matrix (zero-mean normal).

    The only method that sees the full covariance structure instead of a
    single return series -- which estimator's matrix to feed it is exactly
    what the v0.2.0 harness decided. Seeded: same inputs, same answer.
    """
    if not 0.5 < confidence < 1.0:
        raise VaRError(f"confidence must be in (0.5, 1), got {confidence}")
    if horizon_days < 1:
        raise VaRError(f"horizon_days must be >= 1, got {horizon_days}")
    if n_sims < math.ceil(1.0 / (1.0 - confidence)) * 10:
        raise VaRError(f"n_sims={n_sims} is too few to resolve the {confidence:.0%} tail")
    missing = sorted(set(cov.columns) - set(weights.index))
    if missing:
        raise VaRError(f"weights are missing symbol(s) {missing}")
    w = weights.reindex(cov.columns).to_numpy(dtype=float)
    if not np.isfinite(w).all():
        raise VaRError("weights contain NaN or inf")

    rng = np.random.default_rng(seed)
    # Variance scales linearly with horizon (i.i.d. assumption), so draw
    # straight from the h-day distribution rather than compounding paths.
    draws = rng.multivariate_normal(
        np.zeros(len(w)), cov.to_numpy(dtype=float) * horizon_days, size=n_sims
    )
    sims = draws @ w
    q = float(np.quantile(sims, 1.0 - confidence))
    tail = sims[sims <= q]
    return VaREstimate("monte_carlo", confidence, horizon_days, var=-q, es=-float(tail.mean()))


# Methods available to rolling_var. Monte Carlo is excluded: it needs the
# full covariance matrix, not a single portfolio return series.
ROLLING_METHODS = ("parametric", "historical", "filtered_historical")


def rolling_var(
    returns: pd.Series,
    method: str = "historical",
    confidence: float = 0.99,
    window: int = 750,
    lam: float = 0.94,
) -> pd.DataFrame:
    """Day-by-day 1-day VaR with no lookahead: the validation suite's input.

    Row t contains: the VaR estimated from data STRICTLY BEFORE t, the
    realized return of day t, and whether it breached (realized < -var).
    The .shift(1) calls below are the entire no-lookahead guarantee --
    a day never participates in its own VaR.
    """
    values = _check_common(returns, confidence, 1)
    clean = pd.Series(values, index=returns.sort_index().index)
    if len(clean) < window + 2:
        raise VaRError(f"need at least window+2={window + 2} rows, got {len(clean)}")
    if method not in ROLLING_METHODS:
        raise VaRError(f"method must be one of {ROLLING_METHODS}, got {method!r}")

    if method == "historical":
        q = clean.rolling(window).quantile(1.0 - confidence).shift(1)
        var = -q
    elif method == "parametric":
        z = float(norm.ppf(1.0 - confidence))
        mu = clean.rolling(window).mean().shift(1)
        sigma = clean.rolling(window).std(ddof=1).shift(1)
        var = -(mu + z * sigma)
    else:  # filtered_historical
        burn = min(30, window // 4)
        sigma, _ = _ewma_sigma(clean.to_numpy(), lam, burn)
        sigma = pd.Series(sigma, index=clean.index)
        z_series = clean / sigma  # sigma[t] uses through t-1; z_t is day t's surprise
        # Quantile of past standardized returns, rescaled by TODAY's sigma
        # (sigma[t] itself only knows data through t-1, so no shift needed on it).
        q = z_series.rolling(window).quantile(1.0 - confidence).shift(1)
        var = -q * sigma

    out = pd.DataFrame({"var": var, "realized": clean})
    out = out.dropna()
    out["breach"] = out["realized"] < -out["var"]
    return out
