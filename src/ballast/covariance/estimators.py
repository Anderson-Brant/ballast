"""Covariance estimators. v0.2.0 -- implemented.

Notes
-----
What this file does: every way Ballast can turn a returns matrix (dates x
symbols, the load_returns output) into a covariance matrix, behind one
shared signature: DataFrame in, symbol-labeled DataFrame out, DAILY units.
Annualize only at the display layer (helper provided).

The estimators and why each exists:
- sample_cov: the textbook estimator and the baseline every alternative
  must beat in the harness. Honest but noisy: with N assets it estimates
  N(N+1)/2 numbers, and when T isn't much bigger than N the noise
  dominates (eigenvalues spread out, min-variance weights explode).
- ewma_cov: RiskMetrics exponential weighting, lambda=0.94 daily. Recent
  days count more, so it adapts to volatility regimes fast. Uses the
  zero-mean convention (standard for daily data: the true daily mean is
  ~0.03% and estimating it adds more noise than assuming it away).
- ledoit_wolf_cov: shrink the sample matrix toward a scaled identity by a
  data-driven amount (Ledoit & Wolf 2004, "A well-conditioned estimator
  for large-dimensional covariance matrices"). The shrinkage intensity is
  estimated, not tuned; implemented from scratch below and cross-checked
  against scikit-learn in the tests.
- oas_cov: same target, different intensity formula (Chen, Wiesel,
  Eldar & Hero 2010). Derived under Gaussian assumptions; usually shrinks
  a bit harder than LW at small T.
- (v0.4.0) factor-implied covariance, assembled from the factor model.

Design rules:
- Pure functions, no state.
- Every output passes through _ensure_psd: symmetrize and clip negative
  eigenvalue dust, so downstream solvers never see a broken matrix.
- Input NaNs are rejected, not skipped -- load_returns' inner alignment
  means NaNs here are a bug upstream, and pairwise-deletion covariances
  aren't even guaranteed PSD.
- Which estimator becomes the default is decided by the v0.2.0 horse race
  in harness.py, not by preference.
"""

import numpy as np
import pandas as pd

__all__ = [
    "CovarianceError",
    "sample_cov",
    "ewma_cov",
    "ledoit_wolf_cov",
    "oas_cov",
    "annualize",
]

TRADING_DAYS = 252


class CovarianceError(ValueError):
    """Raised for inputs no estimator can do anything meaningful with."""


def _validate(returns: pd.DataFrame) -> pd.DataFrame:
    """Shared input checks; returns the frame sorted by date."""
    if not isinstance(returns, pd.DataFrame):
        raise CovarianceError(f"returns must be a DataFrame, got {type(returns).__name__}")
    if returns.shape[0] < 2:
        raise CovarianceError(f"need at least 2 rows of returns, got {returns.shape[0]}")
    if returns.shape[1] < 1:
        raise CovarianceError("returns frame has no columns")
    values = returns.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        # NaN/inf silently propagate through matrix math and come out the
        # other side as a garbage covariance; refuse at the door instead.
        raise CovarianceError("returns contain NaN or inf; fix alignment upstream")
    # EWMA weights depend on row order, so make order deterministic for all.
    return returns.sort_index()


def _ensure_psd(matrix: np.ndarray) -> np.ndarray:
    """Symmetrize and clip negative eigenvalue dust.

    All four estimators are PSD by construction; what this guards against
    is floating-point asymmetry (A[i,j] != A[j,i] in the 16th digit) and
    eigenvalues like -1e-17, either of which can make a downstream solver
    (cvxpy at v0.5.0) reject the matrix.
    """
    sym = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(sym)
    if eigenvalues.min() >= 0.0:
        return sym
    clipped = np.clip(eigenvalues, 0.0, None)
    return eigenvectors @ np.diag(clipped) @ eigenvectors.T


def _as_frame(matrix: np.ndarray, like: pd.DataFrame) -> pd.DataFrame:
    """Numpy matrix -> DataFrame labeled with the input's symbols."""
    return pd.DataFrame(_ensure_psd(matrix), index=like.columns, columns=like.columns)


def sample_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """Textbook sample covariance (demeaned, ddof=1). The baseline."""
    clean = _validate(returns)
    return _as_frame(clean.cov(ddof=1).to_numpy(), clean)


def ewma_cov(returns: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    """RiskMetrics EWMA covariance: recent observations weighted heaviest.

    S = sum_i w_i * r_i r_i'  with  w_i proportional to lam^(age_i), and
    weights normalized to sum to 1 (the finite-sample correction; without
    it, short histories understate variance by a factor of 1 - lam^T).
    Zero-mean convention: see module notes.
    """
    if not 0.0 < lam < 1.0:
        raise CovarianceError(f"lambda must be in (0, 1), got {lam}")
    clean = _validate(returns)
    values = clean.to_numpy(dtype=float)
    n_rows = len(values)

    # Newest row gets lam^0, oldest gets lam^(T-1). Index is sorted
    # ascending by _validate, so age decreases down the array.
    weights = lam ** np.arange(n_rows - 1, -1, -1)
    weights = weights * (1.0 - lam)
    weights = weights / weights.sum()  # exact normalization

    # Weighted sum of outer products, done as one matrix product:
    # (X * w)' X == sum_i w_i * x_i x_i'  (each row scaled by its weight).
    weighted = values * weights[:, None]
    return _as_frame(weighted.T @ values, clean)


def ledoit_wolf_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """Ledoit-Wolf 2004 shrinkage toward a scaled identity.

    shrunk = (1 - delta) * S + delta * mu * I, where mu preserves the
    average variance and delta (in [0, 1]) is chosen to minimize expected
    error: shrink harder when the sample matrix is noisy (small T, big N),
    barely at all when there's plenty of data. The formulas below follow
    the paper (and match scikit-learn's implementation, which the tests
    verify when sklearn is installed).
    """
    clean = _validate(returns)
    x = clean.to_numpy(dtype=float)
    n_rows, n_assets = x.shape
    x = x - x.mean(axis=0)  # the paper works with demeaned data, ddof=0

    emp = x.T @ x / n_rows
    mu = np.trace(emp) / n_assets

    # delta_hat: distance between the sample matrix and the target --
    # how much there is to shrink across.
    identity = np.eye(n_assets)
    delta_hat = ((emp - mu * identity) ** 2).sum() / n_assets

    # beta_hat: estimation error in the sample matrix itself -- variance
    # of the per-observation outer products around their mean.
    x2 = x**2
    beta_hat = ((x2.T @ x2 / n_rows).sum() - (emp**2).sum()) / (n_assets * n_rows)
    beta = min(beta_hat, delta_hat)  # error can't exceed total distance

    shrinkage = 0.0 if delta_hat == 0.0 else beta / delta_hat
    shrunk = (1.0 - shrinkage) * emp + shrinkage * mu * identity
    return _as_frame(shrunk, clean)


def oas_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """Oracle Approximating Shrinkage (Chen et al. 2010), same target as LW.

    Closed-form intensity derived under Gaussian returns:

        rho = [(1 - 2/N) tr(S^2) + tr(S)^2]
              / [(T + 1 - 2/N) (tr(S^2) - tr(S)^2 / N)]

    capped at 1. Included so the harness can say which intensity rule
    actually wins out of sample, rather than picking one on reputation.
    """
    clean = _validate(returns)
    x = clean.to_numpy(dtype=float)
    n_rows, n_assets = x.shape
    x = x - x.mean(axis=0)

    emp = x.T @ x / n_rows
    mu = np.trace(emp) / n_assets
    tr_s2 = (emp**2).sum()  # tr(S^2) for symmetric S
    tr_s_sq = np.trace(emp) ** 2

    numerator = (1.0 - 2.0 / n_assets) * tr_s2 + tr_s_sq
    denominator = (n_rows + 1.0 - 2.0 / n_assets) * (tr_s2 - tr_s_sq / n_assets)
    rho = 1.0 if denominator <= 0.0 else min(numerator / denominator, 1.0)

    shrunk = (1.0 - rho) * emp + rho * mu * np.eye(n_assets)
    return _as_frame(shrunk, clean)


def annualize(cov: pd.DataFrame, periods_per_year: int = TRADING_DAYS) -> pd.DataFrame:
    """Daily covariance -> annualized (display layer only).

    Variances scale linearly with time under independence, so the whole
    matrix multiplies by 252. Vols (sqrt of diagonal) then scale by sqrt(252).
    """
    return cov * periods_per_year
