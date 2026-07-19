"""Portfolio risk -> factor / specific split. v0.4.0 -- implemented.

Notes
-----
What this file does: the headline feature. Takes portfolio weights plus
the fitted factor model (exposures B, factor covariance F, specific
variances D) and splits total variance

    w' (B F B' + D) w

into named factor bets vs stock-specific risk, with per-position
contributions. This is the calculation behind "you think you own 14
stocks; you actually own one big momentum bet."

The pieces, and the algebra that makes them honest:
- Portfolio factor exposure x = B' w, where B gets a column of ones for
  the market factor (every stock has market exposure 1 by construction of
  the regression intercept). The market entry of x equals the invested
  fraction: 80% invested reads as 0.8 market exposure, as it should.
- Per-factor SIGNED contribution: c_k = x_k * (F x)_k. These sum to the
  factor variance x'Fx exactly. A negative c_k is a hedge -- a short
  value tilt offsetting long-value covariance shows up negative, not
  hidden inside an absolute value.
- Specific variance: sum of w_i^2 D_i (residuals are uncorrelated across
  names by model assumption).
- Per-position contribution: w_i * (Sigma w)_i with Sigma = BFB' + D;
  sums to total variance exactly. Shares of VARIANCE are what get quoted
  as "31% of risk" -- variance is the thing that adds up.
- Effective number of bets: 1 / sum(share_i^2), the Herfindahl inverse of
  the position shares. With any negative share (short hedges) the measure
  loses its meaning and is reported as None rather than nonsense.
- Display vols are signed square roots of contributions: sqrt preserves
  interpretability ("momentum: 5.1%") but squares don't add -- the
  renderer says so in a caption; the dataclass carries variances too.

Annualization happens HERE, exactly once, via model.periods_per_year.
The factor model arrives in per-period units and never leaves them.

Strictness: a portfolio symbol missing exposures or specific variance is
an ERROR naming the symbol, not a silent hole. A decomposition with a
hole misallocates every remaining number; refusing is the only honest
output. (Fix: widen the model universe or lengthen the panel.)
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ballast.factors.regression import FactorModel

__all__ = ["DecompositionError", "DecompositionResult", "decompose_portfolio"]


class DecompositionError(ValueError):
    """Raised when the model cannot cover the portfolio."""


@dataclass(frozen=True, slots=True)
class DecompositionResult:
    """Everything the decomposition renderer needs. All vols annualized."""

    name: str
    as_of: Any  # the exposure date the model was built for
    total_vol: float
    factor_vol: float
    specific_vol: float
    factor_share: float  # of variance
    specific_share: float
    # factor -> (signed annualized variance contribution, signed "vol" = sign*sqrt|.|)
    factor_contributions: pd.DataFrame  # index: factors; columns: variance, vol, share
    portfolio_exposures: pd.Series  # x = B'w, per factor
    # per symbol: weight, variance contribution, share of total variance
    position_contributions: pd.DataFrame
    effective_bets: float | None  # None when short hedges break the measure
    n_positions: int


def decompose_portfolio(
    weights: pd.Series,
    exposures: pd.DataFrame,
    model: FactorModel,
    name: str = "portfolio",
    as_of: Any = None,
) -> DecompositionResult:
    """Split w'(BFB'+D)w into named parts. Pure math, no I/O.

    weights: asset weights (cash excluded -- cash has no factor risk; the
    market exposure coming out below reflects the invested fraction).
    exposures: symbols x style factors, the cross-section at the as-of date.
    model: the fitted FactorModel (factor_cov must cover market + styles).
    """
    if weights.empty:
        raise DecompositionError("no positions to decompose")
    symbols = list(weights.index)

    # --- strict coverage checks: name every hole, fix upstream -----------
    missing_rows = sorted(set(symbols) - set(exposures.index))
    if missing_rows:
        raise DecompositionError(f"no exposures for symbol(s) {missing_rows}")
    b_style = exposures.loc[symbols]
    holes = sorted(b_style.index[b_style.isna().any(axis=1)])
    if holes:
        raise DecompositionError(
            f"exposures contain NaN for symbol(s) {holes} "
            "(missing fundamentals or too little price history)"
        )
    spec = model.specific_variance.reindex(symbols)
    bad_spec = sorted(spec.index[~np.isfinite(spec)])
    if bad_spec:
        raise DecompositionError(
            f"no specific variance for symbol(s) {bad_spec}; "
            "lengthen the estimation panel or widen the universe"
        )

    factor_names = ["market", *b_style.columns]
    cov = model.factor_cov
    if list(cov.index) != factor_names:
        raise DecompositionError(
            f"factor covariance covers {list(cov.index)}, exposures imply {factor_names}"
        )

    # --- the algebra (per-period units until the very end) ---------------
    w = weights.to_numpy(dtype=float)
    b_full = np.column_stack([np.ones(len(symbols)), b_style.to_numpy(dtype=float)])
    f = cov.to_numpy(dtype=float)
    d = spec.to_numpy(dtype=float)

    x = b_full.T @ w  # portfolio factor exposures
    fx = f @ x
    factor_var_parts = x * fx  # signed; sums to x'Fx exactly
    factor_var = float(x @ fx)
    specific_parts = (w**2) * d
    specific_var = float(specific_parts.sum())
    total_var = factor_var + specific_var
    if total_var <= 0:
        raise DecompositionError("total variance is zero; nothing to decompose")

    # Per-position: w_i * (Sigma w)_i, Sigma = BFB' + D.
    sigma_w = b_full @ (f @ (b_full.T @ w)) + d * w
    position_parts = w * sigma_w
    # Invariant, not input validation: if these disagree the algebra above
    # has a bug. (Same pattern as resolve_weights.)
    assert np.isclose(position_parts.sum(), total_var, rtol=1e-9)

    ppy = model.periods_per_year  # annualize once, here

    shares = position_parts / total_var
    effective_bets = None if (shares < 0).any() else float(1.0 / (shares**2).sum())

    factor_table = pd.DataFrame(
        {
            "variance": factor_var_parts * ppy,
            "vol": np.sign(factor_var_parts) * np.sqrt(np.abs(factor_var_parts) * ppy),
            "share": factor_var_parts / total_var,
        },
        index=factor_names,
    )
    position_table = pd.DataFrame(
        {"weight": w, "variance": position_parts * ppy, "share": shares}, index=symbols
    ).sort_values("share", ascending=False)

    return DecompositionResult(
        name=name,
        as_of=as_of,
        total_vol=float(np.sqrt(total_var * ppy)),
        factor_vol=float(np.sqrt(factor_var * ppy)),
        specific_vol=float(np.sqrt(specific_var * ppy)),
        factor_share=factor_var / total_var,
        specific_share=specific_var / total_var,
        factor_contributions=factor_table,
        portfolio_exposures=pd.Series(x, index=factor_names),
        position_contributions=position_table,
        effective_bets=effective_bets,
        n_positions=len(symbols),
    )
