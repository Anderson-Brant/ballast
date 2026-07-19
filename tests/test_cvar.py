"""Tests for optimize/cvar.py.

Notes
-----
Two tests carry the weight. The elliptical anchor: on Gaussian scenarios,
min-CVaR must approximately reproduce the GMV weights (for elliptical
distributions CVaR is a monotone function of variance -- same optimum).
The skew-aversion test: two assets with MATCHED variance, one symmetric
and one crash-prone; MVO cannot tell them apart, CVaR must dodge the
crashes. That difference is the entire reason this optimizer exists.
"""

import numpy as np
import pandas as pd
import pytest

from ballast.covariance.estimators import sample_cov
from ballast.covariance.harness import min_variance_weights
from ballast.optimize.cvar import cvar_weights
from ballast.optimize.mvo import OptimizationError, mvo_weights

cvxpy = pytest.importorskip("cvxpy")  # [opt] extra; skip cleanly where absent


def gaussian_scenarios(n_rows=4000, seed=17) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cov = np.array(
        [
            [4e-4, 1.0e-4, 0.0],
            [1.0e-4, 2.25e-4, 0.5e-4],
            [0.0, 0.5e-4, 1e-4],
        ]
    )
    values = rng.multivariate_normal(np.zeros(3), cov, size=n_rows)
    return pd.DataFrame(values, columns=["A", "B", "C"])


def crash_vs_smooth_scenarios() -> pd.DataFrame:
    """Two assets, (approximately) equal variance, opposite personalities.

    SMOOTH alternates +/-2% (variance 4e-4 exactly). CRASH gains +0.4% on
    24 of 25 days and drops -9.8% on the 25th: mean ~0, variance ~4e-4 --
    the same to any variance-based eye, but the tail is 5x deeper.
    """
    smooth = [0.02 if i % 2 == 0 else -0.02 for i in range(500)]
    crash = [-0.098 if i % 25 == 24 else 0.004 for i in range(500)]
    return pd.DataFrame({"SMOOTH": smooth, "CRASH": crash})


# ----------------------------------------------------------------- anchors


def test_gaussian_scenarios_reproduce_gmv():
    # Elliptical world: min-CVaR == min-variance. Sampling noise on 4000
    # scenarios earns a loose tolerance, but the portfolios must be close.
    scenarios = gaussian_scenarios()
    w_cvar = cvar_weights(scenarios, confidence=0.95)
    w_gmv = min_variance_weights(sample_cov(scenarios)).clip(lower=0)
    w_gmv /= w_gmv.sum()  # comparable long-only normalization
    for s in scenarios.columns:
        assert w_cvar[s] == pytest.approx(w_gmv[s], abs=0.08)


def test_skew_aversion_is_the_point():
    scenarios = crash_vs_smooth_scenarios()
    # Confirm the trap is set: variances match to ~1%, so MVO splits ~50/50.
    cov = sample_cov(scenarios)
    assert cov.loc["SMOOTH", "SMOOTH"] == pytest.approx(cov.loc["CRASH", "CRASH"], rel=0.02)
    w_mvo = mvo_weights(cov)
    assert w_mvo["SMOOTH"] == pytest.approx(0.5, abs=0.05)  # variance is blind here
    # CVaR sees the tail and walks away from it.
    w_cvar = cvar_weights(scenarios, confidence=0.95)
    assert w_cvar["SMOOTH"] > 0.65
    assert w_cvar.sum() == pytest.approx(1.0)


# -------------------------------------------------------------- properties


def test_long_only_and_unit_sum():
    w = cvar_weights(gaussian_scenarios(n_rows=1000), confidence=0.9)
    assert (w >= 0).all()
    assert w.sum() == pytest.approx(1.0)


def test_max_weight_binds():
    scenarios = gaussian_scenarios(n_rows=1000)
    w = cvar_weights(scenarios, confidence=0.9, max_weight=0.4)
    assert (w <= 0.4 + 1e-8).all()
    assert w.sum() == pytest.approx(1.0)


def test_deterministic():
    scenarios = gaussian_scenarios(n_rows=800)
    a = cvar_weights(scenarios, confidence=0.9)
    b = cvar_weights(scenarios, confidence=0.9)
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_shorts_allowed_when_asked():
    w = cvar_weights(gaussian_scenarios(n_rows=1000), confidence=0.9, long_only=False)
    assert w.sum() == pytest.approx(1.0)  # may or may not short; must sum to 1


# ------------------------------------------------------------------- guards


def test_too_few_tail_scenarios_refused():
    scenarios = gaussian_scenarios(n_rows=100)
    with pytest.raises(OptimizationError, match="tail observations"):
        cvar_weights(scenarios, confidence=0.99)  # 1 tail point per 100 rows


def test_bad_inputs_named():
    scenarios = gaussian_scenarios(n_rows=500)
    poisoned = scenarios.copy()
    poisoned.iloc[3, 1] = float("nan")
    with pytest.raises(OptimizationError, match="NaN"):
        cvar_weights(poisoned, confidence=0.9)
    with pytest.raises(OptimizationError, match="confidence"):
        cvar_weights(scenarios, confidence=1.2)
    with pytest.raises(OptimizationError, match="cannot sum to 1"):
        cvar_weights(scenarios, confidence=0.9, max_weight=0.2)  # 3 x 0.2 < 1
