"""Tests for covariance/estimators.py.

Notes
-----
Three layers of evidence, cheapest to strongest:
1. hand-computed cases (worked in comments -- trust the paper over the code)
2. properties every covariance must have (PSD, symmetry, shrinkage direction)
3. recovery: with 8000 draws from a KNOWN matrix, every estimator must land
   close to the truth. This is what the conftest synthetic fixture is for.
Plus a cross-check of the from-scratch Ledoit-Wolf against scikit-learn's,
which runs only where sklearn happens to be installed.
"""

import numpy as np
import pandas as pd
import pytest

from ballast.covariance.estimators import (
    CovarianceError,
    annualize,
    ewma_cov,
    ledoit_wolf_cov,
    oas_cov,
    sample_cov,
)

ALL_ESTIMATORS = [sample_cov, ewma_cov, ledoit_wolf_cov, oas_cov]


def frame(rows: list[list[float]], columns=("X", "Y")) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=list(columns), index=pd.bdate_range("2024-01-01", periods=len(rows))
    )


# ------------------------------------------------------------ hand examples


def test_sample_cov_hand_example():
    # x = [.01, -.01, .02], y = [.02, .00, .01]
    # means: .006667, .01; deviations x: [.003333, -.016667, .013333],
    # y: [.01, -.01, 0]. With ddof=1 (divide by 2):
    #   var(x) = (1.111e-5 + 2.778e-4 + 1.778e-4)/2 = 2.3333e-4
    #   var(y) = (1e-4 + 1e-4 + 0)/2 = 1.0e-4
    #   cov    = (3.333e-5 + 1.667e-4 + 0)/2 = 1.0e-4
    cov = sample_cov(frame([[0.01, 0.02], [-0.01, 0.00], [0.02, 0.01]]))
    assert cov.loc["X", "X"] == pytest.approx(2.3333e-4, rel=1e-3)
    assert cov.loc["Y", "Y"] == pytest.approx(1.0e-4, rel=1e-3)
    assert cov.loc["X", "Y"] == pytest.approx(1.0e-4, rel=1e-3)


def test_ewma_cov_hand_example():
    # Two observations, lam=0.94, zero-mean convention. Normalized weights:
    # newest 1/(1+lam), oldest lam/(1+lam). Var = (b^2 + lam*a^2)/(1+lam)
    # with a=0.01 (old), b=0.02 (new): (4e-4 + 0.94e-4)/1.94 = 2.54639e-4
    cov = ewma_cov(frame([[0.01, 0.0], [0.02, 0.0]]), lam=0.94)
    assert cov.loc["X", "X"] == pytest.approx(2.54639e-4, rel=1e-4)


def test_ewma_constant_series_recovers_exactly():
    # Every row identical: weighting scheme must not matter. var = r^2 exactly
    # (this is what the weight normalization buys).
    cov = ewma_cov(frame([[0.01, 0.0]] * 10), lam=0.94)
    assert cov.loc["X", "X"] == pytest.approx(1e-4)


def test_ewma_weights_recent_data_more():
    # Same numbers, opposite order: quiet-then-wild must show higher EWMA
    # variance than wild-then-quiet. Sample cov can't tell them apart.
    quiet_then_wild = frame([[0.001, 0.0]] * 20 + [[0.03, 0.0]] * 5)
    wild_then_quiet = frame([[0.03, 0.0]] * 5 + [[0.001, 0.0]] * 20)
    recent_wild = ewma_cov(quiet_then_wild).loc["X", "X"]
    old_wild = ewma_cov(wild_then_quiet).loc["X", "X"]
    assert recent_wild > old_wild * 2
    assert sample_cov(quiet_then_wild).loc["X", "X"] == pytest.approx(
        sample_cov(wild_then_quiet).loc["X", "X"]
    )


def test_ewma_bad_lambda_rejected():
    data = frame([[0.01, 0.0], [0.02, 0.0]])
    for bad in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(CovarianceError, match="lambda"):
            ewma_cov(data, lam=bad)


# --------------------------------------------------------------- properties


@pytest.mark.parametrize("estimator", ALL_ESTIMATORS)
def test_output_is_symmetric_psd_and_labeled(estimator, synthetic_returns):
    cov = estimator(synthetic_returns)
    assert list(cov.index) == list(synthetic_returns.columns)
    assert list(cov.columns) == list(synthetic_returns.columns)
    values = cov.to_numpy()
    assert np.allclose(values, values.T)  # symmetric
    assert np.linalg.eigvalsh(values).min() >= -1e-12  # PSD (dust tolerance)


def test_shrinkage_moves_toward_the_target(synthetic_returns):
    # LW pulls off-diagonals toward 0 relative to sample -- never inflates.
    short = synthetic_returns.iloc[:30]  # small T: shrinkage should be visible
    s = sample_cov(short)
    lw = ledoit_wolf_cov(short)
    assert abs(lw.loc["AAA", "BBB"]) < abs(s.loc["AAA", "BBB"])


def test_shrinkage_fixes_singular_sample():
    # 10 assets, 6 observations: the sample matrix is rank-deficient
    # (singular), which is the whole reason shrinkage estimators exist.
    rng = np.random.default_rng(7)
    wide = pd.DataFrame(
        rng.normal(0, 0.01, size=(6, 10)),
        columns=[f"S{i}" for i in range(10)],
        index=pd.bdate_range("2024-01-01", periods=6),
    )
    sample_min = np.linalg.eigvalsh(sample_cov(wide).to_numpy()).min()
    lw_min = np.linalg.eigvalsh(ledoit_wolf_cov(wide).to_numpy()).min()
    oas_min = np.linalg.eigvalsh(oas_cov(wide).to_numpy()).min()
    assert sample_min == pytest.approx(0.0, abs=1e-10)  # singular, as expected
    assert lw_min > 1e-8  # shrunk matrices are invertible
    assert oas_min > 1e-8


# ------------------------------------------------------------------ recovery


@pytest.mark.parametrize("estimator", [sample_cov, ledoit_wolf_cov, oas_cov])
def test_recovery_of_known_covariance(estimator, synthetic_returns, true_cov):
    # 8000 draws: the full-sample estimators must land near the generating
    # matrix. Tolerance is statistical, not numerical -- sampling error at
    # T=8000 is ~2%, so 10% on variances is comfortable but not toothless.
    cov = estimator(synthetic_returns)
    for sym in true_cov.columns:
        assert cov.loc[sym, sym] == pytest.approx(true_cov.loc[sym, sym], rel=0.10)
    # And the strong correlation must come through with the right sign/size.
    est_corr = cov.loc["AAA", "BBB"] / np.sqrt(cov.loc["AAA", "AAA"] * cov.loc["BBB", "BBB"])
    assert est_corr == pytest.approx(0.4, abs=0.05)


def test_ewma_recovery_is_necessarily_looser(synthetic_returns, true_cov):
    # EWMA gets its own tolerance because forgetting IS the estimator:
    # lambda=0.94 concentrates the weight on recent days -- effective sample
    # size 1/sum(w^2) = (1+lam)/(1-lam) ~ 32 observations, no matter how long
    # the history. Relative std of a variance estimate on ~32 obs is
    # sqrt(2/32) ~ 25%, so a tight tolerance would fail on pure sampling
    # noise. Holding it to ~2.5 sigma:
    cov = ewma_cov(synthetic_returns)
    for sym in true_cov.columns:
        assert cov.loc[sym, sym] == pytest.approx(true_cov.loc[sym, sym], rel=0.65)
    est_corr = cov.loc["AAA", "BBB"] / np.sqrt(cov.loc["AAA", "AAA"] * cov.loc["BBB", "BBB"])
    assert est_corr == pytest.approx(0.4, abs=0.30)  # sign and rough size


def test_ledoit_wolf_matches_sklearn(synthetic_returns):
    # Cross-check the from-scratch implementation against the reference one.
    # Skips cleanly where scikit-learn isn't installed (it's not a dependency).
    sklearn_cov = pytest.importorskip("sklearn.covariance")
    ours = ledoit_wolf_cov(synthetic_returns.iloc[:100]).to_numpy()
    theirs = sklearn_cov.LedoitWolf().fit(synthetic_returns.iloc[:100].to_numpy()).covariance_
    assert np.allclose(ours, theirs, rtol=1e-8)


# ------------------------------------------------------------------- guards


def test_nan_rejected():
    data = frame([[0.01, 0.02], [float("nan"), 0.01], [0.02, 0.0]])
    with pytest.raises(CovarianceError, match="NaN"):
        sample_cov(data)


def test_too_few_rows_rejected():
    with pytest.raises(CovarianceError, match="at least 2"):
        sample_cov(frame([[0.01, 0.02]]))


def test_annualize_scales_by_252():
    cov = sample_cov(frame([[0.01, 0.02], [-0.01, 0.0], [0.02, 0.01]]))
    assert annualize(cov).loc["X", "X"] == pytest.approx(cov.loc["X", "X"] * 252)
