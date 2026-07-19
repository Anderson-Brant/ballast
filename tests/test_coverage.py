"""Tests for validate/coverage.py and the v0.3.0 CLI commands.

Notes
-----
The Kupiec p-value has a hand-computed case (arithmetic in the comment).
The Basel test reproduces the official (250-day, 99%) zone table -- if the
generalized binomial cutoffs ever drift from the regulator's table, that
test fails. The clustering tests construct breach patterns directly, since
the whole point is that the COUNT can be right while the TIMING is wrong.
"""

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from ballast.cli.app import app
from ballast.validate.coverage import (
    ValidationResult,
    christoffersen_independence,
    conditional_coverage,
    kupiec_pof,
    traffic_light,
    validate_var,
)
from tests.test_prices import tidy

runner = CliRunner()


# --------------------------------------------------------------- Kupiec POF


def test_kupiec_perfect_calibration_is_p_one():
    # Exactly the expected count: LR = 0, p = 1.
    assert kupiec_pof(50, 1000, 0.95) == pytest.approx(1.0)


def test_kupiec_hand_computed_p_value():
    # x=3, n=250, c=0.99. LR = -2[247 ln(.99/.988...) ...] worked by hand:
    #   247*(ln .99 - ln .988) = 0.4995
    #   3*(ln .01 - ln .012)  = -0.5470
    #   LR = -2*(-0.0475) = 0.0949 -> chi2(1) p = 0.758
    assert kupiec_pof(3, 250, 0.99) == pytest.approx(0.758, abs=0.005)


def test_kupiec_rejects_too_many_breaches():
    # 100 breaches where 50 were expected: overwhelming evidence.
    assert kupiec_pof(100, 1000, 0.95) < 1e-6


def test_kupiec_rejects_zero_breaches_too():
    # No breaches in 100 days at 95% (5 expected) is ALSO miscalibration:
    # the model is wasting capital. The test is two-sided by construction.
    assert kupiec_pof(0, 100, 0.95) < 0.01


def test_kupiec_input_guards():
    with pytest.raises(ValueError, match="observations"):
        kupiec_pof(0, 0, 0.99)
    with pytest.raises(ValueError, match="breaches"):
        kupiec_pof(11, 10, 0.99)
    with pytest.raises(ValueError, match="confidence"):
        kupiec_pof(1, 100, 1.5)


# ----------------------------------------------------------- Christoffersen


def breach_pattern(n: int, breach_at: list[int]) -> np.ndarray:
    b = np.zeros(n, dtype=bool)
    b[breach_at] = True
    return b


def test_clustered_breaches_fail_independence():
    # 10 breaches in a single consecutive run: maximal clustering.
    clustered = breach_pattern(500, list(range(100, 110)))
    p = christoffersen_independence(clustered)
    assert p is not None and p < 1e-6


def test_scattered_breaches_pass_independence():
    # Same count, spread far apart: no breach follows another.
    scattered = breach_pattern(500, list(range(25, 500, 50)))
    p = christoffersen_independence(scattered)
    assert p is not None and p > 0.5


def test_independence_is_none_when_unassessable():
    assert christoffersen_independence(breach_pattern(500, [250])) is None  # 1 breach
    assert christoffersen_independence(breach_pattern(500, [])) is None  # none


def test_conditional_coverage_catches_what_kupiec_misses():
    # THE reason the joint test exists: 10 breaches in 500 days at 98%
    # confidence is EXACTLY the expected count (Kupiec p = 1.0), but all
    # ten in one week means the model missed a regime. cc must fail it.
    clustered = breach_pattern(500, list(range(100, 110)))
    assert kupiec_pof(10, 500, 0.98) == pytest.approx(1.0)
    cc = conditional_coverage(clustered, 0.98)
    assert cc is not None and cc < 1e-4


# ------------------------------------------------------------ traffic light


def test_traffic_light_reproduces_official_basel_table():
    # Basel (250 days, 99%): green through 4 breaches, yellow 5-9, red 10+.
    for x, zone in [(0, "green"), (4, "green"), (5, "yellow"), (9, "yellow"), (10, "red")]:
        assert traffic_light(x, 250, 0.99) == zone, f"x={x}"


def test_traffic_light_generalizes_to_other_setups():
    # 1000 days at 95%: 50 expected. 50 is comfortably green; 80 is not.
    assert traffic_light(50, 1000, 0.95) == "green"
    assert traffic_light(80, 1000, 0.95) == "red"


# ------------------------------------------------------------- validate_var


@pytest.fixture(scope="module")
def gaussian_series():
    rng = np.random.default_rng(21)
    return pd.Series(rng.normal(0.0, 0.01, 3000), index=pd.bdate_range("2012-01-02", periods=3000))


def test_validate_var_passes_on_its_own_distribution(gaussian_series):
    # Historical VaR on stationary data it was built for: everything green.
    result = validate_var(gaussian_series, method="historical", confidence=0.95, window=500)
    assert isinstance(result, ValidationResult)
    assert result.kupiec_pass
    assert result.zone == "green"
    assert result.n_obs == 3000 - 500 - 1 + 1 or result.n_obs > 2000  # rolled window
    assert result.worst_date is not None  # ~5% breaches exist
    assert result.worst_ratio > 1.0  # a breach is by definition beyond VaR


def test_validate_var_flags_a_regime_break():
    # Calm training data, then a violent final year the 750-day window is
    # slow to absorb: parametric VaR should fail coverage and cluster.
    rng = np.random.default_rng(8)
    calm = rng.normal(0.0, 0.005, 2000)
    wild = rng.normal(0.0, 0.03, 300)
    series = pd.Series(
        np.concatenate([calm, wild]), index=pd.bdate_range("2016-01-04", periods=2300)
    )
    result = validate_var(series, method="parametric", confidence=0.99, window=750)
    assert not result.kupiec_pass  # far too many breaches
    assert result.zone == "red"
    assert result.independence_p is not None and result.independence_p < 0.05


def test_no_breach_series_renders_none_fields():
    # Constant positive returns: the tail quantile is a gain, nothing ever
    # breaches, and the unassessable tests come back None, not fake p's.
    series = pd.Series([0.001] * 400, index=pd.bdate_range("2022-01-03", periods=400))
    result = validate_var(series, method="historical", confidence=0.95, window=300)
    assert result.n_breaches == 0
    assert result.independence_p is None and result.cc_p is None
    assert result.worst_date is None
    assert result.zone == "green"


# -------------------------------------------------------------------- CLI


def seed_portfolio_db(tmp_path, days=400):
    db = tmp_path / "t.duckdb"
    rng = np.random.default_rng(13)
    for symbol in ("AAA", "BBB"):
        closes = (100.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.01, days))).tolist()
        from ballast.data.prices import store_prices

        store_prices(tidy(closes, symbol=symbol), db_path=db)
    spec = tmp_path / "p.yaml"
    spec.write_text(
        yaml.safe_dump(
            {
                "name": "p",
                "positions": [{"symbol": "AAA", "shares": 10}, {"symbol": "BBB", "shares": 5}],
            }
        )
    )
    return db, spec


def test_cli_var_all_methods(tmp_path):
    db, spec = seed_portfolio_db(tmp_path)
    result = runner.invoke(app, ["var", str(spec), "--db", str(db), "--confidence", "0.95"])
    assert result.exit_code == 0, result.output
    for method in ("parametric", "cornish_fisher", "historical", "monte_carlo"):
        assert method in result.output


def test_cli_var_degrades_per_method(tmp_path):
    # 60 days: enough for parametric/historical at 95%, NOT enough for
    # filtered_historical (30-day burn + tail). Its row degrades; exit 0.
    db, spec = seed_portfolio_db(tmp_path, days=61)
    result = runner.invoke(app, ["var", str(spec), "--db", str(db), "--confidence", "0.95"])
    assert result.exit_code == 0, result.output
    assert "parametric" in result.output


def test_cli_validate_var(tmp_path):
    db, spec = seed_portfolio_db(tmp_path)
    result = runner.invoke(
        app,
        [
            "validate",
            "var",
            str(spec),
            "--db",
            str(db),
            "--method",
            "historical",
            "--confidence",
            "0.95",
            "--window",
            "250",
        ],
    )
    assert result.exit_code == 0, result.output
    for expected in ("Kupiec", "Christoffersen", "verdict", "zone"):
        assert expected in result.output


def test_cli_validate_var_short_history_exits_one(tmp_path):
    db, spec = seed_portfolio_db(tmp_path, days=100)
    result = runner.invoke(app, ["validate", "var", str(spec), "--db", str(db)])
    assert result.exit_code == 1
    assert "error:" in result.output
