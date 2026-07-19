"""Tests for factors/exposures.py.

Notes
-----
Pure-function tests build price panels and fundamentals frames by hand so
each factor's definition is pinned independently. The integration test at
the bottom runs the whole thing against a seeded database, including the
as-of gate: fundamentals filed after the date must leave value/quality/
size as NaN while momentum and low_vol (price-only) still compute.
"""

import numpy as np
import pandas as pd
import pytest

from ballast.data.prices import load_latest_prices, store_prices
from ballast.factors.exposures import (
    FACTORS,
    MOMENTUM_LONG,
    ExposureError,
    clean_cross_section,
    compute_exposures,
    raw_characteristics,
)
from tests.test_prices import tidy


def price_panel(columns: dict[str, list[float]]) -> pd.DataFrame:
    n = max(len(v) for v in columns.values())
    idx = pd.bdate_range("2023-01-02", periods=n)
    return pd.DataFrame({k: pd.Series(v, index=idx[-len(v) :]) for k, v in columns.items()})


def fundamentals_frame(rows: dict[str, dict]) -> pd.DataFrame:
    fields = [
        "book_equity",
        "net_income",
        "revenue",
        "gross_profit",
        "assets",
        "liabilities",
        "shares_outstanding",
    ]
    return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=fields)


# ------------------------------------------------------- raw characteristics


def test_momentum_12_1_hand_example():
    # A: price 100 until day -253, doubled to 200 by day -21, flat since.
    # Momentum = P(-21)/P(-252) - 1 = 200/100 - 1 = 1.0 exactly (the flat
    # final month is skipped -- that's the "-1" in 12-1).
    a = [100.0] * 30 + list(np.linspace(100, 200, MOMENTUM_LONG - 21)) + [200.0] * 22
    b = [150.0] * len(a)  # flat: momentum 0
    raw = raw_characteristics(
        price_panel({"A": a, "B": b}),
        {"A": 200.0, "B": 150.0},
        fundamentals_frame({}),
    )
    assert raw.loc["A", "momentum"] == pytest.approx(1.0, abs=0.02)
    assert raw.loc["B", "momentum"] == pytest.approx(0.0, abs=1e-12)


def test_momentum_nan_without_13_months():
    raw = raw_characteristics(
        price_panel({"A": [100.0] * 100}), {"A": 100.0}, fundamentals_frame({})
    )
    assert np.isnan(raw.loc["A", "momentum"])  # by definition, not by bug


def test_value_ratios_hand_example():
    # mcap = 50 close x 10 shares = 500; B/P = 250/500 = 0.5; E/P = 50/500 = 0.1
    fundamentals = fundamentals_frame(
        {"A": {"book_equity": 250.0, "net_income": 50.0, "shares_outstanding": 10.0}}
    )
    raw = raw_characteristics(price_panel({"A": [50.0] * 300}), {"A": 50.0}, fundamentals)
    assert raw.loc["A", "bp"] == pytest.approx(0.5)
    assert raw.loc["A", "ep"] == pytest.approx(0.1)
    assert raw.loc["A", "log_mcap"] == pytest.approx(np.log(500.0))


def test_vol_orders_calm_below_wild():
    rng = np.random.default_rng(3)
    calm = (100 * np.cumprod(1 + rng.normal(0, 0.005, 300))).tolist()
    wild = (100 * np.cumprod(1 + rng.normal(0, 0.03, 300))).tolist()
    raw = raw_characteristics(
        price_panel({"CALM": calm, "WILD": wild}),
        {"CALM": 100.0, "WILD": 100.0},
        fundamentals_frame({}),
    )
    assert raw.loc["CALM", "vol"] < raw.loc["WILD", "vol"]


def test_nonpositive_mcap_poisons_ratios_not_momentum():
    fundamentals = fundamentals_frame(
        {"A": {"book_equity": 100.0, "shares_outstanding": 0.0}}  # mcap = 0
    )
    raw = raw_characteristics(price_panel({"A": [50.0] * 300}), {"A": 50.0}, fundamentals)
    assert np.isnan(raw.loc["A", "bp"]) and np.isnan(raw.loc["A", "log_mcap"])
    assert np.isfinite(raw.loc["A", "vol"])  # price-only characteristics survive


# ----------------------------------------------------------- cross-section


def synthetic_raw(n: int = 20, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "bp": rng.uniform(0.1, 2.0, n),
            "ep": rng.uniform(0.01, 0.15, n),
            "log_mcap": rng.uniform(20, 27, n),
            "gp_assets": rng.uniform(0.05, 0.5, n),
            "leverage": rng.uniform(0.1, 0.9, n),
            "momentum": rng.normal(0.1, 0.3, n),
            "vol": rng.uniform(0.005, 0.04, n),
        },
        index=[f"S{i}" for i in range(n)],
    )


def test_clean_outputs_standardized_factors():
    out = clean_cross_section(synthetic_raw())
    assert list(out.columns) == list(FACTORS)
    for factor in ("momentum", "size"):  # single-ingredient columns
        assert out[factor].mean() == pytest.approx(0.0, abs=1e-9)
        assert out[factor].std(ddof=0) == pytest.approx(1.0, rel=1e-9)


def test_winsorize_defangs_an_outlier():
    raw = synthetic_raw()
    spiked = raw.copy()
    spiked.loc["S0", "bp"] = 1e6  # meme-stock accounting artifact
    z_spiked = clean_cross_section(spiked)["value"]
    # Clipped at the 99th percentile: the outlier ends up with the largest
    # value exposure but a BOUNDED one, not a 4-sigma monster.
    assert z_spiked.loc["S0"] == z_spiked.max()
    assert z_spiked.loc["S0"] < 4.0


def test_quality_orientation():
    raw = synthetic_raw()
    raw.loc["GOOD", ["gp_assets", "leverage"]] = [0.6, 0.05]  # profitable, unlevered
    raw.loc["BAD", ["gp_assets", "leverage"]] = [0.01, 0.95]  # neither
    raw.loc[["GOOD", "BAD"], ["bp", "ep", "log_mcap", "momentum", "vol"]] = 0.5
    out = clean_cross_section(raw)
    assert out.loc["GOOD", "quality"] > out.loc["BAD", "quality"]


def test_low_vol_orientation():
    raw = synthetic_raw()
    calmest = raw["vol"].idxmin()
    out = clean_cross_section(raw)
    assert out.loc[calmest, "low_vol"] == out["low_vol"].max()  # calm = high exposure


def test_one_missing_value_ingredient_uses_the_other():
    raw = synthetic_raw()
    raw.loc["S1", "ep"] = np.nan  # no earnings yet: value from B/P alone
    out = clean_cross_section(raw)
    assert np.isfinite(out.loc["S1", "value"])
    raw.loc["S1", "bp"] = np.nan  # now neither ingredient
    out = clean_cross_section(raw)
    assert np.isnan(out.loc["S1", "value"])


def test_sector_neutralization_demeans_within_sector():
    raw = synthetic_raw()
    # Push a level offset into one group's bp: sector-cheapness, not
    # stock-cheapness. Neutralization must remove it.
    tech = [f"S{i}" for i in range(10)]
    fin = [f"S{i}" for i in range(10, 20)]
    raw.loc[fin, "bp"] = raw.loc[fin, "bp"] + 5.0
    sectors = {s: "tech" for s in tech} | {s: "fin" for s in fin}
    out = clean_cross_section(raw, sectors=sectors)
    assert out.loc[tech, "value"].mean() == pytest.approx(0.0, abs=1e-9)
    assert out.loc[fin, "value"].mean() == pytest.approx(0.0, abs=1e-9)


def test_empty_cross_section_raises():
    with pytest.raises(ExposureError, match="empty"):
        clean_cross_section(pd.DataFrame())


# ------------------------------------------------------------- integration


def put_fundamental(db, symbol, field, value, filed):
    import duckdb

    from ballast.data.edgar import _FUNDAMENTALS_SCHEMA

    con = duckdb.connect(str(db))
    try:
        con.execute(_FUNDAMENTALS_SCHEMA)
        con.execute(
            "INSERT OR IGNORE INTO fundamentals VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [symbol, field, value, "2023-12-31", filed, "10-K", 2023, "FY"],
        )
    finally:
        con.close()


def test_compute_exposures_end_to_end_with_as_of_gate(tmp_path):
    db = tmp_path / "t.duckdb"
    rng = np.random.default_rng(17)
    for symbol in ("AAA", "BBB"):
        closes = (100 * np.cumprod(1 + rng.normal(0.0003, 0.01, 320))).tolist()
        store_prices(tidy(closes, symbol=symbol, start="2023-01-02"), db_path=db)

    fields = {
        "book_equity": 500.0,
        "net_income": 60.0,
        "gross_profit": 90.0,
        "assets": 1000.0,
        "liabilities": 400.0,
        "shares_outstanding": 10.0,
    }
    for field, value in fields.items():
        put_fundamental(db, "AAA", field, value, filed="2024-02-15")  # visible
        put_fundamental(db, "BBB", field, value, filed="2024-08-01")  # filed LATER

    as_of = "2024-03-01"  # after AAA's filing, before BBB's
    out = compute_exposures(["AAA", "BBB"], as_of=as_of, db_path=db)

    assert list(out.columns) == list(FACTORS)
    # AAA has fundamentals: all five factors real numbers.
    assert np.isfinite(out.loc["AAA", ["value", "size", "quality"]]).all()
    # BBB's filing wasn't public at as_of: fundamental factors NaN...
    assert out.loc["BBB", ["value", "size", "quality"]].isna().all()
    # ...but price-only factors still compute.
    assert np.isfinite(out.loc["BBB", ["momentum", "low_vol"]]).all()

    # And the as-of price gate: latest price at as_of is from <= as_of.
    px = load_latest_prices(["AAA"], db_path=db, as_of="2023-06-01")
    px_full = load_latest_prices(["AAA"], db_path=db)
    assert px["AAA"] != px_full["AAA"]  # different bar, earlier date
