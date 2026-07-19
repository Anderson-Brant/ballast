"""Tests for data/french.py. Entirely offline.

Notes
-----
make_french_zip() pins the library's actual quirks in a fixture:
description lines before the header, YYYYMMDD dates, PERCENT values,
-99.99 missing markers, and junk after the daily block. The cross-check
tests construct French dailies whose per-period compounding reproduces
the home factor returns EXACTLY, so correlations are 1/-1 by construction
and any deviation is an alignment bug.
"""

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from ballast.data.french import (
    FrenchDataError,
    _parse_french_csv,
    cross_check_factors,
    fetch_french_daily,
)


def make_french_zip(csv_text: str, name: str = "F-F_Research_Data_Factors_daily.CSV") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, csv_text)
    return buffer.getvalue()


FACTORS_CSV = """This file was created by CMPT_ME_BEME_RETS_DAILY using the 202606 CRSP database.
The 1-month TBill return is from Ibbotson and Associates Inc.

,Mkt-RF,SMB,HML,RF
19260701,    0.10,   -0.24,   -0.28,   0.009
19260702,    0.45,   -0.32,   -0.08,   0.009
19260706,  -99.99,    0.13,    0.04,   0.009

  Copyright 2026 Kenneth R. French
"""

MOMENTUM_CSV = """This file was created using the 202606 CRSP database.

,Mom
19260701,    0.56
19260702,   -0.21
19260706,    0.11
"""


# ------------------------------------------------------------------ parsing


def test_parse_handles_the_real_format():
    frame = _parse_french_csv(make_french_zip(FACTORS_CSV))
    assert list(frame.columns) == ["mkt_rf", "smb", "hml", "rf"]
    assert len(frame) == 3
    assert frame.index[0] == pd.Timestamp("1926-07-01")
    # Percent -> decimal exactly once.
    assert frame.loc["1926-07-01", "mkt_rf"] == pytest.approx(0.0010)
    assert frame.loc["1926-07-02", "smb"] == pytest.approx(-0.0032)


def test_parse_maps_missing_markers_to_nan():
    frame = _parse_french_csv(make_french_zip(FACTORS_CSV))
    assert np.isnan(frame.loc["1926-07-06", "mkt_rf"])  # -99.99 marker
    assert frame.loc["1926-07-06", "smb"] == pytest.approx(0.0013)  # neighbors intact


def test_parse_stops_at_the_junk_tail():
    # The copyright line after the data block must not become a row.
    frame = _parse_french_csv(make_french_zip(FACTORS_CSV))
    assert len(frame) == 3


def test_parse_rejects_garbage():
    with pytest.raises(FrenchDataError, match="zip"):
        _parse_french_csv(b"this is not a zip file")
    with pytest.raises(FrenchDataError, match="format"):
        _parse_french_csv(make_french_zip("no data here\nat all\n"))


def test_fetch_merges_both_files(monkeypatch):
    responses = {
        "Research_Data_Factors": make_french_zip(FACTORS_CSV),
        "Momentum": make_french_zip(MOMENTUM_CSV, name="F-F_Momentum_Factor_daily.CSV"),
    }

    def fake_download(url: str) -> bytes:
        for key, payload in responses.items():
            if key in url:
                return payload
        raise AssertionError(url)

    monkeypatch.setattr("ballast.data.french._download", fake_download)
    merged = fetch_french_daily()
    assert list(merged.columns) == ["mkt_rf", "smb", "hml", "rf", "mom"]
    assert merged.loc["1926-07-01", "mom"] == pytest.approx(0.0056)


# -------------------------------------------------------------- cross-check


def weekly_home_returns(n_periods: int = 30, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-07", periods=n_periods, freq="W-FRI")
    return pd.DataFrame(
        rng.normal(0.0, 0.01, (n_periods, 5)),
        index=dates,
        columns=["market", "value", "momentum", "size", "quality"],
    )


def perfect_french_daily(home: pd.DataFrame) -> pd.DataFrame:
    """One French daily row inside each (t, t+1] interval whose value equals
    the matching home return exactly -- compounding a single day is identity,
    so correlations are exactly +1 (or -1 for the sign-flipped size/SMB)."""
    dates = sorted(home.index)
    rows = {}
    for current, nxt in zip(dates[:-1], dates[1:], strict=False):
        mid = current + (nxt - current) / 2  # strictly inside (current, nxt]
        rows[mid] = {
            "mkt_rf": home.loc[current, "market"],
            "hml": home.loc[current, "value"],
            "mom": home.loc[current, "momentum"],
            "smb": -home.loc[current, "size"],  # SMB is small-minus-big
            "rf": 0.0001,
        }
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def test_cross_check_perfect_construction_passes_everything():
    home = weekly_home_returns()
    rows = cross_check_factors(home, perfect_french_daily(home))
    by_name = {r.ballast_factor: r for r in rows}
    assert by_name["momentum"].corr == pytest.approx(1.0)
    assert by_name["value"].corr == pytest.approx(1.0)
    assert by_name["market"].corr == pytest.approx(1.0)
    # The sign discipline: raw correlation is -1, expected sign -1 => pass.
    assert by_name["size"].corr == pytest.approx(-1.0)
    assert all(r.passes for r in rows)


def test_cross_check_flags_a_broken_factor():
    home = weekly_home_returns()
    french = perfect_french_daily(home)
    rng = np.random.default_rng(9)
    french["mom"] = rng.normal(0, 0.01, len(french))  # momentum now unrelated
    rows = cross_check_factors(home, french)
    by_name = {r.ballast_factor: r for r in rows}
    assert not by_name["momentum"].passes  # the acceptance bar catches it
    assert by_name["value"].passes  # others unaffected


def test_cross_check_skips_factors_without_counterparts():
    home = weekly_home_returns()
    rows = cross_check_factors(home, perfect_french_daily(home))
    assert "quality" not in {r.ballast_factor for r in rows}  # no French series


def test_cross_check_needs_history():
    home = weekly_home_returns(n_periods=5)
    with pytest.raises(FrenchDataError, match="periods"):
        cross_check_factors(home, perfect_french_daily(home))


def test_cross_check_needs_overlap():
    home = weekly_home_returns()
    french = perfect_french_daily(home)
    french.index = french.index + pd.Timedelta(days=3650)  # a decade away
    with pytest.raises(FrenchDataError, match="overlap"):
        cross_check_factors(home, french)


# ------------------------------------------------------------------- CLI


def test_cli_validate_factors(tmp_path, monkeypatch):
    # Seeded 10-symbol DB (the decompose recipe) + mocked French download.
    from ballast.data.prices import store_prices
    from tests.test_exposures import put_fundamental
    from tests.test_prices import tidy

    db = tmp_path / "t.duckdb"
    rng = np.random.default_rng(37)
    for symbol in [f"S{i}" for i in range(10)]:
        closes = (100 * np.cumprod(1 + rng.normal(0.0003, 0.012, 340))).tolist()
        store_prices(tidy(closes, symbol=symbol, start="2023-01-02"), db_path=db)
        for field, low, high in [
            ("book_equity", 200, 900),
            ("net_income", 20, 120),
            ("gross_profit", 50, 300),
            ("assets", 800, 2000),
            ("liabilities", 100, 900),
            ("shares_outstanding", 5, 40),
        ]:
            put_fundamental(db, symbol, field, float(rng.uniform(low, high)), filed="2023-03-01")

    def fake_french(start=None):
        dates = pd.bdate_range("2024-01-02", periods=90)
        values = rng.normal(0, 0.01, (90, 5))
        return pd.DataFrame(values, index=dates, columns=["mkt_rf", "smb", "hml", "rf", "mom"])

    monkeypatch.setattr("ballast.data.french.fetch_french_daily", fake_french)

    from typer.testing import CliRunner

    from ballast.cli.app import app

    result = CliRunner().invoke(
        app,
        ["validate", "factors", "--db", str(db), "--start", "2024-02-01", "--end", "2024-04-30"],
    )
    assert result.exit_code == 0, result.output
    for expected in ("momentum", "Mom".lower(), "Correlation", "Verdict"):
        assert expected in result.output.lower() or expected in result.output
