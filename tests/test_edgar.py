"""Tests for data/edgar.py. Entirely offline.

Notes
-----
fixture_facts() fabricates a companyfacts JSON with the shapes that matter:
two fiscal years, an amended filing, a candidate-tag fallback, and a field
that only exists under the dei namespace. The as-of leakage test is the
most important test in this file: a filing must be invisible the day
before it was filed, no matter how old the fiscal period it describes.
"""

import pytest

from ballast.data.edgar import (
    EdgarError,
    ingest_fundamentals,
    latest_fundamentals,
    load_cik_map,
)

CIKS = {"TEST": 12345}


def usd_fact(value: float, end: str, filed: str, form: str = "10-K", fy: int = 2023) -> dict:
    return {"val": value, "end": end, "filed": filed, "form": form, "fy": fy, "fp": "FY"}


def fixture_facts() -> dict:
    """A minimal but structurally faithful companyfacts payload."""
    return {
        "facts": {
            "us-gaap": {
                # Two annual book-equity values, plus an AMENDMENT of the
                # 2023 figure filed later with a corrected number.
                "StockholdersEquity": {
                    "units": {
                        "USD": [
                            usd_fact(1000.0, "2022-12-31", "2023-02-15", fy=2022),
                            usd_fact(1500.0, "2023-12-31", "2024-02-15", fy=2023),
                            usd_fact(1400.0, "2023-12-31", "2024-06-01", form="10-K/A", fy=2023),
                        ]
                    }
                },
                # Revenue is ONLY tagged with the newer contract-revenue tag:
                # the candidate-tag fallback must find it.
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {"USD": [usd_fact(5000.0, "2023-12-31", "2024-02-15")]}
                },
                "Assets": {"units": {"USD": [usd_fact(9000.0, "2023-12-31", "2024-02-15")]}},
                "NetIncomeLoss": {"units": {"USD": [usd_fact(800.0, "2023-12-31", "2024-02-15")]}},
                # An 8-K entry that must be ignored (not a periodic report).
                "Liabilities": {
                    "units": {
                        "USD": [
                            usd_fact(4000.0, "2023-12-31", "2024-02-15"),
                            usd_fact(9999.0, "2023-12-31", "2024-03-01", form="8-K"),
                        ]
                    }
                },
                # No GrossProfit at all -- tolerated, renders as NaN.
            },
            "dei": {
                "EntityCommonStockSharesOutstanding": {
                    "units": {"shares": [usd_fact(100.0, "2023-12-31", "2024-02-15")]}
                }
            },
        }
    }


@pytest.fixture
def seeded(tmp_path, monkeypatch):
    """DB with the fixture filings ingested through the real pipeline."""
    monkeypatch.setattr("ballast.data.edgar._fetch_json", lambda url: fixture_facts())
    db = tmp_path / "t.duckdb"
    written = ingest_fundamentals(["TEST"], db_path=db, cik_map=CIKS)
    return db, written


def test_ingest_extracts_and_counts(seeded):
    _, written = seeded
    # 3 equity + 1 revenue + 1 assets + 1 income + 1 liabilities (8-K
    # dropped) + 1 shares = 8 rows.
    assert written == 8


def test_ingest_is_idempotent(seeded, monkeypatch):
    db, _ = seeded
    monkeypatch.setattr("ballast.data.edgar._fetch_json", lambda url: fixture_facts())
    assert ingest_fundamentals(["TEST"], db_path=db, cik_map=CIKS) == 0


def test_as_of_leakage_contract(seeded):
    # THE test. The FY2023 10-K was filed 2024-02-15. One day earlier,
    # only the FY2022 number may be visible -- the fiscal period being
    # long over is irrelevant; the market hadn't seen the filing.
    db, _ = seeded
    before = latest_fundamentals(["TEST"], as_of="2024-02-14", db_path=db)
    assert before.loc["TEST", "book_equity"] == 1000.0  # FY2022 value
    after = latest_fundamentals(["TEST"], as_of="2024-02-15", db_path=db)
    assert after.loc["TEST", "book_equity"] == 1500.0  # FY2023 visible now


def test_amendment_wins_only_after_its_own_filing_date(seeded):
    # The 10-K/A corrected 1500 -> 1400 on 2024-06-01. Before that date
    # the original number is the truth as the market knew it.
    db, _ = seeded
    original = latest_fundamentals(["TEST"], as_of="2024-05-31", db_path=db)
    assert original.loc["TEST", "book_equity"] == 1500.0
    amended = latest_fundamentals(["TEST"], as_of="2024-06-01", db_path=db)
    assert amended.loc["TEST", "book_equity"] == 1400.0


def test_candidate_tag_fallback_and_dei_namespace(seeded):
    db, _ = seeded
    frame = latest_fundamentals(["TEST"], as_of="2024-12-31", db_path=db)
    assert frame.loc["TEST", "revenue"] == 5000.0  # via the fallback tag
    assert frame.loc["TEST", "shares_outstanding"] == 100.0  # dei namespace


def test_missing_field_is_nan_not_guess(seeded):
    db, _ = seeded
    frame = latest_fundamentals(["TEST"], as_of="2024-12-31", db_path=db)
    assert frame["gross_profit"].isna().all()  # no GrossProfit in filings


def test_non_periodic_forms_are_dropped(seeded):
    # The 8-K liabilities figure (9999) must never surface.
    db, _ = seeded
    frame = latest_fundamentals(["TEST"], as_of="2024-12-31", db_path=db)
    assert frame.loc["TEST", "liabilities"] == 4000.0


def test_unknown_ticker_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("ballast.data.edgar._fetch_json", lambda url: fixture_facts())
    with pytest.raises(EdgarError, match=r"\['NOPE'\]"):
        ingest_fundamentals(["NOPE"], db_path=tmp_path / "t.duckdb", cik_map=CIKS)


def test_symbol_with_no_usable_facts_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("ballast.data.edgar._fetch_json", lambda url: {"facts": {}})
    with pytest.raises(EdgarError, match="no usable"):
        ingest_fundamentals(["TEST"], db_path=tmp_path / "t.duckdb", cik_map=CIKS)


def test_as_of_before_all_filings_raises(seeded):
    db, _ = seeded
    with pytest.raises(EdgarError, match="visible"):
        latest_fundamentals(["TEST"], as_of="2020-01-01", db_path=db)


def test_cik_map_parses_sec_shape(monkeypatch):
    payload = {
        "0": {"cik_str": 320193, "ticker": "aapl", "title": "Apple Inc."},
        "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft"},
    }
    monkeypatch.setattr("ballast.data.edgar._fetch_json", lambda url: payload)
    mapping = load_cik_map()
    assert mapping["AAPL"] == 320193  # upper-cased on the way in
    assert mapping["MSFT"] == 789019
