"""Tests for portfolio/spec.py: loading, every validation rule, resolution math.

Notes
-----
The resolution tests use hand-computed examples (numbers chosen so the
arithmetic is checkable on paper). If one of these fails after an edit,
trust the paper, not the code.
"""

import math

import pytest

from ballast.portfolio.spec import (
    Portfolio,
    PortfolioSpecError,
    Position,
    load_portfolio,
    resolve_weights,
)

# ---------------------------------------------------------------- loading


def test_load_happy_path(write_portfolio):
    # write_portfolio is the conftest.py fixture: dict in, portfolio.yaml
    # path out. pytest injects it by matching the argument name.
    path = write_portfolio(
        {
            "name": "core",
            "positions": [
                {"symbol": "nvda", "shares": 80},
                {"symbol": "V", "shares": 120.5},
                {"symbol": "BRK-B", "weight": 0.10},
            ],
            "cash": 12000,
        }
    )
    p = load_portfolio(path)
    assert p.name == "core"
    assert p.symbols == ("NVDA", "V", "BRK-B")  # lowercased input normalized
    assert p.positions[1].shares == 120.5  # fractional shares allowed
    assert p.positions[2].weight == 0.10
    assert p.cash == 12000.0


def test_load_cash_defaults_to_zero(write_portfolio):
    path = write_portfolio({"name": "p", "positions": [{"symbol": "SPY", "shares": 1}]})
    assert load_portfolio(path).cash == 0.0


def test_missing_file_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_portfolio(tmp_path / "nope.yaml")


def test_invalid_yaml_syntax(tmp_path):
    # match= is a regex tested against the error message: the test fails if
    # the right exception comes with the wrong words. This pins the message
    # quality, not just the exception type.
    path = tmp_path / "bad.yaml"
    path.write_text("name: [unclosed")
    with pytest.raises(PortfolioSpecError, match="invalid YAML"):
        load_portfolio(path)


def test_top_level_must_be_mapping(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- just\n- a\n- list\n")
    with pytest.raises(PortfolioSpecError, match="top level must be a mapping"):
        load_portfolio(path)


def test_unknown_root_key_rejected(write_portfolio):
    path = write_portfolio({"name": "p", "positions": [], "cash": 1, "extra": 5})
    with pytest.raises(PortfolioSpecError, match="unknown key"):
        load_portfolio(path)


def test_unknown_position_key_rejected(write_portfolio):
    # The typo this rule exists for:
    path = write_portfolio({"name": "p", "positions": [{"symbol": "SPY", "wieght": 0.5}]})
    with pytest.raises(PortfolioSpecError, match=r"positions\[0\].*unknown key"):
        load_portfolio(path)


def test_positions_must_be_list(write_portfolio):
    path = write_portfolio({"name": "p", "positions": {"symbol": "SPY"}, "cash": 1})
    with pytest.raises(PortfolioSpecError, match="must be a list"):
        load_portfolio(path)


def test_position_must_be_mapping(write_portfolio):
    path = write_portfolio({"name": "p", "positions": ["SPY"]})
    with pytest.raises(PortfolioSpecError, match="must be a mapping"):
        load_portfolio(path)


def test_missing_symbol_key(write_portfolio):
    path = write_portfolio({"name": "p", "positions": [{"shares": 10}]})
    with pytest.raises(PortfolioSpecError, match="missing required key `symbol`"):
        load_portfolio(path)


def test_error_message_names_file_and_index(write_portfolio):
    path = write_portfolio(
        {"name": "p", "positions": [{"symbol": "A", "shares": 1}, {"symbol": "B"}]}
    )
    with pytest.raises(PortfolioSpecError, match=r"positions\[1\]"):
        load_portfolio(path)


def test_name_required(write_portfolio):
    path = write_portfolio({"positions": [{"symbol": "SPY", "shares": 1}]})
    with pytest.raises(PortfolioSpecError, match="name"):
        load_portfolio(path)


def test_numeric_string_rejected(write_portfolio):
    path = write_portfolio({"name": "p", "positions": [{"symbol": "SPY", "shares": "80"}]})
    with pytest.raises(PortfolioSpecError, match="must be a number"):
        load_portfolio(path)


# ------------------------------------------------------- validation rules


def test_shares_and_weight_both_rejected():
    with pytest.raises(PortfolioSpecError, match="exactly one"):
        Position(symbol="SPY", shares=10, weight=0.5)


def test_shares_and_weight_neither_rejected():
    with pytest.raises(PortfolioSpecError, match="exactly one"):
        Position(symbol="SPY")


# parametrize runs the test once per value: four bad inputs, four test cases,
# each reported separately on failure.
@pytest.mark.parametrize("bad", [0, -5, float("nan"), float("inf")])
def test_bad_shares_rejected(bad):
    with pytest.raises(PortfolioSpecError):
        Position(symbol="SPY", shares=bad)


@pytest.mark.parametrize("bad", [0, -0.1, 1.5, float("nan")])
def test_bad_weight_rejected(bad):
    with pytest.raises(PortfolioSpecError):
        Position(symbol="SPY", weight=bad)


def test_bool_is_not_a_number():
    # bool subclasses int; `shares: true` must not become 1.0
    with pytest.raises(PortfolioSpecError, match="must be a number"):
        Position(symbol="SPY", shares=True)


def test_empty_symbol_rejected():
    with pytest.raises(PortfolioSpecError, match="non-empty"):
        Position(symbol="  ", shares=1)


def test_duplicate_symbols_case_insensitive():
    with pytest.raises(PortfolioSpecError, match="duplicate symbol"):
        Portfolio(
            name="p",
            positions=(Position("BRK-B", shares=1), Position("brk-b", shares=2)),
        )


def test_weights_sum_above_one_rejected():
    with pytest.raises(PortfolioSpecError, match="sum"):
        Portfolio(
            name="p",
            positions=(Position("A", weight=0.6), Position("B", weight=0.5)),
        )


def test_weights_sum_exactly_one_ok_with_float_dust():
    # 0.3 + 0.3 + 0.4 != 1.0 in binary floats; tolerance must absorb it
    Portfolio(
        name="p",
        positions=(
            Position("A", weight=0.3),
            Position("B", weight=0.3),
            Position("C", weight=0.4),
        ),
    )


def test_negative_cash_rejected():
    with pytest.raises(PortfolioSpecError, match="cash"):
        Portfolio(name="p", cash=-1.0)


def test_empty_portfolio_needs_cash():
    with pytest.raises(PortfolioSpecError, match="no positions and no cash"):
        Portfolio(name="p")
    Portfolio(name="p", cash=100.0)  # cash-only is legal


# ------------------------------------------------------------- resolution


def test_resolve_all_shares_hand_example():
    # A: 10 sh @ $50 = $500; B: 30 sh @ $10 = $300; cash $200 -> NAV $1000
    p = Portfolio(
        name="p",
        positions=(Position("A", shares=10), Position("B", shares=30)),
        cash=200.0,
    )
    r = resolve_weights(p, {"A": 50.0, "B": 10.0})
    assert r.nav == 1000.0
    assert r.weights == {"A": 0.5, "B": 0.3}
    assert r.cash_weight == 0.2


def test_resolve_mixed_shares_and_weights_hand_example():
    # NVDA: 80 sh @ $100 = $8000; cash $2000; AAPL claims weight 0.2.
    # N = (8000 + 2000) / (1 - 0.2) = 12500
    # NVDA 8000/12500 = 0.64, AAPL 0.2, cash 2000/12500 = 0.16
    p = Portfolio(
        name="p",
        positions=(Position("NVDA", shares=80), Position("AAPL", weight=0.2)),
        cash=2000.0,
    )
    r = resolve_weights(p, {"NVDA": 100.0})
    assert r.nav == pytest.approx(12500.0)
    assert r.weights["NVDA"] == pytest.approx(0.64)
    assert r.weights["AAPL"] == pytest.approx(0.20)
    assert r.cash_weight == pytest.approx(0.16)
    assert sum(r.weights.values()) + r.cash_weight == pytest.approx(1.0)


def test_resolve_weights_plus_cash_anchored():
    # Weights + cash: cash anchors NAV. W=0.6, C=$5000 -> N = 5000/0.4 = 12500
    p = Portfolio(name="p", positions=(Position("A", weight=0.6),), cash=5000.0)
    r = resolve_weights(p, {})
    assert r.nav == pytest.approx(12500.0)
    assert r.cash_weight == pytest.approx(0.4)


def test_resolve_pure_weights_scale_free():
    p = Portfolio(
        name="p",
        positions=(Position("A", weight=0.7), Position("B", weight=0.3)),
    )
    r = resolve_weights(p, {})  # no prices needed, no NAV knowable
    assert r.nav is None
    assert r.weights == {"A": 0.7, "B": 0.3}
    assert r.cash_weight == 0.0


def test_resolve_cash_only():
    r = resolve_weights(Portfolio(name="p", cash=500.0), {})
    assert r.weights == {}
    assert r.cash_weight == 1.0
    assert r.nav == 500.0


def test_resolve_missing_price_lists_symbols():
    p = Portfolio(
        name="p",
        positions=(Position("A", shares=1), Position("B", shares=1)),
    )
    with pytest.raises(PortfolioSpecError, match=r"\['A', 'B'\]"):
        resolve_weights(p, {})


@pytest.mark.parametrize("bad", [0.0, -10.0, float("nan")])
def test_resolve_bad_price_rejected(bad):
    p = Portfolio(name="p", positions=(Position("A", shares=1),))
    with pytest.raises(PortfolioSpecError, match="price"):
        resolve_weights(p, {"A": bad})


def test_resolve_full_weights_with_share_leg_contradiction():
    # AAPL claims 100% of NAV but NVDA shares + cash also exist: no finite
    # NAV satisfies that; must raise, not divide by zero.
    p = Portfolio(
        name="p",
        positions=(Position("NVDA", shares=10), Position("AAPL", weight=1.0)),
        cash=0.0,
    )
    with pytest.raises(PortfolioSpecError, match="leaving no NAV"):
        resolve_weights(p, {"NVDA": 100.0})


def test_resolved_weights_preserve_spec_order():
    p = Portfolio(
        name="p",
        positions=(
            Position("Z", shares=1),
            Position("A", weight=0.1),
            Position("M", shares=2),
        ),
        cash=100.0,
    )
    r = resolve_weights(p, {"Z": 10.0, "M": 5.0})
    assert list(r.weights) == ["Z", "A", "M"]
    assert math.isclose(sum(r.weights.values()) + r.cash_weight, 1.0)
