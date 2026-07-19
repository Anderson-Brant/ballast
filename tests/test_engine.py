"""Tests for backtest/engine.py.

Notes
-----
The drift and cost examples are worked by hand in the comments -- if one
fails after an edit, redo the arithmetic on paper before touching the
test. The leakage test is the most important one in the file: it pins the
exact boundary of what a weights function is allowed to see.
"""

import numpy as np
import pandas as pd
import pytest

from ballast.backtest.engine import BacktestError, run_backtest


def frame(rows: list[list[float]], columns=("A", "B")) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=list(columns), index=pd.bdate_range("2024-01-01", periods=len(rows))
    )


def fifty_fifty(window_df: pd.DataFrame) -> pd.Series:
    return pd.Series(0.5, index=window_df.columns)


# --------------------------------------------------------- drift and costs


def test_drift_hand_example():
    # window=2 -> simulation covers rows 2 and 3. Weights 50/50, no costs.
    # day 3: gross = .5*.10 + .5*.00 = .05
    #        drift: A -> .5*1.10/1.05 = .52381, B -> .5*1.00/1.05 = .47619
    # day 4: gross = .52381*0 + .47619*.10 = .047619
    # equity: 1.05 * 1.047619 = 1.100000 (equals 50/50 buy-and-hold, as it must)
    returns = frame([[0.01, 0.03], [0.02, 0.00], [0.10, 0.00], [0.00, 0.10]])
    result = run_backtest(returns, fifty_fifty, window=2, step=100, cost_bps=0.0)
    assert result.returns.iloc[0] == pytest.approx(0.05)
    assert result.returns.iloc[1] == pytest.approx(0.047619, rel=1e-4)
    assert result.equity.iloc[-1] == pytest.approx(1.10)
    assert result.n_rebalances == 1


def test_first_position_build_pays_costs():
    # Same setup, 10bps: first rebalance trades |0.5|+|0.5| = 1.0 of
    # notional, cost = 1.0 * 10/10000 = 0.001, charged on day 3.
    returns = frame([[0.01, 0.03], [0.02, 0.00], [0.10, 0.00], [0.00, 0.10]])
    result = run_backtest(returns, fifty_fifty, window=2, step=100, cost_bps=10.0)
    assert result.returns.iloc[0] == pytest.approx(0.05 - 0.001)
    assert result.cost_drag > 0


def test_full_flip_turnover():
    # Strategy flips 100% A -> 100% B at the second rebalance. Trade is
    # |0-1| + |1-0| = 2.0 notional -> one-sided turnover 1.0.
    calls = {"n": 0}

    def flip(window_df: pd.DataFrame) -> pd.Series:
        calls["n"] += 1
        w = [1.0, 0.0] if calls["n"] == 1 else [0.0, 1.0]
        return pd.Series(w, index=window_df.columns)

    returns = frame([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    result = run_backtest(returns, flip, window=2, step=1, cost_bps=0.0)
    # rebalances at t=2 and t=3: build (turnover 0.5), then flip (1.0)
    assert result.n_rebalances == 2
    assert result.avg_turnover == pytest.approx((0.5 + 1.0) / 2)


# ----------------------------------------------------------------- leakage


def test_weights_fn_sees_exactly_yesterday_and_earlier():
    # THE no-lookahead test. Record every window the engine hands out and
    # check its boundary: a window ending at index[t-1] means the weights
    # first apply to index[t] -- never to a day the function has seen.
    seen: list[pd.DataFrame] = []

    def spy(window_df: pd.DataFrame) -> pd.Series:
        seen.append(window_df)
        return pd.Series(0.5, index=window_df.columns)

    returns = frame([[0.01, 0.0]] * 10)
    result = run_backtest(returns, spy, window=4, step=3, cost_bps=0.0)

    assert len(seen[0]) == 4  # exactly `window` rows
    # First window is rows 0..3; first simulated day is row 4.
    assert seen[0].index[-1] == returns.index[3]
    assert result.returns.index[0] == returns.index[4]
    # Second rebalance at t=7: window is rows 3..6.
    assert seen[1].index[-1] == returns.index[6]


# ------------------------------------------------------------------ guards


def test_history_shorter_than_window_rejected():
    with pytest.raises(BacktestError, match="window\\+1"):
        run_backtest(frame([[0.01, 0.0]] * 5), fifty_fifty, window=5, step=1)


def test_weights_not_summing_to_one_rejected():
    def half_invested(window_df):
        return pd.Series(0.25, index=window_df.columns)  # sums to 0.5

    with pytest.raises(BacktestError, match="sum to 1"):
        run_backtest(frame([[0.01, 0.0]] * 4), half_invested, window=2, step=1)


def test_nan_weights_rejected():
    def broken(window_df):
        return pd.Series([float("nan"), 1.0], index=window_df.columns)

    with pytest.raises(BacktestError, match="NaN"):
        run_backtest(frame([[0.01, 0.0]] * 4), broken, window=2, step=1)


def test_missing_symbol_in_weights_rejected():
    def wrong_symbols(window_df):
        return pd.Series({"A": 1.0})  # forgot B

    with pytest.raises(BacktestError, match=r"\['B'\]"):
        run_backtest(frame([[0.01, 0.0]] * 4), wrong_symbols, window=2, step=1)


def test_nan_returns_rejected():
    data = frame([[0.01, 0.0]] * 4)
    data.iloc[2, 0] = float("nan")
    with pytest.raises(BacktestError, match="NaN"):
        run_backtest(data, fifty_fifty, window=2, step=1)


def test_wipeout_raises():
    # Asset A loses 100% in a day while fully held: equity hits zero.
    def all_in_a(window_df):
        return pd.Series([1.0, 0.0], index=window_df.columns)

    returns = frame([[0.01, 0.0], [0.01, 0.0], [-1.0, 0.0]])
    with pytest.raises(BacktestError, match="wiped out"):
        run_backtest(returns, all_in_a, window=2, step=1)


def test_shorts_are_allowed():
    # 130/-30 sums to 1: leverage via shorting is legal at engine level
    # (the harness relies on this -- unconstrained min-variance shorts).
    def long_short(window_df):
        return pd.Series([1.3, -0.3], index=window_df.columns)

    result = run_backtest(frame([[0.01, 0.005]] * 6), long_short, window=2, step=2, cost_bps=0.0)
    # day return = 1.3*.01 - 0.3*.005 = .0115
    assert result.returns.iloc[0] == pytest.approx(0.0115)


def test_result_stats_are_consistent():
    rng = np.random.default_rng(3)
    data = frame(rng.normal(0.0005, 0.01, size=(300, 2)).tolist())
    result = run_backtest(data, fifty_fifty, window=100, step=20)
    # Equity curve and returns must describe the same history.
    assert result.equity.iloc[-1] == pytest.approx(float((1 + result.returns).prod()))
    assert result.ann_vol > 0
    assert result.sharpe is not None
