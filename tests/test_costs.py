"""Tests for trading costs (fees, slippage, holding)."""

from __future__ import annotations

import pandas as pd
import pytest

from bt.backtest import apply_execution
from bt.costs import (
    ExecutionCosts,
    entry_fill_price,
    exit_fill_price,
    holding_fee,
    trading_fee,
)
from bt.strategy import FixedStopTargetStrategy, Position
from bt.trade import extract_trades


def test_entry_slippage_long_and_short():
    assert entry_fill_price(Position.LONG, 100.0, 10) == pytest.approx(100.1)
    assert entry_fill_price(Position.SHORT, 100.0, 10) == pytest.approx(99.9)


def test_exit_slippage_long_and_short():
    assert exit_fill_price(Position.LONG, 100.0, 10) == pytest.approx(99.9)
    assert exit_fill_price(Position.SHORT, 100.0, 10) == pytest.approx(100.1)


def test_trading_fee_formula():
    assert trading_fee(10_000.0, 10) == pytest.approx(10.0)


def test_apply_execution_long_stop_exit_uses_slippage():
    """Stop fill price is slipped below the stop level for a long."""
    df = pd.DataFrame(
        {
            "timestamp": ["t0", "t1", "t2"],
            "signal": [Position.LONG, None, None],
            "open": [100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 100.0],
            "low": [100.0, 100.0, 97.0],
            "close": [100.0, 100.0, 97.0],
        }
    )
    out = apply_execution(
        df,
        FixedStopTargetStrategy(stop_pct=0.02),
        slippage_bps=10,
    )
    entry = out["entry_price"].iloc[1]
    stop_level = out["stop_price"].iloc[1]
    assert entry == pytest.approx(100.1)
    assert stop_level == pytest.approx(entry * 0.98)
    assert out["exit_price"].iloc[2] == pytest.approx(stop_level * 0.999)


def test_apply_execution_short_signal_exit_uses_slippage():
    """Short covering at the open pays more when slippage is applied."""
    df = pd.DataFrame(
        {
            "timestamp": ["t0", "t1", "t2"],
            "signal": [Position.SHORT, Position.LONG, None],
            "open": [100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 100.0],
            "low": [100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0],
        }
    )
    out = apply_execution(df, FixedStopTargetStrategy(), slippage_bps=10)
    # Short entered at t1 open, flipped long at t2 open.
    assert out["entry_price"].iloc[1] == pytest.approx(99.9)
    assert out["exit_price"].iloc[2] == pytest.approx(100.1)


def test_apply_execution_long_entry_uses_slippage():
    df = pd.DataFrame(
        {
            "timestamp": ["t0", "t1"],
            "signal": [Position.LONG, None],
            "open": [100.0, 200.0],
            "high": [100.0, 200.0],
            "low": [100.0, 200.0],
            "close": [100.0, 200.0],
        }
    )
    out = apply_execution(df, FixedStopTargetStrategy(), slippage_bps=10)
    assert out["entry_price"].iloc[1] == pytest.approx(200.2)


def test_extract_trades_deducts_fees_from_profit():
    df = pd.DataFrame(
        {
            "timestamp": ["t0", "t1"],
            "signal": [Position.LONG, None],
            "open": [100.0, 100.0],
            "high": [100.0, 110.0],
            "low": [100.0, 100.0],
            "close": [100.0, 105.0],
            "position": [None, Position.LONG],
            "entry_price": [float("nan"), 100.0],
            "exit_price": [float("nan"), 110.0],
            "exit_reason": [None, "target"],
        }
    )
    costs = ExecutionCosts(fee_bps=10)
    trades = extract_trades(df, costs=costs)
    t = trades[0]
    assert t.gross_pnl == pytest.approx(10.0)
    assert t.slippage_cost == pytest.approx(0.0)
    assert t.fees == pytest.approx(0.21)
    assert t.net_pnl == pytest.approx(9.79)


def test_extract_trades_splits_gross_slippage_and_net():
    df = pd.DataFrame(
        {
            "timestamp": ["t0", "t1", "t2"],
            "signal": [Position.LONG, Position.SHORT, None],
            "open": [100.0, 100.0, 110.0],
            "high": [100.0, 100.0, 110.0],
            "low": [100.0, 100.0, 110.0],
            "close": [100.0, 100.0, 110.0],
        }
    )
    out = apply_execution(df, FixedStopTargetStrategy(), slippage_bps=10)
    trades = extract_trades(out, costs=ExecutionCosts(fee_bps=0))
    t = trades[0]
    assert t.gross_pnl == pytest.approx(10.0)
    assert t.slippage_cost == pytest.approx(0.21)
    assert t.net_pnl == pytest.approx(9.79)


def test_extract_trades_holding_cost_per_bar():
    df = pd.DataFrame(
        {
            "timestamp": ["t0", "t1", "t2"],
            "signal": [Position.LONG, None, None],
            "open": [100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 110.0],
            "low": [100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 110.0],
            "position": [None, Position.LONG, Position.LONG],
            "entry_price": [float("nan"), 100.0, float("nan")],
            "exit_price": [float("nan"), float("nan"), 110.0],
            "exit_reason": [None, None, "target"],
        }
    )
    costs = ExecutionCosts(holding_cost_bps=10)
    trades = extract_trades(df, costs=costs)
    # Bars 1–2: close 100 and 110 → 0.10 + 0.11 holding
    assert trades[0].holding_cost == pytest.approx(0.21)
