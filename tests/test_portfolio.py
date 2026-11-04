"""Tests for `bt.portfolio`."""

from __future__ import annotations

import pytest

from bt.portfolio import (
    DEFAULT_INITIAL_EQUITY,
    EquityPoint,
    equity_curve,
    final_equity,
    total_return_pct,
)
from bt.strategy import Position
from bt.trade import Trade


def _closed_trade(profit: float, exit_time: str = "t1") -> Trade:
    """Build a closed trade whose `profit` evaluates to the desired number."""
    return Trade(
        entry_time="t0",
        entry_price=100.0,
        side=Position.LONG,
        size=1.0,
        exit_time=exit_time,
        exit_price=100.0 + profit,
        exit_reason="target" if profit >= 0 else "stop",
    )


def _open_trade() -> Trade:
    return Trade(entry_time="t0", entry_price=100.0, side=Position.LONG)


def test_empty_trades_returns_only_starting_equity():
    curve = equity_curve([], initial_equity=10000)
    assert curve == [EquityPoint(timestamp="start", equity=10000)]


def test_default_starting_equity_is_10000():
    assert DEFAULT_INITIAL_EQUITY == 10_000.0
    assert final_equity([]) == 10_000.0


def test_single_winning_trade():
    trades = [_closed_trade(profit=50.0, exit_time="t1")]
    curve = equity_curve(trades, initial_equity=10000)
    assert curve == [
        EquityPoint("start", 10000),
        EquityPoint("t0", 10000),
        EquityPoint("t1", 10050),
    ]
    assert final_equity(trades, 10000) == 10050
    assert total_return_pct(trades, 10000) == pytest.approx(0.005)


def test_single_losing_trade():
    trades = [_closed_trade(profit=-30.0, exit_time="t1")]
    assert final_equity(trades, 10000) == 9970
    assert total_return_pct(trades, 10000) == pytest.approx(-0.003)


def test_multiple_trades_accumulate():
    trades = [
        _closed_trade(profit=100, exit_time="t1"),
        _closed_trade(profit=-40, exit_time="t2"),
        _closed_trade(profit=20, exit_time="t3"),
    ]
    curve = equity_curve(trades, initial_equity=10000)
    assert [p.equity for p in curve] == [
        10000,
        10000,
        10100,
        10100,
        10060,
        10060,
        10080,
    ]


def test_open_trades_only_deduct_entry_fee():
    """Unrealized PnL doesn't move the equity curve; entry fee does."""
    trades = [
        _closed_trade(profit=50, exit_time="t1"),
        _open_trade(),
    ]
    curve = equity_curve(trades, initial_equity=10000)
    assert len(curve) == 4  # start + closed entry/exit + open entry
    assert curve[-1].equity == 10050


def test_equity_curve_subtracts_fees_on_entry_and_exit():
    t = Trade(
        entry_time="t0",
        entry_price=100.0,
        side=Position.LONG,
        exit_time="t1",
        exit_price=110.0,
        exit_reason="target",
        gross_pnl=10.0,
        slippage_cost=0.0,
        entry_fee=1.0,
        exit_fee=1.1,
        net_pnl=7.9,
    )
    curve = equity_curve([t], initial_equity=10000)
    assert [p.equity for p in curve] == [
        10000,
        9999.0,   # after entry fee
        10007.9,  # +10 gross - 1.1 exit fee
    ]


def test_rejects_non_positive_initial_equity():
    with pytest.raises(ValueError, match="initial_equity must be positive"):
        equity_curve([], initial_equity=0)
    with pytest.raises(ValueError, match="initial_equity must be positive"):
        equity_curve([], initial_equity=-100)
