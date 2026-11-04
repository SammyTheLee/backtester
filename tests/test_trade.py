"""Tests for `bt.trade`."""

from __future__ import annotations

import pandas as pd
import pytest

from bt.backtest import apply_execution
from bt.strategy import FixedStopTargetStrategy
from bt.strategy import Position
from bt.trade import Trade, extract_trades, save_trades


def test_trade_long_profit_positive():
    t = Trade(
        entry_time="2026-01-01T00:00:00",
        entry_price=100.0,
        side=Position.LONG,
        exit_time="2026-01-01T01:00:00",
        exit_price=110.0,
        exit_reason="target",
    )
    assert t.is_closed
    assert t.profit == 10.0
    assert t.profit_pct == pytest.approx(0.10)


def test_trade_long_profit_negative():
    t = Trade(
        entry_time="2026-01-01T00:00:00",
        entry_price=100.0,
        side=Position.LONG,
        exit_time="2026-01-01T01:00:00",
        exit_price=98.0,
        exit_reason="stop",
    )
    assert t.profit == -2.0
    assert t.profit_pct == pytest.approx(-0.02)


def test_trade_short_profit_positive():
    """A short profits when the price falls."""
    t = Trade(
        entry_time="2026-01-01T00:00:00",
        entry_price=100.0,
        side=Position.SHORT,
        exit_time="2026-01-01T01:00:00",
        exit_price=96.0,
        exit_reason="target",
    )
    assert t.profit == 4.0
    assert t.profit_pct == pytest.approx(0.04)


def test_trade_short_profit_negative():
    """A short loses when the price rises."""
    t = Trade(
        entry_time="2026-01-01T00:00:00",
        entry_price=100.0,
        side=Position.SHORT,
        exit_time="2026-01-01T01:00:00",
        exit_price=102.0,
        exit_reason="stop",
    )
    assert t.profit == -2.0
    assert t.profit_pct == pytest.approx(-0.02)


def test_trade_size_scales_profit():
    """Doubling size doubles profit; profit_pct is size-independent."""
    base = Trade(
        entry_time="t0",
        entry_price=100.0,
        side=Position.LONG,
        size=1.0,
        exit_time="t1",
        exit_price=110.0,
        exit_reason="target",
    )
    big = Trade(
        entry_time="t0",
        entry_price=100.0,
        side=Position.LONG,
        size=10.0,
        exit_time="t1",
        exit_price=110.0,
        exit_reason="target",
    )

    assert base.profit == 10.0
    assert big.profit == 100.0

    assert base.profit_pct == big.profit_pct == pytest.approx(0.10)


def test_trade_default_size_is_one():
    t = Trade(entry_time="t0", entry_price=100.0, side=Position.LONG)
    assert t.size == 1.0


def test_open_trade_has_none_profit():
    t = Trade(
        entry_time="2026-01-01T00:00:00",
        entry_price=100.0,
        side=Position.LONG,
    )
    assert not t.is_closed
    assert t.profit is None
    assert t.profit_pct is None


def test_trade_is_immutable():
    """@dataclass(frozen=True) means you can't accidentally rewrite history."""
    t = Trade(
        entry_time="2026-01-01T00:00:00",
        entry_price=100.0,
        side=Position.LONG,
    )
    with pytest.raises(Exception):
        t.entry_price = 200.0  # type: ignore[misc]


def test_extract_trades_target_hit_long():
    """A target hit on the entry bar produces one closed LONG trade."""
    df = pd.DataFrame(
        {
            "timestamp": [f"2026-01-01T{i:02d}:00:00" for i in range(3)],
            "open":   [100.0, 100.0, 100.0],
            "high":   [100.0, 100.0, 110.0],
            "low":    [100.0, 100.0,  99.5],
            "close":  [100.0, 100.0, 105.0],
            "volume": [1000, 1000, 1000],
            "signal": [None, Position.LONG, None],
        }
    )
    df = apply_execution(
        df,
        FixedStopTargetStrategy(stop_pct=0.02, target_pct=0.04),
    )

    trades = extract_trades(df)

    assert len(trades) == 1
    trade = trades[0]
    assert trade.side == Position.LONG
    assert trade.entry_price == 100.0
    assert trade.exit_price == 104.0
    assert trade.exit_reason == "target"
    assert trade.entry_time == "2026-01-01T02:00:00"
    assert trade.exit_time == "2026-01-01T02:00:00"
    assert trade.size == 1.0
    assert trade.profit == 4.0


def test_extract_trades_target_hit_short():
    """A target hit on the entry bar produces one closed SHORT trade with positive profit."""
    df = pd.DataFrame(
        {
            "timestamp": [f"2026-01-01T{i:02d}:00:00" for i in range(3)],
            "open":   [100.0, 100.0, 100.0],
            "high":   [100.0, 100.0, 101.0],
            "low":    [100.0, 100.0,  90.0],
            "close":  [100.0, 100.0,  95.0],
            "volume": [1000, 1000, 1000],
            "signal": [None, Position.SHORT, None],
        }
    )
    df = apply_execution(
        df,
        FixedStopTargetStrategy(stop_pct=0.02, target_pct=0.04),
    )

    trades = extract_trades(df)

    assert len(trades) == 1
    trade = trades[0]
    assert trade.side == Position.SHORT
    assert trade.entry_price == 100.0
    assert trade.exit_price == 96.0
    assert trade.exit_reason == "target"
    assert trade.profit == 4.0  # (100 - 96) * 1
    assert trade.profit_pct == pytest.approx(0.04)


def test_extract_trades_stop_hit_short():
    """A stop hit on the entry bar produces one closed SHORT trade with negative profit."""
    df = pd.DataFrame(
        {
            "timestamp": [f"2026-01-01T{i:02d}:00:00" for i in range(3)],
            "open":   [100.0, 100.0, 100.0],
            "high":   [100.0, 100.0, 103.0],
            "low":    [100.0, 100.0,  99.5],
            "close":  [100.0, 100.0, 101.5],
            "volume": [1000, 1000, 1000],
            "signal": [None, Position.SHORT, None],
        }
    )
    df = apply_execution(
        df,
        FixedStopTargetStrategy(stop_pct=0.02, target_pct=0.04),
    )

    trades = extract_trades(df)

    assert len(trades) == 1
    trade = trades[0]
    assert trade.side == Position.SHORT
    assert trade.exit_reason == "stop"
    assert trade.profit == -2.0  # (100 - 102) * 1
    assert trade.profit_pct == pytest.approx(-0.02)


def test_extract_trades_applies_size():
    df = pd.DataFrame(
        {
            "timestamp": [f"2026-01-01T{i:02d}:00:00" for i in range(3)],
            "open":   [100.0, 100.0, 100.0],
            "high":   [100.0, 100.0, 110.0],
            "low":    [100.0, 100.0,  99.5],
            "close":  [100.0, 100.0, 105.0],
            "volume": [1000, 1000, 1000],
            "signal": [None, Position.LONG, None],
        }
    )
    df = apply_execution(
        df,
        FixedStopTargetStrategy(stop_pct=0.02, target_pct=0.04),
    )

    trades = extract_trades(df, size=5.0)

    assert trades[0].size == 5.0
    assert trades[0].profit == 20.0  # (104 - 100) * 5


def test_extract_trades_open_when_no_exit():
    """Without stop/target, an entry has no exit yet → returns an open trade."""
    df = pd.DataFrame(
        {
            "timestamp": [f"2026-01-01T{i:02d}:00:00" for i in range(3)],
            "open":   [100.0, 100.0, 100.0],
            "high":   [100.0, 100.0, 101.0],
            "low":    [100.0, 100.0,  99.0],
            "close":  [100.0, 100.0, 100.5],
            "volume": [1000, 1000, 1000],
            "signal": [None, Position.LONG, None],
        }
    )
    df = apply_execution(df, FixedStopTargetStrategy())

    trades = extract_trades(df)

    assert len(trades) == 1
    trade = trades[0]
    assert not trade.is_closed
    assert trade.entry_price == 100.0
    assert trade.exit_price is None
    assert trade.profit is None


def test_extract_trades_missing_columns_raises():
    df = pd.DataFrame({"timestamp": ["t"], "close": [100.0]})
    with pytest.raises(ValueError, match="missing required columns"):
        extract_trades(df)


def test_save_trades_writes_csv(tmp_path):
    """save_trades writes a CSV with one row per trade and the expected columns."""
    trades = [
        Trade(
            entry_time="2026-01-01T00:00:00",
            entry_price=100.0,
            side=Position.LONG,
            size=2.0,
            exit_time="2026-01-01T01:00:00",
            exit_price=110.0,
            exit_reason="target",
        ),
        Trade(
            entry_time="2026-01-01T02:00:00",
            entry_price=110.0,
            side=Position.SHORT,
        ),
    ]

    path = tmp_path / "trades.csv"
    n = save_trades(trades, path)
    assert n == 2
    assert path.exists()

    loaded = pd.read_csv(path)

    expected_columns = [
        "entry_time",
        "entry_price",
        "entry_open",
        "side",
        "size",
        "exit_time",
        "exit_price",
        "exit_raw",
        "exit_reason",
        "gross_pnl",
        "slippage_cost",
        "fees",
        "entry_fee",
        "exit_fee",
        "holding_cost",
        "net_pnl",
        "profit_pct",
    ]
    assert list(loaded.columns) == expected_columns
    assert len(loaded) == 2

    assert loaded["side"].iloc[0] == "long"
    assert loaded["entry_price"].iloc[0] == 100.0
    assert loaded["exit_price"].iloc[0] == 110.0
    assert loaded["net_pnl"].iloc[0] == 20.0  # (110-100) * 2
    assert loaded["exit_reason"].iloc[0] == "target"

    assert loaded["side"].iloc[1] == "short"
    assert pd.isna(loaded["exit_time"].iloc[1])
    assert pd.isna(loaded["net_pnl"].iloc[1])


def test_save_trades_creates_parent_dirs(tmp_path):
    """Nested directories are created on demand."""
    trades = [
        Trade(entry_time="t", entry_price=100.0, side=Position.LONG)
    ]
    path = tmp_path / "deep" / "nested" / "trades.csv"
    save_trades(trades, path)
    assert path.exists()


def test_save_trades_writes_csv(tmp_path):
    path = tmp_path / "trades.csv"
    trades = [
        Trade(
            entry_time="t0",
            entry_price=100.0,
            side=Position.LONG,
            exit_time="t1",
            exit_price=110.0,
            exit_reason="target",
        )
    ]
    n = save_trades(trades, path)
    assert n == 1
    assert path.exists()


def test_save_trades_empty_list_writes_header_only(tmp_path):
    path = tmp_path / "empty.csv"
    n = save_trades([], path)
    assert n == 0
    assert path.exists()
    loaded = pd.read_csv(path)
    assert len(loaded) == 0
    assert "entry_time" in loaded.columns


def test_extract_trades_rejects_non_positive_size():
    df = pd.DataFrame(
        {
            "timestamp": ["t"],
            "position": [None],
            "entry_price": [float("nan")],
            "exit_price": [float("nan")],
            "exit_reason": [None],
        }
    )
    with pytest.raises(ValueError, match="size must be positive"):
        extract_trades(df, size=0)
