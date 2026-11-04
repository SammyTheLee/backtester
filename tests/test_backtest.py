"""Tests for `bt.backtest`."""

from __future__ import annotations

import pandas as pd

from bt.backtest import apply_execution
from bt.strategy import FixedStopTargetStrategy, MaCrossoverStrategy, Position


def _make_candles(closes: list[float], opens: list[float] | None = None) -> pd.DataFrame:
    if opens is None:
        opens = [closes[0]] + closes[:-1]
    assert len(opens) == len(closes)
    return pd.DataFrame(
        {
            "timestamp": [f"2026-01-01T{i:02d}:00:00" for i in range(len(closes))],
            "open": opens,
            "high": [max(o, c) for o, c in zip(opens, closes)],
            "low": [min(o, c) for o, c in zip(opens, closes)],
            "close": closes,
            "volume": [1000] * len(closes),
        }
    )


# ---------- Entry timing ----------------------------------------------------


def test_entry_fills_at_next_bar_open():
    """When signal[N] flips, the entry fills at open[N+1]."""
    closes = [100, 100, 100, 105, 110, 110, 110, 105, 100, 100, 100, 100]
    df = _make_candles(closes)

    strat = MaCrossoverStrategy(fast=2, slow=3)
    df = strat.apply(df)
    df = apply_execution(df, FixedStopTargetStrategy())

    # Bullish cross detected at row 3 → entry at open[4].
    assert df["signal"].iloc[3] is Position.LONG
    assert pd.isna(df["entry_price"].iloc[3])
    assert df["entry_price"].iloc[4] == df["open"].iloc[4]
    assert df["position"].iloc[4] is Position.LONG

    # Bearish cross detected at row 7 → exit-and-flip at open[8].
    assert df["signal"].iloc[7] is Position.SHORT
    assert df["entry_price"].iloc[8] == df["open"].iloc[8]
    assert df["position"].iloc[8] is Position.SHORT


# ---------- Stateful position behavior --------------------------------------


def test_in_position_flag_mirrors_position_column():
    """`in_position` is True iff `position` is non-None."""
    closes = [100, 100, 100, 105, 110, 110, 110, 105, 100, 100, 100, 100]
    df = _make_candles(closes)

    df = MaCrossoverStrategy(fast=2, slow=3).apply(df)
    df = apply_execution(df, FixedStopTargetStrategy())

    for i in range(len(df)):
        position = df["position"].iloc[i]
        in_position = df["in_position"].iloc[i]
        assert in_position == (position is not None), f"row {i}: mismatch"

    # Concretely: flat 0..3, LONG 4..7, SHORT 8..11.
    assert df["in_position"].tolist() == (
        [False] * 4 + [True] * 4 + [True] * 4
    )


def test_position_persists_until_opposing_signal():
    """Once entered, position holds across bars until an opposing signal flips it."""
    closes = [100, 100, 100, 105, 110, 110, 110, 105, 100, 100, 100, 100]
    df = _make_candles(closes)

    df = MaCrossoverStrategy(fast=2, slow=3).apply(df)
    df = apply_execution(df, FixedStopTargetStrategy())

    for i in range(0, 4):
        assert df["position"].iloc[i] is None, f"row {i}: expected flat"
    for i in range(4, 8):
        assert df["position"].iloc[i] is Position.LONG, f"row {i}: expected LONG"
    for i in range(8, 12):
        assert df["position"].iloc[i] is Position.SHORT, f"row {i}: expected SHORT"


def test_opposing_signal_exits_with_signal_change_reason():
    """A LONG → SHORT flip records a signal_change exit at the flip bar's open."""
    closes = [100, 100, 100, 105, 110, 110, 110, 105, 100, 100, 100, 100]
    df = _make_candles(closes)

    df = MaCrossoverStrategy(fast=2, slow=3).apply(df)
    df = apply_execution(df, FixedStopTargetStrategy())

    # The bullish cross has no opposing signal yet → no signal_change exit until row 8.
    assert df["exit_reason"].iloc[8] == "signal_change"
    assert df["exit_price"].iloc[8] == df["open"].iloc[8]


# ---------- Stop / target rules ---------------------------------------------


def test_target_hit_on_entry_bar_long():
    """LONG entry at open[N+1]; bar N+1 high reaches target → exit at target."""
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

    assert df["position"].iloc[2] is Position.LONG
    assert df["entry_price"].iloc[2] == 100.0
    assert df["stop_price"].iloc[2] == 98.0
    assert df["target_price"].iloc[2] == 104.0
    assert df["exit_price"].iloc[2] == 104.0
    assert df["exit_reason"].iloc[2] == "target"


def test_stop_hit_on_entry_bar_long():
    """LONG entry; bar N+1 low pierces stop → exit at stop."""
    df = pd.DataFrame(
        {
            "timestamp": [f"2026-01-01T{i:02d}:00:00" for i in range(3)],
            "open":   [100.0, 100.0, 100.0],
            "high":   [100.0, 100.0, 100.5],
            "low":    [100.0, 100.0,  97.0],
            "close":  [100.0, 100.0,  98.5],
            "volume": [1000, 1000, 1000],
            "signal": [None, Position.LONG, None],
        }
    )

    df = apply_execution(
        df,
        FixedStopTargetStrategy(stop_pct=0.02, target_pct=0.04),
    )

    assert df["exit_price"].iloc[2] == 98.0
    assert df["exit_reason"].iloc[2] == "stop"


def test_target_hit_on_entry_bar_short():
    """SHORT entry; bar N+1 low reaches target → exit at target."""
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

    assert df["position"].iloc[2] is Position.SHORT
    assert df["stop_price"].iloc[2] == 102.0
    assert df["target_price"].iloc[2] == 96.0
    assert df["exit_price"].iloc[2] == 96.0
    assert df["exit_reason"].iloc[2] == "target"


def test_stop_hit_on_entry_bar_short():
    """SHORT entry; bar N+1 high pierces stop → exit at stop."""
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

    assert df["exit_price"].iloc[2] == 102.0
    assert df["exit_reason"].iloc[2] == "stop"


def test_no_exit_when_neither_level_touched():
    df = pd.DataFrame(
        {
            "timestamp": [f"2026-01-01T{i:02d}:00:00" for i in range(3)],
            "open":   [100.0, 100.0, 100.0],
            "high":   [100.0, 100.0, 102.0],
            "low":    [100.0, 100.0,  99.0],
            "close":  [100.0, 100.0, 101.0],
            "volume": [1000, 1000, 1000],
            "signal": [None, Position.LONG, None],
        }
    )

    df = apply_execution(
        df,
        FixedStopTargetStrategy(stop_pct=0.02, target_pct=0.04),
    )

    assert df["position"].iloc[2] is Position.LONG
    assert pd.isna(df["exit_price"].iloc[2])
    assert df["exit_reason"].iloc[2] is None


def test_stop_takes_precedence_when_both_hit():
    """If both stop and target are inside the bar's range, assume stop hit first."""
    df = pd.DataFrame(
        {
            "timestamp": [f"2026-01-01T{i:02d}:00:00" for i in range(3)],
            "open":   [100.0, 100.0, 100.0],
            "high":   [100.0, 100.0, 110.0],
            "low":    [100.0, 100.0,  90.0],
            "close":  [100.0, 100.0, 100.0],
            "volume": [1000, 1000, 1000],
            "signal": [None, Position.LONG, None],
        }
    )

    df = apply_execution(
        df,
        FixedStopTargetStrategy(stop_pct=0.02, target_pct=0.04),
    )

    assert df["exit_price"].iloc[2] == 98.0
    assert df["exit_reason"].iloc[2] == "stop"
