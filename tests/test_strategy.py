"""Tests for `bt.strategy`."""

from __future__ import annotations

import pandas as pd
import pytest

from bt.strategy import (
    FAST_MA_WINDOW,
    SLOW_MA_WINDOW,
    MaCrossoverStrategy,
    Position,
    RegimeGatedMaStrategy,
    RsiStrategy,
    Strategy,
    VolGatedMaStrategy,
    add_fast_ma,
    add_slow_ma,
    add_ma_columns,
    add_rsi,
    apply_signals,
    is_regime_active,
    strategy_from_params,
)


def _make_candles(closes: list[float]) -> pd.DataFrame:
    """Build a minimal candle DataFrame from a list of close prices."""
    return pd.DataFrame(
        {
            "timestamp": [f"2026-01-01T{i:02d}:00:00" for i in range(len(closes))],
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1000] * len(closes),
        }
    )


def test_ma_columns_use_default_windows():
    df = _make_candles([10.0, 20.0, 30.0, 40.0, 50.0])
    out = add_ma_columns(df, fast=2, slow=3)
    assert out.loc[1, "fast_ma"] == 15.0
    assert out.loc[2, "slow_ma"] == 20.0


def test_ma_crossover_satisfies_strategy_protocol():
    strat = MaCrossoverStrategy(fast=5, slow=20)
    assert isinstance(strat, Strategy)


def test_rsi_satisfies_strategy_protocol():
    """Second strategy proves the protocol is not MA-specific."""
    strat = RsiStrategy(period=14, oversold=30, overbought=70)
    assert isinstance(strat, Strategy)
    assert strat.params_dict()["type"] == "rsi"


def test_ma_crossover_implements_strategy_interface():
    strat = MaCrossoverStrategy(fast=2, slow=3, stop_pct=0.02, target_pct=0.04)
    assert strat.params_dict() == {
        "type": "ma_crossover",
        "fast": 2,
        "slow": 3,
        "stop_pct": 0.02,
        "target_pct": 0.04,
    }
    levels = strat.stop_target(Position.LONG, 100.0, None)
    assert levels.stop == 98.0
    assert levels.target == 104.0


def test_apply_signals_matches_apply():
    closes = [100, 100, 100, 105, 110, 110, 110, 105, 100, 100, 100, 100]
    df = _make_candles(closes)
    strat = MaCrossoverStrategy(fast=2, slow=3)
    assert strat.apply(df)["signal"].tolist() == apply_signals(strat, df)["signal"].tolist()


def test_rsi_emits_signal_on_threshold_cross():
    # RSI with period=2 on a short series; craft a cross above oversold (30).
    closes = [100, 100, 100, 95, 90, 88, 92, 96, 100, 100]
    df = _make_candles(closes)
    out = add_rsi(df, period=2)
    strat = RsiStrategy(period=2, oversold=30, overbought=70)
    signals = apply_signals(strat, out)["signal"].tolist()
    assert any(s is Position.LONG for s in signals) or any(
        s is Position.SHORT for s in signals
    )


def test_rsi_strategy_from_config():
    config = {"strategy": {"type": "rsi"}}
    combo = {
        "period": 14,
        "oversold": 30,
        "overbought": 70,
        "stop_pct": 0.02,
        "target_pct": None,
    }
    strat = strategy_from_params(config, combo)
    assert isinstance(strat, RsiStrategy)
    assert strat.params_dict()["type"] == "rsi"


def test_vol_gated_ma_blocks_signals_when_vol_low():
    closes = [100, 100, 100, 105, 110, 110, 110, 105, 100, 100, 100, 100]
    df = _make_candles(closes)
    plain = MaCrossoverStrategy(fast=2, slow=3)
    gated = VolGatedMaStrategy(fast=2, slow=3, vol_window=2, vol_min=1.0)
    plain_signals = apply_signals(plain, df)["signal"].tolist()
    gated_signals = apply_signals(gated, df)["signal"].tolist()
    assert any(s is not None for s in plain_signals)
    assert all(s is None for s in gated_signals)


def test_regime_gated_ma_requires_column():
    df = _make_candles([100, 101, 102, 103])
    with pytest.raises(ValueError, match="risk_on"):
        RegimeGatedMaStrategy(fast=2, slow=3).add_indicators(df)


def test_regime_gated_ma_only_signals_when_regime_on():
    closes = [100, 100, 100, 105, 110, 110, 110, 105, 100, 100, 100, 100]
    df = _make_candles(closes)
    df["risk_on"] = 0
    df.loc[3, "risk_on"] = 1
    df.loc[7, "risk_on"] = 1
    strat = RegimeGatedMaStrategy(fast=2, slow=3)
    signals = apply_signals(strat, df)["signal"].tolist()
    assert signals[3] is Position.LONG
    assert signals[7] is Position.SHORT


def test_is_regime_active():
    assert is_regime_active(1)
    assert is_regime_active("yes")
    assert not is_regime_active(0)
    assert not is_regime_active("off")


def test_ma_crossover_emits_signal_only_at_cross():
    """Signals fire only at the bar where fast crosses slow; otherwise None.

    With fast(2), slow(3), shifted-window MAs and closes:
        [100, 100, 100, 105, 110, 110, 110, 105, 100, 100, 100, 100]

        row | fast_ma | slow_ma | relation       | signal
        ----|---------|---------|----------------|--------
         0  |   NaN   |   NaN   | -              | None
         1  |  100.00 |   NaN   | -              | None
         2  |  100.00 |  100.00 | tied           | None
         3  |  102.50 |  101.67 | fast > slow    | LONG    ← bullish cross
         4  |  107.50 |  105.00 | fast > slow    | None    (no new cross)
         5  |  110.00 |  108.33 | fast > slow    | None
         6  |  110.00 |  110.00 | tied           | None
         7  |  107.50 |  108.33 | fast < slow    | SHORT   ← bearish cross
         8  |  102.50 |  105.00 | fast < slow    | None
         9  |  100.00 |  101.67 | fast < slow    | None
        10  |  100.00 |  100.00 | tied           | None
        11  |  100.00 |  100.00 | tied           | None
    """
    closes = [100, 100, 100, 105, 110, 110, 110, 105, 100, 100, 100, 100]
    df = _make_candles(closes)

    strat = MaCrossoverStrategy(fast=2, slow=3)
    signals = strat.apply(df)["signal"].tolist()

    expected: dict[int, Position] = {3: Position.LONG, 7: Position.SHORT}

    for i, sig in enumerate(signals):
        if i in expected:
            assert sig is expected[i], f"row {i}: expected {expected[i]}, got {sig}"
        else:
            assert sig is None, f"row {i}: expected None, got {sig}"
