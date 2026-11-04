"""Tiny end-to-end demo on a hand-crafted 10-bar dataset.

Run with:
    .venv/bin/python3 tiny_demo.py

This script uses prices so simple that you can compute every fast/slow MA,
signal, entry, exit, and profit value on paper and check the program's
output matches.

Hand computation reference (fast=2, slow=3, no .shift on MAs):

    fast_ma[i] = mean(close[i-1], close[i])
    slow_ma[i] = mean(close[i-2], close[i-1], close[i])

    bar | close | fast_ma  | slow_ma   | relation       | signal
    ----|-------|----------|-----------|----------------|----------
     0  |  100  |   NaN    |   NaN     | -              | None
     1  |  100  |  100.00  |   NaN     | -              | None
     2  |  100  |  100.00  |  100.00   | tied           | None
     3  |  105  |  102.50  |  101.67   | fast > slow    | LONG ←  bullish cross
     4  |  110  |  107.50  |  105.00   | fast > slow    | None
     5  |  115  |  112.50  |  110.00   | fast > slow    | None
     6  |  118  |  116.50  |  114.33   | fast > slow    | None
     7  |  110  |  114.00  |  114.33   | fast < slow    | SHORT ← bearish cross
     8  |  100  |  105.00  |  109.33   | fast < slow    | None
     9  |  100  |  100.00  |  103.33   | fast < slow    | None

Hand-computed execution (no stops/targets):
    bar | event                         | price
    ----|-------------------------------|-------
     4  | LONG entry @ open[4]          | 105
     8  | LONG exit (signal_change)     | 110   ← profit = +5
     8  | SHORT entry @ open[8]         | 110
     9  | SHORT still open at end       | -

Expected closed trades:
    1) LONG  entry@105 exit@110 (signal_change)  profit = +5
Expected open trades:
    2) SHORT entry@110 (no exit yet)
"""

from __future__ import annotations

import pandas as pd

from bt.backtest import apply_execution
from bt.portfolio import equity_curve, final_equity, total_return_pct
from bt.strategy import MaCrossoverStrategy, Position
from bt.trade import extract_trades


INITIAL_EQUITY = 10_000.0


CLOSES = [100, 100, 100, 105, 110, 115, 118, 110, 100, 100]
OPENS = [100] + CLOSES[:-1]  # open[i] = close[i-1] — typical no-gap market

EXPECTED_SIGNALS = {3: Position.LONG, 7: Position.SHORT}
EXPECTED_TRADES = [
    {"side": Position.LONG, "entry": 105.0, "exit": 110.0, "reason": "signal_change", "profit": 5.0},
    {"side": Position.SHORT, "entry": 110.0, "exit": None, "reason": None, "profit": None},
]


def build_candles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [f"2026-01-01T{i:02d}:00:00" for i in range(len(CLOSES))],
            "open": OPENS,
            "high": [max(o, c) for o, c in zip(OPENS, CLOSES)],
            "low": [min(o, c) for o, c in zip(OPENS, CLOSES)],
            "close": CLOSES,
            "volume": [1000] * len(CLOSES),
        }
    )


def main() -> None:
    df = build_candles()

    strat = MaCrossoverStrategy(fast=2, slow=3)
    df = strat.apply(df)
    df = apply_execution(df, strat)
    trades = extract_trades(df)

    print("=== Candles + indicators + signals + execution ===\n")
    cols = ["timestamp", "open", "close", "fast_ma", "slow_ma",
            "signal", "position", "in_position",
            "entry_price", "exit_price", "exit_reason"]
    print(df[cols].to_string(index=True))

    print("\n=== Trades ===\n")
    for i, t in enumerate(trades, 1):
        if t.is_closed:
            print(
                f"{i}) {t.side} entry@{t.entry_price} exit@{t.exit_price} "
                f"({t.exit_reason}) profit={t.profit:+.2f}"
            )
        else:
            print(f"{i}) {t.side} entry@{t.entry_price} (still open)")

    print("\n=== Equity (starting from $10,000) ===\n")
    curve = equity_curve(trades, initial_equity=INITIAL_EQUITY)
    for point in curve:
        print(f"  {point.timestamp:>20}  ${point.equity:,.2f}")
    print(
        f"\n  Final equity: ${final_equity(trades, INITIAL_EQUITY):,.2f}"
        f"  ({total_return_pct(trades, INITIAL_EQUITY) * 100:+.2f}%)"
    )

    _verify(df, trades)
    print("\n✓ All hand-computed expectations match the program output.")


def _verify(df: pd.DataFrame, trades: list) -> None:
    for i, expected in EXPECTED_SIGNALS.items():
        actual = df["signal"].iloc[i]
        assert actual is expected, f"row {i}: expected signal {expected}, got {actual}"
    for i in range(len(df)):
        if i not in EXPECTED_SIGNALS:
            assert df["signal"].iloc[i] is None, f"row {i}: expected no signal"

    assert len(trades) == len(EXPECTED_TRADES), (
        f"expected {len(EXPECTED_TRADES)} trades, got {len(trades)}"
    )
    for actual, expected in zip(trades, EXPECTED_TRADES):
        assert actual.side == expected["side"]
        assert actual.entry_price == expected["entry"]
        assert actual.exit_price == expected["exit"]
        assert actual.exit_reason == expected["reason"]
        if expected["profit"] is None:
            assert actual.profit is None
        else:
            assert actual.profit == expected["profit"]


if __name__ == "__main__":
    main()
