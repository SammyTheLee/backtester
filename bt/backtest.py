"""Trade execution layer.

Converts a sparse `signal` column produced by a strategy into stateful
execution columns. Positions persist across bars until exited.

Timing rules:
    1. `signal[N]` is detected at the close of bar N.
    2. Entries (and any opposing-signal exits) fill at the open of bar N+1.
    3. Stops and targets are checked using each bar's high and low,
       starting from the entry bar (= bar N+1).
    4. If both stop and target are inside the same bar's range, the stop
       is assumed to hit first (conservative).

Slippage (when slippage_bps > 0):
    slippage_rate = slippage_bps / 10000
    long entry  = open * (1 + slippage_rate)
    short entry = open * (1 - slippage_rate)
    long exit   = raw_exit * (1 - slippage_rate)   # sell worse
    short exit  = raw_exit * (1 + slippage_rate)   # cover worse

    `entry_price` and `exit_price` columns store the slipped fills.
    Stops/targets are derived from the slipped entry price.

Exit reasons:
    - "stop"           → bar's range pierced the stop level
    - "target"         → bar's range reached the take-profit level
    - "signal_change"  → opposing signal opened a new position next bar
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from bt.costs import entry_fill_price, exit_fill_price
from bt.strategy import OpenPosition, Strategy


def apply_execution(
    df: pd.DataFrame,
    strategy: Strategy,
    *,
    slippage_bps: float = 0.0,
) -> pd.DataFrame:
    """Return a copy of `df` with execution columns added.

    Required input columns: `signal`, `open`, `high`, `low`.

    Output columns added:
        position      : position held during bar N (None when flat)
        in_position   : True if `position` is non-None on bar N
        entry_price   : slipped entry fill at bar N's open
        entry_open    : raw open before entry slippage (same bar as entry_price)
        stop_price    : stop level set at the entry of the active trade
        target_price  : take-profit level set at the entry of the active trade
        exit_price    : slipped exit fill if the trade closed on bar N
        exit_raw      : raw exit level before exit slippage
        exit_reason   : "stop" | "target" | "signal_change", else None
    """
    _validate(df, strategy, slippage_bps)

    out = df.copy()
    n = len(out)

    position_col: list[Optional[object]] = [None] * n
    entry_price_col: list[float] = [float("nan")] * n
    entry_open_col: list[float] = [float("nan")] * n
    stop_col: list[float] = [float("nan")] * n
    target_col: list[float] = [float("nan")] * n
    exit_price_col: list[float] = [float("nan")] * n
    exit_raw_col: list[float] = [float("nan")] * n
    exit_reason_col: list[Optional[str]] = [None] * n

    current_position: OpenPosition | None = None

    for i in range(n):
        bar_position = current_position.side if current_position is not None else None
        prev_row = out.iloc[i - 1] if i > 0 else None

        # 1. Act on the previous bar's signal at this bar's open.
        prev_signal = _prev_signal(out, i)
        if prev_signal is not None:
            open_price = float(out["open"].iloc[i])

            if (
                current_position is not None
                and current_position.side != prev_signal
            ):
                exit_raw_col[i] = open_price
                exit_price_col[i] = exit_fill_price(
                    current_position.side, open_price, slippage_bps
                )
                exit_reason_col[i] = "signal_change"
                current_position = None

            if current_position is None:
                fill = entry_fill_price(prev_signal, open_price, slippage_bps)
                levels = strategy.stop_target(prev_signal, fill, prev_row)
                current_position = OpenPosition(
                    side=prev_signal,
                    entry_price=fill,
                    stop=levels.stop,
                    target=levels.target,
                )
                entry_open_col[i] = open_price
                entry_price_col[i] = fill
                bar_position = current_position.side
                if current_position.stop is not None:
                    stop_col[i] = current_position.stop
                if current_position.target is not None:
                    target_col[i] = current_position.target

        # 2. Update live position (trailing stops, etc.) then check stop/target.
        if current_position is not None and pd.isna(exit_price_col[i]):
            row = out.iloc[i]
            current_position = strategy.update_position(current_position, row)
            hit_price, hit_reason = _check_hit(
                position=current_position.side,
                high=float(out["high"].iloc[i]),
                low=float(out["low"].iloc[i]),
                stop=current_position.stop,
                target=current_position.target,
            )
            if hit_price is not None:
                exit_raw_col[i] = hit_price
                exit_price_col[i] = exit_fill_price(
                    current_position.side, hit_price, slippage_bps
                )
                exit_reason_col[i] = hit_reason
                current_position = None

        position_col[i] = bar_position

    out["position"] = position_col
    out["in_position"] = [p is not None for p in position_col]
    out["entry_price"] = entry_price_col
    out["entry_open"] = entry_open_col
    out["stop_price"] = stop_col
    out["target_price"] = target_col
    out["exit_price"] = exit_price_col
    out["exit_raw"] = exit_raw_col
    out["exit_reason"] = exit_reason_col
    return out


def _prev_signal(df: pd.DataFrame, i: int):
    if i == 0:
        return None
    sig = df["signal"].iloc[i - 1]
    if pd.isna(sig):
        return None
    return sig


def _validate(
    df: pd.DataFrame,
    strategy: Strategy,
    slippage_bps: float,
) -> None:
    del strategy
    for col in ("signal", "open"):
        if col not in df.columns:
            raise ValueError(f"Input DataFrame must have a '{col}' column.")
    if not {"high", "low"}.issubset(df.columns):
        raise ValueError(
            "Input DataFrame must have 'high' and 'low' columns for stop/target checks."
        )
    if slippage_bps < 0:
        raise ValueError("slippage_bps must be >= 0")


def _check_hit(
    position,
    high: float,
    low: float,
    stop: Optional[float],
    target: Optional[float],
) -> tuple[Optional[float], Optional[str]]:
    """If the bar's range crossed stop or target, return (exit_price, reason).

    When both stop and target are inside the bar's range, we conservatively
    assume the stop hit first (worst case for the trader).
    """
    from bt.strategy import Position

    if position == Position.LONG:
        stop_hit = stop is not None and low <= stop
        target_hit = target is not None and high >= target
        if stop_hit:
            return stop, "stop"
        if target_hit:
            return target, "target"
    else:  # SHORT
        stop_hit = stop is not None and high >= stop
        target_hit = target is not None and low <= target
        if stop_hit:
            return stop, "stop"
        if target_hit:
            return target, "target"
    return None, None
