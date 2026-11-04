"""Strategy interface and built-in strategies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable

import pandas as pd

# Default parameters for CLI and configs.
FAST_MA_WINDOW = 10
SLOW_MA_WINDOW = 30
RSI_PERIOD = 14
RSI_OVERSOLD = 30.0
RSI_OVERBOUGHT = 70.0
VOL_WINDOW = 20
VOL_MIN = 0.005
DEFAULT_REGIME_COLUMN = "risk_on"


def add_fast_ma(df: pd.DataFrame, window: int = FAST_MA_WINDOW) -> pd.DataFrame:
    """Return a copy of `df` with a `fast_ma` column added."""
    out = df.copy()
    out["fast_ma"] = out["close"].rolling(window=window).mean()
    return out


def add_slow_ma(df: pd.DataFrame, window: int = SLOW_MA_WINDOW) -> pd.DataFrame:
    """Return a copy of `df` with a `slow_ma` column added."""
    out = df.copy()
    out["slow_ma"] = out["close"].rolling(window=window).mean()
    return out


def add_ma_columns(df: pd.DataFrame, *, fast: int, slow: int) -> pd.DataFrame:
    """Add both `fast_ma` and `slow_ma` columns."""
    return add_slow_ma(add_fast_ma(df, window=fast), window=slow)


def add_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    """Return a copy of `df` with an ``rsi`` column (0–100)."""
    out = df.copy()
    delta = out["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    out["rsi"] = 100 - (100 / (1 + rs))
    return out


def add_rolling_vol(df: pd.DataFrame, window: int = VOL_WINDOW) -> pd.DataFrame:
    """Rolling std of close-to-close returns (volatility proxy)."""
    out = df.copy()
    out["rolling_vol"] = out["close"].pct_change().rolling(window=window).std()
    return out


def is_regime_active(value: Any) -> bool:
    """True when an external regime flag is on (1, true, yes, risk_on, etc.)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) != 0.0
    text = str(value).strip().lower()
    return text in ("1", "true", "yes", "on", "risk_on")


class Position(Enum):
    LONG = "long"
    SHORT = "short"

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return self.value


@dataclass(frozen=True)
class StopTarget:
    """Stop-loss and take-profit levels for an open position."""

    stop: float | None
    target: float | None


@dataclass(frozen=True)
class OpenPosition:
    """Live position state passed to ``update_position`` each bar."""

    side: Position
    entry_price: float
    stop: float | None
    target: float | None


def pct_stop_target(
    side: Position,
    entry_price: float,
    *,
    stop_pct: float | None,
    target_pct: float | None,
) -> StopTarget:
    """Fixed percentage stop and target from entry price."""
    stop: float | None = None
    target: float | None = None
    if stop_pct is not None:
        if side == Position.LONG:
            stop = entry_price * (1 - stop_pct)
        else:
            stop = entry_price * (1 + stop_pct)
    if target_pct is not None:
        if side == Position.LONG:
            target = entry_price * (1 + target_pct)
        else:
            target = entry_price * (1 - target_pct)
    return StopTarget(stop=stop, target=target)


@runtime_checkable
class Strategy(Protocol):
    """Structural interface for pluggable backtest strategies.

    Implementations must provide:
        add_indicators(candles)
        signal(row, previous, prior)
        stop_target(side, entry_price, previous)
        update_position(position, row)
        params_dict()
    """

    @property
    def name(self) -> str:
        """Short human-readable identifier including parameters."""
        ...

    def add_indicators(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Return candles with strategy-specific indicator columns."""
        ...

    def signal(
        self,
        row: pd.Series,
        previous: pd.Series | None,
        prior: pd.Series | None,
    ) -> Position | None:
        """Return a new entry signal at ``row`` (after that bar closes), else None."""
        ...

    def stop_target(
        self,
        side: Position,
        entry_price: float,
        previous: pd.Series | None,
    ) -> StopTarget:
        """Initial stop and target for a new position."""
        ...

    def update_position(
        self,
        position: OpenPosition,
        row: pd.Series,
    ) -> OpenPosition:
        """Update live position state each bar (e.g. trailing stop)."""
        ...

    def params_dict(self) -> dict[str, Any]:
        """Strategy parameters for metadata, sweeps, and logging."""
        ...


def apply_signals(strategy: Strategy, candles: pd.DataFrame) -> pd.DataFrame:
    """Add indicator columns and a sparse ``signal`` column."""
    out = strategy.add_indicators(candles)
    signals: list[Position | None] = []
    for i in range(len(out)):
        row = out.iloc[i]
        previous = out.iloc[i - 1] if i > 0 else None
        prior = out.iloc[i - 2] if i > 1 else None
        signals.append(strategy.signal(row, previous, prior))
    out["signal"] = signals
    return out


class MaCrossoverStrategy:
    """LONG when fast MA crosses above slow; SHORT on bearish cross."""

    def __init__(
        self,
        fast: int = FAST_MA_WINDOW,
        slow: int = SLOW_MA_WINDOW,
        *,
        stop_pct: float | None = None,
        target_pct: float | None = None,
    ) -> None:
        if fast < 1 or slow < 1:
            raise ValueError("fast and slow must be >= 1")
        if fast >= slow:
            raise ValueError(f"fast ({fast}) must be smaller than slow ({slow})")
        self.fast = fast
        self.slow = slow
        self.stop_pct = stop_pct
        self.target_pct = target_pct

    @property
    def name(self) -> str:
        return f"ma_crossover(fast={self.fast}, slow={self.slow})"

    def add_indicators(self, candles: pd.DataFrame) -> pd.DataFrame:
        return add_ma_columns(candles, fast=self.fast, slow=self.slow)

    def signal(
        self,
        row: pd.Series,
        previous: pd.Series | None,
        prior: pd.Series | None,
    ) -> Position | None:
        return _ma_crossover_signal(row, previous, prior)

    def stop_target(
        self,
        side: Position,
        entry_price: float,
        previous: pd.Series | None,
    ) -> StopTarget:
        del previous
        return pct_stop_target(
            side,
            entry_price,
            stop_pct=self.stop_pct,
            target_pct=self.target_pct,
        )

    def update_position(
        self,
        position: OpenPosition,
        row: pd.Series,
    ) -> OpenPosition:
        del row
        return position

    def params_dict(self) -> dict[str, Any]:
        return {
            "type": "ma_crossover",
            "fast": self.fast,
            "slow": self.slow,
            "stop_pct": self.stop_pct,
            "target_pct": self.target_pct,
        }

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience wrapper: indicators + signals."""
        return apply_signals(self, df)


class RsiStrategy:
    """LONG when RSI crosses above oversold; SHORT when RSI crosses below overbought."""

    def __init__(
        self,
        period: int = RSI_PERIOD,
        oversold: float = RSI_OVERSOLD,
        overbought: float = RSI_OVERBOUGHT,
        *,
        stop_pct: float | None = None,
        target_pct: float | None = None,
    ) -> None:
        if period < 2:
            raise ValueError("period must be >= 2")
        if oversold >= overbought:
            raise ValueError(
                f"oversold ({oversold}) must be less than overbought ({overbought})"
            )
        self.period = period
        self.oversold = float(oversold)
        self.overbought = float(overbought)
        self.stop_pct = stop_pct
        self.target_pct = target_pct

    @property
    def name(self) -> str:
        return (
            f"rsi(period={self.period}, "
            f"oversold={self.oversold:g}, overbought={self.overbought:g})"
        )

    def add_indicators(self, candles: pd.DataFrame) -> pd.DataFrame:
        return add_rsi(candles, period=self.period)

    def signal(
        self,
        row: pd.Series,
        previous: pd.Series | None,
        prior: pd.Series | None,
    ) -> Position | None:
        del prior
        if previous is None:
            return None
        return _classify_threshold_cross(
            previous["rsi"],
            row["rsi"],
            low=self.oversold,
            high=self.overbought,
        )

    def stop_target(
        self,
        side: Position,
        entry_price: float,
        previous: pd.Series | None,
    ) -> StopTarget:
        del previous
        return pct_stop_target(
            side,
            entry_price,
            stop_pct=self.stop_pct,
            target_pct=self.target_pct,
        )

    def update_position(
        self,
        position: OpenPosition,
        row: pd.Series,
    ) -> OpenPosition:
        del row
        return position

    def params_dict(self) -> dict[str, Any]:
        return {
            "type": "rsi",
            "period": self.period,
            "oversold": self.oversold,
            "overbought": self.overbought,
            "stop_pct": self.stop_pct,
            "target_pct": self.target_pct,
        }

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return apply_signals(self, df)


class VolGatedMaStrategy:
    """MA crossover only when rolling volatility is at or above ``vol_min``."""

    def __init__(
        self,
        fast: int = FAST_MA_WINDOW,
        slow: int = SLOW_MA_WINDOW,
        *,
        vol_window: int = VOL_WINDOW,
        vol_min: float = VOL_MIN,
        stop_pct: float | None = None,
        target_pct: float | None = None,
    ) -> None:
        if fast < 1 or slow < 1:
            raise ValueError("fast and slow must be >= 1")
        if fast >= slow:
            raise ValueError(f"fast ({fast}) must be smaller than slow ({slow})")
        if vol_window < 2:
            raise ValueError("vol_window must be >= 2")
        if vol_min <= 0:
            raise ValueError("vol_min must be > 0")
        self.fast = fast
        self.slow = slow
        self.vol_window = vol_window
        self.vol_min = float(vol_min)
        self.stop_pct = stop_pct
        self.target_pct = target_pct

    @property
    def name(self) -> str:
        return (
            f"vol_gated_ma(fast={self.fast}, slow={self.slow}, "
            f"vol_window={self.vol_window}, vol_min={self.vol_min:g})"
        )

    def add_indicators(self, candles: pd.DataFrame) -> pd.DataFrame:
        out = add_ma_columns(candles, fast=self.fast, slow=self.slow)
        return add_rolling_vol(out, window=self.vol_window)

    def signal(
        self,
        row: pd.Series,
        previous: pd.Series | None,
        prior: pd.Series | None,
    ) -> Position | None:
        cross = _ma_crossover_signal(row, previous, prior)
        if cross is None:
            return None
        vol = row.get("rolling_vol")
        if vol is None or pd.isna(vol) or float(vol) < self.vol_min:
            return None
        return cross

    def stop_target(
        self,
        side: Position,
        entry_price: float,
        previous: pd.Series | None,
    ) -> StopTarget:
        del previous
        return pct_stop_target(
            side,
            entry_price,
            stop_pct=self.stop_pct,
            target_pct=self.target_pct,
        )

    def update_position(
        self,
        position: OpenPosition,
        row: pd.Series,
    ) -> OpenPosition:
        del row
        return position

    def params_dict(self) -> dict[str, Any]:
        return {
            "type": "vol_gated_ma",
            "fast": self.fast,
            "slow": self.slow,
            "vol_window": self.vol_window,
            "vol_min": self.vol_min,
            "stop_pct": self.stop_pct,
            "target_pct": self.target_pct,
        }


class RegimeGatedMaStrategy:
    """MA crossover only when an external regime column is active (e.g. ``risk_on``)."""

    def __init__(
        self,
        fast: int = FAST_MA_WINDOW,
        slow: int = SLOW_MA_WINDOW,
        *,
        regime_column: str = DEFAULT_REGIME_COLUMN,
        stop_pct: float | None = None,
        target_pct: float | None = None,
    ) -> None:
        if fast < 1 or slow < 1:
            raise ValueError("fast and slow must be >= 1")
        if fast >= slow:
            raise ValueError(f"fast ({fast}) must be smaller than slow ({slow})")
        if not regime_column:
            raise ValueError("regime_column must be non-empty")
        self.fast = fast
        self.slow = slow
        self.regime_column = regime_column
        self.stop_pct = stop_pct
        self.target_pct = target_pct

    @property
    def name(self) -> str:
        return (
            f"regime_gated_ma(fast={self.fast}, slow={self.slow}, "
            f"regime_column={self.regime_column!r})"
        )

    def add_indicators(self, candles: pd.DataFrame) -> pd.DataFrame:
        if self.regime_column not in candles.columns:
            raise ValueError(
                f"regime column {self.regime_column!r} not found in candle CSV; "
                "add it to your data file or change [strategy].regime_column"
            )
        return add_ma_columns(candles, fast=self.fast, slow=self.slow)

    def signal(
        self,
        row: pd.Series,
        previous: pd.Series | None,
        prior: pd.Series | None,
    ) -> Position | None:
        cross = _ma_crossover_signal(row, previous, prior)
        if cross is None:
            return None
        if not is_regime_active(row.get(self.regime_column)):
            return None
        return cross

    def stop_target(
        self,
        side: Position,
        entry_price: float,
        previous: pd.Series | None,
    ) -> StopTarget:
        del previous
        return pct_stop_target(
            side,
            entry_price,
            stop_pct=self.stop_pct,
            target_pct=self.target_pct,
        )

    def update_position(
        self,
        position: OpenPosition,
        row: pd.Series,
    ) -> OpenPosition:
        del row
        return position

    def params_dict(self) -> dict[str, Any]:
        return {
            "type": "regime_gated_ma",
            "fast": self.fast,
            "slow": self.slow,
            "regime_column": self.regime_column,
            "stop_pct": self.stop_pct,
            "target_pct": self.target_pct,
        }


class FixedStopTargetStrategy:
    """Execution helper for tests with a pre-built ``signal`` column."""

    def __init__(
        self,
        *,
        stop_pct: float | None = None,
        target_pct: float | None = None,
    ) -> None:
        self.stop_pct = stop_pct
        self.target_pct = target_pct

    @property
    def name(self) -> str:
        return "fixed_stop_target"

    def add_indicators(self, candles: pd.DataFrame) -> pd.DataFrame:
        return candles.copy()

    def signal(
        self,
        row: pd.Series,
        previous: pd.Series | None,
        prior: pd.Series | None,
    ) -> Position | None:
        del row, previous, prior
        return None

    def stop_target(
        self,
        side: Position,
        entry_price: float,
        previous: pd.Series | None,
    ) -> StopTarget:
        del previous
        return pct_stop_target(
            side,
            entry_price,
            stop_pct=self.stop_pct,
            target_pct=self.target_pct,
        )

    def update_position(
        self,
        position: OpenPosition,
        row: pd.Series,
    ) -> OpenPosition:
        del row
        return position

    def params_dict(self) -> dict[str, Any]:
        return {
            "type": "fixed_stop_target",
            "stop_pct": self.stop_pct,
            "target_pct": self.target_pct,
        }


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def combo_from_row(row: pd.Series, config: dict[str, Any]) -> dict[str, Any]:
    """Rebuild a parameter combo from a sweep/validation/OOS results row."""
    from bt.sweep import resolve_base_params

    combo = resolve_base_params(config)
    float_keys = {"oversold", "overbought", "stop_pct", "target_pct", "vol_min"}
    int_keys = {"fast", "slow", "period", "vol_window"}
    for key, value in combo.items():
        if key not in row.index or pd.isna(row[key]):
            continue
        raw = row[key]
        if key in float_keys:
            combo[key] = float(raw)
        elif key in int_keys:
            combo[key] = int(raw)
        else:
            combo[key] = raw
    return combo


def strategy_type(config: dict[str, Any]) -> str:
    return str(config.get("strategy", {}).get("type", "ma_crossover"))


def strategy_from_params(
    config: dict[str, Any],
    params: dict[str, Any],
) -> Strategy:
    """Build a strategy instance from config and a resolved parameter dict."""
    typ = strategy_type(config)
    if typ == "ma_crossover":
        return MaCrossoverStrategy(
            fast=int(params["fast"]),
            slow=int(params["slow"]),
            stop_pct=_optional_float(params.get("stop_pct")),
            target_pct=_optional_float(params.get("target_pct")),
        )
    if typ == "rsi":
        return RsiStrategy(
            period=int(params["period"]),
            oversold=float(params["oversold"]),
            overbought=float(params["overbought"]),
            stop_pct=_optional_float(params.get("stop_pct")),
            target_pct=_optional_float(params.get("target_pct")),
        )
    if typ == "vol_gated_ma":
        return VolGatedMaStrategy(
            fast=int(params["fast"]),
            slow=int(params["slow"]),
            vol_window=int(params["vol_window"]),
            vol_min=float(params["vol_min"]),
            stop_pct=_optional_float(params.get("stop_pct")),
            target_pct=_optional_float(params.get("target_pct")),
        )
    if typ == "regime_gated_ma":
        return RegimeGatedMaStrategy(
            fast=int(params["fast"]),
            slow=int(params["slow"]),
            regime_column=str(params.get("regime_column", DEFAULT_REGIME_COLUMN)),
            stop_pct=_optional_float(params.get("stop_pct")),
            target_pct=_optional_float(params.get("target_pct")),
        )
    raise ValueError(f"unsupported strategy type {typ!r}")


def validate_strategy_params(typ: str, params: dict[str, Any]) -> None:
    """Raise ValueError when a parameter combination is invalid for ``typ``."""
    if typ == "ma_crossover":
        fast = int(params["fast"])
        slow = int(params["slow"])
        if fast >= slow:
            raise ValueError(
                f"invalid MA windows: fast ({fast}) must be less than slow ({slow})"
            )
        return
    if typ == "rsi":
        oversold = float(params["oversold"])
        overbought = float(params["overbought"])
        if oversold >= overbought:
            raise ValueError(
                f"invalid RSI thresholds: oversold ({oversold}) "
                f"must be less than overbought ({overbought})"
            )
        return
    if typ == "vol_gated_ma":
        fast = int(params["fast"])
        slow = int(params["slow"])
        if fast >= slow:
            raise ValueError(
                f"invalid MA windows: fast ({fast}) must be less than slow ({slow})"
            )
        if float(params["vol_min"]) <= 0:
            raise ValueError("vol_min must be > 0")
        return
    if typ == "regime_gated_ma":
        fast = int(params["fast"])
        slow = int(params["slow"])
        if fast >= slow:
            raise ValueError(
                f"invalid MA windows: fast ({fast}) must be less than slow ({slow})"
            )
        return
    raise ValueError(f"unsupported strategy type {typ!r}")


def supported_strategy_types() -> tuple[str, ...]:
    return ("ma_crossover", "rsi", "vol_gated_ma", "regime_gated_ma")


def _ma_crossover_signal(
    row: pd.Series,
    previous: pd.Series | None,
    prior: pd.Series | None,
) -> Position | None:
    del prior
    if previous is None:
        return None
    return _classify_cross(
        previous["fast_ma"],
        previous["slow_ma"],
        row["fast_ma"],
        row["slow_ma"],
    )


def _classify_cross(
    prev_fast: float, prev_slow: float, fast: float, slow: float
) -> Optional[Position]:
    """Return LONG/SHORT/None depending on whether a cross happened at this bar."""
    if any(pd.isna(v) for v in (prev_fast, prev_slow, fast, slow)):
        return None
    if prev_fast <= prev_slow and fast > slow:
        return Position.LONG
    if prev_fast >= prev_slow and fast < slow:
        return Position.SHORT
    return None


def _classify_threshold_cross(
    prev_value: float,
    value: float,
    *,
    low: float,
    high: float,
) -> Optional[Position]:
    """LONG on cross above ``low``; SHORT on cross below ``high``."""
    if any(pd.isna(v) for v in (prev_value, value)):
        return None
    if prev_value <= low and value > low:
        return Position.LONG
    if prev_value >= high and value < high:
        return Position.SHORT
    return None
