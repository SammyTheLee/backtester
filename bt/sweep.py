"""Parameter grid expansion for backtest sweeps."""

from __future__ import annotations

import itertools
from typing import Any

import pandas as pd

from bt.strategy import (
    DEFAULT_REGIME_COLUMN,
    FAST_MA_WINDOW,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    RSI_PERIOD,
    SLOW_MA_WINDOW,
    VOL_MIN,
    VOL_WINDOW,
    strategy_type,
)

# Canonical name -> keys accepted in [strategy] or [execution] TOML.
SWEEP_PARAMS_BY_TYPE: dict[str, dict[str, tuple[str, ...]]] = {
    "ma_crossover": {
        "fast": ("fast", "fast_ma"),
        "slow": ("slow", "slow_ma"),
        "stop_pct": ("stop_pct",),
        "target_pct": ("target_pct", "take_profit_pct"),
    },
    "rsi": {
        "period": ("period", "rsi_period"),
        "oversold": ("oversold",),
        "overbought": ("overbought",),
        "stop_pct": ("stop_pct",),
        "target_pct": ("target_pct", "take_profit_pct"),
    },
    "vol_gated_ma": {
        "fast": ("fast", "fast_ma"),
        "slow": ("slow", "slow_ma"),
        "vol_window": ("vol_window",),
        "vol_min": ("vol_min",),
        "stop_pct": ("stop_pct",),
        "target_pct": ("target_pct", "take_profit_pct"),
    },
    "regime_gated_ma": {
        "fast": ("fast", "fast_ma"),
        "slow": ("slow", "slow_ma"),
        "regime_column": ("regime_column",),
        "stop_pct": ("stop_pct",),
        "target_pct": ("target_pct", "take_profit_pct"),
    },
}

DEFAULTS_BY_TYPE: dict[str, dict[str, Any]] = {
    "ma_crossover": {
        "fast": FAST_MA_WINDOW,
        "slow": SLOW_MA_WINDOW,
        "stop_pct": None,
        "target_pct": None,
    },
    "rsi": {
        "period": RSI_PERIOD,
        "oversold": RSI_OVERSOLD,
        "overbought": RSI_OVERBOUGHT,
        "stop_pct": None,
        "target_pct": None,
    },
    "vol_gated_ma": {
        "fast": FAST_MA_WINDOW,
        "slow": SLOW_MA_WINDOW,
        "vol_window": VOL_WINDOW,
        "vol_min": VOL_MIN,
        "stop_pct": None,
        "target_pct": None,
    },
    "regime_gated_ma": {
        "fast": FAST_MA_WINDOW,
        "slow": SLOW_MA_WINDOW,
        "regime_column": DEFAULT_REGIME_COLUMN,
        "stop_pct": None,
        "target_pct": None,
    },
}

# Legacy alias for MA-only callers.
SWEEP_PARAMS = SWEEP_PARAMS_BY_TYPE["ma_crossover"]


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def _sweep_params_for(config: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    typ = strategy_type(config)
    if typ not in SWEEP_PARAMS_BY_TYPE:
        raise ValueError(f"unsupported strategy type {typ!r}")
    return SWEEP_PARAMS_BY_TYPE[typ]


def _read_param(
    config: dict[str, Any],
    aliases: tuple[str, ...],
) -> Any | None:
    """Read a sweep axis from ``[sweep]``, then ``[strategy]``, then ``[execution]``."""
    for section in ("sweep", "strategy", "execution"):
        block = config.get(section, {})
        if not isinstance(block, dict):
            continue
        for key in aliases:
            if key in block:
                return block[key]
    return None


def extract_sweep_axes(config: dict[str, Any]) -> dict[str, list[Any]]:
    """Collect parameters that should be swept (any value given as a list)."""
    axes: dict[str, list[Any]] = {}
    for canonical, aliases in _sweep_params_for(config).items():
        raw = _read_param(config, aliases)
        if raw is None:
            continue
        values = _as_list(raw)
        if len(values) > 1:
            axes[canonical] = values
    return axes


def is_valid_ma_windows(fast: int | float, slow: int | float) -> bool:
    """True when fast MA period is strictly shorter than slow MA period."""
    return int(fast) < int(slow)


def is_valid_rsi_thresholds(oversold: float, overbought: float) -> bool:
    return float(oversold) < float(overbought)


def is_valid_combo(config: dict[str, Any], combo: dict[str, Any]) -> bool:
    typ = strategy_type(config)
    if typ == "ma_crossover":
        fast = combo.get("fast")
        slow = combo.get("slow")
        if fast is None or slow is None:
            return True
        return is_valid_ma_windows(fast, slow)
    if typ in ("vol_gated_ma", "regime_gated_ma"):
        fast = combo.get("fast")
        slow = combo.get("slow")
        if fast is None or slow is None:
            return True
        return is_valid_ma_windows(fast, slow)
    if typ == "rsi":
        oversold = combo.get("oversold")
        overbought = combo.get("overbought")
        if oversold is None or overbought is None:
            return True
        return is_valid_rsi_thresholds(oversold, overbought)
    return True


def skip_reason(config: dict[str, Any], combo: dict[str, Any]) -> str | None:
    typ = strategy_type(config)
    if typ == "ma_crossover":
        fast = combo.get("fast")
        slow = combo.get("slow")
        if fast is not None and slow is not None and not is_valid_ma_windows(fast, slow):
            return "fast_ma >= slow_ma"
    if typ in ("vol_gated_ma", "regime_gated_ma"):
        fast = combo.get("fast")
        slow = combo.get("slow")
        if fast is not None and slow is not None and not is_valid_ma_windows(fast, slow):
            return "fast_ma >= slow_ma"
    if typ == "rsi":
        oversold = combo.get("oversold")
        overbought = combo.get("overbought")
        if (
            oversold is not None
            and overbought is not None
            and not is_valid_rsi_thresholds(oversold, overbought)
        ):
            return "oversold >= overbought"
    return None


def resolve_base_params(config: dict[str, Any]) -> dict[str, Any]:
    """Scalar values for each sweep axis (defaults when omitted)."""
    typ = strategy_type(config)
    defaults = dict(DEFAULTS_BY_TYPE[typ])
    for canonical, aliases in _sweep_params_for(config).items():
        raw = _read_param(config, aliases)
        if raw is None:
            continue
        values = _as_list(raw)
        if len(values) == 1:
            defaults[canonical] = values[0]
    return defaults


def _iter_raw_grid(
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Full cartesian product with no strategy-specific filtering."""
    axes = extract_sweep_axes(config)
    fixed = resolve_base_params(config)
    if not axes:
        return [dict(fixed)]
    keys = list(axes.keys())
    combos: list[dict[str, Any]] = []
    for values in itertools.product(*(axes[k] for k in keys)):
        combos.append({**fixed, **dict(zip(keys, values))})
    return combos


def iter_skipped_combinations(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Combinations removed by strategy-specific validation rules."""
    skipped: list[dict[str, Any]] = []
    for combo in _iter_raw_grid(config):
        reason = skip_reason(config, combo)
        if reason is not None:
            skipped.append({**combo, "skip_reason": reason})
    return skipped


def iter_param_combinations(
    config: dict[str, Any],
    *,
    require_fast_lt_slow: bool = True,
) -> list[dict[str, Any]]:
    """Cartesian product of list-valued parameters, skipping invalid combos."""
    del require_fast_lt_slow  # kept for CLI compatibility
    combos: list[dict[str, Any]] = []
    for combo in _iter_raw_grid(config):
        if skip_reason(config, combo) is not None:
            continue
        combos.append(combo)
    return combos


def combination_count(config: dict[str, Any]) -> int:
    """Number of valid combinations after strategy-specific filtering."""
    return len(iter_param_combinations(config))


def expected_grid_size(config: dict[str, Any]) -> int:
    """Raw cartesian product size before filtering."""
    return len(_iter_raw_grid(config))


def combinations_dataframe(combos: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per combination with combo_id and label columns."""
    rows: list[dict[str, Any]] = []
    for combo_id, combo in enumerate(combos, start=1):
        rows.append(
            {
                "combo_id": combo_id,
                "label": combo_label(combo),
                **combo,
            }
        )
    return pd.DataFrame(rows)


def combo_label(combo: dict[str, Any]) -> str:
    """Short label for one combination."""
    keys = (
        "fast",
        "slow",
        "period",
        "oversold",
        "overbought",
        "vol_window",
        "vol_min",
        "regime_column",
        "stop_pct",
        "target_pct",
    )
    parts = []
    for key in keys:
        if key not in combo:
            continue
        val = combo[key]
        if isinstance(val, float):
            parts.append(f"{key}={val:g}")
        else:
            parts.append(f"{key}={val}")
    return "_".join(parts)
