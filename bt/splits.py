"""Train/test date splits for candle data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from bt.import_candles import CANONICAL_COLUMNS

REQUIRED_CANDLE_COLUMNS = CANONICAL_COLUMNS

SplitName = Literal["all", "train", "test", "oos"]


@dataclass(frozen=True)
class DateRange:
    start: str | None = None
    end: str | None = None


@dataclass(frozen=True)
class SplitConfig:
    """Train, validation, and out-of-sample windows from TOML ``[split]``."""

    train: DateRange
    test: DateRange  # validation / hold-out
    oos: DateRange
    active: SplitName = "all"

    @property
    def is_enabled(self) -> bool:
        return any(
            (
                self.train.start,
                self.train.end,
                self.test.start,
                self.test.end,
                self.oos.start,
                self.oos.end,
            )
        )


def _validation_range(split_cfg: dict[str, Any]) -> DateRange:
    """Validation / hold-out window (``validation_*`` or ``test_*`` keys)."""
    return DateRange(
        start=_optional_str(
            split_cfg.get("validation_start") or split_cfg.get("test_start")
        ),
        end=_optional_str(
            split_cfg.get("validation_end") or split_cfg.get("test_end")
        ),
    )


def _oos_range(split_cfg: dict[str, Any]) -> DateRange:
    """Out-of-sample window (never used for parameter sweeps)."""
    return DateRange(
        start=_optional_str(
            split_cfg.get("oos_start") or split_cfg.get("out_of_sample_start")
        ),
        end=_optional_str(
            split_cfg.get("oos_end") or split_cfg.get("out_of_sample_end")
        ),
    )


def _split_from_config(
    config: dict[str, Any],
    *,
    active: SplitName,
) -> SplitConfig | None:
    split_cfg = config.get("split")
    if not isinstance(split_cfg, dict) or not split_cfg:
        return None
    return SplitConfig(
        train=DateRange(
            start=_optional_str(split_cfg.get("train_start")),
            end=_optional_str(split_cfg.get("train_end")),
        ),
        test=_validation_range(split_cfg),
        oos=_oos_range(split_cfg),
        active=active,
    )


def parse_split_config(config: dict[str, Any]) -> SplitConfig | None:
    """Parse ``[split]`` and ``data.split`` from a run config."""
    base = _split_from_config(config, active="all")
    if base is None:
        return None

    data_cfg = config.get("data", {})
    active_raw = str(data_cfg.get("split", "train")).lower()
    if active_raw not in ("all", "train", "test", "oos"):
        raise ValueError(
            f"data.split must be 'train', 'test', 'oos', or 'all'; got {active_raw!r}"
        )
    active: SplitName = active_raw  # type: ignore[assignment]
    return SplitConfig(
        train=base.train, test=base.test, oos=base.oos, active=active
    )


def split_config_for_oos(config: dict[str, Any]) -> SplitConfig:
    """Re-run top validation configs on the out-of-sample window."""
    base = _split_from_config(config, active="oos")
    if base is None:
        raise ValueError(
            "Out-of-sample testing requires [split] with oos_start/oos_end "
            "(or out_of_sample_start/out_of_sample_end)."
        )
    window = base.oos
    if window.start is None and window.end is None:
        raise ValueError(
            "Out-of-sample requires oos_start/oos_end under [split]."
        )
    return base


def require_walkforward_splits(config: dict[str, Any]) -> SplitConfig:
    """Require train, validation, and OOS date windows for walk-forward validation."""
    base = _split_from_config(config, active="train")
    if base is None:
        raise ValueError(
            "Walk-forward requires [split] with train_start/train_end, "
            "validation_start/validation_end, and oos_start/oos_end."
        )
    missing: list[str] = []
    if base.train.start is None and base.train.end is None:
        missing.append("train_start/train_end")
    if base.test.start is None and base.test.end is None:
        missing.append("validation_start/validation_end (or test_start/test_end)")
    if base.oos.start is None and base.oos.end is None:
        missing.append("oos_start/oos_end")
    if missing:
        raise ValueError(
            "Walk-forward requires all three windows in [split]; missing: "
            + ", ".join(missing)
        )
    return base


def split_config_for_validation(config: dict[str, Any]) -> SplitConfig:
    """Re-run top train configs on the validation (hold-out) window."""
    base = _split_from_config(config, active="test")
    if base is None:
        raise ValueError(
            "Validation requires a [split] section with validation_start/validation_end "
            "(or test_start/test_end)."
        )
    window = base.test
    if window.start is None and window.end is None:
        raise ValueError(
            "Validation requires validation_start/validation_end "
            "(or test_start/test_end) under [split]."
        )
    return base


def split_config_for_sweep(config: dict[str, Any]) -> SplitConfig:
    """Parameter sweeps always use the train window (in-sample only)."""
    base = _split_from_config(config, active="train")
    if base is None:
        raise ValueError(
            "Parameter sweep requires a [split] section with train_start and train_end. "
            "All combinations are backtested on train data only; use "
            "'python -m bt run' with data.split = \"test\" for out-of-sample."
        )
    if base.train.start is None and base.train.end is None:
        raise ValueError(
            "Parameter sweep requires train_start and/or train_end under [split]."
        )
    return base


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        raise ValueError("candle CSV must have a 'timestamp' column for date splits")
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    return out


def filter_by_date_range(
    df: pd.DataFrame,
    *,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    """Inclusive filter on ``timestamp`` (parsed as datetime)."""
    work = _parse_timestamps(df)
    mask = pd.Series(True, index=work.index)
    if start is not None:
        mask &= work["timestamp"] >= pd.to_datetime(start)
    if end is not None:
        mask &= work["timestamp"] <= pd.to_datetime(end)
    filtered = work.loc[mask]
    if filtered.empty:
        raise ValueError(
            f"no candles in date range start={start!r} end={end!r} "
            f"(file has {work['timestamp'].iloc[0]} .. {work['timestamp'].iloc[-1]})"
        )
    return filtered.reset_index(drop=True)


def apply_split(df: pd.DataFrame, split: SplitConfig) -> pd.DataFrame:
    """Return rows for the active split (train, test, or full series)."""
    if split.active == "all" or not split.is_enabled:
        return _parse_timestamps(df)

    if split.active == "train":
        window = split.train
    elif split.active == "oos":
        window = split.oos
    else:
        window = split.test

    if window.start is None and window.end is None:
        raise ValueError(
            f"split={split.active!r} has no dates in config "
            f"(set {split.active}_start / {split.active}_end in [split])"
        )
    return filter_by_date_range(df, start=window.start, end=window.end)


def load_candles(
    csv_path: Path | str,
    *,
    split: SplitConfig | None = None,
) -> tuple[pd.DataFrame, SplitName]:
    """Load OHLCV CSV and apply optional train/test split."""
    path = Path(csv_path)
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_CANDLE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"candle CSV missing required columns {missing}; got {list(df.columns)}"
        )
    if split is None:
        return _parse_timestamps(df), "all"
    return apply_split(df, split), split.active


def split_summary(split: SplitConfig | None) -> dict[str, Any]:
    """JSON-friendly description of split settings."""
    if split is None:
        return {"active": "all"}
    return {
        "active": split.active,
        "train": {"start": split.train.start, "end": split.train.end},
        "validation": {"start": split.test.start, "end": split.test.end},
        "oos": {"start": split.oos.start, "end": split.oos.end},
        "test": {"start": split.test.start, "end": split.test.end},
    }
