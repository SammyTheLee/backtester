"""Tests for train/test date splits."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bt.pipeline import load_run_config
from bt.splits import (
    filter_by_date_range,
    load_candles,
    parse_split_config,
    require_walkforward_splits,
    split_config_for_oos,
)


def test_parse_split_config():
    config = {
        "data": {"csv_path": "x.csv", "split": "test"},
        "split": {
            "train_start": "2026-01-01",
            "train_end": "2026-01-08",
            "test_start": "2026-01-09",
            "test_end": "2026-01-13",
        },
    }
    split = parse_split_config(config)
    assert split is not None
    assert split.active == "test"
    assert split.train.end == "2026-01-08"


def test_filter_by_date_range():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01", "2026-01-05", "2026-01-10"]
            ),
            "close": [1.0, 2.0, 3.0],
        }
    )
    out = filter_by_date_range(df, start="2026-01-02", end="2026-01-08")
    assert len(out) == 1
    assert out["close"].iloc[0] == 2.0


def test_load_candles_train_split(demo_candles_csv: Path):
    config = {
        "data": {"split": "train"},
        "split": {
            "train_start": "2026-01-01",
            "train_end": "2026-01-05",
            "test_start": "2026-01-09",
            "test_end": "2026-01-13",
        },
    }
    split = parse_split_config(config)
    df, name = load_candles(demo_candles_csv, split=split)
    assert name == "train"
    assert len(df) < 300
    assert df["timestamp"].min() >= pd.Timestamp("2026-01-01")
    assert df["timestamp"].max() <= pd.Timestamp("2026-01-05")


def test_run_config_loads_split():
    root = Path(__file__).resolve().parents[1]
    loaded = load_run_config(root / "configs" / "example.toml")
    assert loaded.split is not None
    assert loaded.data_split == "train"


def test_sweep_config_forces_train_even_if_data_split_is_test():
    from bt.splits import split_config_for_sweep

    config = {
        "data": {"csv_path": "x.csv", "split": "test"},
        "split": {
            "train_start": "2026-01-01",
            "train_end": "2026-01-08",
            "test_start": "2026-01-09",
            "test_end": "2026-01-13",
        },
    }
    split = split_config_for_sweep(config)
    assert split.active == "train"


def test_load_run_config_for_sweep_uses_train(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    loaded = load_run_config(root / "configs" / "param_sweep.toml", for_sweep=True)
    assert loaded.data_split == "train"
    assert loaded.split is not None
    assert loaded.split.active == "train"


def test_require_walkforward_splits_rejects_missing_oos():
    with pytest.raises(ValueError, match="oos"):
        require_walkforward_splits(
            {
                "split": {
                    "train_start": "2026-01-01",
                    "train_end": "2026-01-05",
                    "validation_start": "2026-01-06",
                    "validation_end": "2026-01-09",
                }
            }
        )


def test_load_candles_oos_split(demo_candles_csv: Path):
    config = {
        "data": {"split": "oos"},
        "split": {
            "train_start": "2026-01-01",
            "train_end": "2026-01-05",
            "validation_start": "2026-01-06",
            "validation_end": "2026-01-09",
            "oos_start": "2026-01-10",
            "oos_end": "2026-01-13",
        },
    }
    split = parse_split_config(config)
    df, name = load_candles(demo_candles_csv, split=split)
    assert name == "oos"
    assert df["timestamp"].min() >= pd.Timestamp("2026-01-10")
    assert df["timestamp"].max() <= pd.Timestamp("2026-01-13")


def test_split_config_for_oos_aliases():
    config = {
        "split": {
            "out_of_sample_start": "2026-01-10",
            "out_of_sample_end": "2026-01-13",
        }
    }
    split = split_config_for_oos(config)
    assert split.oos.start == "2026-01-10"


def test_sweep_requires_split_section(tmp_path: Path):
    cfg = tmp_path / "bare.toml"
    cfg.write_text(
        '[data]\ncsv_path = "data/candles/DEMO.csv"\n'
        '[strategy]\ntype = "ma_crossover"\nfast = 5\nslow = 20\n'
    )
    with pytest.raises(ValueError, match=r"requires a \[split\]"):
        load_run_config(cfg, for_sweep=True)
