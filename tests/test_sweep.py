"""Tests for parameter sweep grid expansion (offline, in-process)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bt.pipeline import load_run_config, params_from_config
from bt.sweep import (
    combo_label,
    extract_sweep_axes,
    is_valid_ma_windows,
    iter_param_combinations,
    iter_skipped_combinations,
)
from bt.sweep_run import generate_combinations_only, run_parameter_sweep


def test_extract_sweep_axes_from_sweep_section():
    config = {
        "sweep": {
            "fast_ma": [5, 10],
            "slow_ma": [20, 40],
            "stop_pct": [0.01, 0.02],
        },
    }
    axes = extract_sweep_axes(config)
    assert axes["fast"] == [5, 10]
    assert axes["slow"] == [20, 40]


def test_extract_sweep_axes_from_user_style_config():
    config = {
        "strategy": {"fast_ma": [5, 10, 20], "slow_ma": [30, 50, 100]},
        "execution": {"stop_pct": [0.01, 0.02], "take_profit_pct": [0.02, 0.04]},
    }
    axes = extract_sweep_axes(config)
    assert set(axes) == {"fast", "slow", "stop_pct", "target_pct"}
    assert len(axes["fast"]) == 3
    assert len(axes["target_pct"]) == 2


def test_is_valid_ma_windows():
    assert is_valid_ma_windows(5, 20)
    assert not is_valid_ma_windows(20, 5)
    assert not is_valid_ma_windows(10, 10)


def test_iter_skipped_combinations():
    config = {
        "strategy": {"fast_ma": [5, 30], "slow_ma": [10, 50]},
        "execution": {"stop_pct": 0.01},
    }
    skipped = iter_skipped_combinations(config)
    assert len(skipped) == 1
    assert skipped[0]["fast"] == 30 and skipped[0]["slow"] == 10


def test_iter_param_combinations_filters_fast_ge_slow():
    config = {
        "strategy": {"fast_ma": [5, 30], "slow_ma": [10, 50]},
        "execution": {"stop_pct": 0.01},
    }
    combos = iter_param_combinations(config)
    assert len(combos) == 3
    assert not any(c["fast"] >= c["slow"] for c in combos)
    assert (30, 10) not in {(c["fast"], c["slow"]) for c in combos}


def test_iter_param_combinations_full_grid_count():
    config = {
        "strategy": {"fast_ma": [5, 10, 20], "slow_ma": [30, 50, 100]},
        "execution": {"stop_pct": [0.01, 0.02], "take_profit_pct": [0.02, 0.04]},
    }
    combos = iter_param_combinations(config)
    assert len(combos) == 36


def test_combo_label():
    assert "fast=5" in combo_label({"fast": 5, "slow": 30, "stop_pct": 0.01})


def test_params_from_config_reads_fast_ma_alias():
    config = {
        "data": {"csv_path": "data/candles/DEMO.csv"},
        "strategy": {"fast_ma": 7, "slow_ma": 21},
        "execution": {"take_profit_pct": 0.03},
    }
    p = params_from_config(config)
    assert p.fast == 7
    assert p.slow == 21
    assert p.target_pct == 0.03


def test_params_from_config_rejects_invalid_ma_pair():
    config = {
        "data": {"csv_path": "x.csv"},
        "strategy": {"fast_ma": 50, "slow_ma": 10},
    }
    with pytest.raises(ValueError, match="fast .* must be less than slow"):
        params_from_config(config)


def test_sweep_runs_single_scalar_config(
    demo_candles_csv: Path, isolated_results: Path, tmp_path: Path
):
    cfg_path = tmp_path / "example.toml"
    cfg_path.write_text(
        f"""
[data]
csv_path = "{demo_candles_csv}"

[split]
train_start = "2026-01-01"
train_end = "2026-01-05"
validation_start = "2026-01-06"
validation_end = "2026-01-09"
oos_start = "2026-01-10"
oos_end = "2026-01-13"

[strategy]
type = "ma_crossover"
fast = 5
slow = 20

[execution]
fee_bps = 0
slippage_bps = 0

[portfolio]
initial_equity = 10000.0
size = 1.0

[output]
summary_sort_by = "net_pnl"
runs_log = "{isolated_results / 'runs.csv'}"
ranked_summary = "{isolated_results / 'summary.csv'}"
""",
        encoding="utf-8",
    )
    loaded = load_run_config(cfg_path, for_sweep=True)
    sweep_dir, df = run_parameter_sweep(loaded, sweep_base=isolated_results)
    assert sweep_dir.exists()
    assert len(df) == 1


def test_run_rejects_sweep_config(project_root: Path):
    loaded = load_run_config(project_root / "configs" / "sweep.toml")
    assert extract_sweep_axes(loaded.config)


def test_combinations_writes_skipped_csv_for_overlap_config(
    demo_candles_csv: Path, isolated_results: Path, tmp_path: Path
):
    cfg_path = tmp_path / "overlap.toml"
    cfg_path.write_text(
        f"""
[data]
csv_path = "{demo_candles_csv}"

[split]
train_start = "2026-01-01"
train_end = "2026-01-05"
validation_start = "2026-01-06"
validation_end = "2026-01-09"
oos_start = "2026-01-10"
oos_end = "2026-01-13"

[strategy]
type = "ma_crossover"
fast_ma = [5, 30]
slow_ma = [10, 50]

[execution]
stop_pct = 0.01
fee_bps = 0
slippage_bps = 0

[portfolio]
initial_equity = 10000.0
size = 1.0
""",
        encoding="utf-8",
    )
    loaded = load_run_config(cfg_path, for_sweep=True)
    out_dir, df = generate_combinations_only(loaded, output_dir=isolated_results / "combinations_test")
    skipped_csv = out_dir / "skipped_combinations.csv"
    assert skipped_csv.exists()
    skipped = pd.read_csv(skipped_csv)
    assert len(skipped) == 1
    assert skipped.iloc[0]["fast"] == 30
    assert len(df) == 3


def test_combinations_writes_all_rows(
    demo_candles_csv: Path, isolated_results: Path, tmp_path: Path
):
    cfg_path = tmp_path / "sweep.toml"
    cfg_path.write_text(
        f"""
[data]
csv_path = "{demo_candles_csv}"

[split]
train_start = "2026-01-01"
train_end = "2026-01-05"
validation_start = "2026-01-06"
validation_end = "2026-01-09"
oos_start = "2026-01-10"
oos_end = "2026-01-13"

[strategy]
type = "ma_crossover"
fast_ma = [5, 10, 20]
slow_ma = [30, 50, 100]

[execution]
stop_pct = [0.01, 0.02]
take_profit_pct = [0.02, 0.04]
fee_bps = 0
slippage_bps = 0

[portfolio]
initial_equity = 10000.0
size = 1.0
""",
        encoding="utf-8",
    )
    loaded = load_run_config(cfg_path, for_sweep=True)
    out_dir, df = generate_combinations_only(loaded, output_dir=isolated_results / "grid")
    assert len(df) == 36
    assert list(df.columns[:2]) == ["combo_id", "label"]
    assert (out_dir / "combinations.csv").exists()


def test_sweep_writes_results_csv(
    demo_candles_csv: Path, isolated_results: Path, tmp_path: Path
):
    cfg_path = tmp_path / "sweep.toml"
    cfg_path.write_text(
        f"""
[data]
csv_path = "{demo_candles_csv}"

[split]
train_start = "2026-01-01"
train_end = "2026-01-05"
validation_start = "2026-01-06"
validation_end = "2026-01-09"
oos_start = "2026-01-10"
oos_end = "2026-01-13"

[strategy]
type = "ma_crossover"
fast_ma = [5, 10]
slow_ma = [30, 50]
stop_pct = 0.01
target_pct = 0.02

[execution]
fee_bps = 0
slippage_bps = 0

[portfolio]
initial_equity = 10000.0
size = 1.0

[output]
summary_sort_by = "net_pnl"
runs_log = "{isolated_results / 'runs.csv'}"
ranked_summary = "{isolated_results / 'summary.csv'}"
""",
        encoding="utf-8",
    )
    loaded = load_run_config(cfg_path, for_sweep=True)
    sweep_dir, df = run_parameter_sweep(loaded, sweep_base=isolated_results)
    results_csv = sweep_dir / "sweep_results.csv"
    assert results_csv.exists()
    assert len(df) == 4
