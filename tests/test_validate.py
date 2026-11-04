"""Tests for validation (hold-out) runs of top train configs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bt.pipeline import load_run_config
from bt.validate_run import params_from_row, run_validation
from bt.splits import split_config_for_validation


def test_params_from_row():
    row = pd.Series(
        {
            "fast": 5,
            "slow": 30,
            "stop_pct": 0.02,
            "target_pct": 0.04,
        }
    )
    p = params_from_row(row)
    assert p.fast == 5 and p.slow == 30
    assert p.stop_pct == 0.02
    assert p.combo["target_pct"] == 0.04


def test_split_config_for_validation_accepts_validation_keys():
    config = {
        "split": {
            "train_start": "2026-01-01",
            "train_end": "2026-01-08",
            "validation_start": "2026-01-09",
            "validation_end": "2026-01-13",
        }
    }
    split = split_config_for_validation(config)
    assert split.active == "test"
    assert split.test.start == "2026-01-09"


def test_run_validation_on_demo(
    demo_candles_csv: Path, isolated_results: Path, tmp_path: Path, project_root: Path
):
    cfg_path = tmp_path / "param_sweep.toml"
    cfg_path.write_text(
        (project_root / "configs" / "param_sweep.toml")
        .read_text(encoding="utf-8")
        .replace(
            'csv_path = "data/candles/DEMO.csv"',
            f'csv_path = "{demo_candles_csv}"',
        ),
        encoding="utf-8",
    )
    loaded = load_run_config(cfg_path)

    leaderboard = tmp_path / "train_top.csv"
    rows = [
        {
            "rank": 1,
            "fast": 5,
            "slow": 30,
            "stop_pct": 0.02,
            "target_pct": 0.04,
            "composite_score": 0.05,
            "net_pnl": 100.0,
            "data_split": "train",
        },
        {
            "rank": 2,
            "fast": 10,
            "slow": 50,
            "stop_pct": 0.01,
            "target_pct": 0.02,
            "composite_score": 0.03,
            "net_pnl": 50.0,
            "data_split": "train",
        },
    ]
    pd.DataFrame(rows).to_csv(leaderboard, index=False)

    out_dir, results = run_validation(
        loaded,
        top_n=2,
        source=leaderboard,
        append_runs_log=False,
        update_summary=False,
        output_base=isolated_results,
    )
    assert (out_dir / "validation_results.csv").exists()
    assert len(results) == 2
    assert "val_net_pnl" in results.columns
    assert "train_rank" in results.columns
