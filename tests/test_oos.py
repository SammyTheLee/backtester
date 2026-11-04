"""Tests for out-of-sample runs of top validation configs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bt.oos_run import load_top_validation_configs, run_oos
from bt.pipeline import load_run_config
from bt.splits import split_config_for_oos


def test_split_config_for_oos():
    config = {
        "split": {
            "train_start": "2026-01-01",
            "train_end": "2026-01-05",
            "validation_start": "2026-01-06",
            "validation_end": "2026-01-09",
            "oos_start": "2026-01-10",
            "oos_end": "2026-01-13",
        }
    }
    split = split_config_for_oos(config)
    assert split.active == "oos"
    assert split.oos.start == "2026-01-10"


def test_load_top_validation_configs_by_val_rank(tmp_path):
    path = tmp_path / "val.csv"
    rows = [
        {"val_rank": 2, "fast": 10, "slow": 50, "stop_pct": 0.01, "target_pct": 0.02},
        {"val_rank": 1, "fast": 5, "slow": 30, "stop_pct": 0.02, "target_pct": 0.04},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    top = load_top_validation_configs(path, 2, "composite_score")
    assert int(top.iloc[0]["fast"]) == 5


def test_run_oos_on_demo(
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

    sweep_path = tmp_path / "sweep_results.csv"
    val_dir = tmp_path / "validation_run"
    val_dir.mkdir()
    leaderboard = val_dir / "validation_results.csv"
    train_rows = [
        {
            "rank": 1,
            "fast": 5,
            "slow": 30,
            "stop_pct": 0.02,
            "target_pct": 0.04,
            "net_pnl": 100.0,
            "composite_score": 0.05,
            "gross_pnl": 100.0,
            "total_return_pct": 0.01,
            "trade_count": 1,
            "win_rate": 0.5,
            "profit_factor": 1.0,
            "max_drawdown": 1.0,
            "average_hold_hours": 1.0,
            "time_in_market_pct": 0.1,
        },
        {
            "rank": 2,
            "fast": 10,
            "slow": 50,
            "stop_pct": 0.01,
            "target_pct": 0.02,
            "net_pnl": 50.0,
            "composite_score": 0.03,
            "gross_pnl": 50.0,
            "total_return_pct": 0.005,
            "trade_count": 1,
            "win_rate": 0.5,
            "profit_factor": 1.0,
            "max_drawdown": 1.0,
            "average_hold_hours": 1.0,
            "time_in_market_pct": 0.1,
        },
    ]
    pd.DataFrame(train_rows).to_csv(sweep_path, index=False)
    rows = [
        {
            "val_rank": 1,
            "train_rank": 1,
            "fast": 5,
            "slow": 30,
            "stop_pct": 0.02,
            "target_pct": 0.04,
            "val_composite_score": 0.05,
            "val_net_pnl": 80.0,
        },
        {
            "val_rank": 2,
            "train_rank": 2,
            "fast": 10,
            "slow": 50,
            "stop_pct": 0.01,
            "target_pct": 0.02,
            "val_composite_score": 0.03,
            "val_net_pnl": 40.0,
        },
    ]
    pd.DataFrame(rows).to_csv(leaderboard, index=False)
    (val_dir / "validation_metadata.json").write_text(
        f'{{"leaderboard": "{sweep_path}"}}\n',
        encoding="utf-8",
    )

    out_dir, results, wf_path = run_oos(
        loaded,
        top_n=2,
        source=leaderboard,
        append_runs_log=False,
        update_summary=False,
        output_base=isolated_results,
    )
    assert (out_dir / "oos_results.csv").exists()
    assert len(results) == 2
    assert "oos_net_pnl" in results.columns
    assert "val_rank" in results.columns
    assert wf_path.exists()
    wf = pd.read_csv(wf_path)
    assert "train_composite_score" in wf.columns
    assert "val_net_pnl" in wf.columns
    assert "oos_net_pnl" in wf.columns
