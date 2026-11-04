"""Tests for walkforward_summary.csv (train + validation + OOS)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bt.report import SUMMARY_CSV_COLUMNS
from bt.walkforward import build_walkforward_summary, write_walkforward_summary


def _metric_row(prefix: str, net_pnl: float, score: float) -> dict:
    base = {f"{prefix}_{c}": 0.0 for c in SUMMARY_CSV_COLUMNS}
    base[f"{prefix}_net_pnl"] = net_pnl
    base[f"{prefix}_composite_score"] = score
    base[f"{prefix}_gross_pnl"] = net_pnl
    return base


def test_build_walkforward_summary_includes_all_phases():
    params = {"fast": 5, "slow": 30, "stop_pct": 0.02, "target_pct": 0.04}
    sweep = pd.DataFrame(
        [
            {
                "rank": 1,
                **params,
                "net_pnl": 100.0,
                "composite_score": 0.1,
                "gross_pnl": 100.0,
                "total_return_pct": 0.01,
                "trade_count": 2,
                "win_rate": 0.5,
                "profit_factor": 1.5,
                "max_drawdown": 1.0,
                "average_hold_hours": 1.0,
                "time_in_market_pct": 0.1,
            }
        ]
    )
    validation = pd.DataFrame(
        [
            {
                "val_rank": 1,
                "train_rank": 1,
                **params,
                "train_composite_score": 0.1,
                "train_net_pnl": 100.0,
                **_metric_row("val", 50.0, 0.05),
            }
        ]
    )
    oos = pd.DataFrame(
        [
            {
                "val_rank": 1,
                "train_rank": 1,
                **params,
                "val_composite_score": 0.05,
                "val_net_pnl": 50.0,
                **_metric_row("oos", 10.0, 0.01),
            }
        ]
    )
    wf = build_walkforward_summary(oos, validation, sweep, sort_by="composite_score")
    assert len(wf) == 1
    assert wf.iloc[0]["train_net_pnl"] == 100.0
    assert wf.iloc[0]["val_net_pnl"] == 50.0
    assert wf.iloc[0]["oos_net_pnl"] == 10.0
    assert wf.iloc[0]["oos_rank"] == 1


def test_write_walkforward_summary(tmp_path: Path):
    params = {"fast": 10, "slow": 50, "stop_pct": 0.01, "target_pct": 0.04}
    sweep_path = tmp_path / "sweep_results.csv"
    val_dir = tmp_path / "validation_20260101_120000"
    val_dir.mkdir()
    val_path = val_dir / "validation_results.csv"

    pd.DataFrame(
        [{"rank": 1, **params, **{c: 1.0 for c in SUMMARY_CSV_COLUMNS}}]
    ).to_csv(sweep_path, index=False)
    pd.DataFrame(
        [
            {
                "val_rank": 1,
                "train_rank": 1,
                **params,
                "train_net_pnl": 1.0,
                "train_composite_score": 0.01,
                **_metric_row("val", 2.0, 0.02),
            }
        ]
    ).to_csv(val_path, index=False)
    (val_dir / "validation_metadata.json").write_text(
        f'{{"leaderboard": "{sweep_path}"}}\n',
        encoding="utf-8",
    )

    oos_df = pd.DataFrame(
        [
            {
                "val_rank": 1,
                "train_rank": 1,
                **params,
                **_metric_row("oos", 3.0, 0.03),
            }
        ]
    )

    out = write_walkforward_summary(
        oos_df,
        val_path,
        global_path=tmp_path / "walkforward_summary.csv",
        run_dir=tmp_path / "oos_run",
    )
    assert out.exists()
    loaded = pd.read_csv(out)
    for prefix in ("train", "val", "oos"):
        assert f"{prefix}_net_pnl" in loaded.columns
        assert f"{prefix}_composite_score" in loaded.columns
