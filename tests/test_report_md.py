"""Tests for plain-language report.md generation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from bt import __main__ as bt_main
from bt.report import StrategyReportCard
from bt.report_md import (
    build_report_md,
    build_walkforward_report_md,
    card_from_summary_row,
    generate_report_md_from_run_dir,
    load_run_artifacts,
    save_report_md,
)


def _sample_metadata() -> dict:
    return {
        "config": {
            "data": {"csv_path": "data/candles/DEMO.csv"},
            "strategy": {"type": "ma_crossover", "fast": 5, "slow": 20},
            "execution": {"stop_pct": 0.02, "target_pct": 0.04},
            "portfolio": {"initial_equity": 10000.0},
        },
        "resolved": {
            "data_csv": "data/candles/DEMO.csv",
            "strategy": "ma_crossover(fast=5, slow=20)",
            "fast": 5,
            "slow": 20,
            "stop_pct": 0.02,
            "target_pct": 0.04,
            "data_split": "all",
        },
        "assumptions": {
            "timing": [
                "Signal is known at the close of bar N.",
                "Entries fill at the open of the next bar.",
            ],
            "costs_applied": {
                "fee_bps": 10,
                "slippage_bps": 5,
                "holding_cost_bps": 0,
            }
        },
        "data": {
            "split": {"active": "all"},
            "rows": 300,
            "time_start": "2026-01-01 00:00:00",
            "time_end": "2026-01-13 11:00:00",
        },
    }


def _sample_card(**kwargs) -> StrategyReportCard:
    defaults = dict(
        gross_pnl=3.0,
        net_pnl=0.5,
        total_return_pct=0.00005,
        trade_count=11,
        win_rate=0.27,
        profit_factor=1.02,
        max_drawdown=4.9,
        average_hold_hours=11.5,
        time_in_market_pct=0.45,
    )
    defaults.update(kwargs)
    return StrategyReportCard(**defaults)


def test_report_md_is_plain_language(tmp_path: Path):
    text = build_report_md(_sample_metadata(), _sample_card())
    assert "# Backtest Report" in text
    assert "## What data was used?" in text
    assert "## What strategy was tested?" in text
    assert "## What costs were assumed?" in text
    assert "## What dates were train, validation, and out-of-sample?" in text
    assert "## What was the best out-of-sample config?" in text
    assert "## How many trades happened?" in text
    assert "## What is the biggest weakness of the result?" in text
    assert "Moving-average crossover" in text
    assert "DEMO.csv" in text
    assert "11" in text
    assert "synthetic demo data" in text.lower() or "demo" in text.lower()


def test_report_md_no_jargon_required_to_understand(tmp_path: Path):
    text = build_report_md(_sample_metadata(), _sample_card(trade_count=0, net_pnl=-1.0))
    assert "did not take any trades" in text.lower() or "did not trade" in text.lower()


def test_walkforward_report_md_covers_required_questions():
    manifest = {
        "config_path": "configs/param_sweep.toml",
        "sort_by": "composite_score",
        "sweep_combination_count": 36,
        "validation_candidate_count": 2,
        "oos_candidate_count": 2,
        "train_window": {"start": "2026-01-01", "end": "2026-01-05"},
        "validation_window": {"start": "2026-01-06", "end": "2026-01-09"},
        "oos_window": {"start": "2026-01-10", "end": "2026-01-13"},
    }
    wf = pd.DataFrame(
        [
            {
                "fast": 10,
                "slow": 50,
                "stop_pct": 0.01,
                "target_pct": 0.04,
                "train_trade_count": 2,
                "train_net_pnl": 2.45,
                "val_trade_count": 1,
                "val_net_pnl": -1.23,
                "oos_trade_count": 0,
                "oos_net_pnl": 0.0,
                "oos_win_rate": 0.0,
            },
            {
                "fast": 10,
                "slow": 30,
                "stop_pct": 0.01,
                "target_pct": 0.02,
                "train_trade_count": 3,
                "train_net_pnl": 2.07,
                "val_trade_count": 1,
                "val_net_pnl": -1.24,
                "oos_trade_count": 2,
                "oos_net_pnl": 0.47,
                "oos_win_rate": 0.5,
            },
        ]
    )
    text = build_walkforward_report_md(manifest, wf)
    assert "Walk-Forward Backtest Report" in text
    assert "2026-01-01" in text
    assert "2026-01-10" in text
    assert "did not trade" in text.lower()
    assert "validation" in text.lower()


def test_report_md_includes_assumptions_section():
    text = build_report_md(_sample_metadata(), _sample_card())
    assert "## Model assumptions" in text
    assert "Signal is known at the close" in text


def test_generate_report_from_run_folder(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    metadata = _sample_metadata()
    (run / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    card = _sample_card()
    summary_row = {
        "run_at": "2026-01-01T00:00:00+00:00",
        "config": "configs/demo.toml",
        "data_path": "data/candles/DEMO.csv",
        "data_split": "all",
        "strategy": "ma_crossover(fast=5, slow=20)",
        "fast": 5,
        "slow": 20,
        **card.to_dict(),
    }
    pd.DataFrame([summary_row]).to_csv(run / "summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "entry_time": "2026-01-01T00:00:00",
                "exit_time": "2026-01-01T05:00:00",
                "side": "long",
                "net_pnl": 1.0,
                "exit_reason": "target",
            },
            {
                "entry_time": "2026-01-02T00:00:00",
                "exit_time": "2026-01-02T03:00:00",
                "side": "short",
                "net_pnl": -0.5,
                "exit_reason": "stop",
            },
        ]
    ).to_csv(run / "trades.csv", index=False)

    out = generate_report_md_from_run_dir(run)
    text = out.read_text(encoding="utf-8")
    assert "Backtest Report" in text
    assert "Model assumptions" in text
    assert "Winning trades:** 1" in text
    assert "Worst single trade" in text


def test_write_report_cli(tmp_path: Path, capsys):
    run = tmp_path / "run"
    run.mkdir()
    metadata = _sample_metadata()
    (run / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    card = _sample_card(trade_count=2)
    pd.DataFrame(
        [
            {
                "run_at": "2026-01-01",
                "config": "c.toml",
                "data_path": "d.csv",
                "data_split": "all",
                "strategy": "ma",
                "fast": 5,
                "slow": 20,
                **card.to_dict(),
            }
        ]
    ).to_csv(run / "summary.csv", index=False)
    pd.DataFrame(
        [{"entry_time": "t", "exit_time": "t2", "net_pnl": 1.0, "exit_reason": "target"}]
    ).to_csv(run / "trades.csv", index=False)

    rc = bt_main.main(["write-report", str(run)])
    assert rc == 0
    assert (run / "report.md").exists()
    assert "Wrote plain-language report" in capsys.readouterr().out


def test_write_report_walkforward_folder(tmp_path: Path, capsys):
    run = tmp_path / "walkforward_20260805_081744"
    run.mkdir()
    manifest = {
        "config_path": "configs/walkforward.toml",
        "sort_by": "composite_score",
        "sweep_combination_count": 36,
        "validation_candidate_count": 10,
        "oos_candidate_count": 10,
        "train_window": {"start": "2026-01-01", "end": "2026-01-05"},
        "validation_window": {"start": "2026-01-06", "end": "2026-01-09"},
        "oos_window": {"start": "2026-01-10", "end": "2026-01-13"},
    }
    (run / "walkforward_metadata.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "fast": 5,
                "slow": 100,
                "stop_pct": 0.02,
                "target_pct": 0.04,
                "train_trade_count": 0,
                "train_net_pnl": 0.0,
                "val_trade_count": 0,
                "val_net_pnl": 0.0,
                "oos_trade_count": 0,
                "oos_net_pnl": 0.0,
                "oos_win_rate": 0.0,
            }
        ]
    ).to_csv(run / "walkforward_summary.csv", index=False)

    rc = bt_main.main(["write-report", str(run)])
    assert rc == 0
    text = (run / "report.md").read_text(encoding="utf-8")
    assert "Walk-Forward Backtest Report" in text
    assert "Wrote plain-language report" in capsys.readouterr().out


def test_load_run_artifacts_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_run_artifacts(tmp_path)


def test_save_report_md_writes_file(tmp_path: Path):
    path = save_report_md("# Hello\n\nWorld", tmp_path / "report.md")
    assert path.read_text(encoding="utf-8") == "# Hello\n\nWorld\n"
