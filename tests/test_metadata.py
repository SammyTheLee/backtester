"""Tests for `bt.metadata`."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bt.costs import ExecutionCosts
from bt.metadata import build_run_metadata, save_metadata


def test_save_metadata_writes_config_and_assumptions(tmp_path: Path):
    config = {
        "data": {"csv_path": "data/candles/DEMO.csv"},
        "strategy": {"type": "ma_crossover", "fast": 5, "slow": 20},
        "execution": {"fee_bps": 10, "slippage_bps": 5},
        "portfolio": {"initial_equity": 10000.0, "size": 1.0},
    }
    candles = pd.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:00", "2026-01-01T01:00:00"],
            "close": [100.0, 101.0],
        }
    )
    meta = build_run_metadata(
        config,
        config_path="configs/example.toml",
        csv_path="data/candles/DEMO.csv",
        strategy_name="ma_crossover(fast=5, slow=20)",
        fast=5,
        slow=20,
        stop_pct=0.02,
        target_pct=0.04,
        costs=ExecutionCosts(fee_bps=10, slippage_bps=5),
        initial_equity=10_000.0,
        size=1.0,
        summary_sort_by="net_pnl",
        candles_df=candles,
        run_at="2026-01-01T12:00:00+00:00",
    )
    path = tmp_path / "metadata.json"
    save_metadata(meta, path)

    loaded = json.loads(path.read_text())
    assert loaded["config_path"] == "configs/example.toml"
    assert loaded["config"]["strategy"]["fast"] == 5
    assert "timing" in loaded["assumptions"]
    assert loaded["assumptions"]["costs_applied"]["fee_bps"] == 10
    assert loaded["resolved"]["initial_equity"] == 10_000.0
    assert loaded["data"]["rows"] == 2
    assert loaded["data"]["time_start"] == "2026-01-01T00:00:00"
