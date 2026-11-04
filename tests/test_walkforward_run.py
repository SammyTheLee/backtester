"""Tests for end-to-end walk-forward validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bt.pipeline import load_run_config
from bt.splits import require_walkforward_splits
from bt.walkforward_run import (
    resolve_train_top_n,
    resolve_validation_top_n,
    run_walkforward,
)


def test_require_walkforward_splits():
    root = Path(__file__).resolve().parents[1]
    loaded = load_run_config(root / "configs" / "param_sweep.toml")
    splits = require_walkforward_splits(loaded.config)
    assert splits.train.end == "2026-01-05"
    assert splits.test.end == "2026-01-09"
    assert splits.oos.end == "2026-01-13"


def test_resolve_top_n_from_config():
    root = Path(__file__).resolve().parents[1]
    loaded = load_run_config(root / "configs" / "param_sweep.toml")
    assert resolve_train_top_n(loaded) == 10
    assert resolve_validation_top_n(loaded) == 10


def test_run_walkforward_end_to_end(
    demo_candles_csv: Path, isolated_results: Path, tmp_path: Path, monkeypatch, project_root: Path
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
    loaded = load_run_config(cfg_path, for_sweep=True)

    monkeypatch.setattr("bt.paths.RESULTS_DIR", isolated_results)
    import bt.walkforward_run as wf_mod

    monkeypatch.setattr(wf_mod, "RESULTS_DIR", isolated_results)

    result = run_walkforward(
        loaded,
        train_top_n=2,
        validation_top_n=2,
        append_runs_log=False,
    )

    assert result.root_dir.exists()
    assert (result.sweep_dir / "sweep_results.csv").exists()
    assert len(result.sweep_results) > 2
    assert len(result.validation_results) == 2
    assert len(result.oos_results) == 2
    assert result.walkforward_summary.exists()

    wf = pd.read_csv(result.walkforward_summary)
    assert "train_net_pnl" in wf.columns
    assert "val_net_pnl" in wf.columns
    assert "oos_net_pnl" in wf.columns
    assert (result.root_dir / "walkforward_metadata.json").exists()


def test_train_top_n_cannot_exceed_grid(
    demo_candles_csv: Path, isolated_results: Path, tmp_path: Path, monkeypatch, project_root: Path
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
    loaded = load_run_config(cfg_path, for_sweep=True)

    monkeypatch.setattr("bt.paths.RESULTS_DIR", isolated_results)
    import bt.walkforward_run as wf_mod

    monkeypatch.setattr(wf_mod, "RESULTS_DIR", isolated_results)

    with pytest.raises(ValueError, match="train_top_n"):
        run_walkforward(loaded, train_top_n=9999, validation_top_n=1, append_runs_log=False)
