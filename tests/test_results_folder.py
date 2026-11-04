"""Integration test: backtest run writes timestamped CSV/JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from bt.paths import (
    EQUITY_CSV,
    METADATA_JSON,
    REPORT_JSON,
    SUMMARY_CSV,
    TRADES_CSV,
    latest_run_dir,
)
from bt.pipeline import execute_backtest, load_run_config, params_from_config


def test_run_writes_timestamped_results_artifacts(
    demo_candles_csv: Path, isolated_results: Path, tmp_path: Path
):
    cfg_path = tmp_path / "run.toml"
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
    loaded = load_run_config(cfg_path)
    result = execute_backtest(
        loaded,
        params_from_config(loaded.config),
        append_global_log=True,
        update_ranked_summary=False,
    )
    assert result.output_paths is not None

    run_dir = latest_run_dir(isolated_results)
    assert run_dir is not None

    for name in (TRADES_CSV, EQUITY_CSV, SUMMARY_CSV, METADATA_JSON, REPORT_JSON):
        path = run_dir / name
        assert path.exists(), f"missing {path}"
        assert path.stat().st_size > 0, f"empty {path}"

    meta = json.loads((run_dir / METADATA_JSON).read_text(encoding="utf-8"))
    assert "config" in meta
    assert "assumptions" in meta

    runs_log = isolated_results / "runs.csv"
    assert runs_log.exists()
    assert runs_log.stat().st_size > 0
