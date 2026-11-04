"""Walk-forward validation: train sweep → top train on validation → top val on OOS."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from bt.paths import (
    OOS_RESULTS_CSV,
    REPORT_MD,
    RESULTS_DIR,
    SWEEP_RESULTS_CSV,
    VALIDATION_RESULTS_CSV,
    WALKFORWARD_SUMMARY_CSV,
    make_timestamped_run_dir,
)
from bt.pipeline import LoadedRunConfig
from bt.report_md import build_walkforward_report_md, save_report_md
from bt.splits import require_walkforward_splits
from bt.sweep_run import run_parameter_sweep
from bt.oos_run import run_oos
from bt.validate_run import run_validation


@dataclass(frozen=True)
class WalkforwardResult:
    """Paths and tables from a full walk-forward run."""

    root_dir: Path
    sweep_dir: Path
    validation_dir: Path
    oos_dir: Path
    sweep_results: pd.DataFrame
    validation_results: pd.DataFrame
    oos_results: pd.DataFrame
    walkforward_summary: Path


def _cfg_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    section = config.get(name, {})
    return section if isinstance(section, dict) else {}


def resolve_train_top_n(loaded: LoadedRunConfig, override: int | None = None) -> int:
    """How many train-ranked configs to run on validation."""
    if override is not None:
        return override
    wf = _cfg_section(loaded.config, "walkforward")
    if wf.get("train_top_n") is not None:
        return int(wf["train_top_n"])
    val = _cfg_section(loaded.config, "validation")
    if val.get("top_n") is not None:
        return int(val["top_n"])
    return loaded.summary_top_n if loaded.summary_top_n is not None else 10


def resolve_validation_top_n(
    loaded: LoadedRunConfig, override: int | None = None
) -> int:
    """How many validation-ranked configs to run on OOS."""
    if override is not None:
        return override
    wf = _cfg_section(loaded.config, "walkforward")
    if wf.get("validation_top_n") is not None:
        return int(wf["validation_top_n"])
    oos = _cfg_section(loaded.config, "oos")
    if oos.get("top_n") is not None:
        return int(oos["top_n"])
    val = _cfg_section(loaded.config, "validation")
    if val.get("top_n") is not None:
        return int(val["top_n"])
    return loaded.summary_top_n if loaded.summary_top_n is not None else 10


def run_walkforward(
    loaded: LoadedRunConfig,
    *,
    export_each: bool = False,
    train_top_n: int | None = None,
    validation_top_n: int | None = None,
    append_runs_log: bool = True,
) -> WalkforwardResult:
    """Run all three walk-forward phases in order with linked outputs.

    1. **Train** — full parameter sweep on the train window only.
    2. **Validation** — backtest only the top *train_top_n* train configs.
    3. **OOS** — backtest only the top *validation_top_n* validation configs.
    """
    splits = require_walkforward_splits(loaded.config)
    n_train = resolve_train_top_n(loaded, train_top_n)
    n_val = resolve_validation_top_n(loaded, validation_top_n)

    started = datetime.now(timezone.utc)
    root = make_timestamped_run_dir(RESULTS_DIR, run_at=started, prefix="walkforward_")

    print(
        "Walk-forward validation\n"
        f"  Train      : {splits.train.start} .. {splits.train.end} (full sweep)\n"
        f"  Validation : {splits.test.start} .. {splits.test.end} (top {n_train} train configs)\n"
        f"  OOS        : {splits.oos.start} .. {splits.oos.end} (top {n_val} validation configs)\n"
    )

    print("=== Phase 1/3: Parameter sweep on train (full grid) ===")
    sweep_dir, sweep_df = run_parameter_sweep(
        loaded,
        export_each=export_each,
        sweep_base=root,
    )
    sweep_path = sweep_dir / SWEEP_RESULTS_CSV
    grid_size = len(sweep_df)
    if n_train > grid_size:
        raise ValueError(
            f"train_top_n={n_train} exceeds sweep grid size ({grid_size} combinations)"
        )

    print(
        f"\n=== Phase 2/3: Validation (top {n_train} of {grid_size} train configs) ==="
    )
    val_dir, val_df = run_validation(
        loaded,
        top_n=n_train,
        sweep_results=sweep_path,
        append_runs_log=append_runs_log,
        update_summary=False,
        output_base=root,
    )
    val_path = val_dir / VALIDATION_RESULTS_CSV
    if n_val > len(val_df):
        raise ValueError(
            f"validation_top_n={n_val} exceeds validation candidates ({len(val_df)})"
        )

    print(
        f"\n=== Phase 3/3: Out-of-sample (top {n_val} of {len(val_df)} validation configs) ==="
    )
    oos_dir, oos_df, wf_summary = run_oos(
        loaded,
        top_n=n_val,
        validation_results=val_path,
        append_runs_log=append_runs_log,
        update_summary=True,
        output_base=root,
        walkforward_run_dir=root,
    )

    manifest = {
        "walkforward_at": started.isoformat(),
        "config_path": str(loaded.config_path),
        "root_dir": str(root),
        "train_window": {"start": splits.train.start, "end": splits.train.end},
        "validation_window": {"start": splits.test.start, "end": splits.test.end},
        "oos_window": {"start": splits.oos.start, "end": splits.oos.end},
        "sweep_dir": str(sweep_dir),
        "sweep_results_csv": str(sweep_path),
        "sweep_combination_count": grid_size,
        "validation_dir": str(val_dir),
        "validation_results_csv": str(val_path),
        "validation_candidate_count": len(val_df),
        "train_top_n": n_train,
        "oos_dir": str(oos_dir),
        "oos_results_csv": str(oos_dir / OOS_RESULTS_CSV),
        "validation_top_n": n_val,
        "oos_candidate_count": len(oos_df),
        "walkforward_summary_csv": str(wf_summary),
        "sort_by": loaded.summary_sort_by,
    }

    root_summary = root / WALKFORWARD_SUMMARY_CSV
    if wf_summary != root_summary and wf_summary.exists():
        pd.read_csv(wf_summary).to_csv(root_summary, index=False)

    summary_df = pd.read_csv(root_summary if root_summary.exists() else wf_summary)
    report_path = root / REPORT_MD
    save_report_md(build_walkforward_report_md(manifest, summary_df), report_path)
    manifest["report_md"] = str(report_path)
    (root / "walkforward_metadata.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8"
    )

    return WalkforwardResult(
        root_dir=root,
        sweep_dir=sweep_dir,
        validation_dir=val_dir,
        oos_dir=oos_dir,
        sweep_results=sweep_df,
        validation_results=val_df,
        oos_results=oos_df,
        walkforward_summary=root_summary if root_summary.exists() else wf_summary,
    )
