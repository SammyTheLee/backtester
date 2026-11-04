"""Walk-forward summary: train, validation, and OOS metrics per configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from bt.paths import RESULTS_DIR, WALKFORWARD_SUMMARY_CSV
from bt.report import CONFIG_PARAM_COLUMNS, SUMMARY_CSV_COLUMNS
from bt.validate_run import latest_sweep_results


def _normalize_params(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in CONFIG_PARAM_COLUMNS:
        if col not in out.columns:
            continue
        if col in ("stop_pct", "target_pct"):
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _prefix_metrics(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Rename summary metric columns to ``{prefix}_{column}``."""
    rename = {
        c: f"{prefix}_{c}"
        for c in SUMMARY_CSV_COLUMNS
        if c in df.columns and f"{prefix}_{c}" not in df.columns
    }
    return df.rename(columns=rename)


def resolve_sweep_results_path(validation_results_path: Path) -> Path | None:
    """Sweep CSV that seeded validation (from metadata or latest sweep)."""
    meta_path = validation_results_path.parent / "validation_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        leaderboard = meta.get("leaderboard")
        if leaderboard:
            path = Path(leaderboard)
            if path.exists():
                return path
    return latest_sweep_results()


def build_walkforward_summary(
    oos_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    sweep_df: pd.DataFrame,
    *,
    sort_by: str = "composite_score",
) -> pd.DataFrame:
    """One row per config with ``train_*``, ``val_*``, and ``oos_*`` metric columns."""
    if oos_df.empty:
        return pd.DataFrame()

    param_cols = [c for c in CONFIG_PARAM_COLUMNS if c in oos_df.columns]
    oos = _normalize_params(oos_df)
    val = _normalize_params(validation_df)
    sweep = _normalize_params(sweep_df)

    train = sweep.copy()
    if "rank" in train.columns:
        train = train.rename(columns={"rank": "train_rank"})
    train = _prefix_metrics(train, "train")
    train_cols = (
        param_cols
        + (["train_rank"] if "train_rank" in train.columns else [])
        + [f"train_{c}" for c in SUMMARY_CSV_COLUMNS if f"train_{c}" in train.columns]
    )
    train = train[[c for c in train_cols if c in train.columns]]

    val_only = [c for c in val.columns if c.startswith("val_") or c in ("val_rank", "train_rank")]
    val_keep = param_cols + [c for c in val_only if c in val.columns]
    val = val[val_keep]

    oos_only = [c for c in oos.columns if c.startswith("oos_")]
    oos_keep = param_cols + [c for c in oos_only if c in oos.columns]
    oos_part = oos[oos_keep]
    if "oos_rank" in oos_part.columns:
        oos_part = oos_part.drop(columns=["oos_rank"])

    merged = oos_part.merge(val, on=param_cols, how="left")
    merged = merged.merge(train, on=param_cols, how="left", suffixes=("", "_sweep"))

    for col in ("train_rank", "val_rank"):
        sweep_col = f"{col}_sweep"
        if sweep_col in merged.columns:
            merged[col] = merged[col].fillna(merged[sweep_col])
            merged = merged.drop(columns=[sweep_col])
    for metric in SUMMARY_CSV_COLUMNS:
        primary = f"train_{metric}"
        sweep_col = f"{primary}_sweep"
        if sweep_col in merged.columns:
            if primary in merged.columns:
                merged[primary] = merged[primary].fillna(merged[sweep_col])
            else:
                merged[primary] = merged[sweep_col]
            merged = merged.drop(columns=[sweep_col])

    sort_col = f"oos_{sort_by}"
    if sort_col in merged.columns:
        merged = merged.sort_values(sort_col, ascending=False, kind="mergesort")
    if "oos_rank" in merged.columns:
        merged = merged.drop(columns=["oos_rank"])
    merged.insert(0, "oos_rank", range(1, len(merged) + 1))

    lead = ["oos_rank", "train_rank", "val_rank"] + param_cols
    train_metrics = [f"train_{c}" for c in SUMMARY_CSV_COLUMNS]
    val_metrics = [f"val_{c}" for c in SUMMARY_CSV_COLUMNS]
    oos_metrics = [f"oos_{c}" for c in SUMMARY_CSV_COLUMNS]
    ordered = lead + [c for c in train_metrics + val_metrics + oos_metrics if c in merged.columns]
    extra = [c for c in merged.columns if c not in ordered]
    return merged[ordered + extra]


def write_walkforward_summary(
    oos_df: pd.DataFrame,
    validation_results_path: Path,
    *,
    sweep_results_path: Path | None = None,
    sort_by: str = "composite_score",
    global_path: Path | None = None,
    run_dir: Path | None = None,
) -> Path:
    """Write ``walkforward_summary.csv`` (global and optional per-run folder)."""
    validation_df = pd.read_csv(validation_results_path)
    sweep_path = sweep_results_path or resolve_sweep_results_path(
        validation_results_path
    )
    if sweep_path is None or not sweep_path.exists():
        raise FileNotFoundError(
            "Cannot build walkforward summary: no sweep_results.csv found."
        )
    sweep_df = pd.read_csv(sweep_path)

    summary = build_walkforward_summary(
        oos_df,
        validation_df,
        sweep_df,
        sort_by=sort_by,
    )

    dest = Path(global_path or RESULTS_DIR / WALKFORWARD_SUMMARY_CSV)
    dest.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(dest, index=False)

    if run_dir is not None:
        run_copy = Path(run_dir) / WALKFORWARD_SUMMARY_CSV
        run_copy.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(run_copy, index=False)

    return dest
