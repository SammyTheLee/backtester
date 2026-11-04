"""Run a parameter sweep and write ranked comparison CSV."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from bt.paths import (
    RESULTS_DIR,
    SWEEP_COMBINATIONS_CSV,
    SWEEP_METADATA_JSON,
    SWEEP_RESULTS_CSV,
    SWEEP_SKIPPED_CSV,
    make_timestamped_run_dir,
)
from bt.pipeline import BacktestParams, LoadedRunConfig, execute_backtest
from bt.report import rank_dataframe, summary_row
from bt.sweep import (
    combo_label,
    combinations_dataframe,
    expected_grid_size,
    extract_sweep_axes,
    iter_param_combinations,
    iter_skipped_combinations,
)


def _params_from_combo(combo: dict[str, Any]) -> BacktestParams:
    return BacktestParams(combo=dict(combo))


def _write_skipped_combinations(
    config: dict[str, Any],
    dest: Path,
) -> pd.DataFrame | None:
    """Write invalid MA pairs to CSV when any exist."""
    skipped = iter_skipped_combinations(config)
    if not skipped:
        return None
    df = pd.DataFrame(skipped)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return df


def write_all_combinations(
    config: dict[str, Any],
    dest: Path,
    *,
    skipped_dest: Path | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], int]:
    """Write valid combinations to CSV; return table, combo list, skipped count."""
    combos = iter_param_combinations(config)
    if not combos:
        raise ValueError(
            "no valid parameter combinations: every fast/slow pair has fast >= slow"
        )
    df = combinations_dataframe(combos)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    skipped_n = len(iter_skipped_combinations(config))
    if skipped_dest is not None and skipped_n:
        _write_skipped_combinations(config, skipped_dest)
    return df, combos, skipped_n


def generate_combinations_only(
    loaded: LoadedRunConfig,
    *,
    output_dir: Path | None = None,
) -> tuple[Path, pd.DataFrame]:
    """Create timestamped folder with combinations.csv only (no backtests)."""
    combos = iter_param_combinations(loaded.config)
    if not combos:
        raise ValueError(
            "no valid parameter combinations (check list values and fast < slow)"
        )
    started = datetime.now(timezone.utc)
    out_dir = output_dir or make_timestamped_run_dir(
        RESULTS_DIR, run_at=started, prefix="combinations_"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / SWEEP_COMBINATIONS_CSV
    skipped_path = out_dir / SWEEP_SKIPPED_CSV
    df, _, skipped_n = write_all_combinations(
        loaded.config, path, skipped_dest=skipped_path
    )
    axes = extract_sweep_axes(loaded.config)
    grid = expected_grid_size(loaded.config)
    meta = {
        "generated_at": started.isoformat(),
        "config_path": str(loaded.config_path),
        "axes": {k: list(v) for k, v in axes.items()},
        "grid_size": grid,
        "combination_count": len(combos),
        "skipped_invalid_ma_pairs": skipped_n,
        "combinations_csv": str(path),
    }
    if skipped_n:
        meta["skipped_combinations_csv"] = str(skipped_path)
    (out_dir / "combinations_metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return out_dir, df


def run_parameter_sweep(
    loaded: LoadedRunConfig,
    *,
    export_each: bool = False,
    sweep_base: Path | None = None,
) -> tuple[Path, pd.DataFrame]:
    """Run all parameter combinations; return sweep folder and results table."""
    combos = iter_param_combinations(loaded.config)
    if not combos:
        raise ValueError(
            "no valid parameter combinations (check list values and fast < slow)"
        )

    started = datetime.now(timezone.utc)
    sweep_dir = make_timestamped_run_dir(
        sweep_base or RESULTS_DIR, run_at=started, prefix="sweep_"
    )

    comb_path = sweep_dir / SWEEP_COMBINATIONS_CSV
    skipped_path = sweep_dir / SWEEP_SKIPPED_CSV
    _, combos, skipped_n = write_all_combinations(
        loaded.config, comb_path, skipped_dest=skipped_path
    )
    grid = expected_grid_size(loaded.config)
    print(f"Generated {len(combos)} valid combinations (full grid: {grid})")
    if skipped_n:
        print(
            f"Skipped {skipped_n} invalid pair(s) where fast MA length >= slow MA length"
        )
        print(f"Skipped CSV    : {skipped_path}")
    print(f"Combinations CSV: {comb_path}")
    print()

    rows: list[dict[str, Any]] = []
    sort_by = loaded.summary_sort_by

    for index, combo in enumerate(combos, start=1):
        params = _params_from_combo(combo)
        label = f"{loaded.config_path}#{combo_label(combo)}"
        sub_paths = None
        if export_each:
            from bt.paths import RunOutputPaths

            sub_paths = RunOutputPaths(
                run_dir=sweep_dir / combo_label(combo)
            )

        result = execute_backtest(
            loaded,
            params,
            run_at=started,
            output_paths=sub_paths,
            export_artifacts=export_each,
            append_global_log=True,
            update_ranked_summary=False,
            config_label=label,
        )
        card = result.card
        row: dict[str, Any] = {
            **params.combo,
            "data_split": loaded.data_split,
            "strategy": result.strat.name,
            **summary_row(
                card,
                initial_equity=loaded.initial_equity,
                drawdown_penalty=loaded.drawdown_penalty,
            ),
        }
        rows.append(row)
        print(
            f"[{index}/{len(combos)}] {combo_label(params.combo)} "
            f"-> net_pnl=${card.net_pnl:,.2f}"
        )

    df = rank_dataframe(
        pd.DataFrame(rows),
        sort_by,
        initial_equity=loaded.initial_equity,
        drawdown_penalty=loaded.drawdown_penalty,
    )

    results_path = sweep_dir / SWEEP_RESULTS_CSV
    df.to_csv(results_path, index=False)

    runs_count = 0
    if loaded.runs_log_path.exists():
        runs_count = len(pd.read_csv(loaded.runs_log_path))
    print(
        f"\nRuns log      : {loaded.runs_log_path} ({runs_count} rows, all backtests)"
    )
    print(
        f"Final leaderboard: {loaded.ranked_summary_path} "
        f"(OOS only — run: python -m bt oos --config ...)"
    )

    meta = {
        "sweep_at": started.isoformat(),
        "config_path": str(loaded.config_path),
        "data_csv": str(loaded.csv_path),
        "grid_size": grid,
        "combination_count": len(combos),
        "skipped_invalid_ma_pairs": skipped_n,
        "combinations_csv": str(comb_path),
        "combinations": combos,
        "skipped_combinations_csv": str(skipped_path) if skipped_n else None,
        "sort_by": sort_by,
        "export_each": export_each,
        "results_csv": str(results_path),
        "runs_log_rows": runs_count,
    }
    meta_path = sweep_dir / SWEEP_METADATA_JSON
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    return sweep_dir, df
