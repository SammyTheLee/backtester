"""Run top in-sample configs on the validation (hold-out) window."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from bt.paths import (
    RESULTS_DIR,
    SWEEP_RESULTS_CSV,
    VALIDATION_RESULTS_CSV,
    make_timestamped_run_dir,
)
from bt.pipeline import BacktestParams, LoadedRunConfig, execute_backtest
from bt.strategy import combo_from_row
from bt.report import rank_dataframe, summary_row
from bt.splits import split_config_for_validation
from bt.sweep import combo_label


def params_from_row(row: pd.Series, config: dict[str, Any] | None = None) -> BacktestParams:
    """Build backtest params from a summary or sweep results row."""
    cfg = config or {"strategy": {"type": "ma_crossover"}}
    return BacktestParams(combo=combo_from_row(row, cfg))


def latest_sweep_results(base: Path | None = None) -> Path | None:
    """Newest ``results/sweep_*/sweep_results.csv``, if any."""
    base = Path(base or RESULTS_DIR)
    if not base.exists():
        return None
    for sweep_dir in sorted(base.glob("sweep_*"), key=lambda p: p.name, reverse=True):
        path = sweep_dir / SWEEP_RESULTS_CSV
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def resolve_leaderboard_path(
    loaded: LoadedRunConfig,
    *,
    source: Path | None = None,
    sweep_results: Path | None = None,
) -> Path:
    """Pick CSV of train-ranked configs: explicit path, sweep folder, or summary.csv."""
    if source is not None:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"leaderboard not found: {path}")
        return path

    if sweep_results is not None:
        path = Path(sweep_results)
        if not path.exists():
            raise FileNotFoundError(f"sweep results not found: {path}")
        return path

    val_cfg = loaded.config.get("validation", {})
    if isinstance(val_cfg, dict):
        if val_cfg.get("source"):
            path = Path(val_cfg["source"])
            if not path.exists():
                raise FileNotFoundError(f"validation.source not found: {path}")
            return path
        if val_cfg.get("sweep_dir"):
            path = Path(val_cfg["sweep_dir"]) / SWEEP_RESULTS_CSV
            if not path.exists():
                raise FileNotFoundError(f"validation.sweep_dir missing {path}")
            return path

    latest = latest_sweep_results()
    if latest is not None:
        return latest

    raise FileNotFoundError(
        "No train leaderboard found. Run a sweep first, set validation.source, "
        "or pass --from / --sweep-results. "
        "(summary.csv is OOS-only and cannot seed validation.)"
    )


def load_top_train_configs(
    path: Path,
    top_n: int,
    sort_by: str,
    *,
    train_only: bool = True,
    initial_equity: float = 10_000.0,
    drawdown_penalty: float = 1.0,
) -> pd.DataFrame:
    """Top N configurations from a ranked CSV (optionally train rows only)."""
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"leaderboard is empty: {path}")

    if train_only and "data_split" in df.columns:
        mask = df["data_split"].astype(str).str.lower() == "train"
        df = df.loc[mask]
        if df.empty:
            raise ValueError(
                f"no train rows in {path}; use sweep_results.csv or filter data_split"
            )

    if "rank" in df.columns:
        df = df.sort_values("rank", kind="mergesort")
    else:
        df = rank_dataframe(
            df,
            sort_by,
            initial_equity=initial_equity,
            drawdown_penalty=drawdown_penalty,
        )

    return df.head(top_n).reset_index(drop=True)


def run_validation(
    loaded: LoadedRunConfig,
    *,
    top_n: int | None = None,
    source: Path | None = None,
    sweep_results: Path | None = None,
    append_runs_log: bool = True,
    update_summary: bool = False,
    output_base: Path | None = None,
) -> tuple[Path, pd.DataFrame]:
    """Backtest top train configs on validation; return output folder and results table."""
    val_cfg = loaded.config.get("validation", {})
    n = top_n
    if n is None and isinstance(val_cfg, dict) and val_cfg.get("top_n") is not None:
        n = int(val_cfg["top_n"])
    if n is None:
        n = loaded.summary_top_n if loaded.summary_top_n is not None else 10

    leaderboard_path = resolve_leaderboard_path(
        loaded, source=source, sweep_results=sweep_results
    )
    train_only = True
    if isinstance(val_cfg, dict) and "train_only" in val_cfg:
        train_only = bool(val_cfg["train_only"])

    top = load_top_train_configs(
        leaderboard_path,
        n,
        loaded.summary_sort_by,
        train_only=train_only,
        initial_equity=loaded.initial_equity,
        drawdown_penalty=loaded.drawdown_penalty,
    )
    if top.empty:
        raise ValueError("no configurations to validate")

    sweep_total: int | None = None
    if leaderboard_path.name == "sweep_results.csv" and leaderboard_path.exists():
        sweep_total = len(pd.read_csv(leaderboard_path))

    val_split = split_config_for_validation(loaded.config)
    started = datetime.now(timezone.utc)
    out_dir = make_timestamped_run_dir(
        output_base or RESULTS_DIR, run_at=started, prefix="validation_"
    )
    if sweep_total is not None:
        print(
            f"Validation: testing top {n} of {sweep_total} train configs "
            f"(from {leaderboard_path.name})"
        )

    rows: list[dict[str, Any]] = []
    for _, train_row in top.iterrows():
        params = params_from_row(train_row, loaded.config)
        train_rank = int(train_row["rank"]) if "rank" in train_row else None
        label = (
            f"{loaded.config_path}#train_rank={train_rank}_"
            f"{combo_label(params.combo)}"
        )

        result = execute_backtest(
            loaded,
            params,
            run_at=started,
            export_artifacts=False,
            append_global_log=append_runs_log,
            update_ranked_summary=False,
            split=val_split,
            data_split="validation",
            config_label=label,
        )
        val_metrics = summary_row(
            result.card,
            initial_equity=loaded.initial_equity,
            drawdown_penalty=loaded.drawdown_penalty,
        )
        row: dict[str, Any] = {
            "train_rank": train_rank,
            **params.combo,
            "train_composite_score": train_row.get("composite_score"),
            "train_net_pnl": train_row.get("net_pnl"),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        rows.append(row)
        print(
            f"train_rank={train_rank} {combo_label(params.combo)} "
            f"-> val net_pnl=${result.card.net_pnl:,.2f} "
            f"(train was ${float(train_row.get('net_pnl', 0)):,.2f})"
        )

    df = pd.DataFrame(rows)
    sort_col = f"val_{loaded.summary_sort_by}"
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False, kind="mergesort")
    df.insert(0, "val_rank", range(1, len(df) + 1))

    out_path = out_dir / VALIDATION_RESULTS_CSV
    df.to_csv(out_path, index=False)

    meta = {
        "validated_at": started.isoformat(),
        "config_path": str(loaded.config_path),
        "leaderboard": str(leaderboard_path),
        "validation_window": {
            "start": val_split.test.start,
            "end": val_split.test.end,
        },
        "top_n": n,
        "sort_by": loaded.summary_sort_by,
        "results_csv": str(out_path),
    }
    (out_dir / "validation_metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )

    if append_runs_log and update_summary:
        from bt.report import write_ranked_summary_csv

        write_ranked_summary_csv(
            loaded.runs_log_path,
            loaded.ranked_summary_path,
            sort_by=loaded.summary_sort_by,
            initial_equity=loaded.initial_equity,
            drawdown_penalty=loaded.drawdown_penalty,
            top_n=loaded.summary_top_n,
        )

    return out_dir, df
