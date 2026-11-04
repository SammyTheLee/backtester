"""Run top validation configs on the out-of-sample window."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from bt.paths import (
    OOS_RESULTS_CSV,
    RESULTS_DIR,
    VALIDATION_RESULTS_CSV,
    make_timestamped_run_dir,
)
from bt.walkforward import write_walkforward_summary
from bt.pipeline import LoadedRunConfig, execute_backtest
from bt.report import rank_dataframe, summary_row
from bt.splits import split_config_for_oos
from bt.sweep import combo_label
from bt.validate_run import params_from_row


def latest_validation_results(base: Path | None = None) -> Path | None:
    """Newest ``results/validation_*/validation_results.csv``, if any."""
    base = Path(base or RESULTS_DIR)
    if not base.exists():
        return None
    for val_dir in sorted(base.glob("validation_*"), key=lambda p: p.name, reverse=True):
        path = val_dir / VALIDATION_RESULTS_CSV
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def resolve_validation_leaderboard(
    loaded: LoadedRunConfig,
    *,
    source: Path | None = None,
    validation_results: Path | None = None,
) -> Path:
    if source is not None:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"validation results not found: {path}")
        return path

    if validation_results is not None:
        path = Path(validation_results)
        if not path.exists():
            raise FileNotFoundError(f"validation results not found: {path}")
        return path

    oos_cfg = loaded.config.get("oos", {})
    if isinstance(oos_cfg, dict):
        if oos_cfg.get("source"):
            path = Path(oos_cfg["source"])
            if not path.exists():
                raise FileNotFoundError(f"oos.source not found: {path}")
            return path
        if oos_cfg.get("validation_dir"):
            path = Path(oos_cfg["validation_dir"]) / VALIDATION_RESULTS_CSV
            if not path.exists():
                raise FileNotFoundError(f"oos.validation_dir missing {path}")
            return path

    latest = latest_validation_results()
    if latest is not None:
        return latest

    raise FileNotFoundError(
        "No validation results found. Run: python -m bt validate --config ... "
        "or pass --from / --validation-results."
    )


def load_top_validation_configs(
    path: Path,
    top_n: int,
    sort_by: str,
    *,
    initial_equity: float = 10_000.0,
    drawdown_penalty: float = 1.0,
) -> pd.DataFrame:
    """Top N rows from validation_results.csv (by val_rank or val_{sort_by})."""
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"validation results empty: {path}")

    val_sort = f"val_{sort_by}"
    if "val_rank" in df.columns:
        df = df.sort_values("val_rank", kind="mergesort")
    elif val_sort in df.columns:
        df = df.sort_values(val_sort, ascending=False, kind="mergesort")
    else:
        df = rank_dataframe(
            df,
            sort_by,
            initial_equity=initial_equity,
            drawdown_penalty=drawdown_penalty,
        )

    return df.head(top_n).reset_index(drop=True)


def run_oos(
    loaded: LoadedRunConfig,
    *,
    top_n: int | None = None,
    source: Path | None = None,
    validation_results: Path | None = None,
    append_runs_log: bool = True,
    update_summary: bool = True,
    output_base: Path | None = None,
    walkforward_run_dir: Path | None = None,
) -> tuple[Path, pd.DataFrame, Path]:
    """Backtest top validation configs on OOS; return output dir, results, walkforward CSV."""
    oos_cfg = loaded.config.get("oos", {})
    n = top_n
    if n is None and isinstance(oos_cfg, dict) and oos_cfg.get("top_n") is not None:
        n = int(oos_cfg["top_n"])
    if n is None:
        val_cfg = loaded.config.get("validation", {})
        if isinstance(val_cfg, dict) and val_cfg.get("top_n") is not None:
            n = int(val_cfg["top_n"])
        else:
            n = loaded.summary_top_n if loaded.summary_top_n is not None else 10

    leaderboard_path = resolve_validation_leaderboard(
        loaded, source=source, validation_results=validation_results
    )
    top = load_top_validation_configs(
        leaderboard_path,
        n,
        loaded.summary_sort_by,
        initial_equity=loaded.initial_equity,
        drawdown_penalty=loaded.drawdown_penalty,
    )
    if top.empty:
        raise ValueError("no configurations to test out-of-sample")

    val_total = len(pd.read_csv(leaderboard_path))
    print(
        f"Out-of-sample: testing top {n} of {val_total} validation configs "
        f"(from {leaderboard_path.name})"
    )

    oos_split = split_config_for_oos(loaded.config)
    started = datetime.now(timezone.utc)
    out_dir = make_timestamped_run_dir(
        output_base or RESULTS_DIR, run_at=started, prefix="oos_"
    )

    rows: list[dict[str, Any]] = []
    for _, val_row in top.iterrows():
        params = params_from_row(val_row, loaded.config)
        val_rank = int(val_row["val_rank"]) if "val_rank" in val_row else None
        label = (
            f"{loaded.config_path}#val_rank={val_rank}_"
            f"{combo_label(params.combo)}"
        )

        result = execute_backtest(
            loaded,
            params,
            run_at=started,
            export_artifacts=False,
            append_global_log=append_runs_log,
            update_ranked_summary=False,
            split=oos_split,
            data_split="oos",
            config_label=label,
        )
        oos_metrics = summary_row(
            result.card,
            initial_equity=loaded.initial_equity,
            drawdown_penalty=loaded.drawdown_penalty,
        )
        row: dict[str, Any] = {
            "val_rank": val_rank,
            "train_rank": val_row.get("train_rank"),
            **params.combo,
            "val_composite_score": val_row.get("val_composite_score"),
            "val_net_pnl": val_row.get("val_net_pnl"),
            **{f"oos_{k}": v for k, v in oos_metrics.items()},
        }
        rows.append(row)
        val_pnl = float(val_row.get("val_net_pnl", 0) or 0)
        print(
            f"val_rank={val_rank} {combo_label(params.combo)} "
            f"-> oos net_pnl=${result.card.net_pnl:,.2f} "
            f"(validation was ${val_pnl:,.2f})"
        )

    df = pd.DataFrame(rows)
    sort_col = f"oos_{loaded.summary_sort_by}"
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False, kind="mergesort")
    df.insert(0, "oos_rank", range(1, len(df) + 1))

    out_path = out_dir / OOS_RESULTS_CSV
    df.to_csv(out_path, index=False)

    wf_path = write_walkforward_summary(
        df,
        leaderboard_path,
        sort_by=loaded.summary_sort_by,
        run_dir=walkforward_run_dir or out_dir,
    )

    meta = {
        "oos_at": started.isoformat(),
        "config_path": str(loaded.config_path),
        "validation_leaderboard": str(leaderboard_path),
        "oos_window": {"start": oos_split.oos.start, "end": oos_split.oos.end},
        "top_n": n,
        "sort_by": loaded.summary_sort_by,
        "results_csv": str(out_path),
        "walkforward_summary": str(wf_path),
    }
    (out_dir / "oos_metadata.json").write_text(
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

    return out_dir, df, wf_path
