"""Persist run config and model assumptions to metadata.json."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from bt.costs import ExecutionCosts
from bt.splits import SplitConfig, split_summary

# Documented model assumptions (not inferred from code at runtime).
DEFAULT_ASSUMPTIONS: dict[str, Any] = {
    "timing": [
        "Signal is known at the close of bar N.",
        "Entries and opposing-signal exits fill at the open of bar N+1.",
        "Stops and targets are checked with each bar high/low from the entry bar.",
        "If stop and target both trade inside one bar, stop is assumed first.",
    ],
    "strategy": {
        "type": "ma_crossover",
        "long": "fast_ma crosses above slow_ma",
        "short": "fast_ma crosses below slow_ma",
        "signals": "Sparse events only at crossover bars (not every bar in a trend).",
    },
    "costs": {
        "fee": "fee = notional * fee_bps / 10000 (charged on entry and exit)",
        "slippage_rate": "slippage_bps / 10000",
        "long_entry": "open * (1 + slippage_rate)",
        "short_entry": "open * (1 - slippage_rate)",
        "long_exit": "raw_exit * (1 - slippage_rate)",
        "short_exit": "raw_exit * (1 + slippage_rate)",
        "holding": "optional per-bar: close * size * holding_cost_bps / 10000",
    },
    "pnl": {
        "gross_pnl": "Price PnL at raw entry_open and exit_raw",
        "net_pnl": "gross_pnl - slippage_cost - entry_fee - exit_fee - holding_cost",
        "equity_curve": "Entry fee at open; gross - slippage - exit fee - holding at close",
    },
    "ranking": {
        "runs_log": "results/runs.csv appends one row per backtest (full history)",
        "ranked_summary": "results/summary.csv = top N OOS configs (deduped by fast/slow/stop/target, ranked by summary_sort_by)",
        "composite_score": "total_return_pct - drawdown_penalty * (max_drawdown / initial_equity); higher is better",
        "default_sort": "composite_score descending",
        "run_folder": "Each run also writes trades/equity/summary CSV and metadata/report JSON under results/YYYYMMDD_HHMMSS/",
    },
}


def build_run_metadata(
    config: dict[str, Any],
    *,
    config_path: Path | str,
    csv_path: Path | str,
    strategy_name: str,
    fast: int,
    slow: int,
    stop_pct: float | None,
    target_pct: float | None,
    costs: ExecutionCosts,
    initial_equity: float,
    size: float,
    summary_sort_by: str,
    candles_df: pd.DataFrame | None = None,
    run_at: str | None = None,
    output_dir: str | None = None,
    data_split: str = "all",
    split: SplitConfig | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable metadata document for one backtest run."""
    exec_cfg = config.get("execution", {})
    assumptions = {
        **DEFAULT_ASSUMPTIONS,
        "costs_applied": {
            "fee_bps": costs.fee_bps,
            "slippage_bps": costs.slippage_bps,
            "holding_cost_bps": costs.holding_cost_bps,
            "stop_pct": stop_pct,
            "target_pct": target_pct,
        },
    }

    resolved = {
        "data_csv": str(csv_path),
        "strategy": strategy_name,
        "fast": fast,
        "slow": slow,
        "stop_pct": stop_pct,
        "target_pct": target_pct,
        "initial_equity": initial_equity,
        "size": size,
        "summary_sort_by": summary_sort_by,
        "data_split": data_split,
    }

    data_info: dict[str, Any] = {"split": split_summary(split)}
    if candles_df is not None and len(candles_df) > 0:
        data_info["rows"] = len(candles_df)
        if "timestamp" in candles_df.columns:
            data_info["time_start"] = str(candles_df["timestamp"].iloc[0])
            data_info["time_end"] = str(candles_df["timestamp"].iloc[-1])

    doc: dict[str, Any] = {
        "run_at": run_at or datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config": config,
        "resolved": resolved,
        "assumptions": assumptions,
        "data": data_info,
    }
    if output_dir is not None:
        doc["output_dir"] = output_dir
    return doc


def save_metadata(
    metadata: dict[str, Any],
    path: Path | str,
) -> Path:
    """Write metadata dict to JSON (UTF-8, indented)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metadata, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return path


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
