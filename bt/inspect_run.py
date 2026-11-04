"""Inspect a saved backtest run folder (metadata, report card, artifacts)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from bt.paths import (
    METADATA_JSON,
    OOS_RESULTS_CSV,
    REPORT_JSON,
    REPORT_MD,
    RESULTS_DIR,
    SUMMARY_CSV,
    SWEEP_METADATA_JSON,
    SWEEP_RESULTS_CSV,
    TRADES_CSV,
    VALIDATION_RESULTS_CSV,
    WALKFORWARD_SUMMARY_CSV,
    latest_run_dir,
)
from bt.report import StrategyReportCard, format_report_card


def resolve_results_dir(path: Path | str | None) -> Path:
    """Resolve ``latest``, ``results/latest``, or a concrete run folder."""
    if path is None:
        resolved = latest_run_dir()
        if resolved is None:
            raise FileNotFoundError("no run folders under results/")
        return resolved

    raw = Path(path)
    name = raw.as_posix().rstrip("/").lower()
    if name in ("latest", "results/latest", "results"):
        resolved = latest_run_dir(RESULTS_DIR)
        if resolved is None:
            raise FileNotFoundError(f"no run folders under {RESULTS_DIR}/")
        return resolved

    if not raw.exists():
        raise FileNotFoundError(f"run folder not found: {raw}")

    if raw.is_file():
        raise ValueError(f"expected a directory, got file: {raw}")

    return raw


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _card_from_report(path: Path) -> StrategyReportCard | None:
    data = _load_json(path)
    if not data:
        return None
    return StrategyReportCard(
        gross_pnl=float(data["gross_pnl"]),
        net_pnl=float(data["net_pnl"]),
        total_return_pct=float(data["total_return_pct"]),
        trade_count=int(data["trade_count"]),
        win_rate=float(data["win_rate"]),
        profit_factor=data.get("profit_factor"),
        max_drawdown=float(data["max_drawdown"]),
        average_hold_hours=data.get("average_hold_hours"),
        time_in_market_pct=float(data["time_in_market_pct"]),
    )


def _detect_run_kind(run_dir: Path) -> str:
    name = run_dir.name
    if name.startswith("walkforward_"):
        return "walkforward"
    if name.startswith("sweep_"):
        return "sweep"
    if name.startswith("validation_"):
        return "validation"
    if name.startswith("oos_"):
        return "oos"
    return "backtest"


def inspect_run_folder(run_dir: Path) -> list[str]:
    """Build human-readable lines summarizing a results folder."""
    run_dir = Path(run_dir)
    kind = _detect_run_kind(run_dir)
    lines = [f"Run folder : {run_dir}", f"Kind       : {kind}"]

    meta_candidates = {
        "backtest": run_dir / METADATA_JSON,
        "sweep": run_dir / SWEEP_METADATA_JSON,
        "validation": run_dir / "validation_metadata.json",
        "oos": run_dir / "oos_metadata.json",
        "walkforward": run_dir / "walkforward_metadata.json",
    }
    meta_path = meta_candidates.get(kind, run_dir / METADATA_JSON)
    meta = _load_json(meta_path)
    if meta:
        lines.append(f"Metadata   : {meta_path}")
        for key in (
            "config_path",
            "validated_at",
            "oos_at",
            "walkforward_at",
            "sweep_at",
            "sort_by",
            "top_n",
            "train_top_n",
            "validation_top_n",
        ):
            if key in meta:
                lines.append(f"  {key}: {meta[key]}")
        if "validation_window" in meta:
            w = meta["validation_window"]
            lines.append(f"  validation: {w.get('start')} .. {w.get('end')}")
        if "oos_window" in meta:
            w = meta["oos_window"]
            lines.append(f"  oos       : {w.get('start')} .. {w.get('end')}")
        resolved = meta.get("resolved", {})
        if isinstance(resolved, dict) and resolved:
            lines.append(
                f"  strategy  : {resolved.get('strategy')} "
                f"(split={resolved.get('data_split', 'all')})"
            )
            lines.append(f"  data      : {resolved.get('data_csv')}")

    report_path = run_dir / REPORT_JSON
    card = _card_from_report(report_path)
    if card is not None:
        lines.append("")
        lines.extend(format_report_card(card).splitlines())
    elif (run_dir / SUMMARY_CSV).exists():
        df = pd.read_csv(run_dir / SUMMARY_CSV)
        if not df.empty:
            lines.append("")
            lines.append("=== Summary (from summary.csv) ===")
            lines.append(df.to_string(index=False))

    if (run_dir / REPORT_MD).exists():
        lines.append("")
        lines.append(f"Plain-language report: {run_dir / REPORT_MD}")

    if (run_dir / "portfolio_equity.csv").exists():
        lines.append("  portfolio_equity.csv")
    if (run_dir / "rejected_trades.csv").exists():
        lines.append("  rejected_trades.csv")

    extra_tables = [
        ("sweep_results.csv", SWEEP_RESULTS_CSV),
        ("validation_results.csv", VALIDATION_RESULTS_CSV),
        ("oos_results.csv", OOS_RESULTS_CSV),
        ("walkforward_summary.csv", WALKFORWARD_SUMMARY_CSV),
    ]
    for label, fname in extra_tables:
        p = run_dir / fname
        if p.exists():
            df = pd.read_csv(p)
            lines.append("")
            lines.append(f"=== {label} ({len(df)} rows, top 5) ===")
            lines.append(df.head(5).to_string(index=False))

    lines.append("")
    lines.append("=== Artifacts ===")
    for p in sorted(run_dir.iterdir()):
        if p.is_file():
            lines.append(f"  {p.name}")
        elif p.is_dir() and kind == "sweep":
            lines.append(f"  {p.name}/")

    trades = run_dir / TRADES_CSV
    if trades.exists():
        n = len(pd.read_csv(trades))
        lines.append(f"  trades.csv ({n} rows)")

    return lines
