"""Plain-language report.md for non-technical readers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from bt.paths import METADATA_JSON, REPORT_MD, SUMMARY_CSV, TRADES_CSV, WALKFORWARD_SUMMARY_CSV

WALKFORWARD_METADATA_JSON = "walkforward_metadata.json"
from bt.report import StrategyReportCard

_STRATEGY_LABELS = {
    "ma_crossover": "Moving-average crossover",
    "rsi": "RSI (relative strength index)",
    "vol_gated_ma": "Volatility-gated moving-average crossover",
    "regime_gated_ma": "Regime-gated moving-average crossover",
}


def save_report_md(text: str, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return path


def _money(value: float | None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    return f"${value:,.2f}"


def _pct(value: float | None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    return f"{value * 100:+.2f}%"


def _window_line(label: str, window: dict[str, Any] | None) -> str:
    if not window:
        return f"- **{label}:** not used"
    start = window.get("start") or "?"
    end = window.get("end") or "?"
    return f"- **{label}:** {start} to {end}"


def _strategy_plain(config: dict[str, Any], resolved: dict[str, Any]) -> str:
    strat = config.get("strategy", {})
    if not isinstance(strat, dict):
        strat = {}
    typ = str(strat.get("type", "ma_crossover"))
    label = _STRATEGY_LABELS.get(typ, typ.replace("_", " ").title())
    lines = [f"This test used **{label}**."]

    if typ in ("ma_crossover", "vol_gated_ma", "regime_gated_ma"):
        fast = strat.get("fast") or strat.get("fast_ma") or resolved.get("fast")
        slow = strat.get("slow") or strat.get("slow_ma") or resolved.get("slow")
        if fast is not None and slow is not None:
            lines.append(
                f"It compares a **{fast}-bar fast average** to a **{slow}-bar slow average** "
                "and trades when they cross."
            )
    if typ == "regime_gated_ma":
        col = strat.get("regime_column", "risk_on")
        lines.append(f"Trades are only allowed when the **{col}** column in the data is active.")
    if typ == "rsi":
        period = strat.get("period", strat.get("rsi_period"))
        if period is not None:
            lines.append(f"RSI period: **{period}** bars.")

    stop = resolved.get("stop_pct") or strat.get("stop_pct")
    target = resolved.get("target_pct") or strat.get("target_pct")
    if stop is not None or target is not None:
        parts = []
        if stop is not None:
            parts.append(f"**{float(stop) * 100:g}% stop-loss**")
        if target is not None:
            parts.append(f"**{float(target) * 100:g}% take-profit**")
        lines.append("Each trade uses " + " and ".join(parts) + ".")

    portfolio = config.get("portfolio", {})
    if isinstance(portfolio, dict):
        equity = portfolio.get("initial_equity")
        if equity is not None:
            lines.append(f"Starting account size: **{_money(float(equity))}**.")

    return "\n".join(lines)


def _costs_plain(assumptions: dict[str, Any]) -> str:
    costs = assumptions.get("costs_applied", {})
    if not isinstance(costs, dict):
        costs = {}
    fee = float(costs.get("fee_bps", 0) or 0)
    slip = float(costs.get("slippage_bps", 0) or 0)
    holding = float(costs.get("holding_cost_bps", 0) or 0)
    lines = [
        "Every simulated trade included realistic frictions:",
        f"- **Trading fee:** {fee:g} basis points ({fee / 100:.2f}%) on entry and exit",
        f"- **Slippage:** {slip:g} basis points ({slip / 100:.2f}%) on fills",
    ]
    if holding:
        lines.append(f"- **Holding cost:** {holding:g} basis points per bar while in a position")
    else:
        lines.append("- **Holding cost:** none")
    lines.append(
        "\nThese costs reduce the net profit compared with a perfect-price backtest."
    )
    return "\n".join(lines)


def _assumptions_plain(assumptions: dict[str, Any]) -> str:
    """Turn metadata assumptions into readable prose."""
    if not assumptions:
        return "No model assumptions were recorded for this run."

    lines = [
        "The simulation followed these rules (see `metadata.json` for the full record):",
        "",
    ]

    timing = assumptions.get("timing")
    if isinstance(timing, list) and timing:
        lines.append("**When trades happen**")
        for item in timing:
            lines.append(f"- {item}")
        lines.append("")

    strategy = assumptions.get("strategy")
    if isinstance(strategy, dict):
        lines.append("**How signals work**")
        for key in ("long", "short", "signals", "type"):
            if key in strategy and strategy[key]:
                label = key.replace("_", " ").capitalize()
                lines.append(f"- {label}: {strategy[key]}")
        lines.append("")

    costs = assumptions.get("costs")
    if isinstance(costs, dict):
        lines.append("**How costs are calculated**")
        for key, text in costs.items():
            if text:
                lines.append(f"- {key.replace('_', ' ').capitalize()}: {text}")
        lines.append("")

    pnl = assumptions.get("pnl")
    if isinstance(pnl, dict):
        lines.append("**How profit is measured**")
        for key, text in pnl.items():
            if text:
                lines.append(f"- {key.replace('_', ' ').capitalize()}: {text}")

    return "\n".join(lines).rstrip()


def _trades_plain(trades: pd.DataFrame) -> str:
    """Summarize trades.csv in plain English."""
    if trades.empty:
        return "No trades were recorded in `trades.csv`."

    closed = trades[trades["exit_time"].notna()] if "exit_time" in trades.columns else trades
    if closed.empty:
        open_count = len(trades)
        return f"**{open_count}** position(s) were still open at the end of the test."

    net_col = "net_pnl" if "net_pnl" in closed.columns else None
    wins = losses = 0
    best = worst = 0.0
    if net_col:
        vals = pd.to_numeric(closed[net_col], errors="coerce").fillna(0.0)
        wins = int((vals > 0).sum())
        losses = int((vals <= 0).sum())
        if len(vals):
            best = float(vals.max())
            worst = float(vals.min())

    lines = [
        f"- **Closed trades in file:** {len(closed)}",
        f"- **Winning trades:** {wins}",
        f"- **Losing or flat trades:** {losses}",
    ]
    if net_col and len(closed):
        lines.append(f"- **Best single trade (net):** {_money(best)}")
        lines.append(f"- **Worst single trade (net):** {_money(worst)}")

    reasons = closed.get("exit_reason")
    if reasons is not None:
        counts = reasons.value_counts().head(4)
        if not counts.empty:
            lines.append("- **How trades ended:**")
            for reason, n in counts.items():
                lines.append(f"  - {reason}: {n}")

    return "\n".join(lines)


def card_from_summary_row(row: pd.Series) -> StrategyReportCard:
    """Build a report card from one ``summary.csv`` row."""
    pf = row.get("profit_factor")
    if pf is not None and not pd.isna(pf):
        pf_val: float | None = float(pf)
        if pf_val == float("inf"):
            pf_val = float("inf")
    else:
        pf_val = None

    avg_hold = row.get("average_hold_hours")
    return StrategyReportCard(
        gross_pnl=float(row.get("gross_pnl", 0) or 0),
        net_pnl=float(row.get("net_pnl", 0) or 0),
        total_return_pct=float(row.get("total_return_pct", 0) or 0),
        trade_count=int(row.get("trade_count", 0) or 0),
        win_rate=float(row.get("win_rate", 0) or 0),
        profit_factor=pf_val,
        max_drawdown=float(row.get("max_drawdown", 0) or 0),
        average_hold_hours=None if pd.isna(avg_hold) else float(avg_hold),
        time_in_market_pct=float(row.get("time_in_market_pct", 0) or 0),
    )


def load_run_artifacts(
    run_dir: Path | str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Load metadata.json, summary.csv, and trades.csv from a run folder."""
    run_dir = Path(run_dir)
    meta_path = run_dir / METADATA_JSON
    summary_path = run_dir / SUMMARY_CSV
    trades_path = run_dir / TRADES_CSV

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}")
    if not trades_path.exists():
        raise FileNotFoundError(f"Missing {trades_path}")

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    summary = pd.read_csv(summary_path)
    trades = pd.read_csv(trades_path)
    return metadata, summary, trades


def load_walkforward_artifacts(
    run_dir: Path | str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Load walkforward_metadata.json and walkforward_summary.csv."""
    run_dir = Path(run_dir)
    meta_path = run_dir / WALKFORWARD_METADATA_JSON
    summary_path = run_dir / WALKFORWARD_SUMMARY_CSV
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}")
    manifest = json.loads(meta_path.read_text(encoding="utf-8"))
    summary = pd.read_csv(summary_path)
    return manifest, summary


def generate_report_md_from_run_dir(
    run_dir: Path | str,
    *,
    output_path: Path | str | None = None,
) -> Path:
    """Build report.md from a backtest or walk-forward run folder."""
    run_dir = Path(run_dir)
    out = Path(output_path) if output_path is not None else run_dir / REPORT_MD

    wf_meta = run_dir / WALKFORWARD_METADATA_JSON
    wf_summary = run_dir / WALKFORWARD_SUMMARY_CSV
    if wf_meta.exists() or wf_summary.exists():
        if not wf_meta.exists():
            raise FileNotFoundError(f"Missing {wf_meta}")
        if not wf_summary.exists():
            raise FileNotFoundError(f"Missing {wf_summary}")
        manifest, summary = load_walkforward_artifacts(run_dir)
        text = build_walkforward_report_md(manifest, summary)
        return save_report_md(text, out)

    metadata, summary, trades = load_run_artifacts(run_dir)
    if summary.empty:
        raise ValueError(f"{run_dir / SUMMARY_CSV} has no rows")

    card = card_from_summary_row(summary.iloc[0])
    text = build_report_md(metadata, card, trades=trades)
    return save_report_md(text, out)


def _split_section(metadata: dict[str, Any]) -> str:
    data = metadata.get("data", {})
    split = data.get("split", {}) if isinstance(data, dict) else {}
    active = split.get("active", metadata.get("resolved", {}).get("data_split", "all"))

    if active == "all" or not split:
        time_start = data.get("time_start", "?") if isinstance(data, dict) else "?"
        time_end = data.get("time_end", "?") if isinstance(data, dict) else "?"
        return (
            "This was a **single backtest on the full dataset** — there was no "
            "separate train, validation, or out-of-sample period.\n"
            f"- **Data range tested:** {time_start} to {time_end}"
        )

    labels = {
        "train": "Training",
        "test": "Validation",
        "validation": "Validation",
        "oos": "Out-of-sample",
    }
    label = labels.get(str(active), str(active))
    windows = split.get("windows", {}) if isinstance(split, dict) else {}
    if windows:
        parts = [_window_line(label, windows.get(str(active)) or windows.get(active))]
    else:
        parts = [f"- **Period used:** {label}"]
    parts.insert(
        0,
        f"This run tested only the **{label.lower()}** portion of the configured date splits.",
    )
    return "\n".join(parts)


def _weakness_single(card: StrategyReportCard, metadata: dict[str, Any]) -> str:
    issues: list[str] = []
    if card.trade_count == 0:
        issues.append("The strategy **did not take any trades**, so there is no evidence it works.")
    elif card.trade_count < 5:
        issues.append(
            f"Only **{card.trade_count} trades** were taken — too few to draw reliable conclusions."
        )
    if card.net_pnl < 0:
        issues.append(
            f"The strategy **lost money** after costs ({_money(card.net_pnl)} net)."
        )
    if card.win_rate < 0.4 and card.trade_count >= 3:
        issues.append(
            f"The **win rate was low** ({card.win_rate * 100:.0f}% of trades were profitable)."
        )
    if card.max_drawdown > abs(card.net_pnl) + 1 and card.trade_count > 0:
        issues.append(
            f"**Drawdowns were large** ({_money(card.max_drawdown)}) relative to net profit."
        )

    data = metadata.get("data", {})
    rows = data.get("rows") if isinstance(data, dict) else None
    if isinstance(rows, int) and rows < 100:
        issues.append(
            f"The dataset is **very short** ({rows} price bars), so results may not repeat in live trading."
        )

    csv_path = str(metadata.get("resolved", {}).get("data_csv", ""))
    if "DEMO" in csv_path.upper():
        issues.append(
            "The data file looks like **synthetic demo data**, not real market history."
        )

    if not issues:
        return (
            "No major red flags stand out in this single run, but one backtest is "
            "**not proof** the strategy will work going forward. Confirm on fresh "
            "out-of-sample data before risking real capital."
        )
    return "**Biggest concerns:**\n\n" + "\n".join(f"- {item}" for item in issues)


def build_report_md(
    metadata: dict[str, Any],
    card: StrategyReportCard,
    *,
    trades: pd.DataFrame | None = None,
) -> str:
    """Build a plain-language report for one backtest run."""
    config = metadata.get("config", {})
    resolved = metadata.get("resolved", {})
    assumptions = metadata.get("assumptions", {})
    if not isinstance(config, dict):
        config = {}
    if not isinstance(resolved, dict):
        resolved = {}
    if not isinstance(assumptions, dict):
        assumptions = {}

    data_csv = resolved.get("data_csv") or config.get("data", {}).get("csv_path", "unknown")
    data = metadata.get("data", {})
    rows = data.get("rows", "?") if isinstance(data, dict) else "?"
    time_start = data.get("time_start", "?") if isinstance(data, dict) else "?"
    time_end = data.get("time_end", "?") if isinstance(data, dict) else "?"

    pf = card.profit_factor
    pf_text = (
        "infinite (no losing trades)"
        if pf == float("inf")
        else f"{pf:.2f}"
        if pf is not None
        else "n/a"
    )
    hold = (
        f"{card.average_hold_hours:.1f} hours"
        if card.average_hold_hours is not None
        else "n/a"
    )

    trades_section = (
        _trades_plain(trades) if trades is not None else "See `trades.csv` in this folder."
    )

    return f"""# Backtest Report

This report summarizes one strategy test. It is written for readers who do not need to know how the code works.

## What data was used?

- **Price file:** `{data_csv}`
- **Bars in test:** {rows}
- **Date range:** {time_start} to {time_end}

## What strategy was tested?

{_strategy_plain(config, resolved)}

## What costs were assumed?

{_costs_plain(assumptions)}

## What dates were train, validation, and out-of-sample?

{_split_section(metadata)}

## What was the best out-of-sample config?

This was a **single run**, not a parameter search. The configuration tested here is the one listed under “What strategy was tested?” above.

There is **no separate “best out-of-sample” pick** from a sweep for this run.

## How many trades happened?

Summary from `summary.csv`:

- **Closed trades:** {card.trade_count}
- **Net profit (after costs):** {_money(card.net_pnl)}
- **Return on starting equity:** {_pct(card.total_return_pct)}
- **Win rate:** {card.win_rate * 100:.1f}%
- **Profit factor:** {pf_text}
- **Largest peak-to-trough drop:** {_money(card.max_drawdown)}
- **Average time in each trade:** {hold}
- **Share of bars with an open position:** {card.time_in_market_pct * 100:.1f}%

Details from `trades.csv`:

{trades_section}

## Model assumptions

{_assumptions_plain(assumptions)}

## What is the biggest weakness of the result?

{_weakness_single(card, metadata)}
"""


def _config_params_text(row: pd.Series) -> str:
    parts: list[str] = []
    for key, label in (
        ("fast", "fast average"),
        ("slow", "slow average"),
        ("stop_pct", "stop"),
        ("target_pct", "target"),
    ):
        if key in row and pd.notna(row[key]):
            val = row[key]
            if key in ("stop_pct", "target_pct"):
                parts.append(f"{label} {float(val) * 100:g}%")
            else:
                parts.append(f"{label} {int(val)}")
    return ", ".join(parts) if parts else "see walkforward_summary.csv"


def _weakness_walkforward(
    best: pd.Series,
    wf: pd.DataFrame,
    manifest: dict[str, Any],
) -> str:
    issues: list[str] = []

    oos_trades = int(best.get("oos_trade_count", 0) or 0)
    oos_net = float(best.get("oos_net_pnl", 0) or 0)
    val_net = float(best.get("val_net_pnl", 0) or 0)
    train_net = float(best.get("train_net_pnl", 0) or 0)

    if oos_trades == 0:
        issues.append(
            "The top-ranked out-of-sample config **did not trade at all**, so it cannot be trusted as a winner."
        )
    elif oos_trades < 3:
        issues.append(
            f"Out-of-sample activity was **very thin** ({oos_trades} trades only)."
        )
    if oos_net <= 0 and oos_trades > 0:
        issues.append(
            f"Out-of-sample net profit was **not positive** ({_money(oos_net)})."
        )
    if val_net < 0:
        issues.append(
            f"Validation lost money ({_money(val_net)}), so train-period strength **did not carry forward**."
        )
    if train_net > 0 and oos_net < train_net * 0.1:
        issues.append(
            "Profit **shrank sharply** from training to out-of-sample — a classic overfitting warning."
        )

    train_w = manifest.get("train_window", {})
    if train_w.get("start") and train_w.get("end"):
        issues.append(
            "Each walk-forward window is **short**; rankings can change a lot with more data."
        )

    config_path = str(manifest.get("config_path", ""))
    if "DEMO" in config_path or (
        wf.shape[0] > 0 and "DEMO" in str(manifest.get("sweep_results_csv", ""))
    ):
        issues.append("Results use **demo/synthetic data**, not live market history.")

    active = wf[wf["oos_trade_count"].fillna(0) > 0] if "oos_trade_count" in wf.columns else wf
    if oos_trades == 0 and not active.empty:
        alt = active.iloc[0]
        issues.append(
            "A lower-ranked config actually traded out-of-sample — see the full summary table."
        )

    if not issues:
        return (
            "Walk-forward results look reasonable on the surface, but **always confirm** "
            "on longer history and real data before trading live."
        )
    return "**Biggest concerns:**\n\n" + "\n".join(f"- {item}" for item in issues)


def build_walkforward_report_md(
    manifest: dict[str, Any],
    walkforward_summary: pd.DataFrame,
) -> str:
    """Build a plain-language report for a walk-forward run."""
    if walkforward_summary.empty:
        raise ValueError("walkforward_summary is empty")

    best = walkforward_summary.iloc[0]
    config_path = manifest.get("config_path", "unknown")
    sort_by = manifest.get("sort_by", "composite_score")
    n_sweep = manifest.get("sweep_combination_count", "?")
    n_val = manifest.get("validation_candidate_count", "?")
    n_oos = manifest.get("oos_candidate_count", "?")

    strat_type = "ma_crossover"
    # Best-effort: walk-forward uses param_sweep config
    strategy_line = (
        "This was a **moving-average crossover** strategy. "
        "The system tried many combinations of fast/slow averages and stop/target sizes "
        "on training data, kept the best few for validation, then tested survivors out-of-sample."
    )

    exec_assumptions = {
        "costs_applied": {
            "fee_bps": 10,
            "slippage_bps": 5,
            "holding_cost_bps": 0,
        }
    }

    oos_trades = int(best.get("oos_trade_count", 0) or 0)
    oos_net = float(best.get("oos_net_pnl", 0) or 0)

    return f"""# Walk-Forward Backtest Report

This report explains a full **train → validation → out-of-sample** study in plain language.

**Config file:** `{config_path}`  
**Ranking method:** `{sort_by}` (higher is better)

## What data was used?

- **Price file:** from the config (typically `data/candles/DEMO.csv` in demo runs)
- **Parameter combinations tried on training data:** {n_sweep}
- **Configs forwarded to validation:** {n_val}
- **Configs tested out-of-sample:** {n_oos}

## What strategy was tested?

{strategy_line}

**Best out-of-sample settings (rank 1):** {_config_params_text(best)}

## What costs were assumed?

{_costs_plain(exec_assumptions)}

## What dates were train, validation, and out-of-sample?

{_window_line("Training", manifest.get("train_window"))}
{_window_line("Validation", manifest.get("validation_window"))}
{_window_line("Out-of-sample", manifest.get("oos_window"))}

Training is where parameters were searched. Validation checks whether train winners still work on fresh dates. Out-of-sample is the final unseen period.

## What was the best out-of-sample config?

**Rank 1 by {sort_by}:**

| Setting | Value |
|---------|-------|
| Fast average | {best.get("fast", "n/a")} |
| Slow average | {best.get("slow", "n/a")} |
| Stop | {float(best.get("stop_pct", 0) or 0) * 100:g}% |
| Target | {float(best.get("target_pct", 0) or 0) * 100:g}% |

**Out-of-sample results for this config:**

- Net profit: {_money(oos_net)}
- Trades: {oos_trades}
- Win rate: {float(best.get("oos_win_rate", 0) or 0) * 100:.1f}%

## How many trades happened?

Across the three phases for the top config:

| Phase | Trades | Net profit |
|-------|--------|------------|
| Training | {int(best.get("train_trade_count", 0) or 0)} | {_money(float(best.get("train_net_pnl", 0) or 0))} |
| Validation | {int(best.get("val_trade_count", 0) or 0)} | {_money(float(best.get("val_net_pnl", 0) or 0))} |
| Out-of-sample | {oos_trades} | {_money(oos_net)} |

See `walkforward_summary.csv` in the same folder for all configs that were tested out-of-sample.

## What is the biggest weakness of the result?

{_weakness_walkforward(best, walkforward_summary, manifest)}
"""
