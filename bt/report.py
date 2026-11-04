"""Strategy report card — summary metrics for a backtest run."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from bt.paths import GLOBAL_RANKED_SUMMARY_PATH, GLOBAL_RUNS_PATH
from bt.portfolio import DEFAULT_INITIAL_EQUITY, EquityPoint, equity_curve, total_return_pct
from bt.trade import Trade

DEFAULT_SUMMARY_SORT_BY = "composite_score"
DEFAULT_DRAWDOWN_PENALTY = 1.0
DEFAULT_SUMMARY_TOP_N = 10

# Strategy parameters — one leaderboard row per unique set (across train/val/OOS runs).
CONFIG_PARAM_COLUMNS = ("fast", "slow", "stop_pct", "target_pct")

# Legacy alias (dedupe no longer includes data_split).
CONFIG_KEY_COLUMNS = CONFIG_PARAM_COLUMNS + ("data_split",)

FINAL_LEADERBOARD_SPLIT = "oos"


@dataclass(frozen=True)
class BacktestMetrics:
    """Core performance metrics from trades and an equity curve."""

    net_pnl: float
    win_rate: float
    profit_factor: Optional[float]
    max_drawdown: float
    trade_count: int


def compute_metrics(
    trades: list[Trade],
    curve: list[EquityPoint],
) -> BacktestMetrics:
    """Derive summary metrics from closed trades and a pre-built equity curve."""
    closed = [t for t in trades if t.is_closed]
    return BacktestMetrics(
        net_pnl=sum(t.net_pnl or 0.0 for t in closed),
        win_rate=_win_rate(closed),
        profit_factor=_profit_factor(closed),
        max_drawdown=max_drawdown(curve),
        trade_count=len(closed),
    )


@dataclass(frozen=True)
class StrategyReportCard:
    """Summary metrics for one backtest run."""

    gross_pnl: float
    net_pnl: float
    total_return_pct: float
    trade_count: int
    win_rate: float
    profit_factor: Optional[float]
    max_drawdown: float
    average_hold_hours: Optional[float]
    time_in_market_pct: float

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.profit_factor == float("inf"):
            d["profit_factor"] = None
            d["profit_factor_display"] = "inf"
        return d


def build_report_card(
    trades: list[Trade],
    *,
    initial_equity: float = DEFAULT_INITIAL_EQUITY,
    bars_in_market: int | None = None,
    total_bars: int | None = None,
) -> StrategyReportCard:
    """Compute report metrics from closed trades and optional bar counts."""
    closed = [t for t in trades if t.is_closed]
    curve = equity_curve(trades, initial_equity=initial_equity)
    core = compute_metrics(trades, curve)

    return StrategyReportCard(
        gross_pnl=sum(t.gross_pnl or 0.0 for t in closed),
        net_pnl=core.net_pnl,
        total_return_pct=total_return_pct(trades, initial_equity),
        trade_count=core.trade_count,
        win_rate=core.win_rate,
        profit_factor=core.profit_factor,
        max_drawdown=core.max_drawdown,
        average_hold_hours=_average_hold_hours(closed),
        time_in_market_pct=_time_in_market_pct(bars_in_market, total_bars),
    )


def format_report_card(card: StrategyReportCard) -> str:
    """Human-readable report card."""
    pf = (
        "inf"
        if card.profit_factor == float("inf")
        else f"{card.profit_factor:.2f}"
        if card.profit_factor is not None
        else "n/a"
    )
    hold = (
        f"{card.average_hold_hours:.1f} hours"
        if card.average_hold_hours is not None
        else "n/a"
    )
    lines = [
        "=== Strategy Report Card ===",
        f"Gross PnL          : ${card.gross_pnl:,.2f}",
        f"Net PnL            : ${card.net_pnl:,.2f}",
        f"Total return       : {card.total_return_pct * 100:+.2f}%",
        f"Trade count        : {card.trade_count}",
        f"Win rate           : {card.win_rate * 100:.1f}%",
        f"Profit factor      : {pf}",
        f"Max drawdown       : ${card.max_drawdown:,.2f}",
        f"Avg hold time      : {hold}",
        f"Time in market     : {card.time_in_market_pct * 100:.1f}%",
    ]
    return "\n".join(lines)


RUN_META_COLUMNS = [
    "run_at",
    "config",
    "data_path",
    "data_split",
    "strategy",
    "fast",
    "slow",
]

SUMMARY_CSV_COLUMNS = [
    "gross_pnl",
    "net_pnl",
    "total_return_pct",
    "composite_score",
    "trade_count",
    "win_rate",
    "profit_factor",
    "max_drawdown",
    "average_hold_hours",
    "time_in_market_pct",
]


def composite_score(
    total_return_pct: float,
    max_drawdown: float,
    *,
    initial_equity: float = DEFAULT_INITIAL_EQUITY,
    drawdown_penalty: float = DEFAULT_DRAWDOWN_PENALTY,
) -> float:
    """Risk-adjusted score: return (fraction) minus weighted drawdown (fraction of equity).

    Higher is better. Example: +10% return and 2% drawdown with penalty 1.0 → 0.08.
    """
    if initial_equity <= 0:
        return total_return_pct
    drawdown_pct = max_drawdown / initial_equity
    return total_return_pct - drawdown_penalty * drawdown_pct

RUNS_CSV_COLUMNS = RUN_META_COLUMNS + SUMMARY_CSV_COLUMNS
SUMMARY_RANKED_COLUMNS = ["rank"] + RUNS_CSV_COLUMNS


def summary_row(
    card: StrategyReportCard,
    *,
    initial_equity: float = DEFAULT_INITIAL_EQUITY,
    drawdown_penalty: float = DEFAULT_DRAWDOWN_PENALTY,
) -> dict[str, Any]:
    """One dict = one row for summary.csv."""
    pf: Any = card.profit_factor
    if pf == float("inf"):
        pf = None
    score = composite_score(
        card.total_return_pct,
        card.max_drawdown,
        initial_equity=initial_equity,
        drawdown_penalty=drawdown_penalty,
    )
    return {
        "gross_pnl": card.gross_pnl,
        "net_pnl": card.net_pnl,
        "total_return_pct": card.total_return_pct,
        "composite_score": score,
        "trade_count": card.trade_count,
        "win_rate": card.win_rate,
        "profit_factor": pf,
        "max_drawdown": card.max_drawdown,
        "average_hold_hours": card.average_hold_hours,
        "time_in_market_pct": card.time_in_market_pct,
    }


def run_row(
    card: StrategyReportCard,
    *,
    run_at: str | None = None,
    config: str = "",
    data_path: str = "",
    strategy: str = "",
    fast: int | None = None,
    slow: int | None = None,
    initial_equity: float = DEFAULT_INITIAL_EQUITY,
    drawdown_penalty: float = DEFAULT_DRAWDOWN_PENALTY,
    data_split: str = "all",
) -> dict[str, Any]:
    """One dict = one appended row for runs.csv."""
    row: dict[str, Any] = {
        "run_at": run_at or datetime.now(timezone.utc).isoformat(),
        "config": config,
        "data_path": data_path,
        "data_split": data_split,
        "strategy": strategy,
        "fast": fast,
        "slow": slow,
    }
    row.update(
        summary_row(
            card,
            initial_equity=initial_equity,
            drawdown_penalty=drawdown_penalty,
        )
    )
    return row


def append_runs_csv(
    card: StrategyReportCard,
    path: Path | None = None,
    *,
    run_at: str | None = None,
    config: str = "",
    data_path: str = "",
    strategy: str = "",
    fast: int | None = None,
    slow: int | None = None,
    initial_equity: float = DEFAULT_INITIAL_EQUITY,
    drawdown_penalty: float = DEFAULT_DRAWDOWN_PENALTY,
    data_split: str = "all",
) -> None:
    """Append one backtest run (all metrics) as a new row in runs.csv."""
    path = Path(path or GLOBAL_RUNS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = run_row(
        card,
        run_at=run_at,
        config=config,
        data_path=data_path,
        strategy=strategy,
        fast=fast,
        slow=slow,
        initial_equity=initial_equity,
        drawdown_penalty=drawdown_penalty,
        data_split=data_split,
    )
    new_row = pd.DataFrame([row], columns=RUNS_CSV_COLUMNS)
    if path.exists() and path.stat().st_size > 0:
        existing = pd.read_csv(path)
        if "data_split" not in existing.columns:
            existing["data_split"] = "all"
        combined = pd.concat([existing, new_row], ignore_index=True)
        combined = combined.reindex(columns=RUNS_CSV_COLUMNS)
        combined.to_csv(path, index=False)
    else:
        new_row.to_csv(path, index=False)


def ensure_composite_score_column(
    df: pd.DataFrame,
    *,
    initial_equity: float = DEFAULT_INITIAL_EQUITY,
    drawdown_penalty: float = DEFAULT_DRAWDOWN_PENALTY,
) -> pd.DataFrame:
    """Add ``composite_score`` when ranking older runs.csv files that lack it."""
    if "composite_score" in df.columns:
        return df
    out = df.copy()

    def _row_score(row: pd.Series) -> float:
        ret = float(row.get("total_return_pct", 0) or 0)
        if ret == 0 and "net_pnl" in row and initial_equity > 0:
            ret = float(row["net_pnl"]) / initial_equity
        dd = float(row.get("max_drawdown", 0) or 0)
        return composite_score(
            ret, dd, initial_equity=initial_equity, drawdown_penalty=drawdown_penalty
        )

    out["composite_score"] = out.apply(_row_score, axis=1)
    return out


def rank_dataframe(
    df: pd.DataFrame,
    sort_by: str,
    *,
    ascending: bool | None = None,
    initial_equity: float = DEFAULT_INITIAL_EQUITY,
    drawdown_penalty: float = DEFAULT_DRAWDOWN_PENALTY,
) -> pd.DataFrame:
    """Sort *df* for ranking; rank 1 is best. Inserts ``rank`` as first column."""
    if df.empty:
        ranked = df.copy()
        ranked.insert(0, "rank", [])
        return ranked

    work = df
    if sort_by == "composite_score":
        work = ensure_composite_score_column(
            work,
            initial_equity=initial_equity,
            drawdown_penalty=drawdown_penalty,
        )

    if sort_by not in work.columns:
        raise ValueError(
            f"Cannot sort by {sort_by!r}; available columns: {list(work.columns)}"
        )

    if ascending is None:
        ascending = sort_by == "max_drawdown"

    sort_key = work[sort_by].copy()
    if sort_by == "profit_factor":
        sort_key = pd.to_numeric(sort_key, errors="coerce").fillna(-1.0)
        sort_key = sort_key.replace(float("inf"), float("1e308"))

    ranked = (
        work.assign(_sort_key=sort_key)
        .sort_values("_sort_key", ascending=ascending, na_position="last")
        .drop(columns="_sort_key")
    )
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def dedupe_best_per_configuration(
    df: pd.DataFrame,
    sort_by: str,
    *,
    ascending: bool | None = None,
    initial_equity: float = DEFAULT_INITIAL_EQUITY,
    drawdown_penalty: float = DEFAULT_DRAWDOWN_PENALTY,
) -> pd.DataFrame:
    """Keep one row per parameter set — the best by ``sort_by``."""
    if df.empty:
        return df

    keys = [c for c in CONFIG_PARAM_COLUMNS if c in df.columns]
    if not keys:
        return df

    work = df
    if sort_by == "composite_score":
        work = ensure_composite_score_column(
            work,
            initial_equity=initial_equity,
            drawdown_penalty=drawdown_penalty,
        )

    if sort_by not in work.columns:
        return df

    if ascending is None:
        ascending = sort_by == "max_drawdown"

    sort_key = work[sort_by].copy()
    if sort_by == "profit_factor":
        sort_key = pd.to_numeric(sort_key, errors="coerce").fillna(-1.0)
        sort_key = sort_key.replace(float("inf"), float("1e308"))

    ordered = work.assign(_sort_key=sort_key).sort_values(
        "_sort_key", ascending=ascending, na_position="last"
    )
    return ordered.drop_duplicates(subset=keys, keep="first").drop(columns="_sort_key")


def write_ranked_summary_csv(
    runs_path: Path | None = None,
    summary_path: Path | None = None,
    *,
    sort_by: str = DEFAULT_SUMMARY_SORT_BY,
    ascending: bool | None = None,
    initial_equity: float = DEFAULT_INITIAL_EQUITY,
    drawdown_penalty: float = DEFAULT_DRAWDOWN_PENALTY,
    top_n: int | None = DEFAULT_SUMMARY_TOP_N,
    dedupe_configs: bool = True,
    leaderboard_data_split: str | None = FINAL_LEADERBOARD_SPLIT,
) -> tuple[int, int]:
    """Build ``summary.csv``: OOS-only final leaderboard by default.

    ``runs.csv`` keeps every run (train, validation, OOS). ``summary.csv`` ranks only
    out-of-sample rows (``data_split=oos``), best row per parameter set, then top *top_n*.
    Pass ``leaderboard_data_split=None`` to rank all splits (legacy).

    Returns ``(runs_row_count, summary_row_count)``.
    """
    runs_path = Path(runs_path or GLOBAL_RUNS_PATH)
    summary_path = Path(summary_path or GLOBAL_RANKED_SUMMARY_PATH)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if not runs_path.exists() or runs_path.stat().st_size == 0:
        pd.DataFrame(columns=SUMMARY_RANKED_COLUMNS).to_csv(summary_path, index=False)
        return 0, 0

    df = pd.read_csv(runs_path)
    runs_count = len(df)
    if df.empty:
        pd.DataFrame(columns=SUMMARY_RANKED_COLUMNS).to_csv(summary_path, index=False)
        return 0, 0

    if leaderboard_data_split is not None and "data_split" in df.columns:
        split_mask = (
            df["data_split"].astype(str).str.lower() == leaderboard_data_split.lower()
        )
        df = df.loc[split_mask]
        if df.empty:
            pd.DataFrame(columns=SUMMARY_RANKED_COLUMNS).to_csv(summary_path, index=False)
            return runs_count, 0

    if dedupe_configs:
        df = dedupe_best_per_configuration(
            df,
            sort_by,
            ascending=ascending,
            initial_equity=initial_equity,
            drawdown_penalty=drawdown_penalty,
        )

    ranked = rank_dataframe(
        df,
        sort_by,
        ascending=ascending,
        initial_equity=initial_equity,
        drawdown_penalty=drawdown_penalty,
    )
    if top_n is not None and top_n > 0:
        ranked = ranked.head(top_n)

    ranked.to_csv(summary_path, index=False)
    return runs_count, len(ranked)


def save_summary_csv(card: StrategyReportCard, path: Path) -> None:
    """Write one backtest summary row to a CSV file (overwrites, no ranking)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([summary_row(card)], columns=SUMMARY_CSV_COLUMNS)
    df.to_csv(path, index=False)


def save_report_card(card: StrategyReportCard, path: Path) -> None:
    """Write report metrics as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = card.to_dict()
    if card.profit_factor == float("inf"):
        payload["profit_factor"] = None
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def max_drawdown(curve: list[EquityPoint]) -> float:
    """Largest peak-to-trough drop on the equity curve."""
    if len(curve) < 2:
        return 0.0
    peak = curve[0].equity
    max_dd = 0.0
    for point in curve:
        peak = max(peak, point.equity)
        max_dd = max(max_dd, peak - point.equity)
    return max_dd


def _win_rate(closed: list[Trade]) -> float:
    if not closed:
        return 0.0
    wins = sum(1 for t in closed if (t.net_pnl or 0) > 0)
    return wins / len(closed)


def _profit_factor(closed: list[Trade]) -> Optional[float]:
    gross_wins = sum(t.net_pnl for t in closed if (t.net_pnl or 0) > 0)
    gross_losses = abs(sum(t.net_pnl for t in closed if (t.net_pnl or 0) < 0))
    if gross_losses > 0:
        return gross_wins / gross_losses
    if gross_wins > 0:
        return float("inf")
    return 0.0


def _average_hold_hours(closed: list[Trade]) -> Optional[float]:
    if not closed:
        return None
    hours: list[float] = []
    for t in closed:
        if t.exit_time is None:
            continue
        start = pd.to_datetime(t.entry_time)
        end = pd.to_datetime(t.exit_time)
        delta = end - start
        hours.append(delta.total_seconds() / 3600.0)
    if not hours:
        return None
    return sum(hours) / len(hours)


def _time_in_market_pct(
    bars_in_market: int | None, total_bars: int | None
) -> float:
    if bars_in_market is None or total_bars is None or total_bars <= 0:
        return 0.0
    return bars_in_market / total_bars
