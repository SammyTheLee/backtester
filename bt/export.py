"""Export backtest results to timestamped CSV and JSON files."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

import pandas as pd

if TYPE_CHECKING:
    from bt.portfolio_backtest import RejectedTrade

from bt.metadata import save_metadata
from bt.paths import RunOutputPaths
from bt.portfolio import EquityPoint
from bt.portfolio_equity import PortfolioEquityPoint
from bt.report import (
    RUNS_CSV_COLUMNS,
    StrategyReportCard,
    save_report_card,
    summary_row,
)
from bt.report_md import build_report_md, save_report_md
from bt.trade import Trade, save_trades


def save_equity_csv(curve: list[EquityPoint], path) -> None:
    """Write equity curve points to CSV using pandas."""
    pd.DataFrame(
        [{"timestamp": p.timestamp, "equity": p.equity} for p in curve]
    ).to_csv(path, index=False)


def save_portfolio_equity_csv(curve: list[PortfolioEquityPoint], path) -> None:
    """Write portfolio equity with open-position count per bar."""
    pd.DataFrame(
        [
            {
                "timestamp": p.timestamp,
                "equity": p.equity,
                "open_positions": p.open_positions,
            }
            for p in curve
        ]
    ).to_csv(path, index=False)


def save_run_summary_csv(
    card: StrategyReportCard,
    path,
    *,
    run_meta: dict[str, Any],
) -> None:
    """Write this run's metrics (one row) to CSV using pandas."""
    pd.DataFrame([run_meta], columns=RUNS_CSV_COLUMNS).to_csv(path, index=False)


def export_backtest_run(
    paths: RunOutputPaths,
    *,
    trades: list[Trade],
    curve: list[EquityPoint],
    card: StrategyReportCard,
    metadata: dict[str, Any],
    run_meta: dict[str, Any],
) -> RunOutputPaths:
    """Export all artifacts for one run into a timestamped folder."""
    paths.run_dir.mkdir(parents=True, exist_ok=True)

    n_trades = save_trades(trades, paths.trades)
    save_equity_csv(curve, paths.equity)
    save_run_summary_csv(card, paths.summary, run_meta=run_meta)
    save_report_card(card, paths.report)
    trades_df = pd.read_csv(paths.trades)
    save_report_md(build_report_md(metadata, card, trades=trades_df), paths.report_md)

    metadata["output_dir"] = str(paths.run_dir)
    metadata["artifacts"] = {
        "trades_csv": str(paths.trades),
        "equity_csv": str(paths.equity),
        "summary_csv": str(paths.summary),
        "metadata_json": str(paths.metadata),
        "report_json": str(paths.report),
        "report_md": str(paths.report_md),
        "trade_count_written": n_trades,
    }
    save_metadata(metadata, paths.metadata)

    return paths


def export_portfolio_backtest_run(
    paths: RunOutputPaths,
    *,
    trades: list[Trade],
    trade_curve: list[EquityPoint],
    portfolio_curve: list[PortfolioEquityPoint],
    card: StrategyReportCard,
    metadata: dict[str, Any],
    run_meta: dict[str, Any],
    rejected: list["RejectedTrade"],
) -> RunOutputPaths:
    """Export portfolio mode artifacts (includes portfolio_equity.csv)."""
    from bt.portfolio_backtest import save_rejected_trades

    paths.run_dir.mkdir(parents=True, exist_ok=True)

    n_trades = save_trades(trades, paths.trades)
    save_equity_csv(trade_curve, paths.equity)
    save_portfolio_equity_csv(portfolio_curve, paths.portfolio_equity)
    save_rejected_trades(rejected, paths.rejected_trades)
    save_run_summary_csv(card, paths.summary, run_meta=run_meta)
    save_report_card(card, paths.report)
    trades_df = pd.read_csv(paths.trades)
    save_report_md(build_report_md(metadata, card, trades=trades_df), paths.report_md)

    metadata["output_dir"] = str(paths.run_dir)
    metadata["portfolio_mode"] = True
    metadata["artifacts"] = {
        "trades_csv": str(paths.trades),
        "equity_csv": str(paths.equity),
        "portfolio_equity_csv": str(paths.portfolio_equity),
        "rejected_trades_csv": str(paths.rejected_trades),
        "summary_csv": str(paths.summary),
        "metadata_json": str(paths.metadata),
        "report_json": str(paths.report),
        "report_md": str(paths.report_md),
        "trade_count_written": n_trades,
        "rejected_count": len(rejected),
    }
    save_metadata(metadata, paths.metadata)

    return paths
