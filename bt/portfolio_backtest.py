"""Multi-asset portfolio backtest with shared capital and exposure limits."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from bt.pipeline import BacktestParams

from bt.costs import ExecutionCosts
from bt.portfolio import EquityPoint, equity_curve
from bt.portfolio_config import AssetSpec, PortfolioLimits
from bt.portfolio_constraints import (
    _close_prices_at_helper,
    _mark_to_market_prices,
    check_entry_allowed,
    position_notional,
)
from bt.portfolio_equity import PortfolioEquityPoint, build_portfolio_equity_curve
from bt.report import StrategyReportCard, build_report_card
from bt.splits import SplitConfig
from bt.trade import Trade


@dataclass(frozen=True)
class RejectedTrade:
    asset: str
    trade: Trade
    reason: str


@dataclass(frozen=True)
class PortfolioBacktestResult:
    accepted_trades: list[Trade]
    rejected: list[RejectedTrade]
    candles_by_asset: dict[str, pd.DataFrame]
    portfolio_equity: list[PortfolioEquityPoint]
    trade_equity: list[EquityPoint]
    card: StrategyReportCard
    bars_in_market: int
    total_bars: int


def _tag_trades(trades: list[Trade], asset: str) -> list[Trade]:
    return [replace(t, asset=asset) for t in trades]


def _portfolio_events(trades: list[Trade]) -> list[tuple[str, pd.Timestamp, Trade]]:
    events: list[tuple[str, pd.Timestamp, Trade]] = []
    for t in trades:
        events.append(("entry", pd.to_datetime(t.entry_time), t))
        if t.is_closed and t.exit_time is not None:
            events.append(("exit", pd.to_datetime(t.exit_time), t))
    events.sort(key=lambda x: (x[1], 0 if x[0] == "exit" else 1, x[2].asset or ""))
    return events


def apply_portfolio_constraints(
    candidates: list[Trade],
    *,
    initial_equity: float,
    limits: PortfolioLimits,
    candles_by_asset: dict[str, pd.DataFrame],
) -> tuple[list[Trade], list[RejectedTrade]]:
    """Replay entries/exits in time order; reject entries that break caps."""
    events = _portfolio_events(candidates)
    accepted_ids: set[int] = set()
    rejected: list[RejectedTrade] = []
    open_trades: list[Trade] = []
    cash = initial_equity

    for kind, ts, trade in events:
        if kind == "exit":
            if id(trade) not in accepted_ids:
                continue
            open_trades = [t for t in open_trades if id(t) != id(trade)]
            if trade.net_pnl is not None:
                cash += trade.net_pnl
            continue

        prices = _close_prices_at_helper(ts, candles_by_asset)
        equity = cash + _mark_to_market_prices(open_trades, prices)
        asset = trade.asset or "unknown"
        check = check_entry_allowed(
            equity=equity,
            asset=asset,
            new_notional=position_notional(trade),
            open_trades=open_trades,
            limits=limits,
        )
        if not check.allowed:
            rejected.append(
                RejectedTrade(
                    asset=asset,
                    trade=trade,
                    reason=check.reason or "rejected",
                )
            )
            continue
        accepted_ids.add(id(trade))
        open_trades.append(trade)
        cash -= trade.entry_fee

    accepted = [t for t in candidates if id(t) in accepted_ids]
    accepted.sort(key=lambda t: (t.entry_time, t.asset or ""))
    return accepted, rejected


def run_portfolio_backtest(
    assets: list[AssetSpec],
    params: "BacktestParams",
    costs: ExecutionCosts,
    *,
    config: dict,
    initial_equity: float,
    size: float,
    limits: PortfolioLimits,
    split: SplitConfig | None = None,
) -> PortfolioBacktestResult:
    """Backtest each asset, merge trades, enforce shared-capital constraints."""
    from bt.pipeline import run_configured_backtest

    if not assets:
        raise ValueError("portfolio backtest requires at least one asset")

    candidates: list[Trade] = []
    candles_by_asset: dict[str, pd.DataFrame] = {}
    bars_in_market = 0
    total_bars = 0

    for spec in assets:
        if not spec.csv_path.exists():
            raise FileNotFoundError(f"data file not found: {spec.csv_path}")
        df, trades, _strat = run_configured_backtest(
            spec.csv_path,
            params,
            costs,
            config,
            size=size,
            split=split,
        )
        candles_by_asset[spec.symbol] = df
        candidates.extend(_tag_trades(trades, spec.symbol))
        bars_in_market += int(df["in_position"].sum())
        total_bars += len(df)

    accepted, rejected = apply_portfolio_constraints(
        candidates,
        initial_equity=initial_equity,
        limits=limits,
        candles_by_asset=candles_by_asset,
    )

    trade_equity = equity_curve(accepted, initial_equity=initial_equity)
    portfolio_equity = build_portfolio_equity_curve(
        accepted_trades=accepted,
        candles_by_asset=candles_by_asset,
        initial_equity=initial_equity,
    )
    card = build_report_card(
        accepted,
        initial_equity=initial_equity,
        bars_in_market=bars_in_market,
        total_bars=total_bars,
    )

    return PortfolioBacktestResult(
        accepted_trades=accepted,
        rejected=rejected,
        candles_by_asset=candles_by_asset,
        portfolio_equity=portfolio_equity,
        trade_equity=trade_equity,
        card=card,
        bars_in_market=bars_in_market,
        total_bars=total_bars,
    )


def save_rejected_trades(rejected: list[RejectedTrade], path: Path) -> int:
    if not rejected:
        pd.DataFrame(
            columns=["asset", "entry_time", "side", "entry_price", "size", "reason"]
        ).to_csv(path, index=False)
        return 0
    rows = [
        {
            "asset": r.asset,
            "entry_time": r.trade.entry_time,
            "side": str(r.trade.side),
            "entry_price": r.trade.entry_price,
            "size": r.trade.size,
            "reason": r.reason,
        }
        for r in rejected
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return len(rows)
