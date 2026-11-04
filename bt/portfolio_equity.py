"""Bar-level portfolio equity curve (shared capital, multiple open assets)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from bt.portfolio_constraints import _close_prices_at_helper, _mark_to_market_prices
from bt.trade import Trade


@dataclass(frozen=True)
class PortfolioEquityPoint:
    """Portfolio equity and open-position count at a timestamp."""

    timestamp: str
    equity: float
    open_positions: int


def build_portfolio_equity_curve(
    *,
    accepted_trades: list[Trade],
    candles_by_asset: dict[str, pd.DataFrame],
    initial_equity: float,
) -> list[PortfolioEquityPoint]:
    """Equity at each bar: cash plus mark-to-market on open positions."""
    if initial_equity <= 0:
        raise ValueError("initial_equity must be positive")

    events: list[tuple[str, pd.Timestamp, Trade]] = []
    for t in accepted_trades:
        events.append(("entry", pd.to_datetime(t.entry_time), t))
        if t.is_closed and t.exit_time is not None:
            events.append(("exit", pd.to_datetime(t.exit_time), t))
    events.sort(key=lambda x: (x[1], 0 if x[0] == "exit" else 1, x[2].asset or ""))

    all_ts: set[pd.Timestamp] = set()
    for df in candles_by_asset.values():
        all_ts.update(pd.to_datetime(df["timestamp"]))

    timeline = sorted(all_ts)
    cash = initial_equity
    open_trades: list[Trade] = []
    points: list[PortfolioEquityPoint] = [
        PortfolioEquityPoint(
            timestamp="start", equity=initial_equity, open_positions=0
        )
    ]
    event_idx = 0

    for ts in timeline:
        while event_idx < len(events) and events[event_idx][1] <= ts:
            kind, _, trade = events[event_idx]
            if kind == "exit":
                open_trades = [x for x in open_trades if id(x) != id(trade)]
                if trade.net_pnl is not None:
                    cash += trade.net_pnl
            else:
                open_trades.append(trade)
                cash -= trade.entry_fee
            event_idx += 1

        prices = _close_prices_at_helper(ts, candles_by_asset)
        equity = cash + _mark_to_market_prices(open_trades, prices)
        points.append(
            PortfolioEquityPoint(
                timestamp=ts.strftime("%Y-%m-%dT%H:%M:%S"),
                equity=equity,
                open_positions=len(open_trades),
            )
        )

    return points
