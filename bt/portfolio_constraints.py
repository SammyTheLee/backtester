"""Portfolio exposure and position-count limits."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import pandas as pd

from bt.portfolio_config import PortfolioLimits
from bt.strategy import Position
from bt.trade import Trade


def _close_prices_at_helper(
    ts: pd.Timestamp,
    candles_by_asset: dict[str, pd.DataFrame],
) -> dict[str, float]:
    prices: dict[str, float] = {}
    for asset, df in candles_by_asset.items():
        ts_col = pd.to_datetime(df["timestamp"])
        mask = ts_col <= ts
        if not mask.any():
            continue
        prices[asset] = float(df.loc[mask, "close"].iloc[-1])
    return prices


def position_notional(trade: Trade) -> float:
    """Absolute entry notional used for exposure caps."""
    return abs(trade.entry_price * trade.size)


def open_exposure_by_asset(open_trades: list[Trade]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for t in open_trades:
        if t.asset is None:
            continue
        totals[t.asset] += position_notional(t)
    return dict(totals)


def total_open_exposure(open_trades: list[Trade]) -> float:
    return sum(position_notional(t) for t in open_trades)


def _mark_to_market_prices(open_trades: list[Trade], prices: dict[str, float]) -> float:
    total = 0.0
    for t in open_trades:
        if t.asset is None:
            continue
        px = prices.get(t.asset, t.entry_price)
        if t.side == Position.LONG:
            total += (px - t.entry_price) * t.size
        else:
            total += (t.entry_price - px) * t.size
    return total


@dataclass(frozen=True)
class ConstraintCheck:
    allowed: bool
    reason: str | None = None


def check_entry_allowed(
    *,
    equity: float,
    asset: str,
    new_notional: float,
    open_trades: list[Trade],
    limits: PortfolioLimits,
) -> ConstraintCheck:
    """Return whether a new position may open under portfolio caps."""
    if equity <= 0:
        return ConstraintCheck(False, "non_positive_equity")

    if (
        limits.max_open_positions is not None
        and len(open_trades) >= limits.max_open_positions
    ):
        return ConstraintCheck(False, "max_open_positions")

    per_asset = open_exposure_by_asset(open_trades)
    asset_exp = per_asset.get(asset, 0.0) + new_notional
    total_exp = total_open_exposure(open_trades) + new_notional

    if asset_exp / equity > limits.max_exposure_per_asset_pct + 1e-12:
        return ConstraintCheck(False, "max_exposure_per_asset")

    if total_exp / equity > limits.max_total_exposure_pct + 1e-12:
        return ConstraintCheck(False, "max_total_exposure")

    return ConstraintCheck(True)
