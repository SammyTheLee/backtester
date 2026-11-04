"""Trade representation, extraction, and storage.

A `Trade` records one round-trip position: entry time/price, exit time/price,
side (long/short), size, and PnL breakdown (gross, slippage, fees, net).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from bt.costs import ExecutionCosts, holding_fee, slippage_cost, trading_fee
from bt.strategy import Position


@dataclass(frozen=True)
class Trade:
    """A single round-trip trade.

    PnL breakdown (closed trades):
        gross_pnl      — price PnL at raw entry/exit (no slippage, no fees)
        slippage_cost  — PnL lost to entry/exit slippage (>= 0)
        fees           — entry_fee + exit_fee
        net_pnl        — gross_pnl - slippage_cost - fees - holding_cost

    `entry_price` / `exit_price` are slipped fills. Open trades have
    `net_pnl is None` and only `entry_fee` may be charged so far.
    """

    entry_time: str
    entry_price: float
    side: Position
    size: float = 1.0
    asset: str | None = None
    exit_time: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    entry_open: Optional[float] = None
    exit_raw: Optional[float] = None
    gross_pnl: Optional[float] = None
    slippage_cost: float = 0.0
    entry_fee: float = 0.0
    exit_fee: float = 0.0
    holding_cost: float = 0.0
    net_pnl: Optional[float] = None

    def __post_init__(self) -> None:
        if self.exit_price is None:
            return
        if self.gross_pnl is None:
            entry_raw = self.entry_open if self.entry_open is not None else self.entry_price
            exit_raw = self.exit_raw if self.exit_raw is not None else self.exit_price
            object.__setattr__(
                self,
                "gross_pnl",
                _gross_pnl(self.side, entry_raw, exit_raw, self.size),
            )
        if self.slippage_cost == 0.0 and (
            self.entry_open is not None
            and self.exit_raw is not None
            and (
                self.entry_open != self.entry_price
                or self.exit_raw != self.exit_price
            )
        ):
            object.__setattr__(
                self,
                "slippage_cost",
                slippage_cost(
                    self.side,
                    self.entry_open,
                    self.entry_price,
                    self.exit_raw,
                    self.exit_price,
                    self.size,
                ),
            )
        if self.net_pnl is None:
            object.__setattr__(
                self,
                "net_pnl",
                self.gross_pnl
                - self.slippage_cost
                - self.entry_fee
                - self.exit_fee
                - self.holding_cost,
            )

    @property
    def is_closed(self) -> bool:
        return self.exit_price is not None

    @property
    def fees(self) -> float:
        return self.entry_fee + self.exit_fee

    @property
    def trading_fees(self) -> float:
        """Alias for `fees`."""
        return self.fees

    @property
    def gross_profit(self) -> Optional[float]:
        """Alias for `gross_pnl`."""
        return self.gross_pnl

    @property
    def profit(self) -> Optional[float]:
        """Alias for `net_pnl`."""
        return self.net_pnl

    @property
    def profit_pct(self) -> Optional[float]:
        """Net return as a fraction of entry notional. None if open."""
        if self.net_pnl is None:
            return None
        return self.net_pnl / (self.entry_price * self.size)


def extract_trades(
    df: pd.DataFrame,
    size: float = 1.0,
    costs: ExecutionCosts | None = None,
) -> list[Trade]:
    """Walk a DataFrame produced by `apply_execution` and return its trades."""
    _validate(df)

    if size <= 0:
        raise ValueError("size must be positive")

    costs = costs or ExecutionCosts()
    has_raw = "entry_open" in df.columns and "exit_raw" in df.columns
    trades: list[Trade] = []
    open_entry: Optional[tuple[int, str, float, float, Position]] = None

    def _close_trade(
        entry_idx: int,
        entry_time: str,
        entry_fill: float,
        entry_open: float,
        side: Position,
        exit_time: str,
        exit_fill: float,
        exit_raw: float,
        exit_reason: str,
    ) -> Trade:
        gross = _gross_pnl(side, entry_open, exit_raw, size)
        slip = (
            slippage_cost(side, entry_open, entry_fill, exit_raw, exit_fill, size)
            if has_raw
            else 0.0
        )
        entry_fee = trading_fee(entry_fill * size, costs.fee_bps)
        exit_fee = trading_fee(exit_fill * size, costs.fee_bps)
        hold = _holding_cost_for_bars(df, entry_idx, i, size, costs.holding_cost_bps)
        net = gross - slip - entry_fee - exit_fee - hold
        return Trade(
            entry_time=entry_time,
            entry_price=entry_fill,
            side=side,
            size=size,
            exit_time=exit_time,
            exit_price=exit_fill,
            exit_reason=exit_reason,
            entry_open=entry_open,
            exit_raw=exit_raw,
            gross_pnl=gross,
            slippage_cost=slip,
            entry_fee=entry_fee,
            exit_fee=exit_fee,
            holding_cost=hold,
            net_pnl=net,
        )

    for i in range(len(df)):
        entry_price = df["entry_price"].iloc[i]
        exit_price = df["exit_price"].iloc[i]
        exit_reason = df["exit_reason"].iloc[i]

        if (
            not pd.isna(exit_price)
            and exit_reason == "signal_change"
            and open_entry is not None
        ):
            entry_idx, entry_time, entry_fill, entry_open, side = open_entry
            exit_fill = float(exit_price)
            exit_raw = (
                float(df["exit_raw"].iloc[i])
                if has_raw and not pd.isna(df["exit_raw"].iloc[i])
                else exit_fill
            )
            trades.append(
                _close_trade(
                    entry_idx,
                    entry_time,
                    entry_fill,
                    entry_open,
                    side,
                    df["timestamp"].iloc[i],
                    exit_fill,
                    exit_raw,
                    exit_reason,
                )
            )
            open_entry = None

        if not pd.isna(entry_price):
            entry_fill = float(entry_price)
            entry_open = (
                float(df["entry_open"].iloc[i])
                if has_raw and not pd.isna(df["entry_open"].iloc[i])
                else entry_fill
            )
            open_entry = (
                i,
                df["timestamp"].iloc[i],
                entry_fill,
                entry_open,
                df["position"].iloc[i],
            )

        if (
            not pd.isna(exit_price)
            and exit_reason in ("stop", "target")
            and open_entry is not None
        ):
            entry_idx, entry_time, entry_fill, entry_open, side = open_entry
            exit_fill = float(exit_price)
            exit_raw = (
                float(df["exit_raw"].iloc[i])
                if has_raw and not pd.isna(df["exit_raw"].iloc[i])
                else exit_fill
            )
            trades.append(
                _close_trade(
                    entry_idx,
                    entry_time,
                    entry_fill,
                    entry_open,
                    side,
                    df["timestamp"].iloc[i],
                    exit_fill,
                    exit_raw,
                    exit_reason,
                )
            )
            open_entry = None

    if open_entry is not None:
        _idx, entry_time, entry_fill, entry_open, side = open_entry
        trades.append(
            Trade(
                entry_time=entry_time,
                entry_price=entry_fill,
                side=side,
                size=size,
                entry_open=entry_open,
                entry_fee=trading_fee(entry_fill * size, costs.fee_bps),
            )
        )

    return trades


def save_trades(trades: list[Trade], path: Path | str) -> int:
    """Write every trade (closed and still open) to a CSV file.

    Returns the number of rows written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "asset": t.asset,
            "entry_time": t.entry_time,
            "entry_price": t.entry_price,
            "entry_open": t.entry_open,
            "side": str(t.side),
            "size": t.size,
            "exit_time": t.exit_time,
            "exit_price": t.exit_price,
            "exit_raw": t.exit_raw,
            "exit_reason": t.exit_reason,
            "gross_pnl": t.gross_pnl,
            "slippage_cost": t.slippage_cost,
            "fees": t.fees,
            "entry_fee": t.entry_fee,
            "exit_fee": t.exit_fee,
            "holding_cost": t.holding_cost,
            "net_pnl": t.net_pnl,
            "profit_pct": t.profit_pct,
        }
        for t in trades
    ]

    df = pd.DataFrame(
        rows,
        columns=[
            "asset",
            "entry_time",
            "entry_price",
            "entry_open",
            "side",
            "size",
            "exit_time",
            "exit_price",
            "exit_raw",
            "exit_reason",
            "gross_pnl",
            "slippage_cost",
            "fees",
            "entry_fee",
            "exit_fee",
            "holding_cost",
            "net_pnl",
            "profit_pct",
        ],
    )
    df.to_csv(path, index=False)
    return len(rows)


def _gross_pnl(side: Position, entry: float, exit_: float, size: float) -> float:
    per_unit = exit_ - entry if side == Position.LONG else entry - exit_
    return per_unit * size


def _holding_cost_for_bars(
    df: pd.DataFrame,
    entry_idx: int,
    exit_idx: int,
    size: float,
    holding_cost_bps: float,
) -> float:
    if holding_cost_bps <= 0:
        return 0.0
    total = 0.0
    use_close = "close" in df.columns
    for k in range(entry_idx, exit_idx + 1):
        if use_close:
            notional = float(df["close"].iloc[k]) * size
        else:
            notional = float(df["entry_price"].iloc[k]) * size
            if pd.isna(notional):
                continue
        total += holding_fee(notional, holding_cost_bps)
    return total


def _validate(df: pd.DataFrame) -> None:
    required = {"timestamp", "position", "entry_price", "exit_price", "exit_reason"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required columns: {sorted(missing)}. "
            "Did you forget to call apply_execution() first?"
        )
