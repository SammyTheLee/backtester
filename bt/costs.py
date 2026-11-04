"""Trading cost models for backtest execution."""

from __future__ import annotations

from dataclasses import dataclass

from bt.strategy import Position


@dataclass(frozen=True)
class ExecutionCosts:
    """Cost parameters in basis points (1 bps = 0.01%)."""

    fee_bps: float = 0.0
    slippage_bps: float = 0.0
    holding_cost_bps: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("fee_bps", self.fee_bps),
            ("slippage_bps", self.slippage_bps),
            ("holding_cost_bps", self.holding_cost_bps),
        ):
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")


def slippage_rate(slippage_bps: float) -> float:
    return slippage_bps / 10_000


def entry_fill_price(side: Position, open_price: float, slippage_bps: float) -> float:
    """Worse fill on entry: long pays more, short receives less."""
    rate = slippage_rate(slippage_bps)
    if side == Position.LONG:
        return open_price * (1 + rate)
    return open_price * (1 - rate)


def exit_fill_price(side: Position, raw_price: float, slippage_bps: float) -> float:
    """Worse fill on exit: long receives less, short pays more."""
    rate = slippage_rate(slippage_bps)
    if side == Position.LONG:
        return raw_price * (1 - rate)
    return raw_price * (1 + rate)


def trading_fee(notional: float, fee_bps: float) -> float:
    """fee = notional * fee_bps / 10000"""
    return notional * fee_bps / 10_000


def holding_fee(notional: float, holding_cost_bps: float) -> float:
    """Per-bar holding charge using the same bps formula."""
    return notional * holding_cost_bps / 10_000


def slippage_cost(
    side: Position,
    entry_open: float,
    entry_fill: float,
    exit_raw: float,
    exit_fill: float,
    size: float,
) -> float:
    """Total PnL lost to slippage (always >= 0)."""
    gross_at_raw = _pnl(side, entry_open, exit_raw, size)
    gross_at_fill = _pnl(side, entry_fill, exit_fill, size)
    return gross_at_raw - gross_at_fill


def _pnl(side: Position, entry: float, exit_: float, size: float) -> float:
    per_unit = exit_ - entry if side == Position.LONG else entry - exit_
    return per_unit * size


def execution_costs_from_config(exec_cfg: dict) -> ExecutionCosts:
    return ExecutionCosts(
        fee_bps=float(exec_cfg.get("fee_bps", 0.0)),
        slippage_bps=float(exec_cfg.get("slippage_bps", 0.0)),
        holding_cost_bps=float(exec_cfg.get("holding_cost_bps", 0.0)),
    )
