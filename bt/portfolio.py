"""Portfolio-level accounting.

Tracks running equity from a starting balance as closed trades realize their
profit or loss. Open trades are ignored (their PnL is unrealized).
"""

from __future__ import annotations

from dataclasses import dataclass

from bt.trade import Trade


DEFAULT_INITIAL_EQUITY = 10_000.0


@dataclass(frozen=True)
class EquityPoint:
    """Equity value at a point in time (start, or after a closed trade)."""

    timestamp: str
    equity: float


def equity_curve(
    trades: list[Trade],
    initial_equity: float = DEFAULT_INITIAL_EQUITY,
) -> list[EquityPoint]:
    """Walk through `trades` accumulating profit from `initial_equity`.

    The first point is always the starting equity (timestamp = "start").
    Entry fees are subtracted when a trade opens; exit fees, gross PnL, and
    holding costs are applied when a trade closes. Open trades only deduct
    the entry fee so far.
    """
    if initial_equity <= 0:
        raise ValueError("initial_equity must be positive")

    equity = initial_equity
    points = [EquityPoint(timestamp="start", equity=equity)]

    for t in trades:
        equity -= t.entry_fee
        points.append(EquityPoint(timestamp=t.entry_time, equity=equity))

        if not t.is_closed:
            continue

        assert t.net_pnl is not None
        # Entry fee already deducted; realize gross - slippage - exit fee - holding.
        equity += (
            t.gross_pnl
            - t.slippage_cost
            - t.exit_fee
            - t.holding_cost
        )
        points.append(EquityPoint(timestamp=str(t.exit_time), equity=equity))

    return points


def final_equity(
    trades: list[Trade],
    initial_equity: float = DEFAULT_INITIAL_EQUITY,
) -> float:
    """Convenience: equity after the last closed trade."""
    return equity_curve(trades, initial_equity)[-1].equity


def total_return_pct(
    trades: list[Trade],
    initial_equity: float = DEFAULT_INITIAL_EQUITY,
) -> float:
    """Total realized return as a fraction (0.05 means +5%)."""
    final = final_equity(trades, initial_equity)
    return (final - initial_equity) / initial_equity
