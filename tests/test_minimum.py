"""Minimum contract tests for core backtester guarantees.

Each test maps to one required behavior:
- Candle loading rejects missing columns
- Strategy signal appears on a known crossover
- Entry uses next candle open
- No lookahead in train split
- Fees reduce net pnl
- Stop loss closes a trade
- Parameter sweep skips invalid configs
- Walk-forward only reports final leaderboard from out-of-sample
- Portfolio mode rejects trades when max positions are reached
- External joins do not use future rows
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bt.backtest import apply_execution
from bt.costs import ExecutionCosts
from bt.merge_research import join_external_into_candles, merge_research_into_candles
from bt.pipeline import BacktestParams, load_run_config
from bt.portfolio_backtest import run_portfolio_backtest
from bt.portfolio_config import AssetSpec, PortfolioLimits
from bt.report import RUNS_CSV_COLUMNS, write_ranked_summary_csv
from bt.splits import load_candles, parse_split_config
from bt.strategy import FixedStopTargetStrategy, MaCrossoverStrategy, Position
from bt.sweep import iter_param_combinations, iter_skipped_combinations
from bt.sweep_run import generate_combinations_only
from bt.trade import extract_trades


def _candles(closes: list[float], opens: list[float] | None = None) -> pd.DataFrame:
    if opens is None:
        opens = [closes[0]] + closes[:-1]
    return pd.DataFrame(
        {
            "timestamp": [f"2026-01-01T{i:02d}:00:00" for i in range(len(closes))],
            "open": opens,
            "high": [max(o, c) + 0.5 for o, c in zip(opens, closes)],
            "low": [min(o, c) - 0.5 for o, c in zip(opens, closes)],
            "close": closes,
            "volume": [1000] * len(closes),
        }
    )


def test_candle_loading_rejects_missing_columns(tmp_path: Path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"timestamp": ["2026-01-01"], "close": [1.0]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_candles(bad)


def test_strategy_signal_on_known_crossover():
    closes = [100, 100, 100, 105, 110, 110, 110, 105, 100, 100, 100, 100]
    df = _candles(closes)
    signals = MaCrossoverStrategy(fast=2, slow=3).apply(df)["signal"].tolist()
    assert signals[3] is Position.LONG
    assert signals[7] is Position.SHORT
    assert all(signals[i] is None for i in range(len(signals)) if i not in (3, 7))


def test_entry_uses_next_candle_open():
    closes = [100, 100, 100, 105, 110, 110, 110, 105, 100, 100, 100, 100]
    df = _candles(closes)
    strat = MaCrossoverStrategy(fast=2, slow=3)
    df = strat.apply(df)
    df = apply_execution(df, FixedStopTargetStrategy())
    assert df["signal"].iloc[3] is Position.LONG
    assert pd.isna(df["entry_price"].iloc[3])
    assert df["entry_price"].iloc[4] == df["open"].iloc[4]


def test_train_split_has_no_lookahead(demo_candles_csv: Path):
    split = parse_split_config(
        {
            "data": {"split": "train"},
            "split": {
                "train_start": "2026-01-01",
                "train_end": "2026-01-05",
                "validation_start": "2026-01-06",
                "validation_end": "2026-01-09",
                "oos_start": "2026-01-10",
                "oos_end": "2026-01-13",
            },
        }
    )
    df, name = load_candles(demo_candles_csv, split=split)
    assert name == "train"
    assert df["timestamp"].max() <= pd.Timestamp("2026-01-05")
    assert df["timestamp"].min() >= pd.Timestamp("2026-01-01")
    # Validation/OOS dates must not appear in train data.
    assert (df["timestamp"] < pd.Timestamp("2026-01-06")).all()


def test_fees_reduce_net_pnl():
    df = pd.DataFrame(
        {
            "timestamp": ["t0", "t1"],
            "signal": [Position.LONG, None],
            "open": [100.0, 100.0],
            "high": [100.0, 110.0],
            "low": [100.0, 100.0],
            "close": [100.0, 105.0],
            "position": [None, Position.LONG],
            "entry_price": [float("nan"), 100.0],
            "exit_price": [float("nan"), 110.0],
            "exit_reason": [None, "target"],
        }
    )
    trade = extract_trades(df, costs=ExecutionCosts(fee_bps=10))[0]
    assert trade.gross_pnl == pytest.approx(10.0)
    assert trade.net_pnl < trade.gross_pnl


def test_stop_loss_closes_trade():
    df = pd.DataFrame(
        {
            "timestamp": [f"2026-01-01T{i:02d}:00:00" for i in range(3)],
            "open": [100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 100.5],
            "low": [100.0, 100.0, 97.0],
            "close": [100.0, 100.0, 98.5],
            "volume": [1000, 1000, 1000],
            "signal": [None, Position.LONG, None],
        }
    )
    df = apply_execution(
        df,
        FixedStopTargetStrategy(stop_pct=0.02, target_pct=0.04),
    )
    assert df["exit_price"].iloc[2] == 98.0
    assert df["exit_reason"].iloc[2] == "stop"


def test_parameter_sweep_skips_invalid_configs(
    demo_candles_csv: Path, isolated_results: Path, tmp_path: Path
):
    cfg_path = tmp_path / "overlap.toml"
    cfg_path.write_text(
        f"""
[data]
csv_path = "{demo_candles_csv}"

[split]
train_start = "2026-01-01"
train_end = "2026-01-05"
validation_start = "2026-01-06"
validation_end = "2026-01-09"
oos_start = "2026-01-10"
oos_end = "2026-01-13"

[strategy]
type = "ma_crossover"
fast_ma = [5, 30]
slow_ma = [10, 50]

[execution]
stop_pct = 0.01
fee_bps = 0
slippage_bps = 0

[portfolio]
initial_equity = 10000.0
size = 1.0
""",
        encoding="utf-8",
    )
    loaded = load_run_config(cfg_path, for_sweep=True)
    skipped = iter_skipped_combinations(loaded.config)
    assert len(skipped) == 1
    assert skipped[0]["fast"] == 30
    assert skipped[0]["slow"] == 10
    assert len(iter_param_combinations(loaded.config)) == 3

    out_dir, combos = generate_combinations_only(loaded, output_dir=isolated_results / "min_sweep")
    skipped_csv = out_dir / "skipped_combinations.csv"
    assert skipped_csv.exists()
    assert len(combos) == 3


def test_walkforward_leaderboard_is_oos_only(tmp_path: Path):
    runs = tmp_path / "runs.csv"
    summary = tmp_path / "summary.csv"
    rows = [
        {
            "run_at": "t",
            "config": "train",
            "data_path": "d",
            "data_split": "train",
            "strategy": "s",
            "fast": 5,
            "slow": 30,
            "gross_pnl": 999.0,
            "net_pnl": 999.0,
            "total_return_pct": 0.1,
            "composite_score": 0.1,
            "trade_count": 1,
            "win_rate": 0.5,
            "profit_factor": 1.0,
            "max_drawdown": 1.0,
            "average_hold_hours": 1.0,
            "time_in_market_pct": 0.1,
        },
        {
            "run_at": "t",
            "config": "oos",
            "data_path": "d",
            "data_split": "oos",
            "strategy": "s",
            "fast": 10,
            "slow": 50,
            "gross_pnl": 1.0,
            "net_pnl": 1.0,
            "total_return_pct": 0.0001,
            "composite_score": 0.0001,
            "trade_count": 1,
            "win_rate": 0.5,
            "profit_factor": 1.0,
            "max_drawdown": 1.0,
            "average_hold_hours": 1.0,
            "time_in_market_pct": 0.1,
        },
    ]
    pd.DataFrame(rows, columns=RUNS_CSV_COLUMNS).to_csv(runs, index=False)
    _, n = write_ranked_summary_csv(runs, summary, sort_by="composite_score")
    loaded = pd.read_csv(summary)
    assert n == 1
    assert (loaded["data_split"] == "oos").all()
    assert loaded.iloc[0]["net_pnl"] == pytest.approx(1.0)


def test_portfolio_rejects_when_max_positions_reached(demo_candles_csv: Path):
    params = BacktestParams(
        combo={"fast": 5, "slow": 20, "stop_pct": 0.02, "target_pct": 0.04}
    )
    limits = PortfolioLimits(max_open_positions=1)
    result = run_portfolio_backtest(
        [AssetSpec("A", demo_candles_csv), AssetSpec("B", demo_candles_csv)],
        params,
        ExecutionCosts(),
        config={"strategy": {"type": "ma_crossover"}},
        initial_equity=10_000,
        size=1.0,
        limits=limits,
    )
    assert len(result.accepted_trades) >= 1
    assert len(result.rejected) >= 1
    assert max(p.open_positions for p in result.portfolio_equity) <= 1


def test_external_join_has_no_future_leakage():
    candles = pd.DataFrame(
        {
            "timestamp": ["2026-03-15T12:00:00", "2026-04-15T12:00:00"],
            "open": [1, 1],
            "high": [1, 1],
            "low": [1, 1],
            "close": [1, 1],
            "volume": [1, 1],
        }
    )
    research = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-03-01T00:00:00", "2026-04-01T00:00:00"]
            ),
            "regime": ["risk_on", "risk_off"],
        }
    )
    merged = merge_research_into_candles(
        candles,
        research,
        value_column="regime",
    )
    assert merged["regime"].iloc[0] == "risk_on"
    assert merged["regime"].iloc[1] == "risk_off"

    future_only = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-10T00:00:00"]),
            "regime": ["risk_on"],
        }
    )
    early_candles = pd.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:00", "2026-01-05T00:00:00"],
            "open": [1, 1],
            "high": [1, 1],
            "low": [1, 1],
            "close": [1, 1],
            "volume": [1, 1],
        }
    )
    joined = join_external_into_candles(early_candles, future_only)
    assert pd.isna(joined["regime"].iloc[0])
    assert pd.isna(joined["regime"].iloc[1])
