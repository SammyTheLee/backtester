"""Tests for multi-asset portfolio mode."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bt.costs import ExecutionCosts
from bt.pipeline import BacktestParams, load_run_config
from bt.portfolio_backtest import apply_portfolio_constraints, run_portfolio_backtest
from bt.portfolio_config import AssetSpec, PortfolioLimits, parse_assets
from bt.portfolio_constraints import check_entry_allowed
from bt.strategy import Position
from bt.trade import Trade


def _mini_candles(path: Path, base: float, signal_bar: int = 5) -> None:
    """Tiny uptrend with a crossover-friendly pattern."""
    n = 12
    rows = []
    for i in range(n):
        px = base + i * 0.5
        rows.append(
            {
                "timestamp": f"2026-01-01T{i:02d}:00:00",
                "open": px,
                "high": px + 0.2,
                "low": px - 0.2,
                "close": px,
                "volume": 100,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_parse_assets_list():
    cfg = {
        "data": {
            "assets": [
                "data/a.csv",
                {"symbol": "X", "csv_path": "data/b.csv"},
            ]
        }
    }
    assets = parse_assets(cfg)
    assert len(assets) == 2
    assert assets[1].symbol == "X"


def test_check_entry_max_open_positions():
    open_t = Trade(
        entry_time="t",
        entry_price=100.0,
        side=Position.LONG,
        size=1.0,
        asset="A",
    )
    limits = PortfolioLimits(max_open_positions=1)
    check = check_entry_allowed(
        equity=10_000,
        asset="B",
        new_notional=100,
        open_trades=[open_t],
        limits=limits,
    )
    assert not check.allowed
    assert check.reason == "max_open_positions"


def test_apply_constraints_rejects_second_entry(demo_candles_csv: Path):
    from bt.costs import ExecutionCosts
    from bt.pipeline import BacktestParams
    from bt.portfolio_backtest import run_portfolio_backtest
    from bt.portfolio_config import AssetSpec, PortfolioLimits

    params = BacktestParams(
        combo={"fast": 5, "slow": 20, "stop_pct": 0.02, "target_pct": 0.04}
    )
    costs = ExecutionCosts()
    limits = PortfolioLimits(max_open_positions=1)

    result = run_portfolio_backtest(
        [AssetSpec("A", demo_candles_csv), AssetSpec("B", demo_candles_csv)],
        params,
        costs,
        config={"strategy": {"type": "ma_crossover"}},
        initial_equity=10_000,
        size=1.0,
        limits=limits,
    )
    assert len(result.accepted_trades) >= 1
    assert len(result.rejected) >= 1
    assert len(result.portfolio_equity) > 1
    assert result.portfolio_equity[0].open_positions == 0
    assert max(p.open_positions for p in result.portfolio_equity) <= 1


def test_portfolio_run_exports(tmp_path: Path, monkeypatch, demo_candles_csv: Path):
    cfg_path = tmp_path / "p.toml"
    cfg_path.write_text(
        f"""
[data]
assets = ["{demo_candles_csv}", "{demo_candles_csv}"]

[strategy]
type = "ma_crossover"
fast = 5
slow = 20

[execution]
fee_bps = 0
slippage_bps = 0

[portfolio]
mode = "portfolio"
initial_equity = 10000.0
size = 1.0
max_open_positions = 1
max_total_exposure_pct = 0.5
max_exposure_per_asset_pct = 0.3
""",
        encoding="utf-8",
    )
    loaded = load_run_config(cfg_path)
    assert loaded.is_portfolio

    from bt.pipeline import execute_backtest, params_from_config
    from bt.paths import RESULTS_DIR

    monkeypatch.setattr("bt.paths.RESULTS_DIR", tmp_path / "results")
    result = execute_backtest(
        loaded, params_from_config(loaded.config), export_artifacts=True
    )
    paths = result.output_paths
    assert paths is not None
    assert paths.portfolio_equity.exists()
    assert paths.rejected_trades.exists()
    trades = pd.read_csv(paths.trades)
    assert "asset" in trades.columns
    pf_eq = pd.read_csv(paths.portfolio_equity)
    assert "equity" in pf_eq.columns
    assert "open_positions" in pf_eq.columns
    assert pf_eq.iloc[0]["open_positions"] == 0
