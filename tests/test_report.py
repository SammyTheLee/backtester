"""Tests for `bt.report`."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from bt.portfolio import equity_curve
from bt.report import (
    RUNS_CSV_COLUMNS,
    SUMMARY_CSV_COLUMNS,
    SUMMARY_RANKED_COLUMNS,
    StrategyReportCard,
    append_runs_csv,
    build_report_card,
    composite_score,
    compute_metrics,
    format_report_card,
    rank_dataframe,
    save_report_card,
    dedupe_best_per_configuration,
    write_ranked_summary_csv,
)
from bt.strategy import Position
from bt.trade import Trade


def _closed(
    gross: float,
    net: float,
    *,
    entry: str = "2026-01-01T00:00:00",
    exit_: str = "2026-01-01T05:00:00",
) -> Trade:
    return Trade(
        entry_time=entry,
        entry_price=100.0,
        side=Position.LONG,
        exit_time=exit_,
        exit_price=110.0,
        exit_reason="target",
        gross_pnl=gross,
        net_pnl=net,
    )


def test_compute_metrics_from_trades_and_curve():
    trades = [
        _closed(10.0, 8.0),
        _closed(5.0, 4.0, entry="2026-01-02T00:00:00", exit_="2026-01-02T03:00:00"),
        _closed(-3.0, -4.0, entry="2026-01-03T00:00:00", exit_="2026-01-03T01:00:00"),
    ]
    curve = equity_curve(trades, initial_equity=10_000.0)
    m = compute_metrics(trades, curve)
    assert m.net_pnl == pytest.approx(8.0)
    assert m.trade_count == 3
    assert m.win_rate == pytest.approx(2 / 3)
    assert m.profit_factor == pytest.approx(12.0 / 4.0)
    assert m.max_drawdown >= 0.0


def test_build_report_card_metrics():
    trades = [
        _closed(10.0, 8.0),
        _closed(5.0, 4.0, entry="2026-01-02T00:00:00", exit_="2026-01-02T03:00:00"),
        _closed(-3.0, -4.0, entry="2026-01-03T00:00:00", exit_="2026-01-03T01:00:00"),
    ]
    card = build_report_card(
        trades,
        initial_equity=10_000.0,
        bars_in_market=50,
        total_bars=100,
    )
    assert card.gross_pnl == pytest.approx(12.0)
    assert card.net_pnl == pytest.approx(8.0)
    assert card.trade_count == 3
    assert card.win_rate == pytest.approx(2 / 3)
    assert card.profit_factor == pytest.approx(12.0 / 4.0)
    assert card.time_in_market_pct == pytest.approx(0.5)
    assert card.average_hold_hours == pytest.approx((5 + 3 + 1) / 3)


def test_max_drawdown_uses_peak_to_trough():
    trades = [
        _closed(100.0, 100.0, entry="2026-01-01T00:00:00", exit_="2026-01-01T01:00:00"),
        _closed(-80.0, -80.0, entry="2026-01-02T00:00:00", exit_="2026-01-02T01:00:00"),
    ]
    card = build_report_card(trades, initial_equity=10_000.0)
    curve = equity_curve(trades, initial_equity=10_000.0)
    assert card.max_drawdown == pytest.approx(80.0)
    assert curve[-1].equity == pytest.approx(10_020.0)


def test_profit_factor_inf_when_no_losses():
    card = build_report_card([_closed(5.0, 5.0)], initial_equity=10_000.0)
    assert card.profit_factor == float("inf")


def test_append_runs_csv_accumulates_rows(tmp_path: Path):
    card = StrategyReportCard(
        gross_pnl=1.0,
        net_pnl=0.5,
        total_return_pct=0.001,
        trade_count=1,
        win_rate=1.0,
        profit_factor=1.0,
        max_drawdown=0.1,
        average_hold_hours=2.0,
        time_in_market_pct=0.1,
    )
    path = tmp_path / "runs.csv"
    append_runs_csv(
        card,
        path,
        run_at="2026-01-01T00:00:00+00:00",
        config="configs/a.toml",
        data_path="data/candles/DEMO.csv",
        strategy="ma_crossover(fast=5, slow=20)",
        fast=5,
        slow=20,
    )
    append_runs_csv(
        card,
        path,
        run_at="2026-01-02T00:00:00+00:00",
        config="configs/b.toml",
        data_path="data/candles/BTC_USD.csv",
        strategy="ma_crossover(fast=10, slow=30)",
        fast=10,
        slow=30,
    )
    loaded = pd.read_csv(path)
    assert len(loaded) == 2
    assert list(loaded.columns) == RUNS_CSV_COLUMNS
    assert loaded["config"].iloc[1] == "configs/b.toml"
    assert loaded["fast"].iloc[1] == 10


def test_composite_score_rewards_return_penalizes_drawdown():
    # +10% return, 2% drawdown → score 8%
    assert composite_score(0.10, 200.0, initial_equity=10_000) == pytest.approx(0.08)
    # High PnL but huge drawdown scores lower than modest return + small dd
    good = composite_score(0.05, 100.0, initial_equity=10_000)
    bad = composite_score(0.08, 900.0, initial_equity=10_000)
    assert good > bad


def test_rank_dataframe_by_composite_score(tmp_path: Path):
    rows = [
        {
            "config": "low_score",
            "total_return_pct": 0.08,
            "max_drawdown": 900.0,
            "net_pnl": 800.0,
        },
        {
            "config": "high_score",
            "total_return_pct": 0.05,
            "max_drawdown": 100.0,
            "net_pnl": 500.0,
        },
    ]
    df = rank_dataframe(pd.DataFrame(rows), "composite_score", initial_equity=10_000)
    assert df.iloc[0]["config"] == "high_score"
    assert df.iloc[0]["rank"] == 1


def test_dedupe_best_per_configuration_keeps_best_run():
    df = pd.DataFrame(
        [
            {"fast": 5, "slow": 20, "net_pnl": 1.0, "composite_score": 0.01},
            {"fast": 5, "slow": 20, "net_pnl": 50.0, "composite_score": 0.05},
            {"fast": 10, "slow": 30, "net_pnl": 10.0, "composite_score": 0.02},
        ]
    )
    out = dedupe_best_per_configuration(df, "composite_score")
    assert len(out) == 2
    assert out.iloc[0]["net_pnl"] == pytest.approx(50.0)


def test_write_ranked_summary_csv_top_n(tmp_path: Path):
    runs = tmp_path / "runs.csv"
    summary = tmp_path / "summary.csv"
    rows = [
        {
            "run_at": "t",
            "config": f"c{i}",
            "data_path": "d",
            "data_split": "all",
            "strategy": "s",
            "fast": i,
            "slow": 30,
            "gross_pnl": i,
            "net_pnl": float(i),
            "total_return_pct": i / 1000,
            "composite_score": i / 1000,
            "trade_count": 1,
            "win_rate": 0.5,
            "profit_factor": 1.0,
            "max_drawdown": 1.0,
            "average_hold_hours": 1.0,
            "time_in_market_pct": 0.1,
        }
        for i in range(15)
    ]
    pd.DataFrame(rows, columns=RUNS_CSV_COLUMNS).to_csv(runs, index=False)
    runs_n, summary_n = write_ranked_summary_csv(
        runs,
        summary,
        sort_by="composite_score",
        top_n=5,
        leaderboard_data_split=None,
    )
    assert runs_n == 15
    assert summary_n == 5
    loaded = pd.read_csv(summary)
    assert loaded["rank"].tolist() == [1, 2, 3, 4, 5]
    assert loaded["fast"].iloc[0] == 14


def test_write_ranked_summary_csv_oos_only(tmp_path: Path):
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
            "gross_pnl": 1000.0,
            "net_pnl": 1000.0,
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
            "fast": 5,
            "slow": 30,
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
        {
            "run_at": "t",
            "config": "oos2",
            "data_path": "d",
            "data_split": "oos",
            "strategy": "s",
            "fast": 10,
            "slow": 50,
            "gross_pnl": 50.0,
            "net_pnl": 50.0,
            "total_return_pct": 0.05,
            "composite_score": 0.05,
            "trade_count": 1,
            "win_rate": 0.5,
            "profit_factor": 1.0,
            "max_drawdown": 1.0,
            "average_hold_hours": 1.0,
            "time_in_market_pct": 0.1,
        },
    ]
    pd.DataFrame(rows, columns=RUNS_CSV_COLUMNS).to_csv(runs, index=False)
    _, summary_n = write_ranked_summary_csv(runs, summary, sort_by="composite_score")
    assert summary_n == 2
    loaded = pd.read_csv(summary)
    assert (loaded["data_split"] == "oos").all()
    assert loaded.iloc[0]["fast"] == 10
    assert loaded.iloc[0]["net_pnl"] == pytest.approx(50.0)


def test_write_ranked_summary_csv_orders_by_net_pnl(tmp_path: Path):
    card_low = StrategyReportCard(
        gross_pnl=1.0,
        net_pnl=1.0,
        total_return_pct=0.0001,
        trade_count=1,
        win_rate=1.0,
        profit_factor=1.0,
        max_drawdown=1.0,
        average_hold_hours=1.0,
        time_in_market_pct=0.1,
    )
    card_high = StrategyReportCard(
        gross_pnl=100.0,
        net_pnl=50.0,
        total_return_pct=0.05,
        trade_count=10,
        win_rate=0.6,
        profit_factor=2.0,
        max_drawdown=2.0,
        average_hold_hours=3.0,
        time_in_market_pct=0.5,
    )
    runs = tmp_path / "runs.csv"
    summary = tmp_path / "summary.csv"
    append_runs_csv(card_low, runs, config="a.toml", fast=2, slow=20)
    append_runs_csv(card_high, runs, config="b.toml", fast=5, slow=20)
    runs_n, summary_n = write_ranked_summary_csv(
        runs, summary, sort_by="net_pnl", leaderboard_data_split=None
    )
    assert runs_n == 2
    assert summary_n == 2
    loaded = pd.read_csv(summary)
    assert list(loaded.columns) == SUMMARY_RANKED_COLUMNS
    assert loaded["rank"].tolist() == [1, 2]
    assert loaded["net_pnl"].iloc[0] == pytest.approx(50.0)
    assert loaded["config"].iloc[0] == "b.toml"


def test_format_and_save_report(tmp_path: Path):
    card = StrategyReportCard(
        gross_pnl=10.0,
        net_pnl=8.0,
        total_return_pct=0.001,
        trade_count=2,
        win_rate=0.5,
        profit_factor=2.0,
        max_drawdown=5.0,
        average_hold_hours=4.0,
        time_in_market_pct=0.25,
    )
    text = format_report_card(card)
    assert "Gross PnL" in text
    assert "25.0%" in text

    path = tmp_path / "report.json"
    save_report_card(card, path)
    data = json.loads(path.read_text())
    assert data["net_pnl"] == 8.0
