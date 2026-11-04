"""Tests for inspect-run CLI helper."""

from __future__ import annotations

from pathlib import Path

import pytest

from bt.inspect_run import inspect_run_folder, resolve_results_dir


def test_resolve_latest_under_results(tmp_path: Path, monkeypatch):
    run = tmp_path / "results" / "20260101_120000"
    run.mkdir(parents=True)
    (run / "metadata.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert resolve_results_dir("latest").name == "20260101_120000"
    assert resolve_results_dir("results/latest").name == "20260101_120000"


def test_inspect_backtest_folder(tmp_path: Path):
    run = tmp_path / "20260101_120000"
    run.mkdir()
    (run / "metadata.json").write_text(
        '{"config_path":"c.toml","resolved":{"strategy":"s","data_csv":"d.csv"}}',
        encoding="utf-8",
    )
    (run / "report.json").write_text(
        '{"gross_pnl":1,"net_pnl":1,"total_return_pct":0.0001,'
        '"trade_count":1,"win_rate":1,"profit_factor":1,'
        '"max_drawdown":0.1,"average_hold_hours":1,"time_in_market_pct":0.1}',
        encoding="utf-8",
    )
    text = "\n".join(inspect_run_folder(run))
    assert "Strategy Report Card" in text
    assert "c.toml" in text


def test_resolve_missing_raises(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        resolve_results_dir("results/latest")
