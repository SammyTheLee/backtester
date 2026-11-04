"""Tests for bt.merge_research — offline file merge only."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bt import __main__ as bt_main
from bt.merge_research import (
    load_research_csv,
    merge_research_csv,
    merge_research_into_candles,
)
from tests.conftest import write_deterministic_candles


def _write_research(path: Path) -> None:
    pd.DataFrame(
        {
            "timestamp": ["2026-01-02T00:00:00Z", "2026-01-04T00:00:00Z"],
            "asset": ["BTC", "BTC"],
            "regime": ["risk_on", "risk_off"],
        }
    ).to_csv(path, index=False)


def test_load_research_filters_asset(tmp_path: Path):
    path = tmp_path / "research.csv"
    _write_research(path)
    df = load_research_csv(path, asset="BTC")
    assert list(df.columns) == ["timestamp", "regime"]
    assert len(df) == 2


def test_load_research_keeps_asset_column_when_not_filtered(tmp_path: Path):
    path = tmp_path / "research.csv"
    _write_research(path)
    df = load_research_csv(path)
    assert list(df.columns) == ["timestamp", "asset", "regime"]
    assert len(df) == 2


def test_merge_research_matches_by_asset_column():
    candles = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-03T00:00:00",
                "2026-01-03T00:00:00",
                "2026-01-05T00:00:00",
            ],
            "open": [1, 1, 1],
            "high": [1, 1, 1],
            "low": [1, 1, 1],
            "close": [1, 1, 1],
            "volume": [1, 1, 1],
            "asset": ["BTC", "AAPL", "BTC"],
        }
    )
    research = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-02T00:00:00",
                    "2026-01-02T00:00:00",
                    "2026-01-04T00:00:00",
                ]
            ),
            "asset": ["BTC", "AAPL", "BTC"],
            "regime": ["risk_on", "risk_off", "risk_off"],
        }
    )
    merged = merge_research_into_candles(
        candles,
        research,
        value_column="regime",
        output_column="risk_on",
    )
    btc = merged.loc[merged["asset"] == "BTC"].reset_index(drop=True)
    aapl = merged.loc[merged["asset"] == "AAPL"].reset_index(drop=True)
    assert btc.loc[0, "risk_on"] == "risk_on"
    assert aapl.loc[0, "risk_on"] == "risk_off"
    assert btc.loc[1, "risk_on"] == "risk_off"


def test_merge_research_requires_asset_selector_for_single_asset_candles(tmp_path: Path):
    candles = pd.DataFrame(
        {
            "timestamp": ["2026-01-03T00:00:00"],
            "open": [1],
            "high": [1],
            "low": [1],
            "close": [1],
            "volume": [1],
        }
    )
    research = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-02T00:00:00"]),
            "asset": ["BTC"],
            "regime": ["risk_on"],
        }
    )
    with pytest.raises(ValueError, match="pass --asset"):
        merge_research_into_candles(candles, research, value_column="regime")


def test_merge_research_backward_asof_join(tmp_path: Path):
    candles = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01T00:00:00",
                "2026-01-02T12:00:00",
                "2026-01-03T12:00:00",
                "2026-01-04T12:00:00",
            ],
            "open": [1, 1, 1, 1],
            "high": [1, 1, 1, 1],
            "low": [1, 1, 1, 1],
            "close": [1, 1, 1, 1],
            "volume": [1, 1, 1, 1],
        }
    )
    research = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-02T00:00:00", "2026-01-04T00:00:00"]
            ),
            "regime": ["risk_on", "risk_off"],
        }
    )
    merged = merge_research_into_candles(
        candles,
        research,
        value_column="regime",
        output_column="risk_on",
    )
    assert pd.isna(merged["risk_on"].iloc[0])
    assert merged["risk_on"].iloc[1:].tolist() == ["risk_on", "risk_on", "risk_off"]


def test_candle_sees_latest_prior_regime_not_future():
    """March 15 candle gets March 1 regime, not April 1."""
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
                ["2026-03-01T00:00:00Z", "2026-04-01T00:00:00Z"]
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


def test_future_research_does_not_affect_earlier_candles():
    candles = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01T00:00:00",
                "2026-01-05T00:00:00",
                "2026-01-11T00:00:00",
            ],
            "open": [1, 1, 1],
            "high": [1, 1, 1],
            "low": [1, 1, 1],
            "close": [1, 1, 1],
            "volume": [1, 1, 1],
        }
    )
    research = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-10T00:00:00"]),
            "regime": ["risk_on"],
        }
    )
    merged = merge_research_into_candles(
        candles,
        research,
        value_column="regime",
        output_column="risk_on",
    )
    assert pd.isna(merged["risk_on"].iloc[0])
    assert pd.isna(merged["risk_on"].iloc[1])
    assert merged["risk_on"].iloc[2] == "risk_on"


def test_merge_research_csv_end_to_end(tmp_path: Path):
    candles = tmp_path / "candles.csv"
    research = tmp_path / "research.csv"
    out = tmp_path / "merged.csv"
    write_deterministic_candles(candles, seed=3)
    _write_research(research)

    written = merge_research_csv(
        candles,
        research,
        out,
        asset="BTC",
        value_column="regime",
        output_column="risk_on",
        fill_before="risk_off",
    )

    assert written == out
    df = pd.read_csv(out)
    assert "risk_on" in df.columns
    assert set(df["risk_on"].dropna().unique()) <= {"risk_on", "risk_off"}


def test_merge_research_cli(tmp_path: Path, capsys):
    candles = tmp_path / "candles.csv"
    research = tmp_path / "research.csv"
    out = tmp_path / "out.csv"
    write_deterministic_candles(candles, seed=1)
    _write_research(research)

    rc = bt_main.main(
        [
            "merge-research",
            str(candles),
            str(research),
            "-o",
            str(out),
            "--asset",
            "BTC",
            "--column",
            "regime",
            "--output-column",
            "risk_on",
            "--fill-before",
            "risk_off",
        ]
    )

    assert rc == 0
    assert (out).exists()
    captured = capsys.readouterr()
    assert "Merged" in captured.out
    assert "risk_on" in captured.out


def test_merge_research_cli_missing_candles(tmp_path: Path, capsys):
    rc = bt_main.main(
        [
            "merge-research",
            str(tmp_path / "missing.csv"),
            str(tmp_path / "research.csv"),
        ]
    )
    assert rc == 1
    assert "not found" in capsys.readouterr().err.lower()
