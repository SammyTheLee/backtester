"""Tests for bt.import_candles — offline file import only."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bt import __main__ as bt_main
from bt.import_candles import (
    CANONICAL_COLUMNS,
    format_candles_df,
    import_candles_csv,
    normalize_candle_columns,
)


def _write_yahoo_style(path: Path) -> None:
    pd.DataFrame(
        {
            "Datetime": ["2026-01-02 15:00:00", "2026-01-03 15:00:00", "2026-01-01 15:00:00"],
            "Open": [101.5, 102.0, 100.0],
            "High": [102.0, 103.0, 101.0],
            "Low": [100.5, 101.5, 99.5],
            "Close": [101.8, 102.5, 100.8],
            "Volume": [1000, 2000, 1500],
        }
    ).to_csv(path, index=False)


def _write_lowercase_date(path: Path) -> None:
    pd.DataFrame(
        {
            "date": ["2026-02-01", "2026-02-02"],
            "open": [50.1, 50.5],
            "high": [51.0, 51.2],
            "low": [49.8, 50.0],
            "close": [50.6, 50.9],
            "volume": [500, 600],
        }
    ).to_csv(path, index=False)


def test_normalize_yahoo_columns():
    df = pd.DataFrame(
        {
            "Date": ["2026-01-01"],
            "Open": [1.0],
            "High": [2.0],
            "Low": [0.5],
            "Close": [1.5],
            "Volume": [100],
        }
    )
    out = normalize_candle_columns(df)
    assert list(out.columns) == list(CANONICAL_COLUMNS)


def test_format_candles_sorts_and_formats_timestamps():
    df = pd.DataFrame(
        {
            "date": ["2026-01-02", "2026-01-01"],
            "open": [102.0, 100.0],
            "high": [103.0, 101.0],
            "low": [101.0, 99.0],
            "close": [102.5, 100.5],
            "volume": [200, 100],
        }
    )
    out = format_candles_df(df)
    assert out["timestamp"].tolist() == ["2026-01-01T00:00:00", "2026-01-02T00:00:00"]
    assert out["close"].tolist() == [100.5, 102.5]
    assert out["volume"].tolist() == [100, 200]


def test_import_candles_csv_yahoo_style(tmp_path: Path):
    src = tmp_path / "yahoo_aapl.csv"
    out = tmp_path / "out" / "REAL_SAMPLE.csv"
    _write_yahoo_style(src)

    written = import_candles_csv(src, out)

    assert written == out
    loaded = pd.read_csv(out)
    assert list(loaded.columns) == list(CANONICAL_COLUMNS)
    assert len(loaded) == 3
    assert loaded["timestamp"].iloc[0] == "2026-01-01T15:00:00"
    assert loaded["timestamp"].iloc[-1] == "2026-01-03T15:00:00"


def test_import_candles_csv_lowercase_date(tmp_path: Path):
    src = tmp_path / "vendor.csv"
    out = tmp_path / "candles.csv"
    _write_lowercase_date(src)

    import_candles_csv(src, out)
    loaded = pd.read_csv(out)
    assert loaded["timestamp"].tolist() == [
        "2026-02-01T00:00:00",
        "2026-02-02T00:00:00",
    ]


def test_import_candles_csv_limit(tmp_path: Path):
    src = tmp_path / "yahoo.csv"
    out = tmp_path / "sample.csv"
    _write_yahoo_style(src)

    import_candles_csv(src, out, limit=2)
    assert len(pd.read_csv(out)) == 2


def test_import_candles_csv_missing_columns(tmp_path: Path):
    src = tmp_path / "bad.csv"
    pd.DataFrame({"date": ["2026-01-01"], "close": [1.0]}).to_csv(src, index=False)
    with pytest.raises(ValueError, match="Missing required columns"):
        import_candles_csv(src, tmp_path / "out.csv")


def test_import_data_cli(tmp_path: Path, capsys):
    src = tmp_path / "download.csv"
    out = tmp_path / "REAL_SAMPLE.csv"
    _write_yahoo_style(src)

    rc = bt_main.main(
        ["import-data", str(src), "-o", str(out), "--limit", "2"],
    )

    assert rc == 0
    assert len(pd.read_csv(out)) == 2
    captured = capsys.readouterr()
    assert "Imported 2 rows" in captured.out
    assert str(out) in captured.out


def test_import_data_cli_missing_file(tmp_path: Path, capsys):
    rc = bt_main.main(["import-data", str(tmp_path / "missing.csv")])
    assert rc == 1
    assert "not found" in capsys.readouterr().err.lower()


def test_import_data_join_csv_no_future_leakage(tmp_path: Path):
    """March 15 candle gets March 1 regime, not April 1 (backward as-of only)."""
    market = tmp_path / "market.csv"
    join = tmp_path / "regimes.csv"
    out = tmp_path / "candles.csv"

    pd.DataFrame(
        {
            "date": ["2026-03-15", "2026-04-15"],
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1000, 1100],
        }
    ).to_csv(market, index=False)

    pd.DataFrame(
        {
            "timestamp": ["2026-03-01T00:00:00Z", "2026-04-01T00:00:00Z"],
            "regime": ["risk_on", "risk_off"],
        }
    ).to_csv(join, index=False)

    import_candles_csv(market, out, join_csv=join)
    loaded = pd.read_csv(out)

    assert "regime" in loaded.columns
    assert loaded.loc[0, "regime"] == "risk_on"
    assert loaded.loc[1, "regime"] == "risk_off"


def test_import_data_join_csv_cli(tmp_path: Path, capsys):
    market = tmp_path / "market.csv"
    join = tmp_path / "join.csv"
    out = tmp_path / "out.csv"
    _write_lowercase_date(market)
    pd.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:00Z", "2026-02-02T00:00:00Z"],
            "regime": ["risk_on", "risk_off"],
        }
    ).to_csv(join, index=False)

    rc = bt_main.main(
        [
            "import-data",
            str(market),
            "-o",
            str(out),
            "--join-csv",
            str(join),
        ]
    )

    assert rc == 0
    loaded = pd.read_csv(out)
    captured = capsys.readouterr()
    assert "regime" in loaded.columns
    assert "joined: regime" in captured.out
