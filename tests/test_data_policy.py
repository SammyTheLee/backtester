"""Enforce offline, local-CSV data policy."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tests.conftest import deterministic_candles_df, write_deterministic_candles


def test_bt_package_does_not_import_yfinance():
    for name, mod in list(sys.modules.items()):
        if not name.startswith("bt."):
            continue
        assert "yfinance" not in getattr(mod, "__dict__", {}), name


def test_fetch_data_imports_yfinance_lazily():
    """fetch_data.py is manual-only; yfinance loads inside fetch(), not at import."""
    import importlib
    import inspect

    sys.modules.pop("fetch_data", None)
    import fetch_data  # noqa: F401

    assert "yfinance" not in sys.modules
    assert "import yfinance" in inspect.getsource(fetch_data.fetch)


def test_deterministic_candles_are_repeatable():
    a = deterministic_candles_df(seed=7)
    b = deterministic_candles_df(seed=7)
    assert a.equals(b)
    assert len(a) == 336
    assert list(a.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


def test_write_deterministic_candles(tmp_path: Path):
    path = write_deterministic_candles(tmp_path / "candles" / "TEST.csv", seed=1)
    assert path.exists()
    loaded = __import__("pandas").read_csv(path)
    assert len(loaded) == 336
