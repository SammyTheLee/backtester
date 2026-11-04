"""Shared pytest fixtures — offline, deterministic data only.

Data policy (tests):
- Candle data is written locally under tmp_path (or committed DEMO.csv for humans).
- No network access during tests.
- No live API downloads in the test suite.
"""

from __future__ import annotations

import importlib
import socket
from pathlib import Path

import pandas as pd
import pytest

# Modules that bind RESULTS_DIR at import time.
_RESULTS_DIR_MODULES = (
    "bt.paths",
    "bt.sweep_run",
    "bt.walkforward_run",
    "bt.validate_run",
    "bt.oos_run",
    "bt.walkforward",
)


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests must not call the internet."""

    class _BlockedSocket(socket.socket):
        def connect(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise OSError("Network access is disabled during tests")

    monkeypatch.setattr(socket, "socket", _BlockedSocket)


def deterministic_candles_df(*, hours: int = 336, seed: int = 42) -> pd.DataFrame:
    """Fake OHLCV candles: 336 hourly bars from 2026-01-01 (covers split windows)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-01-01T00:00:00")
    timestamps = [
        (start + pd.Timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%S")
        for i in range(hours)
    ]
    close = 100.0 + np.cumsum(rng.normal(0, 0.25, hours))
    close = np.clip(close, 85.0, 115.0)
    open_ = np.empty(hours)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) + rng.uniform(0.05, 0.35, hours)
    low = np.minimum(open_, close) - rng.uniform(0.05, 0.35, hours)
    volume = rng.integers(1000, 5000, hours)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": np.round(open_, 2),
            "high": np.round(high, 2),
            "low": np.round(low, 2),
            "close": np.round(close, 2),
            "volume": volume,
        }
    )


def write_deterministic_candles(path: Path, *, hours: int = 336, seed: int = 42) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    deterministic_candles_df(hours=hours, seed=seed).to_csv(path, index=False)
    return path


@pytest.fixture
def demo_candles_csv(tmp_path: Path) -> Path:
    """Deterministic candle CSV in an isolated temp directory."""
    return write_deterministic_candles(tmp_path / "DEMO.csv")


@pytest.fixture
def isolated_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect all run output to a temp results/ folder."""
    results = tmp_path / "results"
    results.mkdir()
    for name in _RESULTS_DIR_MODULES:
        mod = importlib.import_module(name)
        monkeypatch.setattr(mod, "RESULTS_DIR", results)
    return results


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]
