"""Portfolio mode configuration: multiple assets, shared capital, exposure limits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AssetSpec:
    symbol: str
    csv_path: Path


@dataclass(frozen=True)
class PortfolioLimits:
    """Caps on simultaneous positions and notional exposure (fractions of equity)."""

    max_open_positions: int | None = None
    max_total_exposure_pct: float = 1.0
    max_exposure_per_asset_pct: float = 1.0

    def __post_init__(self) -> None:
        if self.max_open_positions is not None and self.max_open_positions < 1:
            raise ValueError("max_open_positions must be >= 1")
        for name, value in (
            ("max_total_exposure_pct", self.max_total_exposure_pct),
            ("max_exposure_per_asset_pct", self.max_exposure_per_asset_pct),
        ):
            if value < 0 or value > 1:
                raise ValueError(f"{name} must be between 0 and 1, got {value}")


def _symbol_from_path(path: Path) -> str:
    return path.stem


def parse_assets(config: dict[str, Any]) -> list[AssetSpec]:
    """Read ``data.assets`` (list) or fall back to single ``data.csv_path``."""
    data = config.get("data", {})
    raw = data.get("assets")
    if raw is None:
        if "csv_path" not in data:
            raise ValueError("config needs data.csv_path or data.assets")
        path = Path(data["csv_path"])
        return [AssetSpec(symbol=_symbol_from_path(path), csv_path=path)]

    if not isinstance(raw, list) or not raw:
        raise ValueError("data.assets must be a non-empty list")

    assets: list[AssetSpec] = []
    for item in raw:
        if isinstance(item, str):
            path = Path(item)
            assets.append(AssetSpec(symbol=_symbol_from_path(path), csv_path=path))
        elif isinstance(item, dict):
            if "csv_path" not in item:
                raise ValueError("each asset entry needs csv_path")
            path = Path(item["csv_path"])
            symbol = str(item.get("symbol", _symbol_from_path(path)))
            assets.append(AssetSpec(symbol=symbol, csv_path=path))
        else:
            raise ValueError("data.assets entries must be paths or {symbol, csv_path} tables")
    return assets


def parse_portfolio_limits(config: dict[str, Any]) -> PortfolioLimits:
    port = config.get("portfolio", {})
    if not isinstance(port, dict):
        port = {}
    max_pos = port.get("max_open_positions")
    return PortfolioLimits(
        max_open_positions=None if max_pos is None else int(max_pos),
        max_total_exposure_pct=float(port.get("max_total_exposure_pct", 1.0)),
        max_exposure_per_asset_pct=float(port.get("max_exposure_per_asset_pct", 1.0)),
    )


def is_portfolio_mode(config: dict[str, Any], assets: list[AssetSpec]) -> bool:
    port = config.get("portfolio", {})
    if isinstance(port, dict) and str(port.get("mode", "")).lower() == "portfolio":
        return len(assets) >= 1
    return len(assets) > 1
