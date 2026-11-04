"""Import downloaded market CSVs into backtester candle format.

Reads common column names (Yahoo, lowercase, etc.), normalizes to
``timestamp, open, high, low, close, volume``, and writes a local CSV.
No network access — file in, file out.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CANONICAL_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")

DEFAULT_REAL_SAMPLE_PATH = Path("data/candles/REAL_SAMPLE.csv")

# Lowercase source name -> canonical column.
_COLUMN_ALIASES: dict[str, str] = {
    "timestamp": "timestamp",
    "date": "timestamp",
    "datetime": "timestamp",
    "time": "timestamp",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "adj close": "close",
    "adj_close": "close",
    "volume": "volume",
    "vol": "volume",
}


def normalize_candle_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map vendor column names to canonical OHLCV columns."""
    work = df.copy()
    work.columns = [str(c).strip() for c in work.columns]

    rename: dict[str, str] = {}
    claimed: set[str] = set()
    for col in work.columns:
        canonical = _COLUMN_ALIASES.get(col.lower())
        if canonical is None:
            continue
        if canonical in claimed:
            continue
        rename[col] = canonical
        claimed.add(canonical)

    work = work.rename(columns=rename)

    missing = [c for c in CANONICAL_COLUMNS if c not in work.columns]
    if missing:
        raise ValueError(
            f"Missing required columns {missing}. "
            f"Expected date/open/high/low/close/volume variants; got {list(df.columns)}"
        )

    return work[list(CANONICAL_COLUMNS)].copy()


def format_candles_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize types and timestamp string format for backtester CSVs."""
    out = normalize_candle_columns(df)
    out["timestamp"] = pd.to_datetime(out["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    for col in ("open", "high", "low", "close"):
        out[col] = pd.to_numeric(out[col], errors="raise").round(2)
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0).astype(int)
    out = out.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    return out


def import_candles_csv(
    input_path: Path | str,
    output_path: Path | str | None = None,
    *,
    limit: int | None = None,
    join_csv: Path | str | None = None,
    join_asset: str | None = None,
) -> Path:
    """Read a downloaded market CSV and write backtester-format candles."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    out_path = Path(output_path) if output_path is not None else DEFAULT_REAL_SAMPLE_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(input_path)
    df = format_candles_df(raw)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be >= 1")
        df = df.head(limit)

    if join_csv is not None:
        from bt.merge_research import join_csv_into_candles

        df = join_csv_into_candles(df, join_csv, asset=join_asset)

    df.to_csv(out_path, index=False)
    return out_path
