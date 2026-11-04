"""Merge precomputed research columns into candle CSVs.

Uses backward as-of joining on timestamp: each candle bar gets the latest
research row at or before that bar's time. Future research rows never affect
earlier candles (no lookahead). Example: a candle on March 15 sees the March 1
regime, not the April 1 regime. When an asset column exists, rows are matched
by asset as well as time. No network access.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bt.import_candles import CANONICAL_COLUMNS

ASOF_DIRECTION = "backward"
ASSET_COLUMN = "asset"


def _parse_timestamps(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True).dt.tz_convert(None)


def _normalize_asset(value: object) -> str:
    return str(value).strip().upper()


def _asset_keys(series: pd.Series) -> pd.Series:
    return series.map(_normalize_asset)


def _backward_asof_values(
    candle_ts: pd.Series,
    research_ts: pd.Series,
    research_values: pd.Series,
) -> pd.Series:
    """Map each candle to the latest research value with research_ts <= candle_ts."""
    if research_ts.empty:
        return pd.Series(pd.NA, index=candle_ts.index, dtype=research_values.dtype)

    if not research_ts.is_monotonic_increasing:
        order = research_ts.argsort()
        research_ts = research_ts.iloc[order].reset_index(drop=True)
        research_values = research_values.iloc[order].reset_index(drop=True)

    r_ts = research_ts.to_numpy()
    r_vals = research_values.to_numpy()
    out: list[object] = []
    for ts in candle_ts:
        idx = int(r_ts.searchsorted(ts.to_datetime64(), side="right")) - 1
        out.append(r_vals[idx] if idx >= 0 else pd.NA)
    return pd.Series(out, index=candle_ts.index, dtype=research_values.dtype)


def _assert_no_lookahead(
    candle_ts: pd.Series,
    research_ts: pd.Series,
    research_values: pd.Series,
    merged_values: pd.Series,
) -> None:
    """Verify merged values only use research rows at or before each candle."""
    if research_ts.empty:
        return

    order = research_ts.argsort()
    r_ts = research_ts.iloc[order].reset_index(drop=True)
    r_vals = research_values.iloc[order].reset_index(drop=True)
    r_ts_np = r_ts.to_numpy()

    for ts, value in zip(candle_ts, merged_values):
        if pd.isna(value):
            continue
        idx = int(r_ts_np.searchsorted(ts.to_datetime64(), side="right")) - 1
        if idx < 0 or r_ts_np[idx] > ts.to_datetime64():
            raise RuntimeError(
                f"lookahead leak: candle {ts} must not use research after its timestamp"
            )
        if r_vals.iloc[idx] != value:
            raise RuntimeError(f"merge mismatch at candle {ts}")


def load_research_csv(
    path: Path | str,
    *,
    asset: str | None = None,
) -> pd.DataFrame:
    """Load research CSV with ``timestamp`` and optional ``asset`` column."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Research file not found: {path}")

    raw = pd.read_csv(path)
    raw.columns = [str(c).strip() for c in raw.columns]
    lower = {c.lower(): c for c in raw.columns}

    if "timestamp" not in lower:
        raise ValueError(
            f"Research CSV must have a timestamp column; got {list(raw.columns)}"
        )

    df = raw.rename(columns={lower["timestamp"]: "timestamp"}).copy()
    df["timestamp"] = _parse_timestamps(df["timestamp"])

    if "asset" in lower:
        df = df.rename(columns={lower["asset"]: ASSET_COLUMN})
        if asset is not None:
            mask = _asset_keys(df[ASSET_COLUMN]) == _normalize_asset(asset)
            df = df.loc[mask]
            if df.empty:
                raise ValueError(f"No research rows for asset {asset!r}")
            df = df.drop(columns=[ASSET_COLUMN])

    value_cols = [c for c in df.columns if c != "timestamp"]
    if not value_cols:
        raise ValueError("Research CSV has no value columns to merge")

    subset = ["timestamp"]
    if ASSET_COLUMN in df.columns:
        subset.append(ASSET_COLUMN)
    df = df.sort_values(subset).drop_duplicates(subset=subset, keep="last")
    return df.reset_index(drop=True)


def _merge_one_group(
    candles: pd.DataFrame,
    research: pd.DataFrame,
    *,
    value_column: str,
) -> pd.Series:
    patch = research[["timestamp", value_column]].copy()
    patch["_ts"] = _parse_timestamps(patch["timestamp"])
    patch = patch.drop(columns=["timestamp"]).drop_duplicates(subset=["_ts"], keep="last")

    values = _backward_asof_values(
        candles["_ts"],
        patch["_ts"],
        patch[value_column],
    )
    _assert_no_lookahead(
        candles["_ts"],
        patch["_ts"],
        patch[value_column],
        values,
    )
    return values


def _merge_columns_into_group(
    candles: pd.DataFrame,
    research: pd.DataFrame,
    value_columns: list[str],
) -> pd.DataFrame:
    """Backward as-of merge external columns onto one candle group."""
    merged = candles.copy()
    for col in value_columns:
        merged[col] = _merge_one_group(merged, research, value_column=col)
    return merged


def join_external_into_candles(
    candles: pd.DataFrame,
    external: pd.DataFrame,
    *,
    asset: str | None = None,
    fill_before: str | None = None,
) -> pd.DataFrame:
    """Backward as-of join all external columns onto candle rows."""
    if ASOF_DIRECTION != "backward":
        raise RuntimeError("join_csv only supports backward as-of joins")

    value_columns = [
        c for c in external.columns if c not in ("timestamp", ASSET_COLUMN)
    ]
    if not value_columns:
        raise ValueError("Join CSV has no columns to merge besides timestamp/asset")

    overlap = [c for c in value_columns if c in candles.columns]
    if overlap:
        raise ValueError(
            f"Candle CSV already has join columns {overlap}; "
            "remove them from the input or join CSV"
        )

    research_has_asset = ASSET_COLUMN in external.columns
    candles_has_asset = ASSET_COLUMN in candles.columns

    if research_has_asset and not candles_has_asset and asset is None:
        assets = sorted(_asset_keys(external[ASSET_COLUMN]).unique())
        raise ValueError(
            f"Join CSV has an asset column; pass --asset "
            f"(available: {', '.join(assets)})"
        )

    work = candles.copy()
    work["_ts"] = _parse_timestamps(work["timestamp"])
    merged = work.sort_values("_ts").copy()

    if research_has_asset and candles_has_asset:
        merged["_asset_key"] = _asset_keys(merged[ASSET_COLUMN])
        external = external.copy()
        external["_asset_key"] = _asset_keys(external[ASSET_COLUMN])
        for asset_key in merged["_asset_key"].unique():
            c_mask = merged["_asset_key"] == asset_key
            r_mask = external["_asset_key"] == asset_key
            group = _merge_columns_into_group(
                merged.loc[c_mask],
                external.loc[r_mask],
                value_columns,
            )
            for col in value_columns:
                merged.loc[c_mask, col] = group[col].values
        merged = merged.drop(columns=["_asset_key"])
    else:
        group = _merge_columns_into_group(merged, external, value_columns)
        for col in value_columns:
            merged[col] = group[col].values

    if fill_before is not None:
        for col in value_columns:
            merged[col] = merged[col].fillna(fill_before)

    merged["timestamp"] = merged["_ts"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    merged = merged.drop(columns=["_ts"]).reset_index(drop=True)
    return merged


def join_csv_into_candles(
    candles: pd.DataFrame,
    join_path: Path | str,
    *,
    asset: str | None = None,
    fill_before: str | None = None,
) -> pd.DataFrame:
    """Load an external CSV and backward as-of join its columns onto candles."""
    external = load_research_csv(join_path, asset=asset)
    return join_external_into_candles(
        candles,
        external,
        asset=asset,
        fill_before=fill_before,
    )


def merge_research_into_candles(
    candles: pd.DataFrame,
    research: pd.DataFrame,
    *,
    value_column: str,
    output_column: str | None = None,
    fill_before: str | None = None,
    asset: str | None = None,
) -> pd.DataFrame:
    """Backward as-of merge one research column onto candle rows."""
    if ASOF_DIRECTION != "backward":
        raise RuntimeError("merge_research only supports backward as-of joins")

    out_col = output_column or value_column
    if value_column not in research.columns:
        raise ValueError(
            f"Research column {value_column!r} not found; "
            f"available: {[c for c in research.columns if c not in ('timestamp', ASSET_COLUMN)]}"
        )
    if out_col in candles.columns:
        raise ValueError(
            f"Candle CSV already has column {out_col!r}; "
            "choose a different --output-column or remove it from the input"
        )

    research_has_asset = ASSET_COLUMN in research.columns
    candles_has_asset = ASSET_COLUMN in candles.columns

    if research_has_asset and not candles_has_asset and asset is None:
        assets = sorted(_asset_keys(research[ASSET_COLUMN]).unique())
        raise ValueError(
            f"Research CSV has an asset column; add asset to candles or pass --asset "
            f"(available: {', '.join(assets)})"
        )

    work = candles.copy()
    work["_ts"] = _parse_timestamps(work["timestamp"])
    merged = work.sort_values("_ts").copy()
    merged[value_column] = pd.NA

    if research_has_asset and candles_has_asset:
        merged["_asset_key"] = _asset_keys(merged[ASSET_COLUMN])
        research = research.copy()
        research["_asset_key"] = _asset_keys(research[ASSET_COLUMN])
        for asset_key in merged["_asset_key"].unique():
            c_mask = merged["_asset_key"] == asset_key
            r_mask = research["_asset_key"] == asset_key
            merged.loc[c_mask, value_column] = _merge_one_group(
                merged.loc[c_mask],
                research.loc[r_mask],
                value_column=value_column,
            )
        merged = merged.drop(columns=["_asset_key"])
    else:
        merged[value_column] = _merge_one_group(
            merged,
            research,
            value_column=value_column,
        )

    if fill_before is not None:
        merged[value_column] = merged[value_column].fillna(fill_before)
    merged = merged.rename(columns={value_column: out_col})
    merged["timestamp"] = merged["_ts"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    merged = merged.drop(columns=["_ts"]).reset_index(drop=True)
    return merged


def merge_research_csv(
    candles_path: Path | str,
    research_path: Path | str,
    output_path: Path | str,
    *,
    asset: str | None = None,
    value_column: str = "regime",
    output_column: str | None = None,
    fill_before: str | None = None,
) -> Path:
    """Read candles + research CSVs and write merged candles locally."""
    candles_path = Path(candles_path)
    if not candles_path.exists():
        raise FileNotFoundError(f"Candle file not found: {candles_path}")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    candles = pd.read_csv(candles_path)
    missing = [c for c in CANONICAL_COLUMNS if c not in candles.columns]
    if missing:
        raise ValueError(
            f"Candle CSV missing required columns {missing}; got {list(candles.columns)}"
        )

    research = load_research_csv(research_path, asset=asset)
    merged = merge_research_into_candles(
        candles,
        research,
        value_column=value_column,
        output_column=output_column,
        fill_before=fill_before,
        asset=asset,
    )
    merged.to_csv(out_path, index=False)
    return out_path
