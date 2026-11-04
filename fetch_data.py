"""Download real market candles from Yahoo Finance into data/candles/.

MANUAL TOOL ONLY — not used by the backtest engine or the test suite.
Run this yourself when you want a fresh local CSV:

    python fetch_data.py AAPL --period 1y --interval 1d

The backtester always reads from ``data/candles/*.csv`` on disk.
Tests use fake deterministic data and never call this script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

DEFAULT_OUTPUT_DIR = Path("data/candles")


def fetch(
    ticker: str,
    *,
    period: str = "60d",
    interval: str = "1h",
    output_path: Path | None = None,
) -> Path:
    """Download OHLCV data and save as CSV matching the backtester format."""
    import yfinance as yf

    if output_path is None:
        safe_name = ticker.replace("-", "_").upper()
        output_path = DEFAULT_OUTPUT_DIR / f"{safe_name}.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {ticker}  period={period}  interval={interval} ...")
    raw = yf.download(
        ticker,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=False,
    )

    if raw.empty:
        raise ValueError(
            f"No data returned for {ticker!r}. "
            "Check the ticker symbol, period, and interval."
        )

    # yfinance may return MultiIndex columns for a single ticker; flatten.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    from bt.import_candles import format_candles_df

    df = format_candles_df(raw.reset_index())
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} rows → {output_path}")
    print(f"Time range: {df['timestamp'].iloc[0]}  →  {df['timestamp'].iloc[-1]}")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download Yahoo Finance candles for the backtester.",
    )
    parser.add_argument(
        "ticker",
        nargs="?",
        default="BTC-USD",
        help="Yahoo ticker (default: BTC-USD). Examples: AAPL, BTC-USD",
    )
    parser.add_argument(
        "--period",
        default="60d",
        help="How far back (yfinance period), e.g. 7d, 60d, 1y (default: 60d)",
    )
    parser.add_argument(
        "--interval",
        default="1h",
        help="Candle size: 1h, 1d, etc. (default: 1h)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: data/candles/TICKER.csv)",
    )
    args = parser.parse_args(argv)

    try:
        fetch(
            args.ticker.upper() if args.ticker.isalpha() and len(args.ticker) <= 5 else args.ticker,
            period=args.period,
            interval=args.interval,
            output_path=args.output,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
