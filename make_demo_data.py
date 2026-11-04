"""Generate a deterministic 300-row hourly candle CSV at data/candles/DEMO.csv."""

import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


NUM_ROWS = 300
START_TIME = datetime(2026, 1, 1, 0, 0, 0)
SEED = 42
OUTPUT_PATH = Path("data/candles/DEMO.csv")


def generate_candles(num_rows: int) -> pd.DataFrame:
    rng = random.Random(SEED)

    rows = []
    price = 100.0

    for i in range(num_rows):
        timestamp = START_TIME + timedelta(hours=i)

        open_price = price
        # Random walk: close moves up to ±1% from open.
        close_price = open_price * (1 + rng.uniform(-0.01, 0.01))

        # High/low extend a bit beyond open/close.
        high_price = max(open_price, close_price) * (1 + rng.uniform(0, 0.005))
        low_price = min(open_price, close_price) * (1 - rng.uniform(0, 0.005))

        volume = rng.randint(1000, 5000)

        rows.append(
            {
                "timestamp": timestamp.isoformat(),
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "volume": volume,
            }
        )

        price = close_price

    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = generate_candles(NUM_ROWS)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote {len(df)} rows to {OUTPUT_PATH}")
    print("\nFirst 5 rows:")
    print(df.head(5))
    print("\nLast 5 rows:")
    print(df.tail(5))


if __name__ == "__main__":
    main()
