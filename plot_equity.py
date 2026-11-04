import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from bt.paths import EQUITY_CSV, latest_run_dir

DEFAULT_RESULTS = Path("results")


def resolve_equity_csv(results_dir: Path) -> Path:
    run_dir = latest_run_dir(results_dir)
    if run_dir is None:
        raise FileNotFoundError(
            f"No timestamped run folders under {results_dir}. "
            "Run: python -m bt run --config configs/example.toml"
        )
    path = run_dir / EQUITY_CSV
    if not path.exists():
        raise FileNotFoundError(f"Missing equity file: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot equity curve from latest run.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS,
        help="Base results directory (default: results/)",
    )
    parser.add_argument(
        "--equity-csv",
        type=Path,
        default=None,
        help="Explicit equity.csv path (overrides latest run lookup)",
    )
    args = parser.parse_args()

    equity_path = args.equity_csv or resolve_equity_csv(args.results_dir)
    df = pd.read_csv(equity_path)
    print(f"Loaded {len(df)} rows from {equity_path}")

    plt.figure(figsize=(10, 5))
    plt.plot(df["equity"], marker="o")
    plt.title("Equity Curve")
    plt.xlabel("Trade #")
    plt.ylabel("Equity ($)")
    plt.grid(True)
    initial = float(df["equity"].iloc[0]) if len(df) else 10_000.0
    plt.axhline(y=initial, color="gray", linestyle="--", label=f"Initial ${initial:,.0f}")
    plt.legend()

    out = Path("equity_curve.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved chart to {out}")


if __name__ == "__main__":
    main()
