# Backtester

A Python command-line backtester for local OHLCV candle CSVs. Configure strategies and execution costs in TOML, run individual backtests or parameter sweeps, and export trades, equity curves, metrics, and Markdown reports.

## Features

- Moving-average crossover, RSI, volatility-gated MA, and regime-gated MA strategies.
- Next-bar execution with optional stops, profit targets, fees, slippage, and holding costs.
- Parameter grids with ranked results and invalid-combination filtering.
- Train, validation, and out-of-sample (OOS) evaluation.
- A `walkforward` command that runs a train sweep, evaluates top training configurations on validation data, and evaluates top validation configurations on OOS data using the configured date windows.
- Multi-asset portfolios with shared capital, position limits, and exposure caps.
- Local CSV normalization and backward as-of joins for timestamped research data.
- Deterministic offline demo data and tests that block network connections.

## Setup

Run commands from the repository root. The project runs directly from the checkout; it does not currently include a package manifest or dependency lockfile.

```bash
git clone https://github.com/SammyTheLee/backtester.git
cd backtester
python3 -m venv .venv
source .venv/bin/activate
python -m pip install pandas numpy pytest tomli
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

The core engine uses pandas. NumPy is used by the tests, pytest runs the test suite, and `tomli` provides TOML parsing on Python versions without `tomllib` (before Python 3.11).

The offline tests and demo were verified with Python 3.9.7, pandas 2.3.3, NumPy 2.0.2, pytest 8.4.2, and tomli 2.5.0. These are verification versions, not a locked dependency set.

## Quick start (offline)

Generate 300 deterministic hourly candles, inspect them, and run the example configuration:

```bash
python make_demo_data.py
python -m bt inspect-data data/candles/DEMO.csv
python -m bt run --config configs/example.toml
python -m bt inspect-run latest
python -m pytest -q
```

`make_demo_data.py` overwrites `data/candles/DEMO.csv` with synthetic data seeded with `42`. The example configuration uses only its configured training window, rather than the entire CSV.

For the bundled end-to-end demo (tests, data generation, inspection, a run, a sweep, and train/validation/OOS evaluation):

```bash
bash demo.sh
```

## Configuration

Start with [configs/example.toml](configs/example.toml):

```toml
[data]
csv_path = "data/candles/DEMO.csv"
split = "train"

[split]
train_start = "2026-01-01"
train_end = "2026-01-05"
test_start = "2026-01-06"
test_end = "2026-01-09"
oos_start = "2026-01-10"
oos_end = "2026-01-13"

[strategy]
type = "ma_crossover"
fast = 5
slow = 20

[execution]
stop_pct = 0.02
target_pct = 0.04
fee_bps = 10
slippage_bps = 5

[portfolio]
initial_equity = 10000.0
size = 1.0
```

Paths are resolved from the current working directory. `data.split` accepts `train`, `test` (validation), `oos`, or `all`. Validation date keys may also be written as `validation_start` and `validation_end`.

Stop and target distances are fractions (`0.02` means 2%); costs are in basis points (`10` means 0.1%). `size` is the number of units per trade.

Additional strategy examples are in [configs/rsi.toml](configs/rsi.toml), [configs/vol_gated.toml](configs/vol_gated.toml), and [configs/regime_gated.toml](configs/regime_gated.toml). Regime-gated strategies require the configured research column in the candle data.

## Sweeps and held-out evaluation

Generate a parameter grid without running backtests:

```bash
python -m bt combinations --config configs/param_sweep.toml
```

Run the complete train/validation/OOS workflow:

```bash
python -m bt walkforward --config configs/walkforward.toml
```

Or run each stage separately:

```bash
python -m bt sweep --config configs/param_sweep.toml
python -m bt validate --config configs/param_sweep.toml
python -m bt oos --config configs/param_sweep.toml
```

The separate validation and OOS commands select prior result files by default; use `--from` to select a specific leaderboard. Add `--full-exports` to `sweep` or `walkforward` to retain per-combination artifacts. The bundled walk-forward example uses one configured train/validation/OOS split; it does not automatically roll through multiple windows.

## Multi-asset portfolios

```bash
python -m bt portfolio --config configs/portfolio.toml
```

The example uses the bundled BTC and DEMO CSVs. Configure `data.assets` and the `[portfolio]` settings to change assets, shared initial equity, `max_open_positions`, `max_total_exposure_pct`, and `max_exposure_per_asset_pct`. Exposure limits are fractions of equity.

## Input data

Candle CSVs use this schema:

```csv
timestamp,open,high,low,close,volume
2026-01-01T00:00:00,100.00,101.00,99.00,100.50,1000
```

Normalize a downloaded CSV locally (common date and OHLCV column aliases are supported):

```bash
python -m bt import-data path/to/download.csv -o data/candles/custom.csv
```

Use `python -m bt merge-research --help` for backward as-of joins of timestamped research columns onto candles. Update the configuration's CSV path and split dates to match your data.

The backtest engine reads local files and does not download data. The optional `fetch_data.py` helper requires `yfinance` and internet access; data availability depends on Yahoo Finance:

```bash
python -m pip install yfinance
python fetch_data.py AAPL --period 1y --interval 1d
```

## Results and reports

Single runs write to UTC timestamped folders under `results/`, containing:

- `trades.csv`: trade entries, exits, and costs.
- `equity.csv`: trade-based equity points.
- `summary.csv` and `report.json`: run metrics.
- `metadata.json`: configuration, data details, and artifact paths.
- `report.md`: a readable Markdown report.

Cross-run history and rankings are stored in `results/runs.csv` and `results/summary.csv`. Sweeps and held-out workflows create their own result folders and leaderboards; portfolio runs also export portfolio equity and rejected trades. Generated `results/` files are ignored by Git.

Inspect or regenerate a report for a specific run (replace the path with the folder printed by your run):

```bash
python -m bt inspect-run results/YYYYMMDD_HHMMSS
python -m bt write-report results/YYYYMMDD_HHMMSS
```

For an optional equity plot, install matplotlib and run the helper after a single backtest:

```bash
python -m pip install matplotlib
python plot_equity.py
```

This writes `equity_curve.png` from the latest single-run `equity.csv`.

## Execution assumptions

Signals are detected at a candle's close and entries or opposing-signal exits fill at the next candle's open. Stops and targets are checked against candle highs and lows starting on the entry bar; if both levels are touched in the same bar, the stop is assumed to occur first. Slippage worsens entry and exit fills.

Single-asset equity realizes closed-trade profit and loss and deducts entry fees for open trades; it does not mark unrealized open-position profit and loss to market. Portfolio mode additionally exports a bar-level equity curve. These candle-based assumptions and the synthetic demo data should be considered when interpreting results.

## Project layout

- `bt/`: strategies, execution, accounting, evaluation, exports, and CLI.
- `configs/`: example TOML configurations.
- `data/`: local candle and research samples.
- `tests/`: offline pytest suite.
- `submission/`: saved submission artifacts and their existing [guide](submission/README.md).
- `make_demo_data.py`, `demo.sh`: reproducible offline demo.
- `fetch_data.py`, `plot_equity.py`: optional download and plotting helpers.

Use `python -m bt --help` or `python -m bt <command> --help` for the full CLI reference.
