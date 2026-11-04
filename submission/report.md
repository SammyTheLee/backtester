# Backtest Report

This report summarizes one strategy test. It is written for readers who do not need to know how the code works.

## What data was used?

- **Price file:** `data/candles/DEMO.csv`
- **Bars in test:** 97
- **Date range:** 2026-01-01 00:00:00 to 2026-01-05 00:00:00

## What strategy was tested?

This test used **Moving-average crossover**.
It compares a **5-bar fast average** to a **20-bar slow average** and trades when they cross.
Each trade uses **2% stop-loss** and **4% take-profit**.
Starting account size: **$10,000.00**.

## What costs were assumed?

Every simulated trade included realistic frictions:
- **Trading fee:** 10 basis points (0.10%) on entry and exit
- **Slippage:** 5 basis points (0.05%) on fills
- **Holding cost:** none

These costs reduce the net profit compared with a perfect-price backtest.

## What dates were train, validation, and out-of-sample?

This run tested only the **training** portion of the configured date splits.
- **Period used:** Training

## What was the best out-of-sample config?

This was a **single run**, not a parameter search. The configuration tested here is the one listed under “What strategy was tested?” above.

There is **no separate “best out-of-sample” pick** from a sweep for this run.

## How many trades happened?

Summary from `summary.csv`:

- **Closed trades:** 4
- **Net profit (after costs):** $-0.66
- **Return on starting equity:** -0.01%
- **Win rate:** 25.0%
- **Profit factor:** 0.85
- **Largest peak-to-trough drop:** $3.14
- **Average time in each trade:** 12.8 hours
- **Share of bars with an open position:** 55.7%

Details from `trades.csv`:

- **Closed trades in file:** 4
- **Winning trades:** 1
- **Losing or flat trades:** 3
- **Best single trade (net):** $3.67
- **Worst single trade (net):** $-2.24
- **How trades ended:**
  - signal_change: 2
  - stop: 1
  - target: 1

## Model assumptions

The simulation followed these rules (see `metadata.json` for the full record):

**When trades happen**
- Signal is known at the close of bar N.
- Entries and opposing-signal exits fill at the open of bar N+1.
- Stops and targets are checked with each bar high/low from the entry bar.
- If stop and target both trade inside one bar, stop is assumed first.

**How signals work**
- Long: fast_ma crosses above slow_ma
- Short: fast_ma crosses below slow_ma
- Signals: Sparse events only at crossover bars (not every bar in a trend).
- Type: ma_crossover

**How costs are calculated**
- Fee: fee = notional * fee_bps / 10000 (charged on entry and exit)
- Slippage rate: slippage_bps / 10000
- Long entry: open * (1 + slippage_rate)
- Short entry: open * (1 - slippage_rate)
- Long exit: raw_exit * (1 - slippage_rate)
- Short exit: raw_exit * (1 + slippage_rate)
- Holding: optional per-bar: close * size * holding_cost_bps / 10000

**How profit is measured**
- Gross pnl: Price PnL at raw entry_open and exit_raw
- Net pnl: gross_pnl - slippage_cost - entry_fee - exit_fee - holding_cost
- Equity curve: Entry fee at open; gross - slippage - exit fee - holding at close

## What is the biggest weakness of the result?

**Biggest concerns:**

- Only **4 trades** were taken — too few to draw reliable conclusions.
- The strategy **lost money** after costs ($-0.66 net).
- The **win rate was low** (25% of trades were profitable).
- **Drawdowns were large** ($3.14) relative to net profit.
- The dataset is **very short** (97 price bars), so results may not repeat in live trading.
- The data file looks like **synthetic demo data**, not real market history.
