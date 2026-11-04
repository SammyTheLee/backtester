# Walk-Forward Backtest Report

This report explains a full **train → validation → out-of-sample** study in plain language.

**Config file:** `configs/walkforward.toml`  
**Ranking method:** `composite_score` (higher is better)

## What data was used?

- **Price file:** from the config (typically `data/candles/DEMO.csv` in demo runs)
- **Parameter combinations tried on training data:** 36
- **Configs forwarded to validation:** 10
- **Configs tested out-of-sample:** 10

## What strategy was tested?

This was a **moving-average crossover** strategy. The system tried many combinations of fast/slow averages and stop/target sizes on training data, kept the best few for validation, then tested survivors out-of-sample.

**Best out-of-sample settings (rank 1):** fast average 5, slow average 100, stop 2%, target 4%

## What costs were assumed?

Every simulated trade included realistic frictions:
- **Trading fee:** 10 basis points (0.10%) on entry and exit
- **Slippage:** 5 basis points (0.05%) on fills
- **Holding cost:** none

These costs reduce the net profit compared with a perfect-price backtest.

## What dates were train, validation, and out-of-sample?

- **Training:** 2026-01-01 to 2026-01-05
- **Validation:** 2026-01-06 to 2026-01-09
- **Out-of-sample:** 2026-01-10 to 2026-01-13

Training is where parameters were searched. Validation checks whether train winners still work on fresh dates. Out-of-sample is the final unseen period.

## What was the best out-of-sample config?

**Rank 1 by composite_score:**

| Setting | Value |
|---------|-------|
| Fast average | 5.0 |
| Slow average | 100.0 |
| Stop | 2% |
| Target | 4% |

**Out-of-sample results for this config:**

- Net profit: $0.00
- Trades: 0
- Win rate: 0.0%

## How many trades happened?

Across the three phases for the top config:

| Phase | Trades | Net profit |
|-------|--------|------------|
| Training | 0 | $0.00 |
| Validation | 0 | $0.00 |
| Out-of-sample | 0 | $0.00 |

See `walkforward_summary.csv` in the same folder for all configs that were tested out-of-sample.

## What is the biggest weakness of the result?

**Biggest concerns:**

- The top-ranked out-of-sample config **did not trade at all**, so it cannot be trusted as a winner.
- Each walk-forward window is **short**; rankings can change a lot with more data.
- A lower-ranked config actually traded out-of-sample — see the full summary table.
