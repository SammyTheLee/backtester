"""CLI entry point: `python -m bt <command> [args...]`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from bt.backtest import apply_execution
from bt.costs import ExecutionCosts, execution_costs_from_config
from bt.strategy import (
    FAST_MA_WINDOW,
    SLOW_MA_WINDOW,
    MaCrossoverStrategy,
    Position,
    Strategy,
    add_fast_ma,
    add_slow_ma,
)
from bt.portfolio import (
    DEFAULT_INITIAL_EQUITY,
    equity_curve,
    final_equity,
    total_return_pct,
)
from bt.pipeline import execute_backtest, load_run_config, params_from_config
from bt.report import build_report_card, format_report_card
from bt.sweep import extract_sweep_axes
from bt.sweep_run import generate_combinations_only, run_parameter_sweep
from bt.oos_run import run_oos
from bt.validate_run import run_validation
from bt.inspect_run import inspect_run_folder, resolve_results_dir
from bt.import_candles import DEFAULT_REAL_SAMPLE_PATH
from bt.walkforward_run import run_walkforward
from bt.splits import split_config_for_oos, split_config_for_validation
from bt.trade import Trade, extract_trades


def _apply_execution(
    df: pd.DataFrame,
    strategy: Strategy,
    costs: ExecutionCosts,
) -> pd.DataFrame:
    return apply_execution(df, strategy, slippage_bps=costs.slippage_bps)


def _run_ma_backtest(
    csv_path: Path,
    fast: int,
    slow: int,
    stop_pct: float | None,
    target_pct: float | None,
    costs: ExecutionCosts,
    size: float = 1.0,
) -> tuple[pd.DataFrame, list[Trade], Strategy]:
    df = pd.read_csv(csv_path)
    strat = MaCrossoverStrategy(
        fast=fast,
        slow=slow,
        stop_pct=stop_pct,
        target_pct=target_pct,
    )
    df = strat.apply(df)
    df = _apply_execution(df, strat, costs)
    trades = extract_trades(df, size=size, costs=costs)
    return df, trades, strat


def _print_costs(costs: ExecutionCosts) -> None:
    if costs.fee_bps:
        print(f"Fee bps        : {costs.fee_bps:g}")
    if costs.slippage_bps:
        print(f"Slippage bps   : {costs.slippage_bps:g}")
    if costs.holding_cost_bps:
        print(f"Holding bps/bar: {costs.holding_cost_bps:g}")


def cmd_import_data(
    input_path: Path,
    output_path: Path,
    limit: int | None,
    join_csv: Path | None,
    join_asset: str | None,
) -> int:
    from bt.import_candles import import_candles_csv

    try:
        out = import_candles_csv(
            input_path,
            output_path,
            limit=limit,
            join_csv=join_csv,
            join_asset=join_asset,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    df = pd.read_csv(out)
    msg = f"Imported {len(df)} rows → {out}"
    if join_csv is not None:
        joined = [c for c in df.columns if c not in ("timestamp", "open", "high", "low", "close", "volume")]
        if joined:
            msg += f" (joined: {', '.join(joined)})"
    print(msg)
    if len(df):
        print(f"Time range: {df['timestamp'].iloc[0]}  →  {df['timestamp'].iloc[-1]}")
    return 0


def cmd_merge_research(
    candles_path: Path,
    research_path: Path,
    output_path: Path | None,
    asset: str | None,
    value_column: str,
    output_column: str | None,
    fill_before: str | None,
) -> int:
    from bt.merge_research import merge_research_csv

    if output_path is None:
        output_path = candles_path.with_name(f"{candles_path.stem}_research{candles_path.suffix}")

    try:
        out = merge_research_csv(
            candles_path,
            research_path,
            output_path,
            asset=asset,
            value_column=value_column,
            output_column=output_column,
            fill_before=fill_before,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    df = pd.read_csv(out)
    out_col = output_column or value_column
    filled = int(df[out_col].notna().sum()) if out_col in df.columns else 0
    print(f"Merged {filled}/{len(df)} rows with {out_col!r} → {out}")
    if len(df):
        print(f"Time range: {df['timestamp'].iloc[0]}  →  {df['timestamp'].iloc[-1]}")
    return 0


def cmd_inspect_data(csv_path: Path, fast_ma: int, slow_ma: int) -> int:
    if not csv_path.exists():
        print(f"Error: file not found: {csv_path}", file=sys.stderr)
        return 1

    if fast_ma < 1 or slow_ma < 1:
        print("Error: --fast-ma and --slow-ma must be >= 1", file=sys.stderr)
        return 1

    if fast_ma >= slow_ma:
        print(
            f"Warning: fast_ma ({fast_ma}) should be smaller than slow_ma ({slow_ma}).",
            file=sys.stderr,
        )

    df = pd.read_csv(csv_path)
    df = add_fast_ma(df, window=fast_ma)
    df = add_slow_ma(df, window=slow_ma)

    print(f"File: {csv_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Fast MA window: {fast_ma}")
    print(f"Slow MA window: {slow_ma}")

    if "timestamp" in df.columns:
        print(f"Time range: {df['timestamp'].iloc[0]}  →  {df['timestamp'].iloc[-1]}")

    print("\nFirst 5 rows:")
    print(df.head(5))

    print("\nLast 5 rows:")
    print(df.tail(5))

    print("\nSummary stats:")
    print(df.describe())

    return 0


def cmd_indicators(csv_path: Path, fast: int, slow: int) -> int:
    if not csv_path.exists():
        print(f"Error: file not found: {csv_path}", file=sys.stderr)
        return 1

    if fast < 1 or slow < 1:
        print("Error: --fast and --slow must be >= 1", file=sys.stderr)
        return 1

    if fast >= slow:
        print(
            f"Warning: fast ({fast}) should be smaller than slow ({slow}).",
            file=sys.stderr,
        )

    df = pd.read_csv(csv_path)
    df = add_fast_ma(df, window=fast)
    df = add_slow_ma(df, window=slow)

    cols = [c for c in ("timestamp", "close", "fast_ma", "slow_ma") if c in df.columns]
    view = df[cols]

    print(f"File: {csv_path}")
    print(f"Fast MA window: {fast}")
    print(f"Slow MA window: {slow}")
    print(f"Rows: {len(view)}  (first non-NaN row: {view['slow_ma'].first_valid_index()})")

    print("\nFirst 5 rows:")
    print(view.head(5).to_string(index=True))

    print("\nLast 5 rows:")
    print(view.tail(5).to_string(index=True))

    return 0


def cmd_signals(csv_path: Path, fast: int, slow: int) -> int:
    if not csv_path.exists():
        print(f"Error: file not found: {csv_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(csv_path)

    try:
        strat = MaCrossoverStrategy(fast=fast, slow=slow)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    df = strat.apply(df)

    long_count = (df["signal"] == Position.LONG).sum()
    short_count = (df["signal"] == Position.SHORT).sum()
    none_count = df["signal"].isna().sum()

    print(f"File: {csv_path}")
    print(f"Strategy: {strat.name}")
    print(f"Rows: {len(df)}")
    print(f"  LONG signals : {long_count}")
    print(f"  SHORT signals: {short_count}")
    print(f"  no signal    : {none_count}")

    events = df[df["signal"].notna()][
        [c for c in ("timestamp", "close", "fast_ma", "slow_ma", "signal") if c in df.columns]
    ]
    if events.empty:
        print("\nNo signal events found. Try different --fast/--slow values.")
    else:
        print(f"\nSignal events ({len(events)} total):")
        print(events.to_string(index=True))

    return 0


def cmd_positions(
    csv_path: Path,
    fast: int,
    slow: int,
    stop_pct: float | None,
    target_pct: float | None,
    costs: ExecutionCosts,
) -> int:
    if not csv_path.exists():
        print(f"Error: file not found: {csv_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(csv_path)

    try:
        strat = MaCrossoverStrategy(
            fast=fast,
            slow=slow,
            stop_pct=stop_pct,
            target_pct=target_pct,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    df = strat.apply(df)

    try:
        df = _apply_execution(df, strat, costs)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    entries = df[df["entry_price"].notna()].copy()
    bars_in_market = int(df["in_position"].sum())

    print(f"File: {csv_path}")
    print(f"Strategy: {strat.name}")
    if stop_pct is not None:
        print(f"Stop pct  : {stop_pct:.4f}")
    if target_pct is not None:
        print(f"Target pct: {target_pct:.4f}")
    _print_costs(costs)
    print(f"Total positions opened: {len(entries)}")
    print(
        f"Bars in market        : {bars_in_market}/{len(df)} "
        f"({bars_in_market / len(df) * 100:.1f}%)"
    )

    if entries.empty:
        print("\nNo positions opened. Try different parameters.")
        return 0

    cols = [
        c
        for c in ("timestamp", "position", "entry_price", "stop_price", "target_price")
        if c in entries.columns
    ]
    print(f"\nEntries ({len(entries)} total):")
    print(entries[cols].to_string(index=True))

    return 0


def cmd_exits(
    csv_path: Path,
    fast: int,
    slow: int,
    stop_pct: float | None,
    target_pct: float | None,
    costs: ExecutionCosts,
    save_path: Path | None = None,
) -> int:
    if not csv_path.exists():
        print(f"Error: file not found: {csv_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(csv_path)

    try:
        strat = MaCrossoverStrategy(
            fast=fast,
            slow=slow,
            stop_pct=stop_pct,
            target_pct=target_pct,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    df = strat.apply(df)

    try:
        df = _apply_execution(df, strat, costs)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    trades = extract_trades(df, costs=costs)
    closed = [t for t in trades if t.is_closed]
    open_count = sum(1 for t in trades if not t.is_closed)

    by_reason: dict[str, int] = {"stop": 0, "target": 0, "signal_change": 0}
    for t in closed:
        if t.exit_reason in by_reason:
            by_reason[t.exit_reason] += 1

    print(f"File: {csv_path}")
    print(f"Strategy: {strat.name}")
    if stop_pct is not None:
        print(f"Stop pct  : {stop_pct:.4f}")
    if target_pct is not None:
        print(f"Target pct: {target_pct:.4f}")
    _print_costs(costs)
    print(f"Total positions closed: {len(closed)}")
    print(f"  stop          : {by_reason['stop']}")
    print(f"  target        : {by_reason['target']}")
    print(f"  signal_change : {by_reason['signal_change']}")
    if open_count:
        print(f"Still open at end     : {open_count}")

    if not closed:
        print("\nNo closed positions yet. Try different parameters or a longer dataset.")
        return 0

    rows = []
    for t in closed:
        rows.append(
            {
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "side": str(t.side),
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "exit_reason": t.exit_reason,
                "gross_pnl": t.gross_pnl,
                "slippage_cost": t.slippage_cost,
                "fees": t.fees,
                "net_pnl": t.net_pnl,
            }
        )
    print(f"\nExits ({len(closed)} total):")
    print(pd.DataFrame(rows).to_string(index=False))

    if save_path is not None:
        n = save_trades(trades, save_path)
        print(f"\nSaved {n} trade(s) to {save_path}")

    return 0


def cmd_report(
    csv_path: Path,
    fast: int,
    slow: int,
    stop_pct: float | None,
    target_pct: float | None,
    initial_equity: float,
    costs: ExecutionCosts,
    size: float,
    save_path: Path | None = None,
) -> int:
    if not csv_path.exists():
        print(f"Error: file not found: {csv_path}", file=sys.stderr)
        return 1

    try:
        df, trades, strat = _run_ma_backtest(
            csv_path, fast, slow, stop_pct, target_pct, costs, size=size
        )
        card = build_report_card(
            trades,
            initial_equity=initial_equity,
            bars_in_market=int(df["in_position"].sum()),
            total_bars=len(df),
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"File: {csv_path}")
    print(f"Strategy: {strat.name}")
    if stop_pct is not None:
        print(f"Stop pct  : {stop_pct:.4f}")
    if target_pct is not None:
        print(f"Target pct: {target_pct:.4f}")
    _print_costs(costs)
    print()
    print(format_report_card(card))

    if save_path is not None:
        path = Path(save_path)
        if path.suffix.lower() == ".csv":
            write_ranked_summary_csv(
                summary_path=path,
                leaderboard_data_split=None,
            )
        else:
            save_report_card(card, path)
        print(f"\nSaved report to {save_path}")

    return 0


def cmd_equity(
    csv_path: Path,
    fast: int,
    slow: int,
    stop_pct: float | None,
    target_pct: float | None,
    initial_equity: float,
    costs: ExecutionCosts,
    size: float = 1.0,
    save_path: Path | None = None,
) -> int:
    if not csv_path.exists():
        print(f"Error: file not found: {csv_path}", file=sys.stderr)
        return 1

    try:
        df, trades, strat = _run_ma_backtest(
            csv_path, fast, slow, stop_pct, target_pct, costs, size=size
        )
        curve = equity_curve(trades, initial_equity=initial_equity)
        card = build_report_card(
            trades,
            initial_equity=initial_equity,
            bars_in_market=int(df["in_position"].sum()),
            total_bars=len(df),
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"File: {csv_path}")
    print(f"Strategy: {strat.name}")
    if stop_pct is not None:
        print(f"Stop pct  : {stop_pct:.4f}")
    if target_pct is not None:
        print(f"Target pct: {target_pct:.4f}")
    _print_costs(costs)
    print()
    print(format_report_card(card))

    print(f"\nEquity curve ({len(curve)} points):")
    rows = [{"timestamp": p.timestamp, "equity": p.equity} for p in curve]
    print(pd.DataFrame(rows).to_string(index=False))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(save_path, index=False)
        print(f"\nSaved equity curve to {save_path}")

    return 0


def cmd_run(config_path: Path) -> int:
    """Run a full backtest pipeline from a TOML config file."""
    if not config_path.exists():
        print(f"Error: config not found: {config_path}", file=sys.stderr)
        return 1

    try:
        loaded = load_run_config(config_path)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if extract_sweep_axes(loaded.config):
        print(
            "Error: config contains list parameters for sweeping.\n"
            "Use: python -m bt sweep --config "
            f"{config_path}",
            file=sys.stderr,
        )
        return 1

    params = params_from_config(loaded.config)
    try:
        result = execute_backtest(loaded, params)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Config         : {config_path}")
    if loaded.is_portfolio:
        print("Mode           : portfolio (shared capital)")
        for spec in loaded.assets:
            print(f"  Asset        : {spec.symbol} -> {spec.csv_path}")
        lim = loaded.portfolio_limits
        print(
            f"Limits         : max_positions={lim.max_open_positions} "
            f"total_exp={lim.max_total_exposure_pct:.0%} "
            f"per_asset_exp={lim.max_exposure_per_asset_pct:.0%}"
        )
    else:
        print(f"Data           : {loaded.csv_path}")
    if loaded.split is not None:
        print(f"Data split       : {loaded.data_split}")
        if loaded.split.train.start or loaded.split.train.end:
            print(
                f"Train window     : {loaded.split.train.start} .. {loaded.split.train.end}"
            )
        if loaded.split.test.start or loaded.split.test.end:
            print(
                f"Test window      : {loaded.split.test.start} .. {loaded.split.test.end}"
            )
    print(f"Strategy       : {result.strat.name}")
    if params.stop_pct is not None:
        print(f"Stop pct       : {params.stop_pct:.4f}")
    if params.target_pct is not None:
        print(f"Target pct     : {params.target_pct:.4f}")
    _print_costs(loaded.costs)
    print(f"Position size  : {loaded.size}")
    print()
    print(format_report_card(result.card))

    paths = result.output_paths
    if paths is not None:
        print(f"\nRun output folder: {paths.run_dir}")
        print(f"Saved {len(result.trades)} trade(s) to {paths.trades}")
        print(f"Saved equity curve to {paths.equity}")
        if loaded.is_portfolio and paths.portfolio_equity.exists():
            print(f"Saved portfolio equity to {paths.portfolio_equity}")
        if loaded.is_portfolio and paths.rejected_trades.exists():
            print(f"Rejected trades log: {paths.rejected_trades}")
        print(f"Saved metadata to {paths.metadata}")
    _print_runs_and_summary(loaded)

    return 0


def _csv_row_count(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    return max(0, len(pd.read_csv(path)))


def _print_runs_and_summary(loaded) -> None:
    runs_n = _csv_row_count(loaded.runs_log_path)
    summary_n = _csv_row_count(loaded.ranked_summary_path)
    print(f"Runs log      : {loaded.runs_log_path} ({runs_n} rows, all backtests)")
    print(
        f"Final leaderboard: {loaded.ranked_summary_path} "
        f"({summary_n} OOS configs, ranked by {loaded.summary_sort_by})"
    )


def cmd_combinations(config_path: Path) -> int:
    """Generate every parameter combination (CSV only, no backtests)."""
    if not config_path.exists():
        print(f"Error: config not found: {config_path}", file=sys.stderr)
        return 1

    try:
        loaded = load_run_config(config_path, for_sweep=True)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    from bt.sweep import expected_grid_size, extract_sweep_axes, iter_param_combinations

    combos = iter_param_combinations(loaded.config)
    if not combos:
        print(
            "Error: no valid combinations (check fast < slow and config values).",
            file=sys.stderr,
        )
        return 1
    grid = expected_grid_size(loaded.config)

    try:
        out_dir, df = generate_combinations_only(loaded)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    axes = extract_sweep_axes(loaded.config)
    grid_detail = (
        " x ".join(str(len(v)) for v in axes.values()) if axes else "single config"
    )
    print(f"Config       : {config_path}")
    print(f"Grid size    : {grid} ({grid_detail})")
    from bt.sweep import iter_skipped_combinations

    skipped_n = len(iter_skipped_combinations(loaded.config))
    print(f"Combinations : {len(combos)} valid (skipped {skipped_n} with fast_ma >= slow_ma)")
    print(f"Output folder: {out_dir}")
    print(f"CSV          : {out_dir / 'combinations.csv'}")
    print()
    print(df.to_string(index=False))

    return 0


def cmd_sweep(config_path: Path, *, export_each: bool = False, list_only: bool = False) -> int:
    """Run every combination of list-valued parameters in the config."""
    if not config_path.exists():
        print(f"Error: config not found: {config_path}", file=sys.stderr)
        return 1

    try:
        loaded = load_run_config(config_path, for_sweep=True)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not loaded.csv_path.exists():
        print(f"Error: data file not found: {loaded.csv_path}", file=sys.stderr)
        return 1

    from bt.sweep import expected_grid_size, iter_param_combinations

    combos = iter_param_combinations(loaded.config)
    if not combos:
        print(
            "Error: no valid combinations (check fast < slow and config values).",
            file=sys.stderr,
        )
        return 1
    grid = expected_grid_size(loaded.config)
    print(f"Config    : {config_path}")
    print(f"Data      : {loaded.csv_path}")
    print(
        f"Sweep data: train only "
        f"({loaded.split.train.start} .. {loaded.split.train.end})"
    )
    print(f"Grid      : {grid} -> {len(combos)} combinations")
    _print_costs(loaded.costs)
    print()

    if list_only:
        return cmd_combinations(config_path)

    try:
        sweep_dir, results = run_parameter_sweep(
            loaded, export_each=export_each
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"\nSweep folder : {sweep_dir}")
    print(f"Combinations : {sweep_dir / 'combinations.csv'}")
    print(f"Results CSV  : {sweep_dir / 'sweep_results.csv'} (this sweep only)")
    _print_runs_and_summary(loaded)
    print(f"\nTop 5 by {loaded.summary_sort_by}:")
    cols = [
        "rank",
        "fast",
        "slow",
        "stop_pct",
        "target_pct",
        "composite_score",
        "total_return_pct",
        "max_drawdown",
        "net_pnl",
    ]
    show = [c for c in cols if c in results.columns]
    print(results[show].head(5).to_string(index=False))

    return 0


def cmd_validate(
    config_path: Path,
    *,
    top_n: int | None = None,
    source: Path | None = None,
    sweep_results: Path | None = None,
) -> int:
    """Backtest top train configs on the validation (hold-out) window."""
    if not config_path.exists():
        print(f"Error: config not found: {config_path}", file=sys.stderr)
        return 1

    try:
        loaded = load_run_config(config_path)
        val_split = split_config_for_validation(loaded.config)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not loaded.csv_path.exists():
        print(f"Error: data file not found: {loaded.csv_path}", file=sys.stderr)
        return 1

    try:
        out_dir, results = run_validation(
            loaded,
            top_n=top_n,
            source=source,
            sweep_results=sweep_results,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Config           : {config_path}")
    print(f"Validation window: {val_split.test.start} .. {val_split.test.end}")
    print(f"Output folder    : {out_dir}")
    print(f"Results CSV      : {out_dir / 'validation_results.csv'}")
    _print_runs_and_summary(loaded)
    print(f"\nTop validation results (by val_{loaded.summary_sort_by}):")
    cols = [
        "val_rank",
        "train_rank",
        "fast",
        "slow",
        "train_composite_score",
        "val_composite_score",
        "val_net_pnl",
    ]
    show = [c for c in cols if c in results.columns]
    print(results[show].head(min(5, len(results))).to_string(index=False))

    return 0


def cmd_oos(
    config_path: Path,
    *,
    top_n: int | None = None,
    source: Path | None = None,
    validation_results: Path | None = None,
) -> int:
    """Backtest top validation configs on the out-of-sample window."""
    if not config_path.exists():
        print(f"Error: config not found: {config_path}", file=sys.stderr)
        return 1

    try:
        loaded = load_run_config(config_path)
        oos_split = split_config_for_oos(loaded.config)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not loaded.csv_path.exists():
        print(f"Error: data file not found: {loaded.csv_path}", file=sys.stderr)
        return 1

    try:
        out_dir, results, wf_path = run_oos(
            loaded,
            top_n=top_n,
            source=source,
            validation_results=validation_results,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Config       : {config_path}")
    print(f"OOS window   : {oos_split.oos.start} .. {oos_split.oos.end}")
    print(f"Output folder: {out_dir}")
    print(f"Results CSV  : {out_dir / 'oos_results.csv'}")
    print(f"Walk-forward : {wf_path}")
    _print_runs_and_summary(loaded)
    print(f"\nTop OOS results (by oos_{loaded.summary_sort_by}):")
    cols = [
        "oos_rank",
        "val_rank",
        "train_rank",
        "fast",
        "slow",
        "val_composite_score",
        "oos_composite_score",
        "oos_net_pnl",
    ]
    show = [c for c in cols if c in results.columns]
    print(results[show].head(min(5, len(results))).to_string(index=False))

    return 0


def cmd_inspect_run(run_path: Path | None) -> int:
    try:
        run_dir = resolve_results_dir(run_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    for line in inspect_run_folder(run_dir):
        print(line)
    return 0


def cmd_write_report(run_path: Path | None, output_path: Path | None) -> int:
    from bt.report_md import generate_report_md_from_run_dir

    try:
        run_dir = resolve_results_dir(run_path)
        out = generate_report_md_from_run_dir(run_dir, output_path=output_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote plain-language report → {out}")
    return 0


def cmd_walkforward(
    config_path: Path,
    *,
    export_each: bool = False,
    train_top: int | None = None,
    validation_top: int | None = None,
) -> int:
    """Walk-forward validation: full train sweep → top train → top validation → OOS."""
    if not config_path.exists():
        print(f"Error: config not found: {config_path}", file=sys.stderr)
        return 1

    try:
        loaded = load_run_config(config_path, for_sweep=True)
        from bt.splits import require_walkforward_splits

        require_walkforward_splits(loaded.config)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not loaded.csv_path.exists():
        print(f"Error: data file not found: {loaded.csv_path}", file=sys.stderr)
        return 1

    from bt.sweep import expected_grid_size, iter_param_combinations

    combos = iter_param_combinations(loaded.config)
    if not combos:
        print("Error: config has no sweep list parameters.", file=sys.stderr)
        return 1

    print(f"Config : {config_path}")
    print(f"Data   : {loaded.csv_path}")
    print(f"Grid   : {expected_grid_size(loaded.config)} -> {len(combos)} train combinations")
    _print_costs(loaded.costs)
    print()

    try:
        result = run_walkforward(
            loaded,
            export_each=export_each,
            train_top_n=train_top,
            validation_top_n=validation_top,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"\nWalk-forward folder : {result.root_dir}")
    print(f"  Sweep ({len(result.sweep_results)} combos) : {result.sweep_dir}")
    print(f"  Validation ({len(result.validation_results)} configs) : {result.validation_dir}")
    print(f"  OOS ({len(result.oos_results)} configs) : {result.oos_dir}")
    print(f"  Summary (train+val+oos) : {result.walkforward_summary}")
    _print_runs_and_summary(loaded)

    wf = pd.read_csv(result.walkforward_summary)
    print(f"\nTop walk-forward configs (by oos_{loaded.summary_sort_by}):")
    cols = [
        "oos_rank",
        "train_rank",
        "val_rank",
        "fast",
        "slow",
        "train_composite_score",
        "val_composite_score",
        "oos_composite_score",
    ]
    show = [c for c in cols if c in wf.columns]
    print(wf[show].head(min(5, len(wf))).to_string(index=False))

    return 0


def _add_cost_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--fee-bps",
        type=float,
        default=0.0,
        help="Trading fee per side in bps (10 = 0.10%%)",
    )
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=0.0,
        help="Slippage on entry/exit in bps (5 = 0.05%%)",
    )
    parser.add_argument(
        "--holding-cost-bps",
        type=float,
        default=0.0,
        help="Recurring holding cost per bar in bps (optional)",
    )


def _costs_from_args(args: argparse.Namespace) -> ExecutionCosts:
    return ExecutionCosts(
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        holding_cost_bps=args.holding_cost_bps,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bt", description="Tiny backtester CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_run = subparsers.add_parser(
        "inspect-run",
        help="Inspect a saved run under results/ (default: latest)",
    )
    inspect_run.add_argument(
        "run_path",
        type=Path,
        nargs="?",
        default=None,
        help="Run folder, or 'latest' / 'results/latest' (default: newest backtest)",
    )

    write_report = subparsers.add_parser(
        "write-report",
        help="Generate report.md from summary.csv, trades.csv, and metadata.json",
    )
    write_report.add_argument(
        "run_path",
        type=Path,
        nargs="?",
        default=None,
        help="Run folder, or 'latest' (default: newest backtest)",
    )
    write_report.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (default: <run_dir>/report.md)",
    )

    inspect = subparsers.add_parser("inspect-data", help="Inspect a candle CSV file")
    inspect.add_argument("csv_path", type=Path, help="Path to a candle CSV file")
    inspect.add_argument(
        "--fast-ma",
        type=int,
        default=FAST_MA_WINDOW,
        help=f"Fast moving average window (default: {FAST_MA_WINDOW})",
    )
    inspect.add_argument(
        "--slow-ma",
        type=int,
        default=SLOW_MA_WINDOW,
        help=f"Slow moving average window (default: {SLOW_MA_WINDOW})",
    )

    import_data = subparsers.add_parser(
        "import-data",
        help="Import a downloaded market CSV into backtester candle format",
    )
    import_data.add_argument(
        "input_path",
        type=Path,
        help="Path to a downloaded market CSV (Yahoo, etc.)",
    )
    import_data.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_REAL_SAMPLE_PATH,
        help=f"Output CSV path (default: {DEFAULT_REAL_SAMPLE_PATH})",
    )
    import_data.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Keep only the first N rows after sorting by timestamp",
    )
    import_data.add_argument(
        "--join-csv",
        type=Path,
        default=None,
        help="Backward as-of join columns from this CSV onto imported candles",
    )
    import_data.add_argument(
        "--asset",
        default=None,
        help="Asset filter when --join-csv has an asset column",
    )

    merge_research = subparsers.add_parser(
        "merge-research",
        help="Backward as-of join research onto candles (no lookahead)",
    )
    merge_research.add_argument(
        "candles_path",
        type=Path,
        help="Path to OHLCV candle CSV",
    )
    merge_research.add_argument(
        "research_path",
        type=Path,
        help="Path to research CSV (timestamp, asset, regime, ...)",
    )
    merge_research.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV (default: <candles>_research.csv)",
    )
    merge_research.add_argument(
        "--asset",
        default=None,
        help="Asset symbol when candles lack an asset column (filters research by asset)",
    )
    merge_research.add_argument(
        "--column",
        default="regime",
        help="Research column to merge (default: regime)",
    )
    merge_research.add_argument(
        "--output-column",
        default=None,
        help="Column name in output CSV (default: same as --column)",
    )
    merge_research.add_argument(
        "--fill-before",
        default=None,
        help="Value for candle rows before the first research timestamp",
    )

    indicators = subparsers.add_parser(
        "indicators", help="Compute and show indicator columns (fast_ma, slow_ma)"
    )
    indicators.add_argument("csv_path", type=Path, help="Path to a candle CSV file")
    indicators.add_argument(
        "--fast",
        type=int,
        default=FAST_MA_WINDOW,
        help=f"Fast moving average window (default: {FAST_MA_WINDOW})",
    )
    indicators.add_argument(
        "--slow",
        type=int,
        default=SLOW_MA_WINDOW,
        help=f"Slow moving average window (default: {SLOW_MA_WINDOW})",
    )

    signals = subparsers.add_parser(
        "signals", help="Generate cross-based signals from a candle CSV"
    )
    signals.add_argument("csv_path", type=Path, help="Path to a candle CSV file")
    signals.add_argument(
        "--fast",
        type=int,
        default=FAST_MA_WINDOW,
        help=f"Fast moving average window (default: {FAST_MA_WINDOW})",
    )
    signals.add_argument(
        "--slow",
        type=int,
        default=SLOW_MA_WINDOW,
        help=f"Slow moving average window (default: {SLOW_MA_WINDOW})",
    )

    positions = subparsers.add_parser(
        "positions",
        help="Open positions from signals (entry timing + optional stop/target)",
    )
    positions.add_argument("csv_path", type=Path, help="Path to a candle CSV file")
    positions.add_argument(
        "--fast",
        type=int,
        default=FAST_MA_WINDOW,
        help=f"Fast moving average window (default: {FAST_MA_WINDOW})",
    )
    positions.add_argument(
        "--slow",
        type=int,
        default=SLOW_MA_WINDOW,
        help=f"Slow moving average window (default: {SLOW_MA_WINDOW})",
    )
    positions.add_argument(
        "--stop-pct",
        type=float,
        default=None,
        help="Stop distance as a fraction of entry price (e.g. 0.02 = 2%%)",
    )
    positions.add_argument(
        "--target-pct",
        type=float,
        default=None,
        help="Take-profit distance as a fraction of entry price (e.g. 0.04 = 4%%)",
    )
    _add_cost_args(positions)

    exits = subparsers.add_parser(
        "exits",
        help="Close positions: show where and why trades were exited",
    )
    exits.add_argument("csv_path", type=Path, help="Path to a candle CSV file")
    exits.add_argument(
        "--fast",
        type=int,
        default=FAST_MA_WINDOW,
        help=f"Fast moving average window (default: {FAST_MA_WINDOW})",
    )
    exits.add_argument(
        "--slow",
        type=int,
        default=SLOW_MA_WINDOW,
        help=f"Slow moving average window (default: {SLOW_MA_WINDOW})",
    )
    exits.add_argument(
        "--stop-pct",
        type=float,
        default=None,
        help="Stop distance as a fraction of entry price (e.g. 0.02 = 2%%)",
    )
    exits.add_argument(
        "--target-pct",
        type=float,
        default=None,
        help="Take-profit distance as a fraction of entry price (e.g. 0.04 = 4%%)",
    )
    exits.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Save trades to this CSV path (parent dirs created if needed)",
    )
    _add_cost_args(exits)

    equity = subparsers.add_parser(
        "equity",
        help="Build an equity curve from the strategy's closed trades",
    )
    equity.add_argument("csv_path", type=Path, help="Path to a candle CSV file")
    equity.add_argument("--fast", type=int, default=FAST_MA_WINDOW)
    equity.add_argument("--slow", type=int, default=SLOW_MA_WINDOW)
    equity.add_argument(
        "--stop-pct",
        type=float,
        default=None,
        help="Stop distance as a fraction of entry price (e.g. 0.02)",
    )
    equity.add_argument(
        "--target-pct",
        type=float,
        default=None,
        help="Take-profit distance as a fraction of entry price (e.g. 0.04)",
    )
    equity.add_argument(
        "--initial-equity",
        type=float,
        default=DEFAULT_INITIAL_EQUITY,
        help=f"Starting equity (default: {DEFAULT_INITIAL_EQUITY:,.0f})",
    )
    equity.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Save the equity curve to this CSV path",
    )
    _add_cost_args(equity)

    report = subparsers.add_parser(
        "report", help="Print a strategy report card (summary metrics)"
    )
    report.add_argument("csv_path", type=Path, help="Path to a candle CSV file")
    report.add_argument("--fast", type=int, default=FAST_MA_WINDOW)
    report.add_argument("--slow", type=int, default=SLOW_MA_WINDOW)
    report.add_argument("--stop-pct", type=float, default=None)
    report.add_argument("--target-pct", type=float, default=None)
    report.add_argument(
        "--initial-equity",
        type=float,
        default=DEFAULT_INITIAL_EQUITY,
    )
    report.add_argument(
        "--size",
        type=float,
        default=1.0,
        help="Position size per trade (default: 1.0)",
    )
    report.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Save report metrics as JSON to this path",
    )
    _add_cost_args(report)

    run = subparsers.add_parser(
        "run", help="Run a full backtest pipeline from a TOML config file"
    )
    run.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a TOML config file (see configs/example.toml)",
    )

    sweep = subparsers.add_parser(
        "sweep",
        help="Parameter sweep: backtest every TOML list combination (fast < slow)",
    )
    sweep.add_argument(
        "--config",
        type=Path,
        required=True,
        help="TOML with list values in [sweep], [strategy], or [execution] (see configs/param_sweep.toml)",
    )
    sweep.add_argument(
        "--full-exports",
        action="store_true",
        help="Save full trades/equity/JSON for each combination (slow)",
    )
    sweep.add_argument(
        "--list-only",
        action="store_true",
        help="Only write combinations.csv (no backtests)",
    )

    combos = subparsers.add_parser(
        "combinations",
        help="Build the parameter grid CSV only (skips fast_ma >= slow_ma)",
    )
    combos.add_argument(
        "--config",
        type=Path,
        required=True,
        help="TOML config with list params",
    )

    validate = subparsers.add_parser(
        "validate",
        help="Run top train configs on validation (hold-out) data",
    )
    validate.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Same TOML as sweep (needs [split] validation dates)",
    )
    validate.add_argument(
        "--top",
        type=int,
        default=None,
        help="Number of train configs to test (default: validation.top_n or output.summary_top_n)",
    )
    validate.add_argument(
        "--from",
        dest="leaderboard",
        type=Path,
        default=None,
        help="Ranked CSV of train results (default: latest sweep_results.csv)",
    )
    validate.add_argument(
        "--sweep-results",
        type=Path,
        default=None,
        help="Path to a specific sweep_results.csv",
    )

    oos = subparsers.add_parser(
        "oos",
        help="Run top validation configs on out-of-sample data",
    )
    oos.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Same TOML as sweep (needs [split] oos_start/oos_end)",
    )
    oos.add_argument(
        "--top",
        type=int,
        default=None,
        help="Number of validation configs to test (default: oos.top_n or validation.top_n)",
    )
    oos.add_argument(
        "--from",
        dest="leaderboard",
        type=Path,
        default=None,
        help="validation_results.csv (default: latest validation_* folder)",
    )
    oos.add_argument(
        "--validation-results",
        type=Path,
        default=None,
        help="Path to a specific validation_results.csv",
    )

    portfolio = subparsers.add_parser(
        "portfolio",
        help="Multi-asset portfolio backtest (shared capital, exposure limits)",
    )
    portfolio.add_argument(
        "--config",
        type=Path,
        required=True,
        help="TOML with data.assets and portfolio limits",
    )

    walkforward = subparsers.add_parser(
        "walkforward",
        help="Full walk-forward: train sweep, then top train on validation, top val on OOS",
    )
    walkforward.add_argument(
        "--config",
        type=Path,
        required=True,
        help="TOML with [split], [sweep] lists, [validation]/[oos] top_n",
    )
    walkforward.add_argument(
        "--train-top",
        type=int,
        default=None,
        help="Top train configs for validation (default: walkforward.train_top_n or validation.top_n)",
    )
    walkforward.add_argument(
        "--validation-top",
        type=int,
        default=None,
        help="Top validation configs for OOS (default: walkforward.validation_top_n or oos.top_n)",
    )
    walkforward.add_argument(
        "--full-exports",
        action="store_true",
        help="Write per-combo artifacts during the train sweep",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "inspect-run":
        return cmd_inspect_run(args.run_path)

    if args.command == "write-report":
        return cmd_write_report(args.run_path, args.output)

    if args.command == "inspect-data":
        return cmd_inspect_data(args.csv_path, args.fast_ma, args.slow_ma)

    if args.command == "import-data":
        return cmd_import_data(
            args.input_path,
            args.output,
            args.limit,
            args.join_csv,
            args.asset,
        )

    if args.command == "merge-research":
        return cmd_merge_research(
            args.candles_path,
            args.research_path,
            args.output,
            args.asset,
            args.column,
            args.output_column,
            args.fill_before,
        )

    if args.command == "indicators":
        return cmd_indicators(args.csv_path, args.fast, args.slow)

    if args.command == "signals":
        return cmd_signals(args.csv_path, args.fast, args.slow)

    if args.command == "positions":
        return cmd_positions(
            args.csv_path,
            args.fast,
            args.slow,
            args.stop_pct,
            args.target_pct,
            _costs_from_args(args),
        )

    if args.command == "exits":
        return cmd_exits(
            args.csv_path,
            args.fast,
            args.slow,
            args.stop_pct,
            args.target_pct,
            _costs_from_args(args),
            args.save,
        )

    if args.command == "equity":
        return cmd_equity(
            args.csv_path,
            args.fast,
            args.slow,
            args.stop_pct,
            args.target_pct,
            args.initial_equity,
            _costs_from_args(args),
            getattr(args, "size", 1.0),
            args.save,
        )

    if args.command == "report":
        return cmd_report(
            args.csv_path,
            args.fast,
            args.slow,
            args.stop_pct,
            args.target_pct,
            args.initial_equity,
            _costs_from_args(args),
            args.size,
            args.save,
        )

    if args.command == "run":
        return cmd_run(args.config)

    if args.command == "sweep":
        return cmd_sweep(
            args.config,
            export_each=args.full_exports,
            list_only=args.list_only,
        )

    if args.command == "combinations":
        return cmd_combinations(args.config)

    if args.command == "validate":
        return cmd_validate(
            args.config,
            top_n=args.top,
            source=args.leaderboard,
            sweep_results=args.sweep_results,
        )

    if args.command == "oos":
        return cmd_oos(
            args.config,
            top_n=args.top,
            source=args.leaderboard,
            validation_results=args.validation_results,
        )

    if args.command == "portfolio":
        return cmd_run(args.config)

    if args.command == "walkforward":
        return cmd_walkforward(
            args.config,
            export_each=args.full_exports,
            train_top=args.train_top,
            validation_top=args.validation_top,
        )

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
