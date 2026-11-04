"""Shared backtest run pipeline for single runs and parameter sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from bt.backtest import apply_execution
from bt.costs import ExecutionCosts, execution_costs_from_config
from bt.export import export_backtest_run, export_portfolio_backtest_run
from bt.portfolio_backtest import run_portfolio_backtest
from bt.portfolio_config import (
    AssetSpec,
    PortfolioLimits,
    is_portfolio_mode,
    parse_assets,
    parse_portfolio_limits,
)
from bt.metadata import build_run_metadata
from bt.paths import (
    GLOBAL_RANKED_SUMMARY_PATH,
    GLOBAL_RUNS_PATH,
    RunOutputPaths,
)
from bt.portfolio import DEFAULT_INITIAL_EQUITY, equity_curve
from bt.report import (
    DEFAULT_DRAWDOWN_PENALTY,
    DEFAULT_SUMMARY_SORT_BY,
    DEFAULT_SUMMARY_TOP_N,
    StrategyReportCard,
    append_runs_csv,
    build_report_card,
    run_row,
    write_ranked_summary_csv,
)
from bt.splits import (
    SplitConfig,
    SplitName,
    load_candles,
    parse_split_config,
    split_config_for_sweep,
)
from bt.strategy import (
    Strategy,
    apply_signals,
    strategy_from_params,
    strategy_type,
    supported_strategy_types,
    validate_strategy_params,
)
from bt.sweep import resolve_base_params
from bt.trade import Trade, extract_trades


@dataclass(frozen=True)
class BacktestResult:
    card: StrategyReportCard
    trades: list[Trade]
    df: pd.DataFrame
    strat: Strategy
    output_paths: RunOutputPaths | None = None


@dataclass(frozen=True)
class BacktestParams:
    """Resolved strategy + execution parameters for one backtest run."""

    combo: dict[str, Any]

    @property
    def stop_pct(self) -> float | None:
        value = self.combo.get("stop_pct")
        return None if value is None else float(value)

    @property
    def target_pct(self) -> float | None:
        value = self.combo.get("target_pct")
        return None if value is None else float(value)

    @property
    def fast(self) -> int | None:
        value = self.combo.get("fast")
        return None if value is None else int(value)

    @property
    def slow(self) -> int | None:
        value = self.combo.get("slow")
        return None if value is None else int(value)

    @property
    def period(self) -> int | None:
        value = self.combo.get("period")
        return None if value is None else int(value)


@dataclass(frozen=True)
class LoadedRunConfig:
    config: dict[str, Any]
    config_path: Path
    csv_path: Path
    assets: tuple[AssetSpec, ...]
    portfolio_limits: PortfolioLimits
    is_portfolio: bool
    costs: ExecutionCosts
    initial_equity: float
    size: float
    summary_sort_by: str
    drawdown_penalty: float
    summary_top_n: int | None
    split: SplitConfig | None
    data_split: SplitName
    runs_log_path: Path
    ranked_summary_path: Path


def load_run_config(
    config_path: Path,
    *,
    for_sweep: bool = False,
) -> LoadedRunConfig:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]

    with open(config_path, "rb") as f:
        config: dict[str, Any] = tomllib.load(f)

    strat_cfg = config.get("strategy", {})
    strat_type = strat_cfg.get("type", "ma_crossover")
    if strat_type not in supported_strategy_types():
        raise ValueError(
            f"unsupported strategy type {strat_type!r}; "
            f"supported: {', '.join(supported_strategy_types())}"
        )

    assets = tuple(parse_assets(config))
    csv_path = assets[0].csv_path
    portfolio_limits = parse_portfolio_limits(config)
    portfolio = is_portfolio_mode(config, list(assets))
    exec_cfg = config.get("execution", {})
    costs = execution_costs_from_config(exec_cfg)
    port_cfg = config.get("portfolio", {})
    initial_equity = float(port_cfg.get("initial_equity", DEFAULT_INITIAL_EQUITY))
    size = float(port_cfg.get("size", 1.0))
    out_cfg = config.get("output", {})
    summary_sort_by = str(out_cfg.get("summary_sort_by", DEFAULT_SUMMARY_SORT_BY))
    drawdown_penalty = float(
        out_cfg.get("drawdown_penalty", DEFAULT_DRAWDOWN_PENALTY)
    )
    top_n_raw = out_cfg.get("summary_top_n", DEFAULT_SUMMARY_TOP_N)
    summary_top_n = None if top_n_raw is None else int(top_n_raw)
    runs_log_path = Path(out_cfg.get("runs_log", GLOBAL_RUNS_PATH))
    ranked_summary_path = Path(
        out_cfg.get("ranked_summary", GLOBAL_RANKED_SUMMARY_PATH)
    )
    if for_sweep:
        split = split_config_for_sweep(config)
        data_split: SplitName = "train"
    else:
        split = parse_split_config(config)
        data_split = split.active if split else "all"
    return LoadedRunConfig(
        config=config,
        config_path=config_path,
        csv_path=csv_path,
        assets=assets,
        portfolio_limits=portfolio_limits,
        is_portfolio=portfolio,
        costs=costs,
        initial_equity=initial_equity,
        size=size,
        summary_sort_by=summary_sort_by,
        drawdown_penalty=drawdown_penalty,
        summary_top_n=summary_top_n,
        split=split,
        data_split=data_split,
        runs_log_path=runs_log_path,
        ranked_summary_path=ranked_summary_path,
    )


def params_from_config(config: dict[str, Any]) -> BacktestParams:
    """Build backtest params from scalar (or single-element) config values."""
    combo = resolve_base_params(config)
    validate_strategy_params(strategy_type(config), combo)
    return BacktestParams(combo=combo)


def run_strategy_backtest(
    csv_path: Path,
    strategy: Strategy,
    costs: ExecutionCosts,
    *,
    size: float = 1.0,
    split: SplitConfig | None = None,
) -> tuple[pd.DataFrame, list[Trade], Strategy]:
    df, _ = load_candles(csv_path, split=split)
    df = apply_signals(strategy, df)
    df = apply_execution(df, strategy, slippage_bps=costs.slippage_bps)
    trades = extract_trades(df, size=size, costs=costs)
    return df, trades, strategy


def run_configured_backtest(
    csv_path: Path,
    params: BacktestParams,
    costs: ExecutionCosts,
    config: dict[str, Any],
    *,
    size: float = 1.0,
    split: SplitConfig | None = None,
) -> tuple[pd.DataFrame, list[Trade], Strategy]:
    """Run one asset backtest from resolved config + parameter combo."""
    strategy = strategy_from_params(config, params.combo)
    return run_strategy_backtest(
        csv_path,
        strategy,
        costs,
        size=size,
        split=split,
    )


def run_ma_backtest(
    csv_path: Path,
    params: BacktestParams,
    costs: ExecutionCosts,
    *,
    size: float = 1.0,
    split: SplitConfig | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[Trade], Strategy]:
    """Backward-compatible alias for ``run_configured_backtest``."""
    cfg = config or {"strategy": {"type": "ma_crossover"}}
    return run_configured_backtest(
        csv_path,
        params,
        costs,
        cfg,
        size=size,
        split=split,
    )


def execute_backtest(
    loaded: LoadedRunConfig,
    params: BacktestParams,
    *,
    run_at: datetime | None = None,
    output_paths: RunOutputPaths | None = None,
    export_artifacts: bool = True,
    append_global_log: bool = True,
    update_ranked_summary: bool = True,
    split: SplitConfig | None = None,
    data_split: SplitName | str | None = None,
    config_label: str | None = None,
) -> BacktestResult:
    """Run one backtest; optionally export artifacts and append global logs."""
    use_split = split if split is not None else loaded.split
    use_data_split = data_split if data_split is not None else loaded.data_split

    if loaded.is_portfolio:
        for spec in loaded.assets:
            if not spec.csv_path.exists():
                raise FileNotFoundError(f"data file not found: {spec.csv_path}")
        pf = run_portfolio_backtest(
            list(loaded.assets),
            params,
            loaded.costs,
            config=loaded.config,
            initial_equity=loaded.initial_equity,
            size=loaded.size,
            limits=loaded.portfolio_limits,
            split=use_split,
        )
        trades = pf.accepted_trades
        curve = pf.portfolio_equity
        card = pf.card
        df = next(iter(pf.candles_by_asset.values()))
        strat = strategy_from_params(loaded.config, params.combo)
        if not export_artifacts:
            return BacktestResult(card=card, trades=trades, df=df, strat=strat)
    else:
        if not loaded.csv_path.exists():
            raise FileNotFoundError(f"data file not found: {loaded.csv_path}")
        df, trades, strat = run_configured_backtest(
            loaded.csv_path,
            params,
            loaded.costs,
            loaded.config,
            size=loaded.size,
            split=use_split,
        )
        curve = equity_curve(trades, initial_equity=loaded.initial_equity)
        card = build_report_card(
            trades,
            initial_equity=loaded.initial_equity,
            bars_in_market=int(df["in_position"].sum()),
            total_bars=len(df),
        )
        if not export_artifacts:
            return BacktestResult(card=card, trades=trades, df=df, strat=strat)

    started = run_at or datetime.now(timezone.utc)
    run_at_iso = started.isoformat()
    paths = output_paths or RunOutputPaths.create(run_at=started)
    label = config_label or str(loaded.config_path)

    metadata = build_run_metadata(
        loaded.config,
        config_path=loaded.config_path,
        csv_path=loaded.csv_path,
        strategy_name=strat.name,
        fast=params.fast,
        slow=params.slow,
        stop_pct=params.stop_pct,
        target_pct=params.target_pct,
        costs=loaded.costs,
        initial_equity=loaded.initial_equity,
        size=loaded.size,
        summary_sort_by=loaded.summary_sort_by,
        candles_df=df,
        run_at=run_at_iso,
        output_dir=str(paths.run_dir),
        data_split=str(use_data_split),
        split=use_split,
    )
    if loaded.is_portfolio:
        metadata["portfolio"] = {
            "assets": [
                {"symbol": a.symbol, "csv_path": str(a.csv_path)}
                for a in loaded.assets
            ],
            "limits": {
                "max_open_positions": loaded.portfolio_limits.max_open_positions,
                "max_total_exposure_pct": loaded.portfolio_limits.max_total_exposure_pct,
                "max_exposure_per_asset_pct": (
                    loaded.portfolio_limits.max_exposure_per_asset_pct
                ),
            },
            "rejected_trades": len(pf.rejected),
        }
    run_meta = run_row(
        card,
        run_at=run_at_iso,
        config=label,
        data_path=str(loaded.csv_path),
        strategy=strat.name,
        fast=params.fast,
        slow=params.slow,
        initial_equity=loaded.initial_equity,
        drawdown_penalty=loaded.drawdown_penalty,
        data_split=str(use_data_split),
    )
    if loaded.is_portfolio:
        export_portfolio_backtest_run(
            paths,
            trades=trades,
            trade_curve=pf.trade_equity,
            portfolio_curve=pf.portfolio_equity,
            card=card,
            metadata=metadata,
            run_meta=run_meta,
            rejected=pf.rejected,
        )
    else:
        export_backtest_run(
            paths,
            trades=trades,
            curve=curve,
            card=card,
            metadata=metadata,
            run_meta=run_meta,
        )

    if append_global_log:
        append_runs_csv(
            card,
            loaded.runs_log_path,
            run_at=run_at_iso,
            config=label,
            data_path=str(loaded.csv_path),
            strategy=strat.name,
            fast=params.fast,
            slow=params.slow,
            initial_equity=loaded.initial_equity,
            drawdown_penalty=loaded.drawdown_penalty,
            data_split=str(use_data_split),
        )
        if update_ranked_summary and str(use_data_split).lower() == "oos":
            write_ranked_summary_csv(
                loaded.runs_log_path,
                loaded.ranked_summary_path,
                sort_by=loaded.summary_sort_by,
                initial_equity=loaded.initial_equity,
                drawdown_penalty=loaded.drawdown_penalty,
                top_n=loaded.summary_top_n,
            )

    return BacktestResult(
        card=card,
        trades=trades,
        df=df,
        strat=strat,
        output_paths=paths,
    )
