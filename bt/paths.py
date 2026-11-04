"""Output paths for backtest runs (timestamped folders under ``results/``)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

RESULTS_DIR = Path("results")

# Cross-run logs (append / ranked across many backtests).
GLOBAL_RUNS_PATH = RESULTS_DIR / "runs.csv"
GLOBAL_RANKED_SUMMARY_PATH = RESULTS_DIR / "summary.csv"

TRADES_CSV = "trades.csv"
EQUITY_CSV = "equity.csv"
SUMMARY_CSV = "summary.csv"
METADATA_JSON = "metadata.json"
REPORT_JSON = "report.json"
REPORT_MD = "report.md"
SWEEP_COMBINATIONS_CSV = "combinations.csv"
SWEEP_SKIPPED_CSV = "skipped_combinations.csv"
SWEEP_RESULTS_CSV = "sweep_results.csv"
SWEEP_METADATA_JSON = "sweep_metadata.json"
VALIDATION_RESULTS_CSV = "validation_results.csv"
OOS_RESULTS_CSV = "oos_results.csv"
WALKFORWARD_SUMMARY_CSV = "walkforward_summary.csv"
PORTFOLIO_EQUITY_CSV = "portfolio_equity.csv"
REJECTED_TRADES_CSV = "rejected_trades.csv"


def make_timestamped_run_dir(
    base: Path | None = None,
    run_at: datetime | None = None,
    *,
    prefix: str = "",
) -> Path:
    """Return ``results/[prefix]YYYYMMDD_HHMMSS`` (UTC) and create the directory."""
    base = Path(base or RESULTS_DIR)
    when = run_at or datetime.now(timezone.utc)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    stamp = f"{prefix}{when.strftime('%Y%m%d_%H%M%S')}"
    run_dir = base / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


@dataclass(frozen=True)
class RunOutputPaths:
    """CSV + JSON artifact paths for one backtest run."""

    run_dir: Path

    @property
    def trades(self) -> Path:
        return self.run_dir / TRADES_CSV

    @property
    def equity(self) -> Path:
        return self.run_dir / EQUITY_CSV

    @property
    def summary(self) -> Path:
        return self.run_dir / SUMMARY_CSV

    @property
    def metadata(self) -> Path:
        return self.run_dir / METADATA_JSON

    @property
    def report(self) -> Path:
        return self.run_dir / REPORT_JSON

    @property
    def report_md(self) -> Path:
        return self.run_dir / REPORT_MD

    @property
    def portfolio_equity(self) -> Path:
        return self.run_dir / PORTFOLIO_EQUITY_CSV

    @property
    def rejected_trades(self) -> Path:
        return self.run_dir / REJECTED_TRADES_CSV

    @classmethod
    def create(
        cls,
        base: Path | None = None,
        run_at: datetime | None = None,
    ) -> RunOutputPaths:
        return cls(run_dir=make_timestamped_run_dir(base, run_at))

    def artifact_paths(self) -> tuple[Path, ...]:
        return (self.trades, self.equity, self.summary, self.metadata, self.report)


_RUN_DIR_TS = re.compile(
    r"^(?:(?:walkforward|sweep|validation|oos)_)?(\d{8}_\d{6})$"
)


def _run_dir_timestamp(name: str) -> str | None:
    match = _RUN_DIR_TS.match(name)
    return match.group(1) if match else None


def latest_run_dir(base: Path | None = None) -> Path | None:
    """Most recent run folder under ``results/``, or None."""
    base = Path(base or RESULTS_DIR)
    if not base.exists():
        return None
    dirs = [
        p
        for p in base.iterdir()
        if p.is_dir() and _run_dir_timestamp(p.name) is not None
    ]
    dirs.sort(key=lambda p: _run_dir_timestamp(p.name), reverse=True)
    return dirs[0] if dirs else None
