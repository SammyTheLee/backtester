# Submission artifacts

All required files for this backtester submission are in this folder.

## Files (attach these)

| Requirement | File in this folder |
|-------------|---------------------|
| summary.csv | `summary.csv` |
| trades.csv | `trades.csv` |
| metadata.json | `metadata.json` |
| report.md | `report.md` |
| CLI output (text) | `submission_cli_output.txt` |
| Test output (text) | `test_output.txt` |

**Source run:** `results/20260805_090248/` (from `python -m bt run --config configs/example.toml`)

**Walk-forward report (bonus):** `walkforward_report.md`

## Screenshots (take these yourself)

Screenshots must be photos of your terminal. Open Terminal, activate venv, then:

### 1. Tests passing

```bash
cd /Users/sammylee/backtester
source .venv/bin/activate
python -m pytest -q
```

Screenshot when you see: `160 passed in ...s`

### 2. CLI output

Either screenshot while running the full demo:

```bash
./demo.sh
```

Or open `submission_cli_output.txt` in a terminal-style view and screenshot that file.

Minimum CLI to show: `run`, `walkforward`, and `inspect-run latest`.

## Regenerate artifacts (optional)

```bash
source .venv/bin/activate
python -m pytest -q
python -m bt run --config configs/example.toml
python -m bt write-report results/$(ls -t results/20* | head -1 | xargs basename)
```

Then copy `summary.csv`, `trades.csv`, `metadata.json`, `report.md` from the new `results/YYYYMMDD_HHMMSS/` folder.
