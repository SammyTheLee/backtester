#!/usr/bin/env bash
# Reproducible final demo (offline, no network).
set -euo pipefail
cd "$(dirname "$0")"

python -m pytest -q
python make_demo_data.py
python -m bt inspect-data data/candles/DEMO.csv
python -m bt run --config configs/example.toml
python -m bt sweep --config configs/example.toml
python -m bt walkforward --config configs/walkforward.toml
python -m bt inspect-run latest
