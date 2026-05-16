#!/usr/bin/env bash
# P4 — fetch live Kalshi events via ai-prophet CLI (requires: pip install ai-prophet)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
prophet forecast events --deadline 2026-05-25 --out data/events.json
echo "Wrote data/events.json"
