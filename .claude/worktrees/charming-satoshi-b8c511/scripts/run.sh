#!/usr/bin/env bash
# Prophet Hacks 2026 - Trading Track agent launcher.
#
# Usage:
#   bash scripts/run.sh                       # 14-day eval defaults
#   bash scripts/run.sh --dry                 # smoke-test pipeline only
#   bash scripts/run.sh --slug nailong_v02    # override slug
#
# Requires `.env` populated from `.env.template`.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
  echo "ERROR: .env not found. Copy .env.template -> .env and fill in keys." >&2
  exit 1
fi

exec python -m agent.run "$@"
