#!/usr/bin/env bash
# Nailong Elite — auto-restart wrapper for the 14-day eval window.
# Restarts the agent on any crash with a 30-second cooldown. Uses the same
# slug so the SDK resumes from the next un-finalized tick.
#
# Usage:
#   bash scripts/run_forever.sh           # interactive (Ctrl-C kills loop)
#   nohup bash scripts/run_forever.sh > trader.log 2>&1 &   # background
#   tmux new -s nailong 'bash scripts/run_forever.sh'        # tmux session

set -u
cd "$(dirname "$0")/.."

SLUG="${NAILONG_SLUG:-eval_nailonguic}"
LOG_FILE="${NAILONG_LOG:-trader.log}"
COOLDOWN_SEC="${NAILONG_COOLDOWN:-30}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting Nailong forever-loop (slug=$SLUG)"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Log: $LOG_FILE  Cooldown: ${COOLDOWN_SEC}s"

attempt=0
while true; do
    attempt=$((attempt + 1))
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ==== Attempt $attempt: starting agent ===="
    python -m agent.run --slug "$SLUG" 2>&1 | tee -a "$LOG_FILE"
    rc=${PIPESTATUS[0]}
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Agent exited rc=$rc; cooldown ${COOLDOWN_SEC}s before restart"
    sleep "$COOLDOWN_SEC"
done
