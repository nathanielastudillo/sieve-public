#!/usr/bin/env bash
# Hourly snapshot of exp007_v009 progress — runs every hour for 14 iterations.
# Writes one block per check to /tmp/exp007_checks.log.
set -u

REDIS="<node-ip>"
BIG_RUN="exp007_v009_20260418_054649"
LOG=/tmp/exp007_checks.log
ORCH_LOG=/home/<user>/dev/sieve/runs/exp007_v009/logs/orchestrator.log

for i in $(seq 1 14); do
    {
        echo "===================================================================="
        echo "check $i/14 at $(date -Iseconds)"
        echo "--- big run: $BIG_RUN ---"
        results=$(redis-cli -h "$REDIS" -p 6379 --scan --pattern "sieve:results:${BIG_RUN}:*" 2>/dev/null | wc -l)
        queue=$(redis-cli -h "$REDIS" -p 6379 LLEN sieve:jobs:train 2>/dev/null)
        progress_tail=$(redis-cli -h "$REDIS" -p 6379 LRANGE "sieve:progress:${BIG_RUN}" -5 -1 2>/dev/null)
        errored=$(for k in $(redis-cli -h "$REDIS" -p 6379 --scan --pattern "sieve:results:${BIG_RUN}:*" 2>/dev/null); do
            redis-cli -h "$REDIS" -p 6379 GET "$k" 2>/dev/null | grep -o '"status": *"error"' ; done | wc -l)
        echo "results_published=$results queue_depth=$queue errored=$errored"
        echo "recent progress (last 5):"
        echo "$progress_tail"
        echo "orchestrator log tail:"
        tail -3 "$ORCH_LOG" 2>/dev/null
        echo
        echo "--- smoke run (db=1) ---"
        smoke_keys=$(redis-cli -h "$REDIS" -p 6379 -n 1 --scan --pattern "sieve:results:*" 2>/dev/null | wc -l)
        smoke_queue=$(redis-cli -h "$REDIS" -p 6379 -n 1 LLEN sieve:jobs:train 2>/dev/null)
        echo "smoke_results=$smoke_keys smoke_queue=$smoke_queue"
    } >>"$LOG" 2>&1
    sleep 3600
done
echo "DONE at $(date -Iseconds)" >>"$LOG"
