#!/usr/bin/env bash
# Sync SIEVE code to a cluster node and launch a worker daemon pointing at
# the coordinator's Redis. Uses (compute-cluster convention).
#
# Usage:
#   REDIS_URL="redis://<node-ip>:6379/0" \
#     scripts/deploy_sieve_daemons.sh node-4 [node-6 node-7 ...]
#
# Env:
#   REDIS_URL   (required)  redis://COORD_IP:6379/0
#   CORPUS      (default corpus_v006)  which corpus subdir to sync
#   TOK         (default v003)        tokenizer subdir to sync
#   SSH_PASS    (required)
#   REMOTE_ROOT (default /home/<user>/sieve)
#   REMOTE_PY   (default /home/<user>/venvs/voyage4nano/bin/python)
set -euo pipefail

: "${REDIS_URL:?Must set REDIS_URL, e.g. redis://<node-ip>:6379/0}"
CORPUS="${CORPUS:-corpus_v006}"
TOK="${TOK:-v003}"
PASS="${SSH_PASS:?set SSH_PASS}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/<user>/sieve}"
REMOTE_PY="${REMOTE_PY:-/home/<user>/venvs/voyage4nano/bin/python}"

SSH="ssh -o StrictHostKeyChecking=no"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <node1> [node2 ...]" >&2
  exit 2
fi

for NODE in "$@"; do
  echo "=== [$NODE] Syncing code ==="
  $SSH "$NODE" "mkdir -p $REMOTE_ROOT/{controller,worker,model,data/tokenizers,config,runs}"

  rsync -az -e "ssh -o StrictHostKeyChecking=no" \
    controller/ "$NODE:$REMOTE_ROOT/controller/"
  rsync -az -e "ssh -o StrictHostKeyChecking=no" \
    worker/ "$NODE:$REMOTE_ROOT/worker/"
  rsync -az -e "ssh -o StrictHostKeyChecking=no" \
    model/ "$NODE:$REMOTE_ROOT/model/"
  rsync -az -e "ssh -o StrictHostKeyChecking=no" \
    config/ "$NODE:$REMOTE_ROOT/config/"
  rsync -az -e "ssh -o StrictHostKeyChecking=no" \
    --include='*.py' --include='__init__.py' --exclude='*' \
    data/ "$NODE:$REMOTE_ROOT/data/"
  rsync -az -e "ssh -o StrictHostKeyChecking=no" \
    "data/$CORPUS/" "$NODE:$REMOTE_ROOT/data/$CORPUS/"
  rsync -az -e "ssh -o StrictHostKeyChecking=no" \
    "data/tokenizers/$TOK/" "$NODE:$REMOTE_ROOT/data/tokenizers/$TOK/"

  echo "=== [$NODE] Ensuring redis-py + pymupdf ==="
  # pymupdf (fitz) is required by data/ imports chained through worker.trainer;
  # missing it causes instant "Training failed: No module named 'fitz'" which
  # burns through queued jobs in milliseconds. Always verify.
  $SSH "$NODE" "$REMOTE_PY -c 'import redis' 2>/dev/null || $REMOTE_PY -m pip install --quiet redis"
  $SSH "$NODE" "$REMOTE_PY -c 'import fitz' 2>/dev/null || $REMOTE_PY -m pip install --quiet pymupdf"

  echo "=== [$NODE] Sanity import ==="
  $SSH "$NODE" "cd $REMOTE_ROOT && $REMOTE_PY -c 'import sys; sys.path.insert(0,\".\"); from worker.daemon import run_job; from controller.dispatch import RedisDispatcher; print(\"ok\")'"

  echo "=== [$NODE] Killing any existing daemon ==="
  $SSH "$NODE" "pkill -f 'worker\\.daemon' 2>/dev/null || true"
  sleep 1

  echo "=== [$NODE] Launching daemon ==="
  LOG="$REMOTE_ROOT/runs/worker_daemon.log"
  # OpenSSH on these nodes hangs on `&` even with full redirections; wrap in
  # `timeout` to force-close the ssh after the daemon is started. nohup keeps
  # the orphaned python alive.
  timeout 6 ssh -o StrictHostKeyChecking=no "$NODE" \
    "mkdir -p $REMOTE_ROOT/runs && cd $REMOTE_ROOT && \
     nohup $REMOTE_PY -m worker.daemon \
       --redis '$REDIS_URL' \
       --node '$NODE' \
       --sieve-root '$REMOTE_ROOT' \
     > $LOG 2>&1 &" || true

  sleep 2
  echo "=== [$NODE] First log lines ==="
  $SSH "$NODE" "tail -5 $LOG 2>/dev/null || echo '(no log yet)'"
  echo
done

echo "All workers deployed. Sanity check heartbeats:"
echo "  (from coordinator node)  redis-cli -p 6379 KEYS 'sieve:workers:heartbeat:*'"
