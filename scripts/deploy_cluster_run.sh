#!/usr/bin/env bash
# Deploy SIEVE to a compute-cluster node and launch a smoke run.
# Usage: deploy_cluster_run.sh <node> <corpus_name> <tokenizer_dir_name> <run_name>
# Example: deploy_cluster_run.sh node-1 corpus_v007 v004 smoke_loop_v007
set -euo pipefail

NODE="$1"
CORPUS="$2"
TOK="$3"
RUN="$4"
PROFILE="${5:-smoke}"

REMOTE_ROOT="/home/<user>/sieve"
REMOTE_PY="/home/<user>/venvs/voyage4nano/bin/python"
: "${SSH_PASS:?set SSH_PASS (fleet password is no longer stored in the repo)}"
SSH="ssh -o StrictHostKeyChecking=no"
SCP="rsync -az --delete -e 'ssh -o StrictHostKeyChecking=no'"

echo "=== [$NODE] Preparing remote ==="
$SSH "$NODE" "mkdir -p $REMOTE_ROOT/{controller,worker,model,data/tokenizers,config,runs,lab-notes,scripts}"

echo "=== [$NODE] Syncing code ==="
eval "$SCP controller/ $NODE:$REMOTE_ROOT/controller/"
eval "$SCP worker/ $NODE:$REMOTE_ROOT/worker/"
eval "$SCP model/ $NODE:$REMOTE_ROOT/model/"
eval "$SCP config/ $NODE:$REMOTE_ROOT/config/"
# Python modules under data/ (loader, __init__ etc.) — NOT the heavy corpus subdirs
rsync -az -e "ssh -o StrictHostKeyChecking=no" \
  --include='*.py' --include='__init__.py' --exclude='*' \
  data/ "$NODE:$REMOTE_ROOT/data/"
# Only need the one corpus + one tokenizer
rsync -az -e "ssh -o StrictHostKeyChecking=no" data/$CORPUS/ "$NODE:$REMOTE_ROOT/data/$CORPUS/"
rsync -az -e "ssh -o StrictHostKeyChecking=no" data/tokenizers/$TOK/ "$NODE:$REMOTE_ROOT/data/tokenizers/$TOK/"

echo "=== [$NODE] Creating run dirs ==="
$SSH "$NODE" "mkdir -p $REMOTE_ROOT/runs/$RUN/{checkpoints,results,logs}"

echo "=== [$NODE] Sanity import check ==="
$SSH "$NODE" "cd $REMOTE_ROOT && $REMOTE_PY -c 'import sys; sys.path.insert(0, \".\"); from controller.orchestrator import SieveOrchestrator; import worker.trainer; print(\"imports ok\")'"

echo "=== [$NODE] Launching orchestrator (background) ==="
# Detach fully so ssh connection closing doesn't kill the process
$SSH "$NODE" "cd $REMOTE_ROOT && nohup $REMOTE_PY -m controller.orchestrator \
  --config config/default.yaml \
  --profile $PROFILE \
  --corpus-dir data/$CORPUS \
  --checkpoint-dir runs/$RUN/checkpoints \
  --results-dir runs/$RUN/results \
  > runs/$RUN/logs/run.log 2>&1 </dev/null & disown" </dev/null

sleep 5
echo "=== [$NODE] First log lines ==="
$SSH "$NODE" "tail -20 $REMOTE_ROOT/runs/$RUN/logs/run.log"
