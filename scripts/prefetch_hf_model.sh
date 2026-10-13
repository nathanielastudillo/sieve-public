#!/usr/bin/env bash
# prefetch_hf_model.sh — kick off a HuggingFace snapshot download on a fleet node.
# Runs in a detached screen on the target so the SSH session can drop cleanly.
#
# Usage:
#   scripts/prefetch_hf_model.sh <hf_repo_id> <node_name> [revision]
#
# Examples:
#   scripts/prefetch_hf_model.sh deepseek-ai/DeepSeek-V4-Flash <standalone>-aa15
#   scripts/prefetch_hf_model.sh deepseek-ai/DeepSeek-V4-Flash-Base <standalone>-d898 main
#
# Resolves <node_name> against scripts/fleet_peers.txt for the LAN IP, then
# fans through the standard gpu-01 jump + pattern. Logs to
# /tmp/hf-prefetch-<safe-model>.log on the target. Tail with:
#   ssh gpu-01 'ssh -o StrictHostKeyChecking=no a@<ip> tail -f /tmp/hf-prefetch-<safe-model>.log'
set -u

MODEL=${1:?usage: $0 <hf_repo_id> <node_name> [revision]}
NODE=${2:?usage: $0 <hf_repo_id> <node_name> [revision]}
REV=${3:-main}

PEERS_FILE=${SIEVE_FLEET_PEERS_FILE:-$(dirname "$0")/fleet_peers.txt}
SSH_JUMP=${SSH_JUMP:-gpu-01}
SSH_PW=${SSH_PW:?set SSH_PW}
INNER='-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=10'

# Forward HF_TOKEN to the remote download. Read from env, fall back to the
# standard token file on this host. Unauthenticated pulls are throttled
# (~10 MB/s); authenticated pulls run at the node's link speed.
HF_TOKEN_VAL=${HF_TOKEN:-}
if [ -z "$HF_TOKEN_VAL" ] && [ -r "$HOME/.cache/huggingface/token" ]; then
  HF_TOKEN_VAL=$(cat "$HOME/.cache/huggingface/token")
fi

ip=""
if [ -f "$PEERS_FILE" ]; then
  ip=$(awk -v n="$NODE" '$1==n {print $2}' "$PEERS_FILE")
fi
# Fall back to the fleet API (broader inventory than the sieve peer list)
if [ -z "$ip" ]; then
  ip=$(curl -fsS "http://127.0.0.1:8081/nodes/$NODE" 2>/dev/null \
       | python3 -c 'import json,sys; print(json.load(sys.stdin).get("lan_ip",""))' 2>/dev/null)
fi
[ -z "$ip" ] && { echo "node $NODE not found in $PEERS_FILE or fleet API"; exit 2; }

# safe-name for screen session + log file
safe=$(echo "$MODEL" | tr '/:' '__')
session="hfp-$safe"
log="/tmp/hf-prefetch-${safe}.log"

# Remote command:
# - Prefer the sieve venv's hf / huggingface-cli (huggingface_hub ships both).
# - Fall back to the system python with snapshot_download.
# - Wrapped in screen -dmS so disconnecting SSH does not SIGHUP the transfer.
remote=$(cat <<REMOTE
set -e
mkdir -p /tmp
if screen -ls 2>/dev/null | grep -q "$session"; then
  echo "session $session already running on \$(hostname); attach with: screen -r $session"
  exit 3
fi
# Probe for an hf-capable venv. compute nodes ship /home/<user>/sieve/.venv;
# <standalone> nodes ship /home/<user>/venvs/voyage4nano (per fleet_health.sh).
HFBIN=""
for cand in /home/<user>/venvs/voyage4nano/bin/hf /home/<user>/sieve/.venv/bin/hf /home/<user>/venvs/*/bin/hf; do
  [ -x "\$cand" ] && HFBIN="\$cand" && break
done
if [ -z "\$HFBIN" ]; then
  for cand in /home/<user>/venvs/voyage4nano/bin/huggingface-cli /home/<user>/sieve/.venv/bin/huggingface-cli /home/<user>/venvs/*/bin/huggingface-cli; do
    [ -x "\$cand" ] && HFBIN="\$cand" && break
  done
fi
if [ -z "\$HFBIN" ]; then
  echo "no hf/huggingface-cli found in any /home/<user>/venvs/*/bin or /home/<user>/sieve/.venv/bin on \$(hostname)"
  exit 4
fi
CMD="\$HFBIN download $MODEL --revision $REV"
echo "[\$(date -Is)] \$(hostname): \$CMD (token=\${HF_TOKEN:+set})" > $log
screen -dmS $session bash -c "export HF_TOKEN=\${HF_TOKEN}; \$CMD >> $log 2>&1; echo \"[\$(date -Is)] exit=\$?\" >> $log"
echo "started screen session '$session' on \$(hostname); log=$log"
REMOTE
)

# base64 the script to keep quoting sane through the double-ssh hop
b64=$(printf '%s' "$remote" | base64 -w0)

# HF_TOKEN is forwarded as an env var to the remote bash invocation so it
# never appears on the command line (visible in /proc only).
ssh -n -o BatchMode=yes "$SSH_JUMP" \
  "ssh $INNER <user>@$ip \"HF_TOKEN='$HF_TOKEN_VAL' bash -c 'echo $b64 | base64 -d | bash'\""
