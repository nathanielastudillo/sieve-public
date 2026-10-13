#!/usr/bin/env bash
# fleet_deploy.sh — rsync the local sieve repo to every compute node in parallel.
# Pushes code + scripts/fleet_peers.txt; excludes data/corpora/runs/checkpoints
# to avoid shipping gigabytes.
#
# Usage:
#   scripts/fleet_deploy.sh              # push to all peers in fleet_peers.txt
#   SIEVE_FLEET_PEERS=node-2,node-12 scripts/fleet_deploy.sh   # subset
set -u

REPO=${REPO:-/storage/lever/sieve}
PEERS_FILE=${SIEVE_FLEET_PEERS_FILE:-$(dirname "$0")/fleet_peers.txt}
SSH_JUMP=${SSH_JUMP:-gpu-01}
SSH_PW=${SSH_PW:?set SSH_PW}

if [ ! -f "$PEERS_FILE" ]; then
  echo "missing $PEERS_FILE"; exit 2
fi

# Filter
if [ -n "${SIEVE_FLEET_PEERS:-}" ]; then
  FILTER="|$SIEVE_FLEET_PEERS|"
  FILTER="${FILTER//,/|}"
  targets=$(awk -v f="$FILTER" '
    /^#/ || NF==0 {next}
    index(f, "|" $1 "|") > 0 {print $1, $2}
  ' "$PEERS_FILE")
else
  targets=$(awk '/^#/ || NF==0 {next} {print $1, $2}' "$PEERS_FILE")
fi

push_one() {
  local name=$1 ip=$2
  local t0=$(date +%s)
  # rsync from gpu-01 to the compute node via the fleet key. gpu-01
  # first needs the repo: sync local→gpu-01, then gpu-01→compute.
  # Simpler: rsync directly from this host to the compute, jumping via ssh.
  RSYNC_RSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=8 -o ProxyJump=$SSH_JUMP"
  SSHPASS="$SSH_PW" rsync -az \
    --exclude=data/corpora --exclude=data/cleaned_* --exclude=data/extracted \
    --exclude=runs --exclude=checkpoints --exclude='*.pyc' --exclude=__pycache__ \
    --exclude=.git --exclude=lab/.next --exclude=dashboard/frontend/node_modules \
    --exclude=dashboard/frontend/.next \
    -e "-e $RSYNC_RSH" \
    "$REPO/" "<user>@$ip:/home/<user>/sieve/" 2>&1 \
    | tail -3 | tr '\n' ' '
  local t1=$(date +%s)
  echo "  ($((t1 - t0))s)"
}

while IFS=' ' read -r name ip; do
  [ -z "$name" ] && continue
  printf "%-6s %-16s " "$name" "$ip"
  push_one "$name" "$ip" &
done <<< "$targets"
wait

echo
echo "fleet_deploy complete"
