#!/usr/bin/env bash
# fleet_health.sh — one-shot status of every sieve fleet node.
# Reports: hostname, fleet-key, sieve repo, venv, SFT deps, worker screen,
# GPU idle, disk free.
# Runs from anywhere (uses ssh to gpu-01 as the jump host for mDNS access).
#
# Exit 0 if all listed nodes pass; 1 otherwise.
set -u

PEERS="${SIEVE_FLEET_PEERS_FILE:-$(dirname "$0")/fleet_peers.txt}"
[ -f "$PEERS" ] || { echo "missing $PEERS (run fleet_refresh_peers.sh first)"; exit 2; }

SSH_JUMP=${SSH_JUMP:-gpu-01}
SSH_PW=${SSH_PW:?set SSH_PW}
export SSH_JUMP SSH_PW

probe() {
  local name=$1 ip=$2
  ssh -n -o BatchMode=yes "$SSH_JUMP" "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=6 <user>@$ip '
    hostname
    test -f /home/<user>/.ssh/id_sieve_fleet && echo KEY_OK || echo KEY_MISSING
    test -d /home/<user>/sieve/openweight && echo SIEVE_OK || echo SIEVE_MISSING
    test -f /home/<user>/venvs/voyage4nano/bin/python && echo VENV_OK || echo VENV_MISSING
    /home/<user>/venvs/voyage4nano/bin/python -c \"import trl, peft, redis\" 2>/dev/null && echo SFT_DEPS_OK || echo SFT_DEPS_MISSING
    screen -ls 2>/dev/null | grep -q sieve-worker && echo WORKER_UP || echo WORKER_DOWN
    df -h /home 2>/dev/null | tail -1 | awk \"{print \\\"DISK \\\" \\\$4 \\\" free (\\\" \\\$5 \\\" used)\\\"}\"
  '" 2>&1 | tr '\n' '|' | head -c 400
}

export -f probe

ok=0; bad=0
while IFS=' ' read -r name ip; do
  [[ "$name" == \#* || -z "$name" ]] && continue
  out=$(timeout 25 bash -c "probe '$name' '$ip'")
  if [[ "$out" == *SIEVE_OK* && "$out" == *VENV_OK* && "$out" == *SFT_DEPS_OK* && "$out" == *WORKER_UP* ]]; then
    ok=$((ok+1))
    status=✓
  else
    bad=$((bad+1))
    status=✗
  fi
  printf "%s %-10s %-16s %s\n" "$status" "$name" "$ip" "$out"
done < "$PEERS"

echo "---"
echo "$ok ok, $bad degraded"
[ $bad -eq 0 ]
