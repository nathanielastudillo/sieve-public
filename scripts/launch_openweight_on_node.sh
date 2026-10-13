#!/usr/bin/env bash
# Dispatch wrapper for dashboard-launched openweight experiments.
#
# The dashboard calls this with a single argument: the python command to
# run on the compute node (e.g. "python -m scripts.bench_openweight_member
# --model-path ...  --member-id ... ...").
#
# We prepend the node-8 python path and route through gpu-01.
# stdout/stderr flow back to the dashboard-managed log file.

set -euo pipefail

CMD="${1:-}"
if [[ -z "$CMD" ]]; then
  echo "usage: $0 '<remote python command>'" >&2
  exit 2
fi

# Replace a bare "python" prefix with the sieve-venv python on node-8.
REMOTE_CMD="${CMD/#python /\/home\/a\/sieve-venv\/bin\/python }"
REMOTE_CMD="cd /home/<user>/sieve && ${REMOTE_CMD}"

echo "# dispatch -> node-8 (a@<node-ip>)"
echo "# remote cmd: ${REMOTE_CMD}"
echo "# ---"

exec ssh gpu-01 \
  "SSH_PW} ssh -o StrictHostKeyChecking=no a@<node-ip> \
   $(printf '%q' "${REMOTE_CMD}")"
