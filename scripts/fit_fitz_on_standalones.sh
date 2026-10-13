#!/usr/bin/env bash
# Install pymupdf (fitz) on the 9 standalone GB10s that were probed as ready
# but crashed workers because the voyage4nano venv didn't have it.
#
# Run this BEFORE attempting to deploy sieve worker daemons to standalones.
# After this, you can point deploy_sieve_daemons.sh at these nodes safely.
#
# NOTE: Does NOT start any daemons. Install-only.
set -u

IPS=(
  "node-ip"   # <standalone>-a640
  "node-ip"   # <standalone>-91a6
  "node-ip"   # <standalone>-519f
  "node-ip"   # <standalone>-8860
  "node-ip"  # <standalone>-9dba
  "node-ip"  # <standalone>-ad3f
  "node-ip"  # <standalone>-98ff
  "node-ip"  # <standalone>-aa15
  "node-ip"  # <standalone>-9560
)

PY=/home/<user>/venvs/voyage4nano/bin/python
PASS="${SSH_PASS:?set SSH_PASS}"

for ip in "${IPS[@]}"; do
  echo "=== $ip ==="
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=6 "<user>@$ip" \
    "$PY -m pip install --quiet pymupdf && $PY -c 'import fitz; print(\"fitz OK\", fitz.__doc__.split(chr(10))[0] if fitz.__doc__ else \"\")'" 2>&1 | tail -5
done
echo
echo "Next step: deploy workers with"
echo "  REDIS_URL=redis://<node-ip>:6379/0 CORPUS=corpus_v009 TOK=v005 \\"
echo "    scripts/deploy_sieve_daemons.sh a@<node-ip> a@<node-ip> ..."
