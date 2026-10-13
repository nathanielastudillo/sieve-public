#!/usr/bin/env bash
# discover_nodes.sh — Discover GB10 worker nodes on LAN.
#
# Scans a /24 for SSH-reachable hosts that authenticate as user `<user>` and
# report an NVIDIA GB10 via nvidia-smi. Outputs a JSON array.
#
# The GB10 fleet lives on <lan> (gpu-01 is <node-ip>).
# As of 2026-04-18, ~31 unique GB10 hosts respond: all 12 compute-cluster-*,
# ~17 <standalone>-* standalones, conversation-logger, serac-actual. Some have
# multiple NICs (wired + WiFi) → >31 IP hits on a full scan.
#
# <fabric>.x was an earlier tailnet-era network; don't default there or the
# scan returns ~1 node.
#
# Usage:
#   ./discover_nodes.sh [--timeout SECONDS] [--subnet CIDR] [--user USER]
#
# Output (stdout): JSON array of {hostname, ip, gpu} objects
# Logs go to stderr so JSON output stays clean.

set -euo pipefail

TIMEOUT=3
SUBNET="<lan>"
SSH_USER="${SSH_USER:-<user>}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --timeout) TIMEOUT="$2"; shift 2 ;;
        --subnet)  SUBNET="$2"; shift 2 ;;
        --user)    SSH_USER="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--timeout SECONDS] [--subnet PREFIX] [--user USER]" >&2
            echo "  Discovers GB10 nodes on LAN. Outputs JSON to stdout." >&2
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# BatchMode=yes → key auth only. Password-only nodes will be skipped; use the
# mDNS sweep in docs/cluster.md for a fuller picture if you need it.
SSH_OPTS="-o ConnectTimeout=${TIMEOUT} -o StrictHostKeyChecking=no -o BatchMode=yes -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

echo "Scanning ${SUBNET}.0/24 for SSH-reachable GB10 nodes..." >&2

# Temporary files for parallel probe results
TMPDIR_DISC=$(mktemp -d)
trap 'rm -rf "$TMPDIR_DISC"' EXIT

# Probe each IP in parallel
probe_host() {
    local ip="$1"
    local tmpdir="$2"

    # Quick TCP check on port 22 before SSH
    if ! timeout "${TIMEOUT}" bash -c "echo >/dev/tcp/${ip}/22" 2>/dev/null; then
        return
    fi

    # Get hostname via SSH
    local hostname
    hostname=$(ssh ${SSH_OPTS} "${SSH_USER}@${ip}" "hostname" 2>/dev/null) || return

    # Verify it looks like a GB10 (check for NVIDIA GPU)
    local has_gpu
    has_gpu=$(ssh ${SSH_OPTS} "${SSH_USER}@${ip}" \
        "test -e /dev/nvidia0 && echo yes || echo no" 2>/dev/null) || has_gpu="unknown"

    echo "{\"hostname\": \"${hostname}\", \"ip\": \"${ip}\", \"gpu\": \"${has_gpu}\"}" \
        > "${tmpdir}/${ip}.json"
    echo "  Found: ${hostname} (${ip}) gpu=${has_gpu}" >&2
}

export -f probe_host
export SSH_OPTS SSH_USER TIMEOUT

# Scan .1 through .254 in parallel (up to 32 at a time)
for i in $(seq 1 254); do
    probe_host "${SUBNET}.${i}" "${TMPDIR_DISC}" &
    # Limit parallelism
    if (( i % 32 == 0 )); then
        wait
    fi
done
wait

# Collect results into JSON array
RESULTS=()
for f in "${TMPDIR_DISC}"/*.json; do
    [[ -f "$f" ]] || continue
    RESULTS+=("$(cat "$f")")
done

N=${#RESULTS[@]}
echo "Discovered ${N} nodes." >&2

# Build JSON array
echo "["
for i in "${!RESULTS[@]}"; do
    if (( i < N - 1 )); then
        echo "  ${RESULTS[$i]},"
    else
        echo "  ${RESULTS[$i]}"
    fi
done
echo "]"
