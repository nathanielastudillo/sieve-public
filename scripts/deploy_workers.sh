#!/usr/bin/env bash
# deploy_workers.sh — Push sieve worker code to GB10 nodes and install deps.
#
# Usage:
#   ./deploy_workers.sh [--nodes NODES_JSON] [--discover]
#
# If --nodes is given, reads a JSON file of [{ip, hostname}, ...].
# If --discover is given (or no --nodes), runs discover_nodes.sh first.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
NODES_FILE=""
DISCOVER=false
SSH_USER="${SSH_USER:-<user>}"
REMOTE_DIR="/home/${SSH_USER}/sieve"
SSH_OPTS="-o ConnectTimeout=5 -o StrictHostKeyChecking=no -o BatchMode=yes -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nodes)    NODES_FILE="$2"; shift 2 ;;
        --discover) DISCOVER=true; shift ;;
        --user)     SSH_USER="$2"; REMOTE_DIR="/home/${SSH_USER}/sieve"; shift 2 ;;
        --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--nodes NODES_JSON] [--discover] [--user USER] [--remote-dir DIR]"
            echo "  Deploys sieve codebase to GB10 worker nodes."
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# Get node list
if [[ -n "$NODES_FILE" ]]; then
    if [[ ! -f "$NODES_FILE" ]]; then
        echo "Error: nodes file not found: $NODES_FILE" >&2
        exit 1
    fi
    NODES_JSON=$(cat "$NODES_FILE")
elif [[ "$DISCOVER" == "true" ]] || [[ -z "$NODES_FILE" ]]; then
    echo "Discovering nodes..." >&2
    NODES_JSON=$("${SCRIPT_DIR}/discover_nodes.sh" 2>/dev/null)
fi

# Parse IPs from JSON (lightweight — no jq dependency)
IPS=$(echo "$NODES_JSON" | grep -oP '"ip"\s*:\s*"\K[^"]+')

if [[ -z "$IPS" ]]; then
    echo "No nodes found. Nothing to deploy." >&2
    exit 1
fi

NODE_COUNT=$(echo "$IPS" | wc -l)
echo "Deploying to ${NODE_COUNT} nodes..." >&2

# Files/dirs to sync (exclude large/irrelevant stuff)
RSYNC_EXCLUDES=(
    --exclude '.git'
    --exclude '__pycache__'
    --exclude '*.pyc'
    --exclude '.venv'
    --exclude 'checkpoints/'
    --exclude 'results/'
    --exclude 'corpus/'
    --exclude '.eggs'
    --exclude '*.egg-info'
    --exclude 'wandb/'
)

deploy_to_node() {
    local ip="$1"
    local label="${2:-$ip}"

    echo "[${label}] Starting deployment..." >&2

    # 1. Create remote directory
    ssh ${SSH_OPTS} "${SSH_USER}@${ip}" "mkdir -p ${REMOTE_DIR}" 2>/dev/null
    if [[ $? -ne 0 ]]; then
        echo "[${label}] ERROR: Cannot SSH to ${ip}" >&2
        return 1
    fi

    # 2. rsync codebase
    echo "[${label}] Syncing codebase..." >&2
    rsync -az --delete \
        "${RSYNC_EXCLUDES[@]}" \
        -e "ssh ${SSH_OPTS}" \
        "${PROJECT_DIR}/" "${SSH_USER}@${ip}:${REMOTE_DIR}/"
    if [[ $? -ne 0 ]]; then
        echo "[${label}] ERROR: rsync failed" >&2
        return 1
    fi

    # 3. Install dependencies
    echo "[${label}] Installing dependencies..." >&2
    ssh ${SSH_OPTS} "${SSH_USER}@${ip}" bash <<'REMOTE_SCRIPT'
        set -e
        cd ~/sieve

        # Find python
        PYTHON=""
        for p in python3.12 python3.11 python3; do
            if command -v "$p" &>/dev/null; then
                PYTHON="$p"
                break
            fi
        done

        if [[ -z "$PYTHON" ]]; then
            echo "ERROR: No Python 3.11+ found"
            exit 1
        fi

        PY_VERSION=$($PYTHON --version 2>&1 | grep -oP '\d+\.\d+')
        echo "Python: $PYTHON ($PY_VERSION)"

        # Create venv if needed
        if [[ ! -d .venv ]]; then
            $PYTHON -m venv .venv
        fi

        # Install
        .venv/bin/pip install --quiet --upgrade pip
        .venv/bin/pip install --quiet -e .

        # Verify torch
        echo -n "PyTorch: "
        .venv/bin/python -c "import torch; print(f'{torch.__version__}, CUDA={torch.cuda.is_available()}')" 2>/dev/null || echo "NOT AVAILABLE"

        echo "Done."
REMOTE_SCRIPT

    if [[ $? -ne 0 ]]; then
        echo "[${label}] ERROR: Dependency install failed" >&2
        return 1
    fi

    echo "[${label}] Deployment complete." >&2
    return 0
}

# Deploy to all nodes in parallel
PIDS=()
FAILED=0

while IFS= read -r ip; do
    deploy_to_node "$ip" "$ip" &
    PIDS+=($!)
done <<< "$IPS"

# Wait for all deployments
for pid in "${PIDS[@]}"; do
    wait "$pid" || ((FAILED++))
done

SUCCEEDED=$((NODE_COUNT - FAILED))
echo "" >&2
echo "Deployment complete: ${SUCCEEDED}/${NODE_COUNT} nodes succeeded." >&2

if [[ $FAILED -gt 0 ]]; then
    echo "WARNING: ${FAILED} node(s) failed deployment." >&2
    exit 1
fi
