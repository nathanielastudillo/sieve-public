#!/usr/bin/env bash
# run_sieve.sh — Main entry point for SIEVE evolutionary ensemble training.
#
# Usage:
#   ./run_sieve.sh [OPTIONS]
#
# Options:
#   --config PATH        Path to config file (default: config/default.yaml)
#   --profile PROFILE    Profile: auto, small, medium, large, full (default: auto)
#   --corpus-dir PATH    Path to training corpus directory
#   --workers FILE       Path to workers JSON file (skip discovery)
#   --discover           Force node discovery even if --workers given
#   --deploy             Deploy code to workers before starting
#   --no-deploy          Skip deployment (assume workers are ready)
#   --dry-run            Show what would happen without executing
#   --help               Show this help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="${PROJECT_DIR}/config"

# Defaults
CONFIG="${CONFIG_DIR}/default.yaml"
PROFILE="auto"
CORPUS_DIR=""
WORKERS_FILE=""
DO_DISCOVER=true
DO_DEPLOY=true
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)      CONFIG="$2"; shift 2 ;;
        --profile)     PROFILE="$2"; shift 2 ;;
        --corpus-dir)  CORPUS_DIR="$2"; shift 2 ;;
        --workers)     WORKERS_FILE="$2"; DO_DISCOVER=false; shift 2 ;;
        --discover)    DO_DISCOVER=true; shift ;;
        --deploy)      DO_DEPLOY=true; shift ;;
        --no-deploy)   DO_DEPLOY=false; shift ;;
        --dry-run)     DRY_RUN=true; shift ;;
        --help|-h)
            head -20 "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

echo "========================================="
echo "  SIEVE — Evolutionary Ensemble Trainer"
echo "========================================="
echo ""

# --- Step 1: Validate config ---
if [[ ! -f "$CONFIG" ]]; then
    echo "Error: Config file not found: $CONFIG" >&2
    exit 1
fi
echo "[1/4] Config: $CONFIG"

# --- Step 2: Discover or load workers ---
WORKERS_TMP=""
if [[ -n "$WORKERS_FILE" ]] && [[ -f "$WORKERS_FILE" ]]; then
    echo "[2/4] Using provided workers file: $WORKERS_FILE"
    WORKERS_TMP="$WORKERS_FILE"
else
    echo "[2/4] Discovering worker nodes..."
    WORKERS_TMP=$(mktemp /tmp/sieve-workers-XXXXXX.json)
    trap 'rm -f "$WORKERS_TMP"' EXIT

    "${SCRIPT_DIR}/discover_nodes.sh" > "$WORKERS_TMP" 2>/dev/null

    if [[ ! -s "$WORKERS_TMP" ]] || [[ "$(cat "$WORKERS_TMP")" == "[]" ]]; then
        echo "Error: No worker nodes found on LAN." >&2
        echo "Make sure GB10 nodes are powered on and SSH-reachable." >&2
        exit 1
    fi
fi

# Count workers (count IPs in JSON)
N_WORKERS=$(grep -c '"ip"' "$WORKERS_TMP" 2>/dev/null || echo "0")
echo "   Found ${N_WORKERS} worker(s)."

if [[ "$N_WORKERS" -eq 0 ]]; then
    echo "Error: No workers available. Cannot proceed." >&2
    exit 1
fi

# --- Step 3: Select profile ---
select_profile() {
    local n=$1
    if   (( n <= 8 ));  then echo "small"
    elif (( n <= 16 )); then echo "medium"
    elif (( n <= 32 )); then echo "large"
    else                     echo "full"
    fi
}

if [[ "$PROFILE" == "auto" ]]; then
    PROFILE=$(select_profile "$N_WORKERS")
    echo "[3/4] Auto-selected profile: ${PROFILE} (${N_WORKERS} workers)"
else
    echo "[3/4] Using profile: ${PROFILE}"
fi

PROFILE_FILE="${CONFIG_DIR}/profiles/${PROFILE}.yaml"
if [[ ! -f "$PROFILE_FILE" ]]; then
    echo "Error: Profile config not found: $PROFILE_FILE" >&2
    echo "Available profiles: small, medium, large, full" >&2
    exit 1
fi
echo "   Profile config: $PROFILE_FILE"

# --- Step 4: Deploy to workers ---
if [[ "$DO_DEPLOY" == "true" ]]; then
    echo "[4/4] Deploying to workers..."
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "   (dry-run: skipping deployment)"
    else
        "${SCRIPT_DIR}/deploy_workers.sh" --nodes "$WORKERS_TMP"
    fi
else
    echo "[4/4] Skipping deployment (--no-deploy)."
fi

echo ""
echo "--- Launch Configuration ---"
echo "  Config:     $CONFIG"
echo "  Profile:    $PROFILE ($PROFILE_FILE)"
echo "  Workers:    $N_WORKERS"
echo "  Corpus:     ${CORPUS_DIR:-$(grep 'corpus_dir' "$CONFIG" | awk '{print $2}')}"
echo "----------------------------"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo "(dry-run: would launch controller now)"
    exit 0
fi

# Build controller args
CONTROLLER_ARGS=(
    --config "$CONFIG"
    --profile "$PROFILE_FILE"
    --workers "$WORKERS_TMP"
)

if [[ -n "$CORPUS_DIR" ]]; then
    CONTROLLER_ARGS+=(--corpus-dir "$CORPUS_DIR")
fi

# Launch the controller
echo "Launching SIEVE controller..."
exec python3 -m controller.orchestrator "${CONTROLLER_ARGS[@]}"
