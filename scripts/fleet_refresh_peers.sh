#!/usr/bin/env bash
# fleet_refresh_peers.sh — rebuild scripts/fleet_peers.txt from live avahi.
#
# Runs on gpu-01 (the mDNS-capable jump host). Call from anywhere via ssh.
# Filters out 10.x fabric IPs (link-local-only, unreachable from the
# warehouse LAN) and picks the first <lan>.x record per hostname.
#
# Captures both GB10 cluster families: compute-cluster (node-*) and <standalone>-*.
set -euo pipefail

OUT="${1:-$(dirname "$0")/fleet_peers.txt}"
TMP=$(mktemp)

trap 'rm -f "$TMP"' EXIT

ssh gpu-01 "avahi-browse -art --resolve --parsable 2>/dev/null \
    | awk -F';' '\$1==\"=\" && \$3==\"IPv4\" && \$4 ~ /SSH/ \
                 && \$8 ~ /^<lan>\./ \
                 && (\$7 ~ /compute-cluster/ || \$7 ~ /^<standalone>-/) \
                 {print \$7, \$8}' \
    | sort -u" > "$TMP"

cat > "$OUT" <<EOF
# sieve fleet peer list — regenerated $(date -Iseconds)
# Format: <name> <ip>
# Covers compute-cluster (node-*) and <standalone>-* families.
EOF

while read -r fqdn ip; do
  name=${fqdn%.local}
  # Normalize compute-cluster-N → node-N; leave <standalone>-* as-is.
  case "$name" in
    compute-cluster-*) name="node-${name#compute-cluster-}" ;;
  esac
  printf "%s %s\n" "$name" "$ip"
done < "$TMP" | sort -V >> "$OUT"

echo "Wrote $(grep -cE '^(node-|<standalone>-)' "$OUT") nodes to $OUT"
