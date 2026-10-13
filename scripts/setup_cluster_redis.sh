#!/usr/bin/env bash
# Install + start Redis on the SIEVE coordinator node (default: node-8).
# Binds 0.0.0.0:6379 so other cluster nodes on <node-ip>/24 can reach it.
#
# Usage:
#   scripts/setup_cluster_redis.sh              # node-8 default
#   scripts/setup_cluster_redis.sh node-4        # alternate coordinator
#
# Idempotent — re-running is safe.
set -euo pipefail

NODE="${1:-node-8}"
PASS="${SSH_PASS:?set SSH_PASS}"
PORT="${REDIS_PORT:-6379}"

SSH="ssh -o StrictHostKeyChecking=no"

echo "=== [$NODE] Installing redis-server if needed ==="
$SSH "$NODE" "dpkg -l | grep -q '^ii  redis-server' || sudo -n apt-get install -y redis-server || echo 'NOTE: need manual install'"

echo "=== [$NODE] Configuring redis (bind 0.0.0.0, no password, no protected mode) ==="
$SSH "$NODE" "sudo tee /etc/redis/redis.conf.d/sieve.conf >/dev/null <<'EOF' || true
bind 0.0.0.0
protected-mode no
port $PORT
maxmemory 4gb
maxmemory-policy allkeys-lru
save ''
appendonly no
EOF
true"

# Some installs use a single redis.conf — patch bind in-place if include dir not honored.
$SSH "$NODE" "sudo sed -i 's/^bind .*/bind 0.0.0.0/' /etc/redis/redis.conf 2>/dev/null || true"
$SSH "$NODE" "sudo sed -i 's/^protected-mode .*/protected-mode no/' /etc/redis/redis.conf 2>/dev/null || true"

echo "=== [$NODE] Restarting redis ==="
$SSH "$NODE" "sudo systemctl restart redis-server || sudo service redis-server restart || true"
sleep 1

echo "=== [$NODE] Verifying reachability ==="
$SSH "$NODE" "redis-cli -p $PORT PING"

# Discover node's <lan>.x address so caller can use it
ADDR=$($SSH "$NODE" "hostname -I | tr ' ' '\n' | grep -E '^192\\.168\\.0\\.' | head -1")
echo
echo "Redis ready at: redis://$ADDR:$PORT/0"
echo "Sanity from another node:"
echo "  ssh node-4 redis-cli -h $ADDR -p $PORT PING"
