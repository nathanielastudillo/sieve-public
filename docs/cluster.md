# SIEVE Cluster — fleet, discovery, parallel dispatch

SIEVE trains populations in parallel across the GB10 fleet. This doc is how to
find workers, stand up a coordinator, deploy the daemon, and launch a run that
actually uses the fleet.

## Fleet layout

All GB10s live on **<lan>** (warehouse LAN). As of 2026-04-18, ~42
GB10 hosts respond:

- **compute-cluster** (12 nodes, CX-7 200G RDMA fabric): `compute-cluster-1`
  through `compute-cluster-12`, IPs scattered in `<node-ips>`.
- **standalone <standalone>-\*** (~29 nodes, WiFi or single-port wired): `<standalone>-XXXX`
  named by last 4 of MAC. Full table in the user's memory
  (`gb10_inventory.md`) and mirrored in `~/dev/sysadmin/topology/hosts.md`.
- **named specials**: `conversation-logger` (<node-ip>, = <standalone>-ac0d),
  `serac-actual`, `gpu-01` (this machine, <node-ip> CX-7 + <node-ip>
  WiFi).

SSH convention: user `<user>`, password from `$SSH_PASS` (never stored in the repo). IPs are DHCP and shift on reboot
— always rediscover before a run.

**Not on the GB10 LAN:**
- `thoughtspace` (<node-ip>) — useful as a coordinator because it's on a
  different LAN but reachable from gpu-01. When the GB10 fleet can't route
  to it, pick a coordinator that's actually on `<lan>` (see below).

## Discovery

```bash
# Full /24 scan around the cluster core — fast SSH probe + GPU check.
scripts/discover_nodes.sh --subnet <lan> > /tmp/gb10.json

# Wider /20 sweep via mDNS (faster, no SSH auth required).
avahi-browse -art --resolve --parsable 2>/dev/null \
  | awk -F';' '$1=="=" && $3=="IPv4" && $8 ~ /^<lan>\./ {print $8" "$7}' \
  | sort -u | grep -iE "<standalone>|compute-cluster|gb10|conversation-logger"
```

`discover_nodes.sh` uses `BatchMode=yes` — it only finds nodes where this
machine's SSH key is already authorized. For the initial fleet view, the mDNS
sweep is more complete.

## Coordinator (Redis)

The orchestrator dispatches `TrainJob`s on a Redis list; workers `BLPOP` and
return results. Redis can live anywhere reachable from both orchestrator and
all workers. Simplest pattern: run it on one of the compute-cluster nodes,
listening on `:6379`.

```bash
# Option A — Redis on a cluster node (preferred; everyone's on <node-ip>.*).
ssh a@compute-cluster-4 \
  'sudo apt-get update -qq && sudo apt-get install -y redis-server && \
   sudo sed -i "s/^bind .*/bind 0.0.0.0/;s/^protected-mode yes/protected-mode no/" /etc/redis/redis.conf && \
   sudo systemctl restart redis-server'

# Quick check
redis-cli -h <node-ip> -p 6379 PING    # → PONG

# Option B — ad-hoc on a spare port with no systemd.
ssh a@<standalone>-XXXX 'redis-server --port 6400 --bind 0.0.0.0 \
  --protected-mode no --daemonize yes --dir /tmp'
```

Use **Option A** when you have multiple-day runs (systemd will keep it up).
Use **Option B** for smoke tests — no install, no config, dies with the node.

## Deploy daemons

```bash
REDIS_URL="redis://<node-ip>:6379/0" \
  scripts/deploy_sieve_daemons.sh \
    compute-cluster-1 compute-cluster-2 compute-cluster-3 \
    compute-cluster-5 compute-cluster-6 compute-cluster-7 \
    <standalone>-f347 <standalone>-974f <standalone>-359a conversation-logger
```

The script rsyncs `controller/`, `worker/`, `model/`, `config/`, the selected
`data/corpus_XXX/`, and the selected `data/tokenizers/vXXX/` to each node,
ensures `redis-py` + `pymupdf` are installed, kills any existing daemon, and
launches a new one under `nohup`. Daemon logs: `/home/<user>/sieve/runs/worker_daemon.log`
on each node.

Heartbeats:

```bash
redis-cli -h <node-ip> -p 6379 KEYS 'sieve:workers:heartbeat:*'
```

## Launch a run

```bash
cd ~/dev/sieve
python -m controller.orchestrator \
  --config    config/default.yaml \
  --profile   vfull \
  --corpus-dir        data/corpus_vfull \
  --checkpoint-dir    runs/vfull_par/checkpoints \
  --results-dir       runs/vfull_par/results \
  --dispatch          redis://<node-ip>:6379/0 \
  --run-id            vfull_par \
  --remote-sieve-root /home/<user>/sieve \
  --remote-corpus-dir data/corpus_vfull \
  --remote-checkpoint-dir runs/vfull_par/checkpoints \
  2>&1 | tee runs/vfull_par/logs/orchestrator.log
```

**Do not** pass `--workers local` for anything other than a 1-member smoke
test. The whole point of SIEVE is parallel population training; a local run
serializes every member onto one GPU and defeats the premise. If only one
node is reachable, rediscover the fleet or power on more nodes before
launching.

## Remote access to cluster-hosted services (dashboards, servers)

**The "Tailnet" is not one network.** `gpu-01` and the user's
workstation are on the `nathanielastudillo@` tailnet; `compute-cluster-*`
hosts are on `tail1fafbc.ts.net` (owned by `xyrasinclair@`). The two
tailscale tailnets are **disjoint** — a `100.x.y.z` address from one
is not routable from the other. On top of that, compute hosts typically run
`ufw` with default `INPUT DROP` and only SSH (22) open.

Practical consequence: to browse a dashboard (e.g. `dashboard.server`
on `node-8:8096`) from your workstation, you cannot go `http://<compute
tailscale ip>:<port>/` directly, and you also can't just
`ssh -L` from the workstation to `node-8` because your workstation
can't reach `node-8` in the first place.

The working pattern: one SSH hop on `gpu-01` (which can reach the
warehouse LAN at `<lan>`), bound on `gpu-01`'s tailnet
interface so every nathanielastudillo@-peer can reach it:

```bash
ssh gpu-01 'nohup ssh -N -o GatewayPorts=yes \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L 0.0.0.0:<local-port>:localhost:<remote-port> a@<node-lan-ip> \
  >/tmp/tunnel-<tag>.log 2>&1 </dev/null &'

# Then from any nathanielastudillo@ tailnet peer:
#   http://gpu-01:<local-port>/...
```

For persistent forwards, wrap with `autossh -M 0 -N ...` and run under
systemd or in a detached tmux.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'fitz'` on a worker** — daemon
  started before `pymupdf` was installed. Rerun `deploy_sieve_daemons.sh`;
  it verifies both `redis` and `fitz` import before launching the daemon.
- **`ConnectionRefusedError` from the orchestrator** — coordinator Redis
  isn't bound to a routable interface. `redis-cli -h <coord-ip> -p <port> PING`
  from gpu-01 to confirm. If that fails, the coordinator is 127.0.0.1-only;
  restart it with `--bind 0.0.0.0 --protected-mode no`.
- **One-way routability** between <fabric>.x and <lan>.x — common. Always
  pick a coordinator that sits on `<lan>` so every GB10 worker can
  reach it directly.
- **Sequential-looking run** — check the orchestrator log for `Workers:
  local`. If you see it, kill the run, verify `--dispatch` is pointing at a
  reachable Redis, and confirm at least one `sieve:workers:heartbeat:*` key
  exists on the coordinator before relaunching.
