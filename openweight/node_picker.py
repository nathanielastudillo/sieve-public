"""Pick a free GB10 node to host a shim — the one-model-per-node rule.

The hard constraint (per the user, 2026-04-24): on a GB10 with ~125 GB
unified memory, two 13–14B fp16 models + their KV caches OOM-kill the
parent process. So inference shims spread across nodes; a 2nd shim never
co-locates on a node that already has one. If the fleet can't satisfy a
request, we panic — return ``NoFreeNode`` and let the caller 503.

This module is a placeholder for FLEET's ``/select`` endpoint. When that
ships, ``pick_node`` becomes a thin HTTP wrapper around it. Until then
we ssh-probe the fleet ourselves: avahi/peers list → ssh-check
weights+memory+no-existing-shim → first fit.

Discovery is intentionally serial-with-early-stop: the fleet has 12+
nodes, picking the first viable one is fine; spawning 12 ssh probes in
parallel would be faster but burn ssh/auth more aggressively for no
real win.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

# Fleet ssh: from any compute node, the fleet key authenticates to all others.
FLEET_KEY = "/home/<user>/.ssh/id_sieve_fleet"
FLEET_USER = "a"
FLEET_PEERS = Path("/home/<user>/sieve/scripts/fleet_peers.txt")
SSH_OPTS = (
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR",
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=5",
)

# Headroom needed to safely load a 13–14B fp16 + KV cache + working set.
# 30 GB is conservative; an empty GB10 has ~125 GB MemAvailable.
MIN_FREE_GB = 30.0


class NoFreeNode(RuntimeError):
    """Raised when the fleet has no node that can host a fresh shim."""


@dataclass(frozen=True)
class Node:
    name: str       # short alias used in logs/UI, e.g., "node-6"
    ip: str         # ssh target — IP or mDNS FQDN. Prefer FQDN
                    # (e.g., "compute-cluster-6.local") so DHCP drift
                    # doesn't break the file.


def load_peers(peers_file: Path = FLEET_PEERS) -> list[Node]:
    """Parse fleet_peers.txt → [Node(name, ip), ...]. Skips comments."""
    nodes: list[Node] = []
    for raw in peers_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        nodes.append(Node(name=parts[0], ip=parts[1]))
    return nodes


def _hf_cache_dir_for(model_id: str, hf_cache: str = "/home/<user>/.cache/huggingface/hub") -> str:
    """HF id 'org/name' → cache dir 'models--org--name'."""
    safe = model_id.replace("/", "--")
    return f"{hf_cache}/models--{safe}"


def _ssh_probe(
    node: Node,
    model_id_or_path: str,
    *,
    timeout_s: float = 12.0,
) -> dict:
    """Probe ``node`` for: no existing shim, MemAvailable headroom, weights present.

    Returns ``{"ok": bool, "reason": str, "free_gb": float}``. Never raises.
    """
    # If the model is a local path (merged checkpoint), check it as a path.
    # If it's an HF id (contains '/' but not starting with '/'), check the HF cache.
    if model_id_or_path.startswith("/"):
        weights_check = f"test -d {shlex.quote(model_id_or_path)}"
        weights_label = model_id_or_path
    else:
        cache_dir = _hf_cache_dir_for(model_id_or_path)
        # We accept either a top-level cache dir OR a snapshots subdir — older
        # HF versions stored differently. Either signal is sufficient.
        weights_check = f"test -d {shlex.quote(cache_dir)}"
        weights_label = cache_dir

    # One ssh round-trip with structured output. Use distinct prefixes so the
    # parser is unambiguous even if the remote shell is noisy.
    #
    # BUSY catches any inference server that holds GB10 RAM, not just our
    # own shim — the fleet hosts the gemma-4-31b "hive" (vllm), a
    # Qwen3.5-122B MoE (llama.cpp llama-server), and a Qwen2.5-72B Ray TP
    # cluster (run_cluster.sh + vllm). Picking a node that's already
    # running any of those would OOM-kill the user's daily LLM. Ollama's
    # idle ``serve`` daemon is intentionally NOT matched: it sits on
    # 11434 with ~0 RAM until called, so it's safe to co-locate.
    remote = "; ".join([
        "pgrep -af 'python.*-m openweight\\.serving_shim|"
        "python.*vllm\\.entrypoints|"
        "llama-server|"
        "run_cluster\\.sh' >/dev/null && echo BUSY=1 || echo BUSY=0",
        "awk '/MemAvailable/ {print \"MEM_KB=\"$2}' /proc/meminfo",
        f"{weights_check} && echo WEIGHTS=1 || echo WEIGHTS=0",
    ])

    cmd = [
        "ssh", "-i", FLEET_KEY, *SSH_OPTS,
        f"{FLEET_USER}@{node.ip}", remote,
    ]
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": f"ssh timeout to {node.name}", "free_gb": 0.0}
    except Exception as ex:  # noqa: BLE001
        return {"ok": False, "reason": f"ssh failed: {ex}", "free_gb": 0.0}

    if r.returncode != 0:
        return {"ok": False, "reason": f"ssh rc={r.returncode}: {r.stderr[:200]}", "free_gb": 0.0}

    parsed: dict[str, str] = {}
    for line in r.stdout.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            parsed[k.strip()] = v.strip()

    busy = parsed.get("BUSY") == "1"
    weights = parsed.get("WEIGHTS") == "1"
    mem_kb_str = parsed.get("MEM_KB", "0")
    try:
        free_gb = float(mem_kb_str) / 1_048_576.0
    except ValueError:
        free_gb = 0.0

    if busy:
        return {"ok": False, "reason": "inference server already running", "free_gb": free_gb}
    if not weights:
        return {"ok": False, "reason": f"weights missing: {weights_label}", "free_gb": free_gb}
    if free_gb < MIN_FREE_GB:
        return {"ok": False, "reason": f"low memory: {free_gb:.1f} GB free", "free_gb": free_gb}
    return {"ok": True, "reason": "ok", "free_gb": free_gb, "weights_present": weights}


def find_weights_source(
    model_id_or_path: str,
    *,
    peers_file: Path = FLEET_PEERS,
) -> Optional[Node]:
    """Return any fleet node (including self + excluded) that has the model cached.

    Used by ``pick_node(auto_replicate=True)`` to find a fanout source when
    every viable peer is missing weights. Searches local first (cheap, no
    ssh) then ssh-probes peers. Returns ``None`` if nobody has it.

    A self-hit returns a ``Node`` with ``ip="127.0.0.1"`` so the caller can
    distinguish "rsync from local" from "rsync from peer".
    """
    if model_id_or_path.startswith("/"):
        local_check_path = model_id_or_path
    else:
        local_check_path = _hf_cache_dir_for(model_id_or_path)
    if Path(local_check_path).is_dir():
        return Node(name=local_hostname_short(), ip="127.0.0.1")

    self_name = local_hostname_short()
    self_ips_set = local_ips()
    for node in load_peers(peers_file):
        if node.name == self_name or node.ip in self_ips_set:
            continue
        check = (
            f"test -d {shlex.quote(local_check_path)} && echo OK || echo MISS"
        )
        cmd = [
            "ssh", "-i", FLEET_KEY, *SSH_OPTS,
            f"{FLEET_USER}@{node.ip}", check,
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if r.returncode == 0 and "OK" in r.stdout:
                return node
        except Exception:  # noqa: BLE001
            continue
    return None


def replicate_weights(
    model_id_or_path: str,
    source: Node,
    target: Node,
    *,
    timeout_s: float = 14400.0,
    log_cb=None,
) -> None:
    """Rsync model weights from ``source`` to ``target`` via the fleet ssh key.

    For HF model ids: replicates ``~/.cache/huggingface/hub/models--ORG--NAME/``.
    For absolute local paths (merged checkpoints): replicates the path itself.

    If ``source.ip == "127.0.0.1"`` rsync runs locally with target as the
    remote destination. Otherwise rsync runs on the source node (via ssh) and
    pushes to the target — every fleet node carries the same fleet key, so
    source→target ssh works. Raises ``RuntimeError`` if rsync exits non-zero.

    Default timeout is 4 hours: compute↔compute throughput in practice is ~5 MB/s
    over ssh (single-stream, encryption-bound), so a 28 GB fp16 13B base
    takes ~90 min. ``--partial`` is set so an interrupted run resumes
    cleanly from the next call rather than restarting the chunk it was on.
    """
    if model_id_or_path.startswith("/"):
        src_path = dst_path = model_id_or_path.rstrip("/")
        parent = str(Path(src_path).parent)
    else:
        cache_dir = _hf_cache_dir_for(model_id_or_path)
        src_path = dst_path = cache_dir
        parent = "/home/<user>/.cache/huggingface/hub"

    ssh_e = (
        f"ssh -i {FLEET_KEY} -o StrictHostKeyChecking=no "
        f"-o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o BatchMode=yes"
    )

    mkparent = [
        "ssh", "-i", FLEET_KEY, *SSH_OPTS,
        f"{FLEET_USER}@{target.ip}", f"mkdir -p {shlex.quote(parent)}",
    ]
    subprocess.run(mkparent, check=False, capture_output=True, timeout=15)

    if log_cb:
        log_cb(f"replicating {model_id_or_path} from {source.name} → {target.name}")

    if source.ip == "127.0.0.1":
        cmd = [
            "rsync", "-a", "--partial", "--info=stats2",
            "-e", ssh_e,
            f"{src_path}/",
            f"{FLEET_USER}@{target.ip}:{dst_path}/",
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s,
        )
    else:
        remote_cmd = (
            f"rsync -a --partial --info=stats2 "
            f"-e {shlex.quote(ssh_e)} "
            f"{shlex.quote(src_path)}/ "
            f"{FLEET_USER}@{target.ip}:{shlex.quote(dst_path)}/"
        )
        cmd = [
            "ssh", "-i", FLEET_KEY, *SSH_OPTS,
            f"{FLEET_USER}@{source.ip}", remote_cmd,
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s,
        )

    if proc.returncode != 0:
        raise RuntimeError(
            f"rsync {model_id_or_path} {source.name}→{target.name} failed "
            f"rc={proc.returncode}: stderr={proc.stderr[:500]}"
        )
    if log_cb:
        log_cb(f"replication complete: {model_id_or_path} → {target.name}")


def pick_node(
    model_id_or_path: str,
    *,
    exclude_hosts: Iterable[str] = (),
    exclude_ips: Iterable[str] = (),
    peers_file: Path = FLEET_PEERS,
    candidate_filter: Optional[Iterable[str]] = None,
    auto_replicate: bool = False,
) -> tuple[Node, dict]:
    """Pick the first node that has the weights AND no shim AND memory headroom.

    Args:
        model_id_or_path: HF id (``org/name``) or absolute local path to a
            merged checkpoint. Determines what "weights present" means.
        exclude_hosts: hostnames (e.g., ``node-8``) to skip — typically the
            dashboard host itself + any node already reserved by another
            shim in this dashboard's registry.
        exclude_ips: IPs to skip — same idea, for callers that have IPs
            instead of names.
        candidate_filter: if given, only consider nodes whose name is in
            this set (e.g., restrict to compute cluster, skip <standalone>).
        auto_replicate: when True, if every viable peer is missing weights
            but a source exists somewhere in the fleet, return a (target,
            probe) pair where ``probe["needs_replication"]`` flags that the
            caller must rsync from ``probe["source_name"]`` before launching.
            The picker does NOT do the rsync itself — that lets the caller
            release any pick lock first and run replication in parallel.

    Returns:
        (Node, probe_result_dict) on success. Probe contains
        ``"needs_replication"`` (bool) and, when True, ``"source_name"`` +
        ``"source_ip"`` so the caller knows where to pull from.

    Raises:
        NoFreeNode: every candidate failed the probe and (if auto_replicate)
            no replication source/target combo was viable.
    """
    excluded_hosts = set(exclude_hosts)
    excluded_ips = set(exclude_ips)
    nodes = load_peers(peers_file)

    if candidate_filter is not None:
        keep = set(candidate_filter)
        nodes = [n for n in nodes if n.name in keep]

    rejected: list[str] = []
    weight_misses: list[tuple[Node, dict]] = []  # nodes that ONLY fail on weights
    for node in nodes:
        if node.name in excluded_hosts or node.ip in excluded_ips:
            rejected.append(f"{node.name}: excluded")
            continue
        probe = _ssh_probe(node, model_id_or_path)
        if probe["ok"]:
            logger.info("pick_node: chose %s (%s) — %s", node.name, node.ip, probe)
            return node, probe
        if probe["reason"].startswith("weights missing"):
            # The first probe already proved no inference server is running
            # (else the reason would be "inference server already running")
            # and gave us free_gb
            # in the same round-trip. Reuse it instead of a second ssh —
            # avoids a race window where a transient ssh failure on the
            # second probe drops every weight-miss out of the candidate set.
            if probe["free_gb"] >= MIN_FREE_GB:
                tprobe = {
                    "ok": True,
                    "reason": "ok",
                    "free_gb": probe["free_gb"],
                    "weights_present": False,
                }
                weight_misses.append((node, tprobe))
        rejected.append(f"{node.name}: {probe['reason']}")

    if not auto_replicate or not weight_misses:
        raise NoFreeNode(
            f"no free node has weights for {model_id_or_path!r}; "
            f"checked {len(nodes)} candidates: " + " | ".join(rejected)
        )

    source = find_weights_source(model_id_or_path, peers_file=peers_file)
    if source is None:
        raise NoFreeNode(
            f"no free node has weights for {model_id_or_path!r} and no "
            f"source has the weights cached anywhere in the fleet; "
            f"checked {len(nodes)} candidates: " + " | ".join(rejected)
        )

    for node, tprobe in weight_misses:
        if node.name == source.name:
            continue
        logger.info(
            "pick_node: chose %s (%s) for %s — needs replication from %s",
            node.name, node.ip, model_id_or_path, source.name,
        )
        return node, {
            **tprobe,
            "needs_replication": True,
            "source_name": source.name,
            "source_ip": source.ip,
        }

    raise NoFreeNode(
        f"no free node has weights for {model_id_or_path!r} and no replication "
        f"target was viable; checked {len(nodes)} candidates: " + " | ".join(rejected)
    )


def local_hostname_short() -> str:
    """Best-effort short hostname for self-exclusion.

    Returns ``node-8`` for ``compute-cluster-8``; passes through anything
    else so ``hostname`` already in fleet_peers.txt form (``node-8``) works.
    """
    import socket
    h = socket.gethostname()
    # Map "compute-cluster-N" → "node-N".
    if h.startswith("compute-cluster-"):
        return "node-" + h.removeprefix("compute-cluster-")
    return h


def local_ips() -> set[str]:
    """All IPv4 addresses bound on this host — for IP-based self-exclusion."""
    import socket
    ips: set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            family, _, _, _, sockaddr = info
            if family == socket.AF_INET:
                ips.add(sockaddr[0])
    except Exception:
        pass
    # Also try the SO_DGRAM trick to discover the primary outbound IP.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ips.add(s.getsockname()[0])
    except Exception:
        pass
    return ips


if __name__ == "__main__":
    import argparse, json, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--exclude-self", action="store_true",
                    help="exclude this host from candidates (default for production use)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    excl_hosts: set[str] = set()
    excl_ips: set[str] = set()
    if args.exclude_self:
        excl_hosts.add(local_hostname_short())
        excl_ips.update(local_ips())
    try:
        node, probe = pick_node(args.model, exclude_hosts=excl_hosts, exclude_ips=excl_ips)
        print(json.dumps({"node": node.name, "ip": node.ip, "probe": probe}, indent=2))
    except NoFreeNode as ex:
        print(json.dumps({"error": str(ex)}, indent=2))
        sys.exit(1)
