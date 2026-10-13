"""P2P adapter replication across the compute fleet.

Adapters live on the local filesystem of whichever worker trained them.
Eval jobs get scheduled to *some* worker, not necessarily the trainer.
To make any worker a valid eval target we rsync each freshly-trained
adapter to every peer at the end of ``run_sft``.

The rsync uses ``~/.ssh/id_sieve_fleet`` (a shared key distributed to
every compute node — see ``scripts/fleet_deploy.sh``). Peer set is read
from ``SIEVE_FLEET_PEERS`` (comma-separated host list); if unset, it
falls back to the hostnames in ``scripts/fleet_peers.txt`` minus the
local hostname.

Failure policy: we log per-peer failures but only raise if
``replica_count < min_replicas``. Default ``min_replicas=1`` — i.e. at
least one other node must have a copy before we call SFT done.
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_KEY = Path.home() / ".ssh" / "id_sieve_fleet"
DEFAULT_USER = "a"
DEFAULT_PEERS_FILE = Path.home() / "sieve" / "scripts" / "fleet_peers.txt"


def load_peers() -> list[str]:
    env = os.environ.get("SIEVE_FLEET_PEERS", "").strip()
    if env:
        return [p.strip() for p in env.split(",") if p.strip()]
    if DEFAULT_PEERS_FILE.exists():
        return [
            line.split()[1] if len(line.split()) >= 2 else line.strip()
            for line in DEFAULT_PEERS_FILE.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    return []


def _local_ips() -> set[str]:
    ips = {"127.0.0.1", "::1"}
    try:
        host = socket.gethostname()
        ips.add(host)
        for info in socket.getaddrinfo(host, None):
            ips.add(info[4][0])
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["hostname", "-I"], capture_output=True, text=True, timeout=5, check=False
        ).stdout
        ips.update(out.split())
    except Exception:
        pass
    return ips


def _is_self(peer: str, self_ips: set[str]) -> bool:
    return peer in self_ips or peer == socket.gethostname()


def _rsync_one(
    adapter_dir: Path,
    peer: str,
    key: Path,
    user: str,
    timeout_s: int,
) -> tuple[str, bool, str]:
    remote = f"{user}@{peer}:{adapter_dir}/"
    cmd = [
        "rsync",
        "-az",
        "--delete",
        "--mkpath",
        "-e",
        f"ssh -i {key} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=8",
        f"{adapter_dir}/",
        remote,
    ]
    t0 = time.time()
    try:
        res = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, check=False
        )
        ok = res.returncode == 0
        msg = f"{time.time() - t0:.1f}s" if ok else res.stderr.strip()[:200]
        return peer, ok, msg
    except subprocess.TimeoutExpired:
        return peer, False, f"timeout>{timeout_s}s"
    except Exception as e:
        return peer, False, f"exc:{e}"


def broadcast_adapter(
    adapter_dir: Path,
    peers: list[str] | None = None,
    key: Path = DEFAULT_KEY,
    user: str = DEFAULT_USER,
    per_peer_timeout_s: int = 300,
    min_replicas: int = 1,
) -> dict[str, str]:
    """rsync ``adapter_dir`` to each peer in parallel.

    Returns a dict ``{peer: status}`` where status is ``"ok(Xs)"`` or the
    short error. Raises ``RuntimeError`` if fewer than ``min_replicas``
    peers succeeded.
    """
    if peers is None:
        peers = load_peers()
    self_ips = _local_ips()
    targets = [p for p in peers if not _is_self(p, self_ips)]
    if not targets:
        logger.warning("broadcast_adapter: no peers (self=%s, loaded=%d)", self_ips, len(peers))
        return {}

    logger.info(
        "broadcast_adapter: %s → %d peers (key=%s)", adapter_dir, len(targets), key
    )
    results: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=min(len(targets), 16)) as ex:
        futs = {
            ex.submit(_rsync_one, adapter_dir, p, key, user, per_peer_timeout_s): p
            for p in targets
        }
        for fut in as_completed(futs):
            peer, ok, msg = fut.result()
            results[peer] = f"ok({msg})" if ok else f"err:{msg}"

    ok_count = sum(1 for v in results.values() if v.startswith("ok"))
    logger.info(
        "broadcast_adapter done: %d/%d peers ok", ok_count, len(targets)
    )
    if ok_count < min_replicas:
        failed = {p: v for p, v in results.items() if not v.startswith("ok")}
        raise RuntimeError(
            f"broadcast_adapter: only {ok_count} replicas (need {min_replicas}); failures={failed}"
        )
    return results


def pull_adapter(
    adapter_dir: Path,
    source_peer: str,
    key: Path = DEFAULT_KEY,
    user: str = DEFAULT_USER,
    timeout_s: int = 300,
) -> bool:
    """Pull an adapter from a specific peer. Used by eval as a fallback
    when the local copy is missing and we know where it was trained."""
    remote = f"{user}@{source_peer}:{adapter_dir}/"
    cmd = [
        "rsync",
        "-az",
        "--mkpath",
        "-e",
        f"ssh -i {key} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=8",
        remote,
        f"{adapter_dir}/",
    ]
    try:
        res = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, check=False
        )
        if res.returncode == 0:
            logger.info("pull_adapter: %s ← %s ok", adapter_dir, source_peer)
            return True
        logger.warning(
            "pull_adapter: %s ← %s failed: %s", adapter_dir, source_peer, res.stderr.strip()[:200]
        )
        return False
    except Exception as e:
        logger.warning("pull_adapter: exception %s", e)
        return False
