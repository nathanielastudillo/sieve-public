"""Launch ``serving_shim`` on a remote GB10 node over ssh.

Companion to ``inference.py``: that module Popens the shim locally; this
one ssh-launches it on a peer node and reaches it via an ssh local-port
forward. The returned ``RemoteServerHandle`` mirrors the local
``ServerHandle`` interface (``stop()``, ``base_url``, ``port``) so
callers can be agnostic about where the shim lives.

The remote shim binds to ``127.0.0.1`` on a free port; the dashboard
opens ``ssh -N -L <local>:127.0.0.1:<remote>`` and uses
``http://127.0.0.1:<local>/v1`` as ``base_url``. Tunneling sidesteps the
fleet's ufw/iptables INPUT-DROP that blocks LAN access to dynamic ports
— ssh on port 22 is allowed everywhere, so as long as we can ssh in we
can reach the shim. Logs are tailed back over ssh on demand by the
remote-log endpoint (see dashboard/server.py).
"""

from __future__ import annotations

import logging
import shlex
import socket
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from openweight.node_picker import FLEET_KEY, FLEET_USER, SSH_OPTS, Node

logger = logging.getLogger(__name__)

REMOTE_SIEVE_DIR = "/home/<user>/sieve"
REMOTE_LOG_DIR = "/tmp/sieve-serve"
REMOTE_MERGED_CACHE = "/tmp/sieve-merged"

# Local source-of-truth for the openweight package — pushed to the target
# right before each launch so a node with a stale or missing
# ``openweight/`` snapshot still ends up with the current shim code. The
# dir is small (~200 KB) so the rsync round-trip is cheap; ``-a`` keeps it
# incremental on subsequent calls.
LOCAL_OPENWEIGHT_DIR = "/home/<user>/sieve/openweight"

# Two venv conventions live on the fleet:
#   /home/<user>/sieve-venv/      → on node-8 (the dashboard host) and a few others
#   /home/<user>/venvs/voyage4nano → fleet-provisioned default (per fleet_health.sh)
# The launcher does shell-side detection so we don't depend on which one
# happens to be on the chosen node.
REMOTE_PYTHON_FALLBACK_CHAIN = (
    "/home/<user>/sieve-venv/bin/python",
    "/home/<user>/venvs/voyage4nano/bin/python",
)


def _python_picker_shell() -> str:
    """Shell expression that prints the first existing python from the chain.

    Used inline in the remote command — no extra ssh round-trip. Returns
    an ``if/elif/else`` chain (not a subshell starting with ``(``) so that
    wrapping in ``$(...)`` doesn't get misparsed as arithmetic ``$((...))``.
    """
    parts = ["if false; then :"]
    for p in REMOTE_PYTHON_FALLBACK_CHAIN:
        parts.append(f"elif [ -x {p} ]; then echo {p}")
    parts.append("else echo 'no usable python venv on remote' >&2; exit 1")
    parts.append("fi")
    return "; ".join(parts)


@dataclass
class RemoteServerHandle:
    """Live remote shim — interface-compatible with ``ServerHandle``.

    ``port`` is the *local* forwarded port (what the dashboard hits);
    ``remote_port`` is the actual bind on the GB10. ``base_url`` always
    points at ``127.0.0.1:port`` so OpenAI clients on the dashboard host
    can talk to it directly. ``stop()`` kills the remote shim pid via ssh
    and also terminates the local ssh-forward process.
    """

    node: Node
    port: int
    base_url: str
    served_model: str
    remote_pid: Optional[int]
    log_path: Path  # remote path; not directly readable from dashboard host
    tunnel_pid: Optional[int] = None
    remote_port: Optional[int] = None

    @property
    def host(self) -> str:
        return self.node.ip

    @property
    def served_model_name(self) -> str:
        return self.served_model

    def stop(self, timeout_s: float = 10.0) -> None:
        # Kill the remote shim first — even if the tunnel teardown trips,
        # we don't want to leak a 60-GB process on the GB10.
        if self.remote_pid is not None:
            cmd = [
                "ssh", "-i", FLEET_KEY, *SSH_OPTS,
                f"{FLEET_USER}@{self.node.ip}",
                f"kill {int(self.remote_pid)} 2>/dev/null; sleep 1; "
                f"kill -9 {int(self.remote_pid)} 2>/dev/null; true",
            ]
            try:
                subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
            except Exception:  # noqa: BLE001
                logger.exception("RemoteServerHandle.stop ssh failed for %s", self.node.name)
        # Local ssh -L process. SIGTERM is enough; ssh hangs up cleanly.
        if self.tunnel_pid is not None:
            try:
                import os
                import signal
                os.kill(int(self.tunnel_pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            except Exception:  # noqa: BLE001
                logger.exception("RemoteServerHandle.stop tunnel kill failed pid=%s", self.tunnel_pid)


def _find_local_free_port() -> int:
    """Pick an unused TCP port on the dashboard host (for the ssh -L bind)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _open_ssh_tunnel(
    node: Node, remote_port: int, *, ready_timeout_s: float = 15.0,
) -> tuple[int, int]:
    """Open ``ssh -N -L 127.0.0.1:LOCAL:127.0.0.1:REMOTE`` to ``node``.

    Returns ``(local_port, ssh_pid)``. Raises if the local listener
    doesn't come up within ``ready_timeout_s``. The tunnel survives until
    ``RemoteServerHandle.stop()`` SIGTERMs the ssh process (or the
    dashboard process exits — children die with the parent).
    """
    local_port = _find_local_free_port()
    cmd = [
        "ssh", "-i", FLEET_KEY, *SSH_OPTS,
        "-o", "ServerAliveInterval=15",
        "-o", "ExitOnForwardFailure=yes",
        "-N",
        "-L", f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}",
        f"{FLEET_USER}@{node.ip}",
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    deadline = time.time() + ready_timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            err = ""
            try:
                err = (proc.stderr.read() or b"").decode(errors="replace")[:300] if proc.stderr else ""
            except Exception:  # noqa: BLE001
                pass
            raise RuntimeError(
                f"ssh -L tunnel to {node.name} died before binding: {err!r}"
            )
        # Probe the local listener. ssh binds it before printing anything,
        # so a successful TCP connect to 127.0.0.1:local_port means the
        # forward is active. The forwarded request itself may still be
        # refused by the remote (shim not bound yet), but that's fine —
        # the readiness loop handles that.
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                s.connect(("127.0.0.1", local_port))
            return local_port, proc.pid
        except OSError:
            time.sleep(0.2)
    # Timed out — kill the ssh process so we don't leak.
    try:
        proc.terminate()
        proc.wait(timeout=2.0)
    except Exception:  # noqa: BLE001
        pass
    raise TimeoutError(
        f"ssh -L tunnel to {node.name} not bound after {ready_timeout_s}s"
    )


def _pick_remote_port(node: Node, timeout_s: float = 8.0) -> int:
    """Ask the remote node for an unused TCP port.

    We can't trust local ``_find_free_port`` — that picks a port on the
    dashboard host. The shim binds on the remote.
    """
    cmd = [
        "ssh", "-i", FLEET_KEY, *SSH_OPTS,
        f"{FLEET_USER}@{node.ip}",
        "python3 -c 'import socket; "
        "s=socket.socket(socket.AF_INET, socket.SOCK_STREAM); "
        "s.bind((\"0.0.0.0\",0)); print(s.getsockname()[1]); s.close()'",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if r.returncode != 0:
        raise RuntimeError(f"remote port pick failed on {node.name}: {r.stderr[:200]}")
    return int(r.stdout.strip())


def _wait_remote_ready(
    base_url: str, timeout_s: float, node: Node, remote_pid: int,
) -> None:
    """Poll ``base_url/models`` until 2xx/4xx, or until pid disappears, or timeout.

    Mirrors ``inference._wait_ready`` but checks pid liveness over ssh
    rather than via Popen.poll(). Cheap-ish: a pid check is one ssh per
    failed http poll, so when the model is loading slowly we keep
    pestering the remote ssh daemon. That's the cost of seeing the early
    "subprocess died" signal — without it, a crashed shim looks identical
    to a slow-loading shim until the timeout.
    """
    deadline = time.time() + timeout_s
    last_err: Optional[str] = None
    last_pid_check = 0.0
    while time.time() < deadline:
        # Fast path: try the http endpoint first.
        try:
            with urllib.request.urlopen(base_url, timeout=2.0) as r:
                if 200 <= r.status < 500:
                    return
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
        # Slow path: every ~10s, check the pid is still alive.
        if time.time() - last_pid_check > 10.0:
            last_pid_check = time.time()
            try:
                pid_cmd = [
                    "ssh", "-i", FLEET_KEY, *SSH_OPTS,
                    f"{FLEET_USER}@{node.ip}",
                    f"kill -0 {int(remote_pid)} 2>/dev/null && echo ALIVE || echo DEAD",
                ]
                pid_r = subprocess.run(pid_cmd, capture_output=True, text=True, timeout=8.0)
                if "DEAD" in pid_r.stdout:
                    raise RuntimeError(
                        f"remote shim pid {remote_pid} died on {node.name} "
                        f"before becoming ready; last_err={last_err}"
                    )
            except subprocess.TimeoutExpired:
                pass  # tolerate ssh hiccups
        time.sleep(2.0)
    raise TimeoutError(
        f"remote shim {base_url} on {node.name} not ready after {timeout_s}s: {last_err}"
    )


def _sync_openweight_pkg(node: Node, *, timeout_s: float = 30.0) -> None:
    """Push the local ``openweight/`` package to the target as Phase -1.

    Fleet nodes accumulate stale snapshots (some haven't been touched
    since exp006); without this, the shim launch dies with
    ``No module named openweight.serving_shim``. Cheap rsync — ~200 KB
    incremental — so we just always run it before launch.
    """
    if not Path(LOCAL_OPENWEIGHT_DIR).is_dir():
        logger.warning(
            "_sync_openweight_pkg: local %s missing — skipping pre-launch code sync",
            LOCAL_OPENWEIGHT_DIR,
        )
        return
    ssh_e = (
        f"ssh -i {FLEET_KEY} -o StrictHostKeyChecking=no "
        f"-o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o BatchMode=yes "
        f"-o ConnectTimeout=5"
    )
    cmd = [
        "rsync", "-a", "--delete", "--info=stats0",
        "-e", ssh_e,
        f"{LOCAL_OPENWEIGHT_DIR}/",
        f"{FLEET_USER}@{node.ip}:{REMOTE_SIEVE_DIR}/openweight/",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        logger.warning("_sync_openweight_pkg: rsync to %s timed out", node.name)
        return
    if r.returncode != 0:
        logger.warning(
            "_sync_openweight_pkg: rsync to %s rc=%d stderr=%s",
            node.name, r.returncode, r.stderr[:200],
        )


def launch_remote_shim(
    node: Node,
    model_path: str,
    *,
    adapter_path: Optional[str] = None,
    served_model_name: Optional[str] = None,
    ready_timeout_s: float = 1800.0,
    on_spawn: Optional[Callable[[dict], None]] = None,
) -> RemoteServerHandle:
    """Launch ``openweight.serving_shim`` on ``node`` via ssh.

    For ``adapter_path``: the merge happens on the remote node. The
    adapter directory is NOT shipped — caller must arrange for it (e.g.,
    rsync) to live at the same path on the chosen node before calling.
    Today the playground only serves ``kind=base`` so this is unused;
    when SFT serving lands, the spawning layer is responsible for adapter
    distribution.

    Phase -1 always rsyncs ``openweight/`` from the dashboard host to the
    target before launch — fleet nodes have drifted snapshots and this
    keeps the shim code in sync without a separate deploy step.
    """
    _sync_openweight_pkg(node)
    remote_port = _pick_remote_port(node)
    served = served_model_name or Path(model_path).name.replace("/", "_")
    log_path = f"{REMOTE_LOG_DIR}/{served}-{remote_port}.log"
    pid_path = f"{REMOTE_LOG_DIR}/{served}-{remote_port}.pid"

    if adapter_path:
        # Merge happens on the remote: same logic as ``inference.merge_lora_to_dir``.
        # We keep this path explicit so the launch command is one shell
        # invocation that returns a pid; doing it in two ssh calls would
        # leak intermediate failures.
        #
        # Quoting: the python script is built with ``{x!r}`` so each
        # argument is a Python string literal in single quotes, then the
        # whole script is fed to ``"$PY" -c`` as a single shell-quoted
        # token. Earlier versions wrapped the script in literal single
        # quotes inline, which collided with the inner ``repr()`` quotes
        # and produced unquoted bare-name args (Qwen/Qwen3-14B parsed as
        # division → SyntaxError).
        merged_dir = f"{REMOTE_MERGED_CACHE}/{Path(adapter_path).name}"
        py_code = (
            f"from openweight.inference import merge_lora_to_dir; "
            f"from pathlib import Path; "
            f"merge_lora_to_dir({model_path!r}, {adapter_path!r}, Path({merged_dir!r}))"
        )
        merge_step = (
            f"if [ ! -f {shlex.quote(merged_dir)}/config.json ]; then "
            f'  "$PY" -c {shlex.quote(py_code)} || exit 1; '
            f"fi; "
        )
        serve_path = merged_dir
    else:
        merge_step = ""
        serve_path = model_path

    # Bind 127.0.0.1 on the remote. Only the local sshd connects through
    # the forward, so LAN exposure is unneeded — and ufw on the compute nodes
    # drops dynamic LAN ports anyway.
    remote_cmd = (
        f"set -e; "
        f"PY=$({_python_picker_shell()}); "
        f"mkdir -p {REMOTE_LOG_DIR} {REMOTE_MERGED_CACHE}; "
        f"cd {REMOTE_SIEVE_DIR}; "
        f"{merge_step}"
        f"setsid nohup \"$PY\" -m openweight.serving_shim "
        f"  --model-path {shlex.quote(serve_path)} "
        f"  --host 127.0.0.1 --port {remote_port} "
        f"  > {shlex.quote(log_path)} 2>&1 < /dev/null & "
        f"echo $! > {shlex.quote(pid_path)}; "
        f"disown; "
        f"cat {shlex.quote(pid_path)}"
    )

    cmd = [
        "ssh", "-i", FLEET_KEY, *SSH_OPTS,
        "-o", "ServerAliveInterval=15",
        f"{FLEET_USER}@{node.ip}",
        remote_cmd,
    ]
    logger.info(
        "launch_remote_shim: %s → %s remote_port=%d", node.name, serve_path, remote_port,
    )
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60.0)
    if r.returncode != 0:
        raise RuntimeError(
            f"remote launch failed on {node.name}: rc={r.returncode} "
            f"stderr={r.stderr[:400]} stdout={r.stdout[:200]}"
        )
    pid_line = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    try:
        remote_pid = int(pid_line)
    except ValueError:
        raise RuntimeError(
            f"remote launch on {node.name} returned no pid (output: {r.stdout!r})"
        )

    # Open the ssh local-port forward so the dashboard can reach the
    # remote 127.0.0.1:remote_port through 127.0.0.1:local_port.
    try:
        local_port, tunnel_pid = _open_ssh_tunnel(node, remote_port)
    except Exception:
        # Shim is up but we can't tunnel to it — kill the orphan.
        _kill_remote_orphan(node, remote_pid)
        raise

    base_url = f"http://127.0.0.1:{local_port}/v1"
    logger.info(
        "launch_remote_shim: tunnel %d→%s:%d (ssh pid=%d)",
        local_port, node.name, remote_port, tunnel_pid,
    )

    if on_spawn is not None:
        try:
            on_spawn({
                "pid": remote_pid,
                "port": local_port,
                "remote_port": remote_port,
                "host": node.ip,
                "node": node.name,
                "log_path": log_path,
                "base_url": base_url,
                "tunnel_pid": tunnel_pid,
            })
        except Exception:
            logger.exception("on_spawn callback raised (non-fatal)")

    try:
        _wait_remote_ready(f"{base_url}/models", ready_timeout_s, node, remote_pid)
    except Exception:
        # Tear down the orphan if it's still alive — don't leak a 60-GB
        # process on the remote node, and don't leak a dangling ssh -L.
        _kill_remote_orphan(node, remote_pid)
        try:
            import os, signal
            os.kill(tunnel_pid, signal.SIGTERM)
        except Exception:  # noqa: BLE001
            pass
        raise

    return RemoteServerHandle(
        node=node,
        port=local_port,
        base_url=base_url,
        served_model=str(serve_path),
        remote_pid=remote_pid,
        log_path=Path(log_path),
        tunnel_pid=tunnel_pid,
        remote_port=remote_port,
    )


def _kill_remote_orphan(node: Node, remote_pid: int) -> None:
    """Best-effort ssh-kill of a remote pid. Used for cleanup-on-failure paths."""
    try:
        kill_cmd = [
            "ssh", "-i", FLEET_KEY, *SSH_OPTS,
            f"{FLEET_USER}@{node.ip}",
            f"kill {remote_pid} 2>/dev/null; sleep 1; kill -9 {remote_pid} 2>/dev/null; true",
        ]
        subprocess.run(kill_cmd, capture_output=True, text=True, timeout=10.0)
    except Exception:  # noqa: BLE001
        logger.exception("cleanup-on-fail kill failed for pid=%s on %s", remote_pid, node.name)


@dataclass
class DiscoveredShim:
    """A live ``serving_shim`` found on the fleet but not in the dashboard registry.

    Used by ``adopt_remote_shim`` to re-attach orphans left over from a
    dashboard restart instead of relaunching them — preserves loaded model
    weight RAM and avoids the launch latency.
    """

    node: Node
    model_path: str
    adapter_path: Optional[str]
    remote_port: int
    remote_pid: int
    log_path: str


def _parse_shim_cmdline(line: str) -> Optional[dict]:
    """Parse a ``pgrep -af`` line for ``serving_shim`` args. ``None`` if unparseable.

    Lines look like:
    ``PID /path/python -m openweight.serving_shim --model-path X --host 127.0.0.1 --port P [--lora-paths A]``
    """
    parts = line.split()
    if not parts or "openweight.serving_shim" not in line:
        return None
    try:
        pid = int(parts[0])
    except ValueError:
        return None
    out: dict = {"pid": pid, "model_path": None, "port": None, "adapter_path": None}
    i = 1
    while i < len(parts):
        a = parts[i]
        nxt = parts[i + 1] if i + 1 < len(parts) else None
        if a == "--model-path" and nxt is not None:
            out["model_path"] = nxt; i += 2; continue
        if a == "--port" and nxt is not None:
            try:
                out["port"] = int(nxt)
            except ValueError:
                pass
            i += 2; continue
        if a == "--lora-paths" and nxt is not None:
            out["adapter_path"] = nxt; i += 2; continue
        i += 1
    if out["model_path"] is None or out["port"] is None:
        return None
    return out


def discover_remote_shims(peers_file: Optional[Path] = None) -> list[DiscoveredShim]:
    """Scan the fleet for live ``serving_shim`` processes via parallel ssh + pgrep.

    Returns one entry per (node, pid). Probes run concurrently with a
    bounded thread pool so a 21-node fleet finishes in roughly the slowest
    single-ssh latency, not the sum.
    """
    from concurrent.futures import ThreadPoolExecutor
    from openweight.node_picker import FLEET_PEERS, load_peers

    nodes = load_peers(peers_file or FLEET_PEERS)

    def probe(node: Node) -> list[DiscoveredShim]:
        cmd = [
            "ssh", "-i", FLEET_KEY, *SSH_OPTS,
            f"{FLEET_USER}@{node.ip}",
            "pgrep -af 'python.*-m openweight\\.serving_shim' || true",
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=8.0)
        except Exception:  # noqa: BLE001
            return []
        found: list[DiscoveredShim] = []
        for line in r.stdout.splitlines():
            parsed = _parse_shim_cmdline(line)
            if parsed is None:
                continue
            served = Path(parsed["model_path"]).name.replace("/", "_")
            log_path = f"{REMOTE_LOG_DIR}/{served}-{parsed['port']}.log"
            found.append(DiscoveredShim(
                node=node,
                model_path=parsed["model_path"],
                adapter_path=parsed["adapter_path"],
                remote_port=parsed["port"],
                remote_pid=parsed["pid"],
                log_path=log_path,
            ))
        return found

    out: list[DiscoveredShim] = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        for shims in ex.map(probe, nodes):
            out.extend(shims)
    return out


def adopt_remote_shim(
    d: DiscoveredShim,
    *,
    ready_check_s: float = 8.0,
) -> RemoteServerHandle:
    """Attach to an already-running ``serving_shim`` — tunnel + handle, no relaunch.

    Companion to ``launch_remote_shim``: opens an ``ssh -L`` forward to the
    discovered shim and verifies it answers ``/v1/models`` within
    ``ready_check_s``. Used to recover orphans across a dashboard restart
    so loaded weights aren't re-paged from scratch.
    """
    local_port, tunnel_pid = _open_ssh_tunnel(d.node, d.remote_port)
    base_url = f"http://127.0.0.1:{local_port}/v1"

    deadline = time.time() + ready_check_s
    healthy = False
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/models", timeout=2.0) as r:
                if 200 <= r.status < 500:
                    healthy = True
                    break
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    if not healthy:
        try:
            import os
            import signal
            os.kill(tunnel_pid, signal.SIGTERM)
        except Exception:  # noqa: BLE001
            pass
        raise RuntimeError(
            f"adopt_remote_shim: shim on {d.node.name}:{d.remote_port} did not "
            f"respond to /v1/models within {ready_check_s}s"
        )

    return RemoteServerHandle(
        node=d.node,
        port=local_port,
        base_url=base_url,
        served_model=d.model_path,
        remote_pid=d.remote_pid,
        log_path=Path(d.log_path),
        tunnel_pid=tunnel_pid,
        remote_port=d.remote_port,
    )


def tail_remote_log(node: Node, log_path: str, tail: int = 40) -> list[str]:
    """ssh-tail the remote shim log. Best-effort, non-raising."""
    tail = max(1, min(int(tail), 400))
    cmd = [
        "ssh", "-i", FLEET_KEY, *SSH_OPTS,
        f"{FLEET_USER}@{node.ip}",
        f"tail -n {tail} {shlex.quote(log_path)} 2>/dev/null || true",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=8.0)
    except Exception:  # noqa: BLE001
        return []
    return [ln.rstrip() for ln in r.stdout.splitlines() if ln.strip()]
