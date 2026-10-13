"""Parallel inference serving across GB10 cluster.

See ``docs/architecture-openweight.md`` §"Inference — parallel serving"
for the three modes. This module implements mode 1 (same-tokenizer-family
output ensembling) as the default, with hooks for modes 2 and 3.

Serving backend: ``transformers serve`` (built into transformers 5.x).
OpenAI-compatible FastAPI. No extra install needed on the compute stack —
transformers 5.3 is already present. sglang / vLLM remain as optional
backends behind the same ``launch_*_server`` interface once they're
validated on Blackwell arm64 + torch nightly.

For LoRA adapters we pre-merge into a standalone model directory before
launching the server (``peft.PeftModel.merge_and_unload``). The
"adapter-on-base" runtime path (``--lora-paths``) is sglang/vLLM-only
and skipped here.

Output-level ensembling is the default per the 2026-04-21 decision —
members produce text (via sampling), the coordinator combines at the
text layer (consensus / voting / judge). No per-token logit coordinator
is needed for the dev loop.
"""

from __future__ import annotations

import logging
import shutil
import socket
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

from openweight.genome import OpenweightMember

logger = logging.getLogger(__name__)


@dataclass
class MemberEndpoint:
    member_id: str
    host: str
    port: int
    tokenizer_family: str

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"


@dataclass
class EnsembleSpec:
    """The final artifact of an evolutionary run.

    Loaded by the serving coordinator; describes which GB10 serves each
    member and how outputs are combined.
    """

    ensemble_id: str
    members: list[MemberEndpoint]
    aggregation: str  # "output_vote" | "output_judge" | "logit_average"
    smoothing_eps: float = 0.0


@dataclass
class ServerHandle:
    """Live server subprocess + its endpoint, for orderly shutdown."""

    proc: subprocess.Popen
    host: str
    port: int
    base_url: str
    log_path: Path
    served_model_name: str

    def stop(self, timeout_s: float = 15.0) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5.0)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_ready(url: str, timeout_s: float, proc: subprocess.Popen) -> None:
    deadline = time.time() + timeout_s
    last_err: Optional[str] = None
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"serving subprocess exited with code {proc.returncode} "
                f"before becoming ready; last_err={last_err}"
            )
        try:
            with urllib.request.urlopen(url, timeout=2.0) as r:
                if 200 <= r.status < 500:  # 404 at /v1/models is fine — server is up
                    return
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            time.sleep(1.0)
    raise TimeoutError(f"serving endpoint {url} not ready after {timeout_s}s: {last_err}")


def merge_lora_to_dir(
    base_model_id: str, adapter_path: str, out_dir: Path,
) -> Path:
    """Merge a LoRA/DoRA adapter into a full model and save to ``out_dir``.

    ``transformers serve`` loads a plain HF model; there is no adapter
    hook. Merging costs ~30s for a 1.5B model and produces a directory
    that behaves exactly like a base model on disk. This also means the
    eval measures the merged behavior — the adapter's contribution is
    fully baked in, no runtime scaling surprises.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("merge_lora_to_dir: base=%s adapter=%s → %s",
                base_model_id, adapter_path, out_dir)
    tok = AutoTokenizer.from_pretrained(base_model_id)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16, device_map="cpu",
    )
    peft_model = PeftModel.from_pretrained(base, adapter_path)
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    tok.save_pretrained(str(out_dir))
    del base, peft_model, merged
    return out_dir


def launch_transformers_server(
    model_path: str,
    host: str = "127.0.0.1",
    port: Optional[int] = None,
    log_dir: Optional[Path] = None,
    served_model_name: Optional[str] = None,
    python_bin: Optional[str] = None,
    extra_args: Optional[list[str]] = None,
    ready_timeout_s: float = 1800.0,
    env_overrides: Optional[dict[str, str]] = None,
    on_spawn: Optional[Callable[[dict], None]] = None,
) -> ServerHandle:
    """Launch ``transformers serve`` against ``model_path``.

    ``model_path`` can be an HF id or a local directory (pre-merged
    adapter). Returns a ``ServerHandle`` whose ``.served_model_name`` is
    what clients must pass as the ``model`` field in requests.
    """
    import os
    import sys

    port = port or _find_free_port()
    log_dir = log_dir or Path("/tmp/sieve-serve")
    log_dir.mkdir(parents=True, exist_ok=True)
    served = served_model_name or Path(model_path).name.replace("/", "_")
    log_path = log_dir / f"{served}-{port}.log"

    # Always invoke via the current python so any deps installed in the
    # active venv (e.g. sieve-venv's ``openai`` on top of story venv's
    # transformers) are importable. Standalone ``transformers`` scripts
    # from sibling venvs don't share site-packages.
    # ``transformers serve`` 5.3.0 has a handler-binding bug on
    # /v1/chat/completions that makes it unusable regardless of fastapi
    # version (``body: dict`` is treated as query). We route through
    # ``openweight.serving_shim`` instead — same OpenAI surface, ~120
    # LOC, served in-subprocess via uvicorn.
    py = python_bin or sys.executable
    cmd = [py, "-m", "openweight.serving_shim",
           "--model-path", str(model_path),
           "--host", host, "--port", str(port)]

    if extra_args:
        cmd += list(extra_args)

    env = dict(os.environ)
    if env_overrides:
        env.update(env_overrides)

    logger.info("launch_transformers_server: %s (log=%s)", " ".join(cmd), log_path)
    lf = log_path.open("w")
    proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
    base_url = f"http://{host}:{port}/v1"
    if on_spawn is not None:
        try:
            on_spawn({
                "pid": proc.pid,
                "port": port,
                "host": host,
                "log_path": str(log_path),
                "base_url": base_url,
            })
        except Exception:
            logger.exception("on_spawn callback raised (non-fatal)")
    try:
        _wait_ready(f"{base_url}/models", ready_timeout_s, proc)
    except Exception:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        raise

    return ServerHandle(
        proc=proc, host=host, port=port, base_url=base_url,
        log_path=log_path, served_model_name=str(model_path),
    )


# Kept under the sglang name so callers (worker/openweight_runner.py)
# don't need to branch on backend. Internally delegates to
# ``launch_transformers_server`` after merging any LoRA adapter.
def launch_sglang_server(
    base_model_id: str,
    adapter_path: Optional[str] = None,
    host: str = "127.0.0.1",
    port: Optional[int] = None,
    log_dir: Optional[Path] = None,
    served_model_name: Optional[str] = None,
    extra_args: Optional[list[str]] = None,
    ready_timeout_s: float = 1800.0,
    merged_cache_root: Optional[Path] = None,
    on_spawn: Optional[Callable[[dict], None]] = None,
) -> ServerHandle:
    """Compat shim: returns a ``transformers serve`` handle.

    When ``adapter_path`` is set, the adapter is merged into
    ``{merged_cache_root}/{adapter_basename}/`` once (cached across
    evals) and that directory becomes the served model.
    """
    if adapter_path:
        cache_root = merged_cache_root or Path("/tmp/sieve-merged")
        merged_dir = cache_root / Path(adapter_path).name
        if not (merged_dir / "config.json").exists():
            merge_lora_to_dir(base_model_id, adapter_path, merged_dir)
        return launch_transformers_server(
            model_path=str(merged_dir),
            host=host, port=port, log_dir=log_dir,
            served_model_name=served_model_name,
            extra_args=extra_args, ready_timeout_s=ready_timeout_s,
            on_spawn=on_spawn,
        )
    return launch_transformers_server(
        model_path=base_model_id,
        host=host, port=port, log_dir=log_dir,
        served_model_name=served_model_name,
        extra_args=extra_args, ready_timeout_s=ready_timeout_s,
        on_spawn=on_spawn,
    )


def launch_for_member(
    member: OpenweightMember,
    host: str = "127.0.0.1",
    port: Optional[int] = None,
    **kwargs,
) -> ServerHandle:
    """Convenience: dispatch on member.kind to pick base vs merged path."""
    if member.kind == "sft":
        if not member.adapter_path:
            raise ValueError(f"sft member {member.id} has no adapter_path")
        return launch_sglang_server(
            base_model_id=member.base_model_id,
            adapter_path=member.adapter_path,
            host=host, port=port, **kwargs,
        )
    elif member.kind in ("merge", "lora_merge"):
        if not member.merged_checkpoint_path:
            raise ValueError(f"{member.kind} member {member.id} has no merged_checkpoint_path")
        return launch_sglang_server(
            base_model_id=member.merged_checkpoint_path,
            adapter_path=None,
            host=host, port=port, **kwargs,
        )
    else:  # pragma: no cover
        raise ValueError(f"unknown member.kind={member.kind}")


def ensemble_generate(
    spec: EnsembleSpec,
    prompt: str,
    max_new_tokens: int = 256,
) -> str:
    """Fan a prompt across members, aggregate at the output layer.

    Not needed for the dev loop (eval goes per-member). Stub kept for
    the serving-coordinator path.
    """
    raise NotImplementedError("output-layer aggregation — wire during serving-coord work")


def ensemble_generate_stream(
    spec: EnsembleSpec, prompt: str, max_new_tokens: int = 256,
) -> Iterable[str]:
    raise NotImplementedError


def cka_radius(members: list[OpenweightMember]) -> float:
    raise NotImplementedError


def soup_cluster(
    members: list[OpenweightMember], weights: Optional[list[float]] = None,
) -> str:
    raise NotImplementedError
