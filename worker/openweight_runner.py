"""Openweight job executors — run on the worker daemon.

One executor per job kind:

- ``run_sft(job)`` — LoRA/DoRA fine-tune via ``openweight.finetune.fine_tune``.
  Saves adapter to worker-local disk; returns ``SftResultMsg``.
- ``run_eval(job)`` — launch sglang on base+adapter (or merged model),
  invoke benchmark adapter against the OpenAI-compatible endpoint,
  tear down server, return ``EvalResultMsg``.
- ``run_merge(job)`` — mergekit via ``openweight.merge.run_mergekit``,
  return ``MergeResultMsg``.

The daemon is thin — it BLPOPs across queues and dispatches here.
Exceptions are caught and become error results so one bad job doesn't
kill the worker.
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from pathlib import Path
from typing import Any

from controller.dispatch import (
    EvalJob,
    EvalResultMsg,
    MergeJob,
    MergeResultMsg,
    SftJob,
    SftResultMsg,
)
from openweight.genome import SftConfig

logger = logging.getLogger(__name__)


def _coerce_sft_config(payload: dict[str, Any]) -> SftConfig:
    # SftConfig is a plain dataclass with typed fields; payload may have
    # been round-tripped through json already.
    return SftConfig(**payload)


def run_sft(job: SftJob, sieve_root: Path, worker_id: str) -> SftResultMsg:
    from openweight.finetune import fine_tune  # lazy — heavy torch import
    from openweight.replicate import broadcast_adapter, load_peers

    t0 = time.time()
    corpus_path = sieve_root / job.corpus_shard_path
    out_dir = sieve_root / job.adapter_out_dir
    try:
        cfg = _coerce_sft_config(job.sft_config)
        logger.info(
            "run_sft: member=%s base=%s rank=%d corpus=%s",
            job.member_id, cfg.base_model_id, cfg.rank, corpus_path,
        )
        adapter_dir = fine_tune(cfg, corpus_path, out_dir)

        # P2P broadcast so any peer can pick up the post-SFT eval.
        # Fail-soft: a local success is still a success. The eval side
        # can retry-pull from this worker if it lands on a node without
        # a replica (see openweight.replicate.pull_adapter).
        peers = load_peers()
        replica_status: dict[str, str] = {}
        if peers:
            try:
                replica_status = broadcast_adapter(
                    adapter_dir, peers=peers, min_replicas=0
                )
                ok_n = sum(1 for v in replica_status.values() if v.startswith("ok"))
                logger.info(
                    "run_sft replicate: member=%s replicas=%d/%d peers=%s",
                    job.member_id, ok_n, len(replica_status),
                    ",".join(replica_status.keys()),
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("run_sft replicate failed for %s: %s", job.member_id, e)
        else:
            logger.info("run_sft: no peers configured, skipping replication")

        return SftResultMsg(
            run_id=job.run_id,
            generation=job.generation,
            member_id=job.member_id,
            worker_id=worker_id,
            status="ok",
            adapter_path=str(adapter_dir),
            base_model_id=cfg.base_model_id,
            final_train_loss=None,  # trl's SFTTrainer logs come from the trainer; wire later
            eval_loss=None,
            wallclock_s=time.time() - t0,
        )
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        logger.error("run_sft failed for %s: %s\n%s", job.member_id, e, tb)
        return SftResultMsg(
            run_id=job.run_id, generation=job.generation, member_id=job.member_id,
            worker_id=worker_id, status="error",
            adapter_path=None, base_model_id=job.sft_config.get("base_model_id"),
            final_train_loss=None, eval_loss=None,
            wallclock_s=time.time() - t0,
            error=f"{type(e).__name__}: {e}",
        )


def run_eval(job: EvalJob, sieve_root: Path, worker_id: str) -> EvalResultMsg:
    from benchmarks.adapters import run_benchmark
    from openweight.inference import launch_sglang_server
    from openweight.replicate import pull_adapter

    t0 = time.time()
    try:
        ref = job.model_ref
        if ref.get("kind") == "lora":
            base = ref["base"]
            adapter = ref["adapter"]
            # P2P fallback: if the adapter isn't on our local FS yet
            # (replication lag, partial failure, cold worker), pull it
            # from whichever peer trained it.
            adapter_p = Path(adapter)
            if not (adapter_p / "adapter_config.json").exists():
                source = ref.get("source_peer")
                if source:
                    logger.info(
                        "run_eval: adapter %s missing locally, pulling from %s",
                        adapter, source,
                    )
                    if not pull_adapter(adapter_p, source):
                        raise FileNotFoundError(
                            f"adapter {adapter} missing and pull from {source} failed"
                        )
                else:
                    raise FileNotFoundError(
                        f"adapter {adapter} missing and no source_peer in model_ref"
                    )
            handle = launch_sglang_server(
                base_model_id=base, adapter_path=adapter,
            )
        elif ref.get("kind") == "merged":
            handle = launch_sglang_server(
                base_model_id=ref["path"], adapter_path=None,
            )
        elif ref.get("kind") == "base":
            handle = launch_sglang_server(
                base_model_id=ref["path"], adapter_path=None,
            )
        else:
            raise ValueError(f"unknown model_ref kind: {ref.get('kind')!r}")

        try:
            score, raw = run_benchmark(
                name=job.benchmark_name,
                base_url=handle.base_url,
                model_name=handle.served_model_name,
                proxy=job.proxy,
            )
        finally:
            handle.stop()

        return EvalResultMsg(
            run_id=job.run_id, generation=job.generation,
            member_id=job.member_id, benchmark_name=job.benchmark_name,
            worker_id=worker_id, status="ok",
            score=float(score), raw=raw,
            wallclock_s=time.time() - t0,
        )
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        logger.error(
            "run_eval failed for member=%s bench=%s: %s\n%s",
            job.member_id, job.benchmark_name, e, tb,
        )
        return EvalResultMsg(
            run_id=job.run_id, generation=job.generation,
            member_id=job.member_id, benchmark_name=job.benchmark_name,
            worker_id=worker_id, status="error",
            score=None, raw=None,
            wallclock_s=time.time() - t0,
            error=f"{type(e).__name__}: {e}",
        )


def run_merge(job: MergeJob, sieve_root: Path, worker_id: str) -> MergeResultMsg:
    from openweight.merge import run_mergekit

    t0 = time.time()
    out_dir = sieve_root / job.out_dir
    try:
        out_path = run_mergekit(job.merge_recipe, out_dir)
        return MergeResultMsg(
            run_id=job.run_id, generation=job.generation, member_id=job.member_id,
            worker_id=worker_id, status="ok",
            merged_checkpoint_path=str(out_path),
            wallclock_s=time.time() - t0,
        )
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        logger.error("run_merge failed for %s: %s\n%s", job.member_id, e, tb)
        return MergeResultMsg(
            run_id=job.run_id, generation=job.generation, member_id=job.member_id,
            worker_id=worker_id, status="error",
            merged_checkpoint_path=None,
            wallclock_s=time.time() - t0,
            error=f"{type(e).__name__}: {e}",
        )
