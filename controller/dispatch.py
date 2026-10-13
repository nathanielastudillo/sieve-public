"""Redis-backed job dispatch for parallel SIEVE training.

Architecture
------------
- One Redis instance on a cluster coordinator node (default: node-8).
- Orchestrator (on any node with line-of-sight to Redis) enqueues training
  jobs and waits for results.
- Worker daemons on each GB10 node BLPOP jobs, run ``train_model``, and
  push results back.

Key schema
----------
  sieve:jobs:train                LIST  — BLPOP source for workers
  sieve:results:{run_id}:{gen}:{member_id}   STRING(json)
  sieve:progress:{run_id}         LIST  — append-only human-readable log
  sieve:workers:heartbeat:{wid}   STRING  with EX TTL
  sieve:run:{run_id}:meta         HASH  — tokenizer, corpus, counts

All values are JSON. No binary blobs — checkpoints are written to each
worker's local disk and referenced by path (workers sync back separately).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Job / Result schemas
# ---------------------------------------------------------------------------


@dataclass
class TrainJob:
    """A single-member training job pulled off the queue by one worker."""

    run_id: str
    generation: int
    member_id: str
    model_config: dict[str, Any]
    train_config: dict[str, Any]
    # Worker resolves paths relative to its own $SIEVE_ROOT so we pass
    # relative strings here (e.g. "data/corpus_v007", "runs/smoke/checkpoints")
    corpus_dir: str
    checkpoint_dir: str
    # Optional — lets the worker tag its output for fan-in
    results_dir: Optional[str] = None
    # Monotonic ordering / debug
    submitted_at: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "TrainJob":
        return cls(**json.loads(s))


@dataclass
class TrainResultMsg:
    """Structured worker → orchestrator message."""

    run_id: str
    generation: int
    member_id: str
    worker_id: str
    status: str  # "ok" | "error"
    val_loss: Optional[float]
    final_loss: Optional[float]
    checkpoint_path: Optional[str]
    wallclock_s: float
    tokens_seen: int = 0
    error: Optional[str] = None
    finished_at: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "TrainResultMsg":
        return cls(**json.loads(s))


@dataclass
class SftResultMsg:
    run_id: str
    generation: int
    member_id: str
    worker_id: str
    status: str  # "ok" | "error"
    adapter_path: Optional[str]
    base_model_id: Optional[str]
    final_train_loss: Optional[float]
    eval_loss: Optional[float]  # held-out if available, else None
    wallclock_s: float
    error: Optional[str] = None
    finished_at: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "SftResultMsg":
        return cls(**json.loads(s))


@dataclass
class EvalResultMsg:
    run_id: str
    generation: int
    member_id: str
    benchmark_name: str
    worker_id: str
    status: str  # "ok" | "error"
    score: Optional[float]
    raw: Optional[dict[str, Any]]
    wallclock_s: float
    error: Optional[str] = None
    finished_at: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "EvalResultMsg":
        return cls(**json.loads(s))


@dataclass
class MergeResultMsg:
    run_id: str
    generation: int
    member_id: str
    worker_id: str
    status: str
    merged_checkpoint_path: Optional[str]
    wallclock_s: float
    error: Optional[str] = None
    finished_at: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "MergeResultMsg":
        return cls(**json.loads(s))


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


TRAIN_QUEUE = "sieve:jobs:train"

# Openweight track queues (2026-04-18). See docs/architecture-openweight.md
# §"Generation loop" step 3. All three carry JSON-encoded dicts.
SFT_QUEUE = "sieve:jobs:sft"
MERGE_QUEUE = "sieve:jobs:merge"
EVAL_QUEUE = "sieve:jobs:eval"


@dataclass
class SftJob:
    """Fine-tune a base model with a LoRA/DoRA config on a corpus shard."""

    run_id: str
    generation: int
    member_id: str
    sft_config: dict[str, Any]  # openweight.genome.SftConfig as dict
    corpus_shard_path: str  # worker-relative path
    adapter_out_dir: str  # worker-relative path
    submitted_at: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "SftJob":
        return cls(**json.loads(s))


@dataclass
class MergeJob:
    """Run a mergekit recipe to produce a merged checkpoint."""

    run_id: str
    generation: int
    member_id: str
    merge_recipe: dict[str, Any]  # openweight.genome.MergeRecipe as dict
    parent_refs: list[dict[str, str]]  # [{"member_id": ..., "path": ...}] or HF ids
    out_dir: str  # worker-relative
    submitted_at: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "MergeJob":
        return cls(**json.loads(s))


@dataclass
class EvalJob:
    """Evaluate one member against one named benchmark."""

    run_id: str
    generation: int
    member_id: str
    benchmark_name: str
    model_ref: dict[str, str]  # {"kind": "lora", "base": ..., "adapter": ...} or {"kind": "merged", "path": ...}
    tokenizer_family: str
    proxy: bool = False  # True = cheap MCQ subsample, full suite skipped
    submitted_at: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "EvalJob":
        return cls(**json.loads(s))


def make_sft_job(
    run_id: str,
    generation: int,
    member_id: str,
    sft_config: dict[str, Any],
    corpus_shard_path: str,
    adapter_out_dir: str,
) -> SftJob:
    return SftJob(
        run_id=run_id,
        generation=generation,
        member_id=member_id,
        sft_config=sft_config,
        corpus_shard_path=corpus_shard_path,
        adapter_out_dir=adapter_out_dir,
    )


def make_merge_job(
    run_id: str,
    generation: int,
    member_id: str,
    merge_recipe: dict[str, Any],
    parent_refs: list[dict[str, str]],
    out_dir: str,
) -> MergeJob:
    return MergeJob(
        run_id=run_id,
        generation=generation,
        member_id=member_id,
        merge_recipe=merge_recipe,
        parent_refs=parent_refs,
        out_dir=out_dir,
    )


def make_eval_job(
    run_id: str,
    generation: int,
    member_id: str,
    benchmark_name: str,
    model_ref: dict[str, str],
    tokenizer_family: str,
    proxy: bool = False,
) -> EvalJob:
    return EvalJob(
        run_id=run_id,
        generation=generation,
        member_id=member_id,
        benchmark_name=benchmark_name,
        model_ref=model_ref,
        tokenizer_family=tokenizer_family,
        proxy=proxy,
    )


class RedisDispatcher:
    """Thin wrapper around a redis-py client for SIEVE job dispatch.

    The orchestrator holds one of these and calls ``submit_generation``
    followed by ``wait_for_generation``. Workers hold one and call
    ``next_job`` / ``publish_result``.
    """

    def __init__(self, url: str):
        try:
            import redis  # lazy — only required when dispatch is used
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "redis-py not installed. `pip install redis` in the sieve venv."
            ) from e
        self.url = url
        self.r = redis.Redis.from_url(url, decode_responses=True)
        # Fail fast if unreachable so ops see it at startup, not mid-run.
        self.r.ping()

    # --- orchestrator side ---------------------------------------------

    def submit_generation(self, jobs: list[TrainJob]) -> None:
        """Atomically LPUSH all jobs for one generation."""
        if not jobs:
            return
        pipe = self.r.pipeline()
        for j in jobs:
            pipe.rpush(TRAIN_QUEUE, j.to_json())
        pipe.execute()
        logger.info("Enqueued %d training jobs on %s", len(jobs), TRAIN_QUEUE)

    def result_key(self, run_id: str, generation: int, member_id: str) -> str:
        return f"sieve:results:{run_id}:{generation}:{member_id}"

    def wait_for_generation(
        self,
        run_id: str,
        generation: int,
        member_ids: list[str],
        timeout_s: float = 3600.0,
        poll_interval_s: float = 2.0,
    ) -> dict[str, TrainResultMsg]:
        """Block until all members of a generation have reported, or timeout.

        Returns a dict member_id → TrainResultMsg. Missing members on
        timeout are represented with status="error".
        """
        deadline = time.time() + timeout_s
        pending = set(member_ids)
        out: dict[str, TrainResultMsg] = {}
        last_log = 0.0

        while pending and time.time() < deadline:
            # Pipelined MGET is cheap; no need for pub/sub yet
            keys = [self.result_key(run_id, generation, mid) for mid in pending]
            vals = self.r.mget(keys)
            for mid, raw in zip(list(pending), vals):
                if raw is None:
                    continue
                try:
                    msg = TrainResultMsg.from_json(raw)
                except Exception as e:
                    logger.warning("Malformed result for %s: %s", mid, e)
                    continue
                out[mid] = msg
                pending.discard(mid)

            now = time.time()
            if now - last_log > 15.0 and pending:
                logger.info(
                    "Generation %d: %d/%d done, still waiting on %s",
                    generation, len(out), len(member_ids),
                    sorted(list(pending))[:5],
                )
                last_log = now
            if pending:
                time.sleep(poll_interval_s)

        # Fill in errors for anyone who didn't report
        for mid in pending:
            out[mid] = TrainResultMsg(
                run_id=run_id, generation=generation, member_id=mid,
                worker_id="<timeout>", status="error",
                val_loss=None, final_loss=None, checkpoint_path=None,
                wallclock_s=0.0, error="timeout",
            )
        return out

    # --- openweight orchestrator side -----------------------------------

    def submit_sft_jobs(self, jobs: list[SftJob]) -> None:
        if not jobs:
            return
        pipe = self.r.pipeline()
        for j in jobs:
            pipe.rpush(SFT_QUEUE, j.to_json())
        pipe.execute()
        logger.info("Enqueued %d SFT jobs on %s", len(jobs), SFT_QUEUE)

    def submit_eval_jobs(self, jobs: list[EvalJob]) -> None:
        if not jobs:
            return
        pipe = self.r.pipeline()
        for j in jobs:
            pipe.rpush(EVAL_QUEUE, j.to_json())
        pipe.execute()
        logger.info("Enqueued %d eval jobs on %s", len(jobs), EVAL_QUEUE)

    def submit_merge_jobs(self, jobs: list[MergeJob]) -> None:
        if not jobs:
            return
        pipe = self.r.pipeline()
        for j in jobs:
            pipe.rpush(MERGE_QUEUE, j.to_json())
        pipe.execute()
        logger.info("Enqueued %d merge jobs on %s", len(jobs), MERGE_QUEUE)

    def sft_result_key(self, run_id: str, generation: int, member_id: str) -> str:
        return f"sieve:results:sft:{run_id}:{generation}:{member_id}"

    def eval_result_key(
        self, run_id: str, generation: int, member_id: str, benchmark: str,
    ) -> str:
        return f"sieve:results:eval:{run_id}:{generation}:{member_id}:{benchmark}"

    def merge_result_key(self, run_id: str, generation: int, member_id: str) -> str:
        return f"sieve:results:merge:{run_id}:{generation}:{member_id}"

    def wait_for_sft(
        self,
        run_id: str,
        generation: int,
        member_ids: list[str],
        timeout_s: float = 7200.0,
        poll_interval_s: float = 2.0,
    ) -> dict[str, SftResultMsg]:
        deadline = time.time() + timeout_s
        pending = set(member_ids)
        out: dict[str, SftResultMsg] = {}
        last_log = 0.0
        while pending and time.time() < deadline:
            keys = [self.sft_result_key(run_id, generation, mid) for mid in pending]
            vals = self.r.mget(keys)
            for mid, raw in zip(list(pending), vals):
                if raw is None:
                    continue
                out[mid] = SftResultMsg.from_json(raw)
                pending.discard(mid)
            now = time.time()
            if now - last_log > 30.0 and pending:
                logger.info(
                    "SFT gen %d: %d/%d done, still waiting on %s",
                    generation, len(out), len(member_ids),
                    sorted(list(pending))[:5],
                )
                last_log = now
            if pending:
                time.sleep(poll_interval_s)
        for mid in pending:
            out[mid] = SftResultMsg(
                run_id=run_id, generation=generation, member_id=mid,
                worker_id="<timeout>", status="error",
                adapter_path=None, base_model_id=None,
                final_train_loss=None, eval_loss=None,
                wallclock_s=0.0, error="timeout",
            )
        return out

    def wait_for_evals(
        self,
        run_id: str,
        generation: int,
        expected: list[tuple[str, str]],  # [(member_id, benchmark_name)]
        timeout_s: float = 7200.0,
        poll_interval_s: float = 2.0,
    ) -> dict[tuple[str, str], EvalResultMsg]:
        deadline = time.time() + timeout_s
        pending = set(expected)
        out: dict[tuple[str, str], EvalResultMsg] = {}
        last_log = 0.0
        while pending and time.time() < deadline:
            keys = [
                self.eval_result_key(run_id, generation, mid, bench)
                for (mid, bench) in pending
            ]
            vals = self.r.mget(keys)
            for (mid, bench), raw in zip(list(pending), vals):
                if raw is None:
                    continue
                out[(mid, bench)] = EvalResultMsg.from_json(raw)
                pending.discard((mid, bench))
            now = time.time()
            if now - last_log > 30.0 and pending:
                logger.info(
                    "Eval gen %d: %d/%d done, still waiting on %s",
                    generation, len(out), len(expected),
                    sorted(list(pending))[:5],
                )
                last_log = now
            if pending:
                time.sleep(poll_interval_s)
        for mid, bench in pending:
            out[(mid, bench)] = EvalResultMsg(
                run_id=run_id, generation=generation, member_id=mid,
                benchmark_name=bench, worker_id="<timeout>",
                status="error", score=None, raw=None,
                wallclock_s=0.0, error="timeout",
            )
        return out

    def wait_for_merges(
        self,
        run_id: str,
        generation: int,
        member_ids: list[str],
        timeout_s: float = 3600.0,
        poll_interval_s: float = 2.0,
    ) -> dict[str, MergeResultMsg]:
        deadline = time.time() + timeout_s
        pending = set(member_ids)
        out: dict[str, MergeResultMsg] = {}
        while pending and time.time() < deadline:
            keys = [self.merge_result_key(run_id, generation, mid) for mid in pending]
            vals = self.r.mget(keys)
            for mid, raw in zip(list(pending), vals):
                if raw is None:
                    continue
                out[mid] = MergeResultMsg.from_json(raw)
                pending.discard(mid)
            if pending:
                time.sleep(poll_interval_s)
        for mid in pending:
            out[mid] = MergeResultMsg(
                run_id=run_id, generation=generation, member_id=mid,
                worker_id="<timeout>", status="error",
                merged_checkpoint_path=None, wallclock_s=0.0, error="timeout",
            )
        return out

    # --- worker side ----------------------------------------------------

    def next_job(self, timeout_s: int = 30) -> Optional[TrainJob]:
        """Blocking pop. Returns None on empty-timeout."""
        res = self.r.blpop(TRAIN_QUEUE, timeout=timeout_s)
        if res is None:
            return None
        _key, raw = res
        return TrainJob.from_json(raw)

    # Multi-queue pop used by the openweight worker. Returns
    # (queue_name, raw_json) so the caller decides how to decode.
    def next_openweight_job(
        self,
        queues: list[str],
        timeout_s: int = 30,
    ) -> Optional[tuple[str, str]]:
        res = self.r.blpop(queues, timeout=timeout_s)
        if res is None:
            return None
        key, raw = res
        return (key, raw)

    def publish_result(self, msg: TrainResultMsg, ttl_s: int = 7 * 24 * 3600) -> None:
        key = self.result_key(msg.run_id, msg.generation, msg.member_id)
        self.r.set(key, msg.to_json(), ex=ttl_s)
        # Human-readable log for observability
        log_line = (
            f"[gen {msg.generation}] {msg.member_id[:8]} "
            f"status={msg.status} val_loss={msg.val_loss} "
            f"worker={msg.worker_id} wall={msg.wallclock_s:.1f}s"
        )
        self.r.rpush(f"sieve:progress:{msg.run_id}", log_line)
        self.r.ltrim(f"sieve:progress:{msg.run_id}", -2000, -1)

    def publish_sft_result(
        self, msg: SftResultMsg, ttl_s: int = 7 * 24 * 3600,
    ) -> None:
        key = self.sft_result_key(msg.run_id, msg.generation, msg.member_id)
        self.r.set(key, msg.to_json(), ex=ttl_s)
        log_line = (
            f"[gen {msg.generation} sft] {msg.member_id[:8]} "
            f"status={msg.status} adapter={msg.adapter_path} "
            f"worker={msg.worker_id} wall={msg.wallclock_s:.1f}s"
        )
        self.r.rpush(f"sieve:progress:{msg.run_id}", log_line)
        self.r.ltrim(f"sieve:progress:{msg.run_id}", -2000, -1)

    def publish_eval_result(
        self, msg: EvalResultMsg, ttl_s: int = 7 * 24 * 3600,
    ) -> None:
        key = self.eval_result_key(
            msg.run_id, msg.generation, msg.member_id, msg.benchmark_name,
        )
        self.r.set(key, msg.to_json(), ex=ttl_s)
        log_line = (
            f"[gen {msg.generation} eval/{msg.benchmark_name}] {msg.member_id[:8]} "
            f"status={msg.status} score={msg.score} "
            f"worker={msg.worker_id} wall={msg.wallclock_s:.1f}s"
        )
        self.r.rpush(f"sieve:progress:{msg.run_id}", log_line)
        self.r.ltrim(f"sieve:progress:{msg.run_id}", -2000, -1)

    def publish_merge_result(
        self, msg: MergeResultMsg, ttl_s: int = 7 * 24 * 3600,
    ) -> None:
        key = self.merge_result_key(msg.run_id, msg.generation, msg.member_id)
        self.r.set(key, msg.to_json(), ex=ttl_s)

    def heartbeat(self, worker_id: str, state: str = "idle", ttl_s: int = 30) -> None:
        key = f"sieve:workers:heartbeat:{worker_id}"
        self.r.set(key, json.dumps({"state": state, "ts": time.time()}), ex=ttl_s)

    def list_workers(self) -> list[tuple[str, dict]]:
        """Introspection: active workers + their state."""
        keys = list(self.r.scan_iter("sieve:workers:heartbeat:*"))
        out = []
        for k in keys:
            raw = self.r.get(k)
            if raw is None:
                continue
            try:
                out.append((k.rsplit(":", 1)[-1], json.loads(raw)))
            except Exception:
                continue
        return out

    # --- utilities ------------------------------------------------------

    def queue_depth(self) -> int:
        return int(self.r.llen(TRAIN_QUEUE))

    def drain_queue(self) -> int:
        """Emergency: clear pending jobs. Returns how many were dropped."""
        n = self.queue_depth()
        self.r.delete(TRAIN_QUEUE)
        return n


def make_job(
    run_id: str,
    generation: int,
    member_id: str,
    model_config: dict,
    train_config: dict,
    corpus_dir: str,
    checkpoint_dir: str,
    results_dir: Optional[str] = None,
) -> TrainJob:
    return TrainJob(
        run_id=run_id,
        generation=generation,
        member_id=member_id,
        model_config=dict(model_config),
        train_config=dict(train_config),
        corpus_dir=corpus_dir,
        checkpoint_dir=checkpoint_dir,
        results_dir=results_dir,
    )


def new_run_id() -> str:
    return uuid.uuid4().hex[:10]
