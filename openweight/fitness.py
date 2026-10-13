"""Fitness evaluation for openweight members.

See ``docs/architecture-openweight.md`` §"Generation loop" step 4 and
§"Compute scheduling" (surrogate screening). This module:

1. Dispatches eval jobs into ``sieve:jobs:eval``, one per (member,
   benchmark) pair.
2. Collects per-benchmark scores from ``wait_for_evals``.
3. Aggregates them into a ``FitnessVector`` for NSGA-II.

Benchmark implementations live under ``benchmarks/adapters/`` and are
addressed by string name (e.g. ``"ifeval"``, ``"smoke_mcq"``). See
``KNOWN_BENCHMARKS`` for the authoritative list.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Optional

from controller.dispatch import (
    EvalResultMsg,
    RedisDispatcher,
    make_eval_job,
)
from openweight.genome import FitnessVector, OpenweightMember

logger = logging.getLogger(__name__)


# Authoritative list of benchmarks the fitness aggregator will
# enqueue. Adding a benchmark is two steps: add it here and add an
# adapter under ``benchmarks/adapters/``.
KNOWN_BENCHMARKS: set[str] = {
    "smoke_mcq",
    "ifeval",
    "hellaswag",
    "mmlu_pro",
    "gpqa_diamond",
}


@dataclass
class BenchmarkResult:
    member_id: str
    benchmark_name: str
    score: float
    wallclock_s: float
    raw: dict


@dataclass
class FitnessConfig:
    benchmarks: list[str]
    benchmark_weights: dict[str, float]
    diversity_prompt_set: Optional[str] = None
    calibration_benchmark: Optional[str] = None

    def __post_init__(self) -> None:
        unknown = [b for b in self.benchmarks if b not in KNOWN_BENCHMARKS]
        if unknown:
            raise ValueError(
                f"unknown benchmarks in FitnessConfig: {unknown} "
                f"(known: {sorted(KNOWN_BENCHMARKS)})"
            )


def _member_ref(member: OpenweightMember) -> dict[str, str]:
    if member.kind == "sft":
        if not member.adapter_path:
            raise ValueError(f"member {member.id}: adapter_path not set")
        return {"kind": "lora", "base": member.base_model_id,
                "adapter": member.adapter_path}
    if member.kind in ("merge", "lora_merge"):
        if not member.merged_checkpoint_path:
            raise ValueError(f"member {member.id}: merged_checkpoint_path not set")
        return {"kind": "merged", "path": member.merged_checkpoint_path}
    raise ValueError(f"unknown member.kind={member.kind!r}")


def enqueue_member_evals(
    dispatcher: RedisDispatcher,
    member: OpenweightMember,
    cfg: FitnessConfig,
    run_id: str,
    proxy: bool = False,
) -> list[tuple[str, str]]:
    """Enqueue one eval job per benchmark. Returns (member_id, bench) pairs."""
    ref = _member_ref(member)
    jobs = []
    expected: list[tuple[str, str]] = []
    for bench in cfg.benchmarks:
        job = make_eval_job(
            run_id=run_id,
            generation=member.generation,
            member_id=member.id,
            benchmark_name=bench,
            model_ref=ref,
            tokenizer_family=member.tokenizer_family,
            proxy=proxy,
        )
        jobs.append(job)
        expected.append((member.id, bench))
    dispatcher.submit_eval_jobs(jobs)
    return expected


def collect_member_results(
    dispatcher: RedisDispatcher,
    run_id: str,
    generation: int,
    expected: list[tuple[str, str]],
    timeout_s: float = 7200.0,
) -> dict[tuple[str, str], EvalResultMsg]:
    return dispatcher.wait_for_evals(
        run_id=run_id, generation=generation,
        expected=expected, timeout_s=timeout_s,
    )


def aggregate_fitness(
    member: OpenweightMember,
    bench_results: list[BenchmarkResult],
    diversity_score: float,
    calibration_ece: float,
    latency_ms: float,
    memory_gb: float,
    cfg: FitnessConfig,
) -> FitnessVector:
    """Weighted-sum benchmark agg + passthrough of the other objectives."""
    weighted = 0.0
    total_w = 0.0
    for r in bench_results:
        w = cfg.benchmark_weights.get(r.benchmark_name, 1.0)
        weighted += w * r.score
        total_w += w
    bench_agg = weighted / total_w if total_w > 0 else 0.0
    return FitnessVector(
        bench_agg=bench_agg,
        diversity=diversity_score,
        calibration_ece=calibration_ece,
        latency_ms=latency_ms,
        memory_gb=memory_gb,
    )


def population_diversity(
    members: list[OpenweightMember], prompt_set_path: str,
) -> dict[str, float]:
    """Pairwise JSD over a held-out prompt set; attribute per-member.

    Not wired in Gen-0 — returns zeros so the FitnessVector populates but
    NSGA-II has no useful diversity signal yet. Implement once the
    ensemble coordinator is up.
    """
    return {m.id: 0.0 for m in members}


def surrogate_screen(
    member: OpenweightMember, threshold: float = 0.5,
) -> Optional[BenchmarkResult]:
    """Cheap proxy eval (hook point). Not wired for Gen-0."""
    return None
