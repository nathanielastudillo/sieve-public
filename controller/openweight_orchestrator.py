"""Openweight Gen-0 orchestrator.

End-to-end loop:

    seed_roster → SftJob × N → wait → EvalJob × (N × |benchmarks|)
                → wait → FitnessVector → lineage.db → (optional) survival

Gen-0 has no evolution operators — it's just "fine-tune every seed,
benchmark every fine-tune, record." That's the smoke-and-Gen-0 path
the user is pushing for: "actually train, run, monitor, and evaluate."

Later generations layer mutation/crossover/immigration on top; the
selection and reproduction helpers are sketched as ``TODO`` so the
shape is visible but they don't block the dev loop.

Resume semantics: on startup, we read any already-published SFT and
eval result keys from Redis before enqueuing. A killed orchestrator
can be restarted with the same ``run_id`` and generation index, and
completed members skip.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

# Allow running from repo root
_here = Path(__file__).resolve().parent
if str(_here.parent) not in sys.path:
    sys.path.insert(0, str(_here.parent))

from controller.dispatch import (  # noqa: E402
    EvalResultMsg,
    RedisDispatcher,
    SftResultMsg,
    make_eval_job,
    make_sft_job,
    new_run_id,
)
from controller.lineage import LineageDB  # noqa: E402
from openweight.fitness import (  # noqa: E402
    BenchmarkResult,
    FitnessConfig,
    aggregate_fitness,
    population_diversity,
)
from openweight.genome import (  # noqa: E402
    OpenweightMember,
    SftConfig,
    new_member_id,
)
from openweight.seed_roster import SeedModel, pick_roster  # noqa: E402

logger = logging.getLogger("sieve.openweight")


DEFAULT_SFT_HYPERS = dict(
    adapter_kind="lora",
    rank=16,
    alpha=32,
    dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    learning_rate=2e-4,
    warmup_ratio=0.03,
    weight_decay=0.0,
    epochs=1,
    batch_size=1,
    max_seq_len=2048,
    optimizer="adamw_torch",
    precision="bf16",
)


def build_sft_config(seed: SeedModel, shard_id: str, rng_seed: int) -> SftConfig:
    return SftConfig(
        base_model_id=seed.hf_id,
        corpus_shard_id=shard_id,
        seed=rng_seed,
        **DEFAULT_SFT_HYPERS,
    )


def build_members_for_roster(
    roster: list[SeedModel],
    shard_id: str,
    generation: int,
    base_seed: int = 1729,
) -> list[OpenweightMember]:
    members: list[OpenweightMember] = []
    for i, seed in enumerate(roster):
        sft = build_sft_config(seed, shard_id=shard_id, rng_seed=base_seed + i)
        members.append(OpenweightMember(
            id=new_member_id(),
            generation=generation,
            kind="sft",
            parent_ids=[],
            lineage=[],
            tokenizer_family=seed.tokenizer_family,
            base_model_id=seed.hf_id,
            sft_config=sft,
        ))
    return members


def _existing_sft_results(
    dispatcher: RedisDispatcher,
    run_id: str,
    generation: int,
    member_ids: list[str],
) -> dict[str, SftResultMsg]:
    """Scan Redis for already-published SFT results (resume support)."""
    out: dict[str, SftResultMsg] = {}
    keys = [dispatcher.sft_result_key(run_id, generation, mid) for mid in member_ids]
    vals = dispatcher.r.mget(keys)
    for mid, raw in zip(member_ids, vals):
        if raw is not None:
            try:
                out[mid] = SftResultMsg.from_json(raw)
            except Exception:
                pass
    return out


def _existing_eval_results(
    dispatcher: RedisDispatcher,
    run_id: str,
    generation: int,
    expected: list[tuple[str, str]],
) -> dict[tuple[str, str], EvalResultMsg]:
    out: dict[tuple[str, str], EvalResultMsg] = {}
    keys = [
        dispatcher.eval_result_key(run_id, generation, mid, bench)
        for (mid, bench) in expected
    ]
    vals = dispatcher.r.mget(keys)
    for (mid, bench), raw in zip(expected, vals):
        if raw is not None:
            try:
                out[(mid, bench)] = EvalResultMsg.from_json(raw)
            except Exception:
                pass
    return out


def run_sft_stage(
    dispatcher: RedisDispatcher,
    run_id: str,
    members: list[OpenweightMember],
    corpus_shard_path: str,
    adapter_root: str,
    sft_timeout_s: float,
) -> dict[str, SftResultMsg]:
    generation = members[0].generation if members else 0
    existing = _existing_sft_results(
        dispatcher, run_id, generation, [m.id for m in members],
    )
    pending_members = [m for m in members if m.id not in existing]
    if existing:
        logger.info(
            "sft: %d/%d results already in Redis, resuming with %d pending",
            len(existing), len(members), len(pending_members),
        )
    jobs = []
    for m in pending_members:
        assert m.sft_config is not None
        adapter_out = f"{adapter_root}/{m.id}"
        jobs.append(make_sft_job(
            run_id=run_id,
            generation=m.generation,
            member_id=m.id,
            sft_config=asdict(m.sft_config),
            corpus_shard_path=corpus_shard_path,
            adapter_out_dir=adapter_out,
        ))
    dispatcher.submit_sft_jobs(jobs)
    fresh = dispatcher.wait_for_sft(
        run_id=run_id, generation=generation,
        member_ids=[m.id for m in pending_members],
        timeout_s=sft_timeout_s,
    )
    return {**existing, **fresh}


def run_eval_stage(
    dispatcher: RedisDispatcher,
    run_id: str,
    generation: int,
    members: list[OpenweightMember],
    benchmarks: list[str],
    eval_timeout_s: float,
    proxy: bool = False,
) -> dict[tuple[str, str], EvalResultMsg]:
    expected = [
        (m.id, b) for m in members for b in benchmarks
        if m.adapter_path or m.merged_checkpoint_path
    ]
    existing = _existing_eval_results(dispatcher, run_id, generation, expected)
    pending = [pair for pair in expected if pair not in existing]
    if existing:
        logger.info(
            "eval: %d/%d results already in Redis, resuming with %d pending",
            len(existing), len(expected), len(pending),
        )
    # Enqueue pending
    pending_by_member = {}
    for mid, bench in pending:
        pending_by_member.setdefault(mid, []).append(bench)
    id_to_member = {m.id: m for m in members}
    jobs = []
    for mid, benches in pending_by_member.items():
        m = id_to_member[mid]
        if m.kind == "sft":
            ref = {"kind": "lora", "base": m.base_model_id, "adapter": m.adapter_path}
            # If SFT succeeded on a specific worker, record it so eval can
            # pull the adapter from there when replication hasn't propagated
            # (openweight.replicate.pull_adapter in worker/openweight_runner).
            # worker_id is "<hostname>-<pid>"; we want the bare hostname.
            if getattr(m, "sft_worker_id", None):
                peer = m.sft_worker_id.rsplit("-", 1)[0]
                if not peer.endswith(".local"):
                    peer = peer + ".local"
                ref["source_peer"] = peer
        elif m.kind in ("merge", "lora_merge"):
            ref = {"kind": "merged", "path": m.merged_checkpoint_path}
        else:
            continue
        for bench in benches:
            jobs.append(make_eval_job(
                run_id=run_id,
                generation=generation,
                member_id=mid,
                benchmark_name=bench,
                model_ref=ref,
                tokenizer_family=m.tokenizer_family,
                proxy=proxy,
            ))
    dispatcher.submit_eval_jobs(jobs)
    fresh = dispatcher.wait_for_evals(
        run_id=run_id, generation=generation,
        expected=pending, timeout_s=eval_timeout_s,
    )
    return {**existing, **fresh}


def record_generation(
    lineage: LineageDB,
    members: list[OpenweightMember],
    bench_results: dict[tuple[str, str], EvalResultMsg],
    cfg: FitnessConfig,
) -> dict[str, float]:
    """Write members + per-benchmark rows + aggregate FitnessVector.

    Returns ``{member_id: bench_agg_score}`` for quick logging.
    """
    diversity = population_diversity(members, cfg.diversity_prompt_set or "")
    scores: dict[str, float] = {}
    for m in members:
        config_blob = (
            asdict(m.sft_config) if m.sft_config else
            asdict(m.merge_recipe) if m.merge_recipe else
            asdict(m.lora_merge_recipe) if m.lora_merge_recipe else {}
        )
        lineage.record_openweight_member(
            member_id=m.id, generation=m.generation, kind=m.kind,
            config_json=json.dumps(config_blob),
            parent_ids=m.parent_ids,
            tokenizer_family=m.tokenizer_family,
            base_model_id=m.base_model_id,
            adapter_path=m.adapter_path,
            merged_checkpoint_path=m.merged_checkpoint_path,
        )
        per_member_results: list[BenchmarkResult] = []
        for bench in cfg.benchmarks:
            er = bench_results.get((m.id, bench))
            if er is None or er.status != "ok" or er.score is None:
                continue
            lineage.record_fitness_eval(
                member_id=m.id, benchmark_name=bench,
                score=float(er.score), wallclock_s=er.wallclock_s,
                raw=er.raw,
            )
            per_member_results.append(BenchmarkResult(
                member_id=m.id, benchmark_name=bench,
                score=float(er.score), wallclock_s=er.wallclock_s,
                raw=er.raw or {},
            ))
        m.fitness = aggregate_fitness(
            member=m,
            bench_results=per_member_results,
            diversity_score=diversity.get(m.id, 0.0),
            calibration_ece=0.0,
            latency_ms=0.0,
            memory_gb=0.0,
            cfg=cfg,
        )
        scores[m.id] = m.fitness.bench_agg
    return scores


def run_gen0(
    redis_url: str,
    roster_name: str,
    pop_size: Optional[int],
    corpus_shard_path: str,
    adapter_root: str,
    benchmarks: list[str],
    benchmark_weights: dict[str, float],
    run_id: Optional[str],
    lineage_db: Path,
    sft_timeout_s: float,
    eval_timeout_s: float,
    proxy_eval: bool,
) -> dict:
    run_id = run_id or new_run_id()
    logger.info("openweight Gen-0: run_id=%s roster=%s pop=%s bench=%s",
                run_id, roster_name, pop_size, benchmarks)

    cfg = FitnessConfig(
        benchmarks=benchmarks,
        benchmark_weights=benchmark_weights,
    )
    roster = pick_roster(roster_name, n=pop_size)
    members = build_members_for_roster(roster, shard_id=corpus_shard_path, generation=0)
    dispatcher = RedisDispatcher(redis_url)

    t_sft0 = time.time()
    sft_results = run_sft_stage(
        dispatcher=dispatcher, run_id=run_id, members=members,
        corpus_shard_path=corpus_shard_path, adapter_root=adapter_root,
        sft_timeout_s=sft_timeout_s,
    )
    t_sft = time.time() - t_sft0

    # Backfill adapter_path + sft_worker_id onto each member from its SFT result
    for m in members:
        sr = sft_results.get(m.id)
        if sr and sr.status == "ok" and sr.adapter_path:
            m.adapter_path = sr.adapter_path
            m.sft_worker_id = sr.worker_id
        else:
            logger.warning("sft failed for %s: %s",
                           m.id, sr.error if sr else "missing")

    ok_members = [m for m in members if m.adapter_path]
    if not ok_members:
        logger.error("no SFT members succeeded; aborting")
        return {"run_id": run_id, "status": "no_sft_ok", "sft_time_s": t_sft}

    t_eval0 = time.time()
    eval_results = run_eval_stage(
        dispatcher=dispatcher, run_id=run_id, generation=0,
        members=ok_members, benchmarks=benchmarks,
        eval_timeout_s=eval_timeout_s, proxy=proxy_eval,
    )
    t_eval = time.time() - t_eval0

    lineage = LineageDB(str(lineage_db))
    scores = record_generation(lineage, members, eval_results, cfg)
    lineage.record_generation(0, {
        "best_individual_loss": None,
        "best_ensemble_loss": None,
        "wallclock_seconds": t_sft + t_eval,
    })
    lineage.close()

    summary = {
        "run_id": run_id, "roster": roster_name, "pop_size": len(members),
        "sft_ok": len(ok_members), "sft_time_s": t_sft,
        "eval_time_s": t_eval,
        "benchmark_agg_per_member": scores,
        "best_member": max(scores, key=scores.get) if scores else None,
    }
    logger.info("Gen-0 complete: %s", json.dumps(summary, indent=2))
    return summary


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--redis", required=True)
    ap.add_argument("--roster", default="llama3_within_family",
                    help="Roster name (see openweight/seed_roster.py::ROSTERS)")
    ap.add_argument("--pop-size", type=int, default=None,
                    help="Subsample the roster; defaults to full roster.")
    ap.add_argument("--corpus-shard", required=True,
                    help="Worker-relative path to the SFT JSONL shard.")
    ap.add_argument("--adapter-root", default="runs/openweight_gen0/adapters",
                    help="Worker-relative directory for adapter outputs.")
    ap.add_argument("--benchmarks", default="smoke_mcq",
                    help="Comma-separated benchmark names.")
    ap.add_argument("--bench-weights", default="",
                    help='JSON, e.g. {"ifeval": 1.0, "mmlu_pro": 2.0}')
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--lineage-db", default="runs/openweight_gen0/lineage.db")
    ap.add_argument("--sft-timeout-s", type=float, default=14400.0)
    ap.add_argument("--eval-timeout-s", type=float, default=7200.0)
    ap.add_argument("--proxy-eval", action="store_true",
                    help="Use the short/sampled benchmark subsets.")
    args = ap.parse_args()

    benches = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    weights = json.loads(args.bench_weights) if args.bench_weights else {b: 1.0 for b in benches}

    lineage_db_path = Path(args.lineage_db)
    lineage_db_path.parent.mkdir(parents=True, exist_ok=True)

    summary = run_gen0(
        redis_url=args.redis,
        roster_name=args.roster,
        pop_size=args.pop_size,
        corpus_shard_path=args.corpus_shard,
        adapter_root=args.adapter_root,
        benchmarks=benches,
        benchmark_weights=weights,
        run_id=args.run_id,
        lineage_db=lineage_db_path,
        sft_timeout_s=args.sft_timeout_s,
        eval_timeout_s=args.eval_timeout_s,
        proxy_eval=args.proxy_eval,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
