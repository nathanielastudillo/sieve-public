"""Experiment 006 runner — multi-model ensemble, pre/post SFT comparison.

Drives the full stack for one roster:

    baseline   — gen=-1, kind="base"    : eval each HF checkpoint as-is
       ↓                                   (no SFT, no LoRA)
    SFT        — gen=0,  kind="sft"     : LoRA-fine-tune each on v009
       ↓
    post-SFT   — gen=0,  kind="sft"     : eval each fine-tuned member
       ↓
    ensembles  — pseudo-members         : majority vote (pre) & (post)
       ↓
    summary    — compare pre vs post, individual vs ensemble

Single ``run_id`` is used across all phases so Redis reuse is trivial
and the lineage DB carries a coherent pre→post history.

Usage
-----
    python -m scripts.run_experiment_006 \\
        --redis redis://<node-ip>:6379/0 \\
        --roster sota_2026_q2 \\
        --pop-size 8 \\
        --corpus-shard data/sft_shards/v009.jsonl \\
        --adapter-root runs/exp006_sota/adapters \\
        --lineage-db runs/exp006_sota/lineage.db \\
        --benchmark smoke_mcq --proxy-eval \\
        --run-id exp006_sota_q2_01
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path

_here = Path(__file__).resolve().parent
_repo = _here.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from controller.dispatch import (  # noqa: E402
    RedisDispatcher,
    make_eval_job,
    new_run_id,
)
from controller.lineage import LineageDB  # noqa: E402
from controller.openweight_ensemble_eval import aggregate_from_lineage  # noqa: E402
from controller.openweight_orchestrator import run_gen0  # noqa: E402
from openweight.seed_roster import pick_roster  # noqa: E402

logger = logging.getLogger("exp006")


def baseline_member_id(run_id: str, hf_id: str) -> str:
    # Stable id across reruns so baselines de-dup on the same run
    safe = hf_id.replace("/", "__").replace(".", "_")
    return f"base-{run_id}-{safe}"


def run_baseline_evals(
    dispatcher: RedisDispatcher,
    lineage: LineageDB,
    run_id: str,
    roster_name: str,
    benchmark: str,
    proxy: bool,
    eval_timeout_s: float,
    pop_size: int | None = None,
) -> dict[str, float]:
    """Enqueue eval jobs against each base HF checkpoint and record them.

    Members are inserted with generation=-1 and kind="base". Each carries
    the same HF ID as base_model_id so the ensemble-eval aggregator can
    walk from the eval → member → base mapping.
    """
    roster = pick_roster(roster_name)
    if pop_size is not None:
        roster = roster[:pop_size]
    logger.info("baseline: %d roster entries for run=%s", len(roster), run_id)

    member_ids: list[str] = []
    expected: list[tuple[str, str]] = []
    jobs = []
    for seed in roster:
        mid = baseline_member_id(run_id, seed.hf_id)
        member_ids.append(mid)

        lineage.record_openweight_member(
            member_id=mid, generation=-1, kind="base",
            config_json=json.dumps({"hf_id": seed.hf_id, "why": seed.why}),
            parent_ids=[],
            tokenizer_family=seed.tokenizer_family,
            base_model_id=seed.hf_id,
            adapter_path=None, merged_checkpoint_path=None,
        )
        expected.append((mid, benchmark))
        jobs.append(make_eval_job(
            run_id=run_id, generation=-1, member_id=mid,
            benchmark_name=benchmark,
            model_ref={"kind": "base", "path": seed.hf_id},
            tokenizer_family=seed.tokenizer_family,
            proxy=proxy,
        ))

    # Skip any already-persisted baseline evals so resume-replay works
    pending = []
    for job, (mid, bench) in zip(jobs, expected):
        key = dispatcher.eval_result_key(run_id, -1, mid, bench)
        if dispatcher.r.get(key) is None:
            pending.append(job)
    if pending:
        dispatcher.submit_eval_jobs(pending)
        logger.info("baseline: dispatched %d eval jobs", len(pending))
    else:
        logger.info("baseline: all %d evals already in Redis", len(expected))

    fresh = dispatcher.wait_for_evals(
        run_id=run_id, generation=-1,
        expected=expected, timeout_s=eval_timeout_s,
    )

    scores: dict[str, float] = {}
    for (mid, bench), msg in fresh.items():
        if msg.status != "ok" or msg.score is None:
            logger.warning("baseline eval failed mid=%s: %s", mid, msg.error)
            continue
        lineage.record_fitness_eval(
            member_id=mid, benchmark_name=bench,
            score=float(msg.score), wallclock_s=msg.wallclock_s, raw=msg.raw,
        )
        scores[mid] = float(msg.score)
    return scores


def _format_scores(scores: dict[str, float], lineage: LineageDB) -> str:
    # Compact per-base summary for the log
    by_base: dict[str, list[tuple[str, float]]] = {}
    for mid, s in scores.items():
        row = lineage.conn.execute(
            "SELECT base_model_id FROM members WHERE id = ?", (mid,),
        ).fetchone()
        base = row["base_model_id"] if row else "?"
        by_base.setdefault(base, []).append((mid, s))
    lines = []
    for base, entries in sorted(by_base.items()):
        for mid, s in entries:
            lines.append(f"  {s:.3f}  {base}  [{mid[:24]}]")
    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--redis", required=True)
    ap.add_argument("--roster", required=True,
                    help="Roster name (see openweight/seed_roster.py::ROSTERS)")
    ap.add_argument("--pop-size", type=int, default=None,
                    help="Subsample the roster (first N); default = full.")
    ap.add_argument("--corpus-shard", required=True,
                    help="Worker-relative path to the SFT JSONL shard.")
    ap.add_argument("--adapter-root", required=True,
                    help="Worker-relative directory for adapter outputs.")
    ap.add_argument("--lineage-db", required=True)
    ap.add_argument("--benchmark", default="smoke_mcq",
                    help="Single benchmark whose raw items drive the vote.")
    ap.add_argument("--strategy", default="weighted_majority",
                    choices=["majority", "weighted_majority"])
    ap.add_argument("--proxy-eval", action="store_true",
                    help="Use the short/sampled benchmark subsets.")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--skip-baseline", action="store_true",
                    help="Skip the pre-SFT eval phase (e.g. for resume).")
    ap.add_argument("--skip-sft", action="store_true",
                    help="Skip SFT+post-eval (e.g. when gen=0 already done).")
    ap.add_argument("--sft-timeout-s", type=float, default=14400.0)
    ap.add_argument("--eval-timeout-s", type=float, default=7200.0)
    args = ap.parse_args()

    run_id = args.run_id or new_run_id()
    lineage_db_path = Path(args.lineage_db)
    lineage_db_path.parent.mkdir(parents=True, exist_ok=True)

    lineage = LineageDB(str(lineage_db_path))
    dispatcher = RedisDispatcher(args.redis)

    # --- Phase 1: baseline (pre-SFT) ---
    baseline_scores: dict[str, float] = {}
    if not args.skip_baseline:
        t0 = time.time()
        logger.info("=== Phase 1: baseline evals (gen=-1, kind=base) ===")
        baseline_scores = run_baseline_evals(
            dispatcher=dispatcher, lineage=lineage,
            run_id=run_id, roster_name=args.roster,
            benchmark=args.benchmark, proxy=args.proxy_eval,
            eval_timeout_s=args.eval_timeout_s,
            pop_size=args.pop_size,
        )
        logger.info("baseline scores (%.0fs):\n%s",
                    time.time() - t0, _format_scores(baseline_scores, lineage))

    # --- Phase 2: SFT + post-SFT eval (gen=0) ---
    if not args.skip_sft:
        logger.info("=== Phase 2: SFT + post-eval (gen=0, kind=sft) ===")
        run_gen0(
            redis_url=args.redis,
            roster_name=args.roster,
            pop_size=args.pop_size,
            corpus_shard_path=args.corpus_shard,
            adapter_root=args.adapter_root,
            benchmarks=[args.benchmark],
            benchmark_weights={args.benchmark: 1.0},
            run_id=run_id,
            lineage_db=lineage_db_path,
            sft_timeout_s=args.sft_timeout_s,
            eval_timeout_s=args.eval_timeout_s,
            proxy_eval=args.proxy_eval,
        )

    # --- Phase 3: posthoc ensemble aggregation (pre + post) ---
    logger.info("=== Phase 3: ensemble aggregation (majority vote) ===")
    pre = aggregate_from_lineage(
        str(lineage_db_path),
        benchmark=args.benchmark,
        roster_name=f"{args.roster}-baseline",
        strategy=args.strategy,
        generation=-1,
        ensemble_id=f"ens-pre-{run_id}",
    )
    post = aggregate_from_lineage(
        str(lineage_db_path),
        benchmark=args.benchmark,
        roster_name=f"{args.roster}-post-sft",
        strategy=args.strategy,
        generation=0,
        ensemble_id=f"ens-post-{run_id}",
    )

    summary = {
        "run_id": run_id,
        "roster": args.roster,
        "benchmark": args.benchmark,
        "strategy": args.strategy,
        "pre": pre,
        "post": post,
    }
    out = json.dumps(summary, indent=2)
    print(out)

    # Terse log lines the user can read at a glance
    def _line(label: str, d: dict) -> None:
        if d.get("status") != "ok":
            logger.info("%s: %s", label, d.get("status"))
            return
        logger.info(
            "%s  best_ind=%.3f  ensemble=%.3f  Δ=%+.3f  voters=%d  fams=%s",
            label,
            d["best_individual_score"], d["ensemble_score"],
            d["delta_vs_best_individual"], d["n_voters"],
            ",".join(d["tokenizer_families"]),
        )

    _line("PRE  (baseline)", pre)
    _line("POST (SFT'd)   ", post)
    if pre.get("status") == "ok" and post.get("status") == "ok":
        d_ind = post["best_individual_score"] - pre["best_individual_score"]
        d_ens = post["ensemble_score"] - pre["ensemble_score"]
        logger.info(
            "SFT lift: best_individual %+.3f  ensemble %+.3f",
            d_ind, d_ens,
        )

    lineage.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
