"""Launch transformers serve, run benchmarks, record to lineage.db.

Reusable for the baseline pass (``--model-path Qwen/Qwen2.5-1.5B-Instruct``,
``--member-id baseline-qwen25-15b``) and the post-SFT pass (``--model-path
path/to/merged_dir``, ``--member-id sft-qwen25-15b-v009``).

Single script so the experiment is a small shell loop, not an orchestrator
run — that comes next once we know the primitives survive end-to-end.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from benchmarks.adapters import run_benchmark
from controller.lineage import LineageDB
from openweight.inference import launch_transformers_server

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True,
                    help="HF id or local directory")
    ap.add_argument("--member-id", required=True,
                    help="e.g. baseline-qwen25-15b")
    ap.add_argument("--kind", default="base",
                    help="base | sft | merge (recorded in lineage)")
    ap.add_argument("--base-model-id", default=None,
                    help="For sft/merge, the underlying base; defaults to --model-path")
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--merged-checkpoint-path", default=None)
    ap.add_argument("--tokenizer-family", default="qwen25")
    ap.add_argument("--generation", type=int, default=0)
    ap.add_argument("--parent-ids", default="[]")
    ap.add_argument("--benchmarks", default="smoke_mcq",
                    help="Comma-separated: smoke_mcq,ifeval,hellaswag,mmlu_pro,gpqa_diamond")
    ap.add_argument("--proxy", action="store_true",
                    help="Pass through to benchmark adapter for a faster subset")
    ap.add_argument("--lineage-db", required=True, type=Path)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--log-dir", default=Path("/tmp/sieve-serve"), type=Path)
    ap.add_argument("--ready-timeout-s", type=float, default=900.0,
                    help="Qwen2.5-1.5B loads in ~30s but first-run cudnn init is slow")
    args = ap.parse_args()

    benches = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    args.lineage_db.parent.mkdir(parents=True, exist_ok=True)

    db = LineageDB(str(args.lineage_db))
    db.record_openweight_member(
        member_id=args.member_id,
        generation=args.generation,
        kind=args.kind,
        config_json=json.dumps({
            "model_path": args.model_path,
            "benchmarks": benches,
            "proxy": args.proxy,
        }),
        parent_ids=json.loads(args.parent_ids),
        tokenizer_family=args.tokenizer_family,
        base_model_id=args.base_model_id or args.model_path,
        adapter_path=args.adapter_path,
        merged_checkpoint_path=args.merged_checkpoint_path,
    )
    logger.info("Recorded member %s (kind=%s)", args.member_id, args.kind)

    logger.info("Launching transformers serve against %s", args.model_path)
    handle = launch_transformers_server(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        log_dir=args.log_dir,
        served_model_name=args.member_id,
        ready_timeout_s=args.ready_timeout_s,
    )
    logger.info("Server ready at %s (model=%s, log=%s)",
                handle.base_url, handle.served_model_name, handle.log_path)

    summary: dict[str, dict] = {}
    try:
        for bench in benches:
            t0 = time.time()
            logger.info("→ Running benchmark %s", bench)
            try:
                score, raw = run_benchmark(
                    name=bench,
                    base_url=handle.base_url,
                    model_name=handle.served_model_name,
                    proxy=args.proxy,
                )
                dt = time.time() - t0
                logger.info("  %s: score=%.4f (%.1fs)", bench, score, dt)
                db.record_fitness_eval(
                    member_id=args.member_id,
                    benchmark_name=bench,
                    score=float(score),
                    wallclock_s=dt,
                    raw=raw,
                )
                summary[bench] = {"score": float(score), "wallclock_s": dt}
            except Exception as e:  # noqa: BLE001
                dt = time.time() - t0
                logger.exception("  %s FAILED after %.1fs: %s", bench, dt, e)
                summary[bench] = {"error": str(e), "wallclock_s": dt}
    finally:
        logger.info("Stopping serve subprocess...")
        handle.stop()
        db.close()

    print(json.dumps({
        "member_id": args.member_id,
        "model_path": args.model_path,
        "results": summary,
    }, indent=2))


if __name__ == "__main__":
    main()
