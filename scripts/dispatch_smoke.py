#!/usr/bin/env python3
"""End-to-end smoke test for the SIEVE Redis dispatch layer.

Submits a tiny training job to the job queue and waits for the first
available worker to execute it. Intended to verify wiring *before*
running a full evolutionary loop.

Usage:
    .venv/bin/python scripts/dispatch_smoke.py \
        --redis redis://<node-ip>:6379/0 \
        --corpus-dir data/corpus_v006 \
        --ckpt-dir runs/smoke_dispatch/checkpoints
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from controller.dispatch import RedisDispatcher, make_job, new_run_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--redis", required=True)
    ap.add_argument("--corpus-dir", default="data/corpus_v006",
                    help="relative path on worker")
    ap.add_argument("--ckpt-dir", default="runs/smoke_dispatch/checkpoints",
                    help="relative path on worker")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--n-embd", type=int, default=128)
    ap.add_argument("--vocab-size", type=int, default=16000)
    ap.add_argument("--timeout-s", type=int, default=900)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    log = logging.getLogger("smoke")

    d = RedisDispatcher(args.redis)
    log.info("Queue depth at start: %d", d.queue_depth())
    workers = d.list_workers()
    log.info("Known workers: %s", [w[0] for w in workers])
    if not workers:
        log.warning("No workers registered yet. Submitting anyway.")

    run_id = new_run_id()
    job = make_job(
        run_id=run_id,
        generation=0,
        member_id="smoketest",
        model_config={
            "n_layer": args.n_layer, "n_head": args.n_head,
            "n_embd": args.n_embd, "block_size": 256,
            "dropout": 0.1, "bias": False,
            "vocab_size": args.vocab_size,
        },
        train_config={
            "epochs": args.epochs, "batch_size": 32,
            "learning_rate": 3e-4, "weight_decay": 0.3,
            "warmup_fraction": 0.05, "lr_schedule": "cosine",
            "optimizer_name": "adamw", "gradient_clip": 1.0,
            "shuffle_per_epoch": True, "seed": 42,
            "checkpoint_every": args.epochs,
        },
        corpus_dir=args.corpus_dir,
        checkpoint_dir=args.ckpt_dir,
    )
    d.submit_generation([job])
    log.info("Submitted run_id=%s", run_id)

    t0 = time.time()
    results = d.wait_for_generation(run_id, 0, ["smoketest"],
                                    timeout_s=args.timeout_s,
                                    poll_interval_s=1.0)
    r = results["smoketest"]
    dt = time.time() - t0
    log.info("Got result in %.1fs: status=%s val_loss=%s worker=%s ckpt=%s",
             dt, r.status, r.val_loss, r.worker_id, r.checkpoint_path)
    if r.status != "ok":
        log.error("FAILED: %s", r.error)
        sys.exit(1)
    log.info("SMOKE OK")


if __name__ == "__main__":
    main()
