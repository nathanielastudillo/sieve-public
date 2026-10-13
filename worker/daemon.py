"""SIEVE worker daemon.

Long-running process on each GB10 node. Pulls jobs from Redis,
executes them, publishes results. One daemon = one GPU's worth of
concurrent work (the daemon IS the training/serving process — no
per-job subprocessing except where inference needs a separate sglang
server).

Roles
-----
A worker advertises a role set at startup (``--roles``):

- ``train`` — small-corpus from-scratch GPT trainer (``TRAIN_QUEUE``)
- ``sft``   — LoRA/DoRA fine-tune of open-weight bases (``SFT_QUEUE``)
- ``eval``  — benchmark evaluation against sglang endpoints (``EVAL_QUEUE``)
- ``merge`` — mergekit recipe execution (``MERGE_QUEUE``)

Default: all four. BLPOP is multi-key so Redis hands the first
available job on any matching queue.

Usage
-----
    python -m worker.daemon \
        --redis redis://<node-ip>:6379/0 \
        --node node-4 \
        --roles train,sft,eval,merge \
        --sieve-root /home/<user>/sieve
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import time
import traceback
from pathlib import Path

# Allow running from repo root OR as installed package
_here = Path(__file__).resolve().parent
if str(_here.parent) not in sys.path:
    sys.path.insert(0, str(_here.parent))

from controller.dispatch import (  # noqa: E402
    EVAL_QUEUE,
    EvalJob,
    MERGE_QUEUE,
    MergeJob,
    RedisDispatcher,
    SFT_QUEUE,
    SftJob,
    TRAIN_QUEUE,
    TrainJob,
    TrainResultMsg,
)

logger = logging.getLogger("sieve.worker")


ROLE_TO_QUEUE = {
    "train": TRAIN_QUEUE,
    "sft":   SFT_QUEUE,
    "eval":  EVAL_QUEUE,
    "merge": MERGE_QUEUE,
}


def run_train_job(job: TrainJob, sieve_root: Path, worker_id: str) -> TrainResultMsg:
    """Execute a small-corpus training job. Exceptions become error results."""
    from worker.trainer import train_model  # imported lazily so daemon starts fast

    t0 = time.time()
    corpus_dir = sieve_root / job.corpus_dir
    ckpt_dir = sieve_root / job.checkpoint_dir / job.member_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = dict(job.train_config)
    train_cfg["checkpoint_dir"] = str(ckpt_dir)

    try:
        result = train_model(
            model_config=job.model_config,
            train_config=train_cfg,
            data_dir=str(corpus_dir),
        )
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        logger.error("Training failed for %s: %s", job.member_id, e)
        logger.debug(tb)
        return TrainResultMsg(
            run_id=job.run_id, generation=job.generation, member_id=job.member_id,
            worker_id=worker_id, status="error",
            val_loss=None, final_loss=None, checkpoint_path=None,
            wallclock_s=time.time() - t0,
            error=f"{type(e).__name__}: {e}",
        )

    if result.best_val_loss != float("inf"):
        reported_val = float(result.best_val_loss)
    else:
        reported_val = float(result.best_loss)

    chk = result.best_val_checkpoint or (
        result.checkpoint_paths[-1] if result.checkpoint_paths else None
    )
    return TrainResultMsg(
        run_id=job.run_id, generation=job.generation, member_id=job.member_id,
        worker_id=worker_id, status="ok",
        val_loss=reported_val,
        final_loss=float(result.final_loss),
        checkpoint_path=chk,
        wallclock_s=result.wallclock_time,
        tokens_seen=result.tokens_seen,
    )


def handle_job(
    queue: str,
    raw: str,
    dispatcher: RedisDispatcher,
    sieve_root: Path,
    worker_id: str,
) -> str:
    """Decode, dispatch, publish. Returns a short log tag for the daemon log."""
    # Lazy import so the daemon doesn't pay torch's import cost until a
    # job actually lands.
    from worker.openweight_runner import run_eval, run_merge, run_sft

    if queue == TRAIN_QUEUE:
        job = TrainJob.from_json(raw)
        dispatcher.heartbeat(worker_id, state=f"train:{job.member_id[:8]}",
                             ttl_s=120)
        result = run_train_job(job, sieve_root, worker_id)
        dispatcher.publish_result(result)
        return f"train/{job.member_id[:8]}/{result.status}"

    elif queue == SFT_QUEUE:
        job = SftJob.from_json(raw)
        dispatcher.heartbeat(worker_id, state=f"sft:{job.member_id[:8]}",
                             ttl_s=600)
        result = run_sft(job, sieve_root, worker_id)
        dispatcher.publish_sft_result(result)
        return f"sft/{job.member_id[:8]}/{result.status}"

    elif queue == EVAL_QUEUE:
        job = EvalJob.from_json(raw)
        dispatcher.heartbeat(
            worker_id,
            state=f"eval:{job.member_id[:8]}:{job.benchmark_name}",
            ttl_s=600,
        )
        result = run_eval(job, sieve_root, worker_id)
        dispatcher.publish_eval_result(result)
        return f"eval/{job.member_id[:8]}/{job.benchmark_name}/{result.status}"

    elif queue == MERGE_QUEUE:
        job = MergeJob.from_json(raw)
        dispatcher.heartbeat(worker_id, state=f"merge:{job.member_id[:8]}",
                             ttl_s=600)
        result = run_merge(job, sieve_root, worker_id)
        dispatcher.publish_merge_result(result)
        return f"merge/{job.member_id[:8]}/{result.status}"

    else:
        raise ValueError(f"unexpected queue: {queue!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--redis", required=True,
                    help="Redis URL, e.g. redis://<node-ip>:6379/0")
    ap.add_argument("--node", default=socket.gethostname())
    ap.add_argument("--sieve-root", default=os.environ.get("SIEVE_ROOT", "/home/<user>/sieve"))
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--heartbeat-s", type=float, default=10.0)
    ap.add_argument("--idle-timeout-s", type=int, default=30,
                    help="BLPOP timeout between heartbeats")
    ap.add_argument("--exit-when-idle-s", type=float, default=0.0,
                    help="If >0, exit cleanly after this many seconds with no work")
    ap.add_argument("--roles", default="train,sft,eval,merge",
                    help="Comma-separated set of {train,sft,eval,merge}.")
    args = ap.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    sieve_root = Path(args.sieve_root).resolve()
    worker_id = f"{args.node}-{os.getpid()}"

    roles = [r.strip() for r in args.roles.split(",") if r.strip()]
    bad = [r for r in roles if r not in ROLE_TO_QUEUE]
    if bad:
        logger.error("Unknown roles: %s (valid: %s)", bad, list(ROLE_TO_QUEUE))
        sys.exit(2)
    queues = [ROLE_TO_QUEUE[r] for r in roles]

    logger.info(
        "Starting worker %s on %s (sieve_root=%s roles=%s queues=%s)",
        worker_id, args.node, sieve_root, roles, queues,
    )

    dispatcher = RedisDispatcher(args.redis)
    dispatcher.heartbeat(worker_id, state="idle", ttl_s=int(args.heartbeat_s * 3))

    last_beat = time.time()
    last_work = time.time()

    while True:
        now = time.time()
        if now - last_beat > args.heartbeat_s:
            dispatcher.heartbeat(worker_id, state="idle",
                                 ttl_s=int(args.heartbeat_s * 3))
            last_beat = now

        if args.exit_when_idle_s > 0 and (now - last_work) > args.exit_when_idle_s:
            logger.info("No work for %.0fs, exiting cleanly.", now - last_work)
            return

        try:
            res = dispatcher.next_openweight_job(
                queues=queues, timeout_s=args.idle_timeout_s,
            )
        except KeyboardInterrupt:
            logger.info("Interrupted, exiting.")
            return
        except Exception as e:  # noqa: BLE001
            logger.warning("next_job error: %s (retry in 5s)", e)
            time.sleep(5)
            continue
        if res is None:
            continue  # BLPOP timeout — loop back for heartbeat

        queue, raw = res
        logger.info("Got job from %s", queue)

        try:
            tag = handle_job(queue, raw, dispatcher, sieve_root, worker_id)
            logger.info("Completed %s", tag)
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            logger.error("handle_job crashed (queue=%s): %s\n%s", queue, e, tb)

        last_work = time.time()
        last_beat = last_work
        dispatcher.heartbeat(worker_id, state="idle",
                             ttl_s=int(args.heartbeat_s * 3))


if __name__ == "__main__":
    main()
