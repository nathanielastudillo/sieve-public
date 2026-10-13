"""Post-hoc ensemble evaluation for a completed SIEVE dispatch run.

Pulls all `status=ok` worker results from Redis for a given run_id,
rsyncs each checkpoint back from the worker node that produced it,
and runs the existing `EnsembleEvaluator` machinery over the
recovered members.

Why this exists
---------------
exp007_v009 (2026-04-18) hit a job_timeout_s mismatch: members trained
5-11h each but the orchestrator gave 2h per generation. Every member
was marked `error="timeout"` in the orchestrator's in-memory state, so
no ensemble was ever evaluated — but 10 of them DID successfully
publish val_loss 3.85-4.26 to Redis with their remote checkpoint paths.
This script recovers that work without needing to rerun training.

See lab-notes/experiments/004-exp007-v009.md for the full incident.

Usage
-----
    # Dry run: list what would be pulled
    SSH_PASS=... python3 scripts/posthoc_ensemble_eval.py \\
        --redis redis://<node-ip>:6379/0 \\
        --run-id exp007_v009_20260418_054649 \\
        --corpus-dir data/corpus_v009 \\
        --output-dir runs/exp007_v009/posthoc \\
        --dry-run

    # Real run: pull + evaluate
    SSH_PASS=... python3 scripts/posthoc_ensemble_eval.py \\
        --redis redis://<node-ip>:6379/0 \\
        --run-id exp007_v009_20260418_054649 \\
        --corpus-dir data/corpus_v009 \\
        --output-dir runs/exp007_v009/posthoc
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

# Allow `python3 scripts/posthoc_ensemble_eval.py` from repo root.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

logger = logging.getLogger("posthoc")


def pull_checkpoint(
    worker_id: str,
    remote_path: str,
    local_root: Path,
    member_id: str,
    ssh_user: str,
    ssh_pass: str,
) -> Path | None:
    """Rsync a remote checkpoint back to local disk.

    Returns the local path on success, None on any failure.
    Matches the user-prefix logic in controller/orchestrator.py so
    compute worker_ids (`node-9-1868887`) and standalone worker_ids
    (`a@<node-ip>-21552`) are both handled.
    """
    node = re.sub(r"-\d+$", "", worker_id)
    target = node if "@" in node else f"{ssh_user}@{node}"

    local_dir = local_root / member_id
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / Path(remote_path).name
    if local_path.exists():
        logger.info("  [%s] already local: %s", member_id, local_path)
        return local_path

    try:
        subprocess.run(
            [
                "rsync", "-az",
                "-e", "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=8",
                f"{target}:{remote_path}", str(local_path),
            ],
            check=True,
            timeout=300,
            capture_output=True,
        )
        logger.info("  [%s] pulled %s -> %s",
                    member_id, target, local_path)
        return local_path
    except subprocess.CalledProcessError as e:
        logger.error("  [%s] rsync failed from %s: %s",
                     member_id, target, e.stderr.decode(errors="replace")[:200])
        return None
    except subprocess.TimeoutExpired:
        logger.error("  [%s] rsync timed out from %s", member_id, target)
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--redis", required=True, help="redis://host:port/db")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--corpus-dir", required=True,
                    help="Local corpus dir with train.bin / val.bin / meta.json")
    ap.add_argument("--output-dir", required=True,
                    help="Where to save pulled checkpoints + best_ensemble.json")
    ap.add_argument("--ensemble-size", type=int, default=4)
    ap.add_argument("--max-combinations", type=int, default=500)
    ap.add_argument("--ssh-user", default=os.environ.get("SSH_USER", "<user>"))
    ap.add_argument("--ssh-pass", default=os.environ.get("SSH_PASS"))
    ap.add_argument("--aggregator", default="logit_mean_smoothed")
    ap.add_argument("--smoothing-eps", type=float, default=0.05)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--block-size", type=int, default=None,
                    help="Override evaluator block_size. Must be ≤ the trained "
                         "model's block_size. exp007 members were trained at 256.")
    ap.add_argument("--dry-run", action="store_true",
                    help="List candidate members without pulling or evaluating")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    import redis as redis_mod

    r = redis_mod.from_url(args.redis, decode_responses=True)
    pattern = f"sieve:results:{args.run_id}:*"
    keys = list(r.scan_iter(pattern))
    logger.info("Found %d result keys for run %s", len(keys), args.run_id)

    oks: list[dict] = []
    bad = 0
    for key in keys:
        raw = r.get(key)
        if not raw:
            continue
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            bad += 1
            continue
        if d.get("status") != "ok":
            continue
        if not d.get("checkpoint_path") or not d.get("worker_id"):
            continue
        oks.append(d)
    logger.info("  %d with status=ok (malformed: %d)", len(oks), bad)

    # Sort best → worst by val_loss so logs are useful
    oks.sort(key=lambda d: d.get("val_loss") if d.get("val_loss") is not None else float("inf"))

    for d in oks:
        logger.info(
            "  gen%d %s val=%.4f wall=%.2fh worker=%s",
            d["generation"], d["member_id"][:8],
            d["val_loss"], d["wallclock_s"] / 3600, d["worker_id"],
        )

    if args.dry_run:
        logger.info("Dry run — not pulling or evaluating.")
        return 0

    if not oks:
        logger.error("No ok results to evaluate; nothing to do.")
        return 1

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root = out_dir / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    # --- pull checkpoints ---
    logger.info("\nPulling %d checkpoints into %s", len(oks), ckpt_root)
    pulled: list[tuple[dict, Path]] = []
    for d in oks:
        local = pull_checkpoint(
            worker_id=d["worker_id"],
            remote_path=d["checkpoint_path"],
            local_root=ckpt_root,
            member_id=d["member_id"],
            ssh_user=args.ssh_user,
            ssh_pass=args.ssh_pass,
        )
        if local is not None:
            pulled.append((d, local))

    logger.info("Recovered %d/%d checkpoints", len(pulled), len(oks))
    if not pulled:
        logger.error("No checkpoints could be pulled; aborting.")
        return 2

    # --- build Members and evaluate ---
    from controller.ensemble_eval import EnsembleEvaluator
    from controller.population import Member

    members: list[Member] = []
    for d, local in pulled:
        m = Member(
            id=d["member_id"],
            generation=d["generation"],
            config={},  # unknown at this stage — not required for eval
            checkpoint_path=str(local),
            val_loss=d["val_loss"],
        )
        members.append(m)

    evaluator = EnsembleEvaluator(
        checkpoint_dir=str(ckpt_root),
        data_dir=str(Path(args.corpus_dir).resolve()),
        device=args.device,
        block_size=args.block_size,
        aggregator=args.aggregator,
        smoothing_eps=args.smoothing_eps,
    )

    eval_size = min(args.ensemble_size, len(members))
    logger.info(
        "\nEvaluating ensembles of size %d over %d members (max %d combos)",
        eval_size, len(members), args.max_combinations,
    )
    rankings = evaluator.evaluate_all_combinations(
        members,
        ensemble_size=eval_size,
        max_combinations=args.max_combinations,
        strategy="random",
    )

    if not rankings:
        logger.error("Evaluator produced no rankings.")
        return 3

    # --- write artifact ---
    member_map = {m.id: m for m in members}
    top = rankings[0]
    best = {
        "run_id": args.run_id,
        "ensemble_id": f"posthoc-{args.run_id}-top1",
        "source": "posthoc_ensemble_eval.py",
        "members": [
            {
                "id": mid,
                "checkpoint": member_map[mid].checkpoint_path,
                "generation": member_map[mid].generation,
                "val_loss": member_map[mid].val_loss,
            }
            for mid in top["member_ids"] if mid in member_map
        ],
        "ensemble_size": len(top["member_ids"]),
        "ensemble_val_loss": top["ensemble_val_loss"],
        "diversity_score": top["diversity_score"],
        "aggregation": args.aggregator,
        "smoothing_eps": args.smoothing_eps,
        "best_individual_val_loss": min(
            (m.val_loss for m in members if m.val_loss is not None),
            default=float("inf"),
        ),
        "n_members_considered": len(members),
        "n_combinations_scored": len(rankings),
    }

    out_json = out_dir / "best_ensemble.json"
    out_json.write_text(json.dumps(best, indent=2))
    logger.info("\nWrote %s", out_json)
    logger.info(
        "best individual val_loss: %.4f | best ensemble val_loss: %.4f | Δ = %+.4f",
        best["best_individual_val_loss"],
        best["ensemble_val_loss"],
        best["ensemble_val_loss"] - best["best_individual_val_loss"],
    )

    # Top-5 ensembles for quick visual
    logger.info("\nTop-5 ensembles:")
    for i, r in enumerate(rankings[:5]):
        logger.info(
            "  %d. val=%.4f div=%.4f members=%s",
            i + 1, r["ensemble_val_loss"], r["diversity_score"],
            [mid[:8] for mid in r["member_ids"]],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
