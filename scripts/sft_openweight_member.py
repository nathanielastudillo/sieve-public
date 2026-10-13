"""Run a single LoRA fine-tune directly (no Redis), record to lineage.db.

Companion to ``bench_openweight_member.py``. Skips the orchestrator/
worker loop so the first smoke experiment can proceed on one box
without depending on the full dispatch path.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

from controller.lineage import LineageDB
from openweight.finetune import fine_tune
from openweight.genome import SftConfig

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model-id", required=True)
    ap.add_argument("--corpus-shard", required=True, type=Path)
    ap.add_argument("--member-id", required=True)
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Where the LoRA adapter gets saved")
    ap.add_argument("--lineage-db", required=True, type=Path)
    ap.add_argument("--tokenizer-family", default="qwen25")
    ap.add_argument("--generation", type=int, default=0)
    ap.add_argument("--parent-ids", default="[]")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--optimizer", default="adamw_torch",
                    help="trl SFTConfig.optim string")
    ap.add_argument("--corpus-shard-id", default=None,
                    help="Stable id for the shard (defaults to shard filename)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.lineage_db.parent.mkdir(parents=True, exist_ok=True)

    cfg = SftConfig(
        base_model_id=args.base_model_id,
        adapter_kind="lora",
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        target_modules=[m.strip() for m in args.target_modules.split(",") if m.strip()],
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        corpus_shard_id=args.corpus_shard_id or args.corpus_shard.name,
        seed=args.seed,
        optimizer=args.optimizer,
        precision="bf16",
    )

    db = LineageDB(str(args.lineage_db))
    db.record_openweight_member(
        member_id=args.member_id,
        generation=args.generation,
        kind="sft",
        config_json=json.dumps(asdict(cfg)),
        parent_ids=json.loads(args.parent_ids),
        tokenizer_family=args.tokenizer_family,
        base_model_id=args.base_model_id,
        adapter_path=str(args.out_dir),
    )
    logger.info("Recorded SFT member %s → adapter dir %s", args.member_id, args.out_dir)

    t0 = time.time()
    try:
        adapter_dir = fine_tune(cfg=cfg, corpus_path=args.corpus_shard, out_dir=args.out_dir)
        dt = time.time() - t0
        logger.info("fine_tune done in %.1fs → %s", dt, adapter_dir)
    finally:
        db.close()

    print(json.dumps({
        "member_id": args.member_id,
        "adapter_path": str(args.out_dir),
        "base_model_id": args.base_model_id,
        "wallclock_s": time.time() - t0,
    }, indent=2))


if __name__ == "__main__":
    main()
