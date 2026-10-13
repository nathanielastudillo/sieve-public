"""Write best_ensemble.json for the completed smoke_loop run.

The orchestrator crashed on spec write because runs/smoke_loop/results/ did
not exist after CLI --results-dir override (fixed in-tree). This script
reconstructs the spec from the final gen-2 checkpoints via one more
ensemble eval.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controller.ensemble_eval import EnsembleEvaluator
from controller.population import Member
from worker.checkpoint import load_checkpoint


def main() -> int:
    checkpoint_root = Path("runs/smoke_loop/checkpoints")
    results_dir = Path("runs/smoke_loop/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir = "data/corpus_v003"

    # Final generation only (gen-002-*)
    members: list[Member] = []
    for member_dir in sorted(checkpoint_root.glob("gen-002-*")):
        ckpts = sorted(member_dir.glob("*.pt"))
        if not ckpts:
            continue
        # Recover model config from checkpoint metadata
        ckpt = load_checkpoint(str(ckpts[-1]), device="cpu")
        config = dict(ckpt["config"])
        members.append(Member(
            id=member_dir.name.split("-", 2)[-1],
            generation=2,
            config=config,
            checkpoint_path=str(ckpts[-1]),
        ))

    print(f"Loaded {len(members)} gen-2 members")
    evaluator = EnsembleEvaluator(
        checkpoint_dir=str(checkpoint_root),
        data_dir=data_dir,
        device="cuda",
        batch_size=32,
        block_size=256,
    )

    rankings = evaluator.evaluate_all_combinations(
        members, ensemble_size=3, max_combinations=20, strategy="random",
    )

    top = rankings[0]
    member_map = {m.id: m for m in members}

    spec = {
        "ensemble_id": "sieve-smoke-gen2-top1",
        "run": "smoke_loop",
        "corpus": "corpus_v003",
        "generation": 2,
        "members": [
            {
                "id": mid,
                "checkpoint": member_map[mid].checkpoint_path,
                "config": member_map[mid].config,
            }
            for mid in top["member_ids"] if mid in member_map
        ],
        "ensemble_size": len(top["member_ids"]),
        "val_loss": top["ensemble_val_loss"],
        "diversity_score": top.get("diversity_score"),
        "aggregation": "logit_average",
        "generation_progression": {
            "gen_0_best_ensemble_loss": 9.3659,
            "gen_1_best_ensemble_loss": 9.1746,
            "gen_2_best_ensemble_loss": 8.7315,
        },
        "notes": (
            "Smoke loop validates SIEVE end-to-end on corpus_v003 (~85K train "
            "tokens, 16K custom BPE). 3 generations × 6 members × ensemble_size=3; "
            "ensemble loss monotonically improved 9.37 → 9.17 → 8.73 (uniform "
            "baseline ln(16000)=9.68). Individual members overfit hard (train loss "
            "<0.1) but diverse-ensemble eval beats baseline — the SIEVE thesis in "
            "miniature."
        ),
    }

    out = results_dir / "best_ensemble.json"
    out.write_text(json.dumps(spec, indent=2))
    print(f"Wrote {out}")
    print(f"Best val_loss: {spec['val_loss']:.4f}")
    print(f"Members: {[m['id'] for m in spec['members']]}")

    # Also persist the full rankings for inspection
    rankings_out = results_dir / "gen2_final_rankings.json"
    rankings_out.write_text(json.dumps(rankings, indent=2, default=str))
    print(f"Wrote {rankings_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
