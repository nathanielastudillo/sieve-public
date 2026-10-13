"""Diagnostic: run ensemble evaluation on existing gen-000-* checkpoints.

Used to validate the ensemble_eval data_dir fix without re-training the
full smoke loop. Builds Member stubs from disk and calls
EnsembleEvaluator.evaluate_all_combinations directly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controller.ensemble_eval import EnsembleEvaluator
from controller.population import Member


def main() -> int:
    checkpoint_root = Path("runs/smoke_loop/checkpoints")
    data_dir = "data/corpus_v003"

    members: list[Member] = []
    for member_dir in sorted(checkpoint_root.glob("gen-000-*")):
        member_id = member_dir.name.split("-", 2)[-1]
        ckpts = sorted(member_dir.glob("*.pt"))
        if not ckpts:
            print(f"  skip {member_id}: no checkpoints")
            continue
        members.append(Member(
            id=member_id,
            generation=0,
            config={},  # config not needed for eval
            checkpoint_path=str(ckpts[-1]),
        ))
        print(f"  loaded {member_id}: {ckpts[-1]}")

    print(f"\nLoaded {len(members)} members for eval")
    if len(members) < 3:
        print("Need >=3 members for ensemble_size=3")
        return 1

    evaluator = EnsembleEvaluator(
        checkpoint_dir=str(checkpoint_root),
        data_dir=data_dir,
        device="cuda",
        batch_size=32,
        block_size=256,
    )

    rankings = evaluator.evaluate_all_combinations(
        members,
        ensemble_size=3,
        max_combinations=20,
        strategy="random",
    )

    print(f"\nEvaluated {len(rankings)} combinations")
    print(f"Best ensemble_val_loss: {rankings[0]['ensemble_val_loss']:.4f}")
    print(f"Best member_ids: {rankings[0]['member_ids']}")

    # Persist for inspection
    out = Path("runs/smoke_loop/results/gen0_eval_diagnostic.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rankings, indent=2, default=str))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
