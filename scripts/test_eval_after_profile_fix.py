"""Verify both fixes (data_dir + block_size) propagate through the orchestrator.

Constructs the orchestrator via the same path as main(), applies profile +
CLI overrides, then uses its evaluator on the existing gen-0 checkpoints.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controller.orchestrator import SieveOrchestrator
from controller.population import Member


def main() -> int:
    orchestrator = SieveOrchestrator(config_path="config/default.yaml")
    orchestrator.load_profile("smoke")
    orchestrator.data_dir = "data/corpus_v003"
    orchestrator.evaluator.data_dir = "data/corpus_v003"

    print(f"evaluator.data_dir   = {orchestrator.evaluator.data_dir}")
    print(f"evaluator.block_size = {orchestrator.evaluator._block_size_override}")
    print(f"evaluator.batch_size = {orchestrator.evaluator.batch_size}")

    checkpoint_root = Path("runs/smoke_loop/checkpoints")
    members: list[Member] = []
    for member_dir in sorted(checkpoint_root.glob("gen-000-*")):
        ckpts = sorted(member_dir.glob("*.pt"))
        if ckpts:
            members.append(Member(
                id=member_dir.name.split("-", 2)[-1],
                generation=0,
                config={},
                checkpoint_path=str(ckpts[-1]),
            ))
    print(f"Loaded {len(members)} members")

    rankings = orchestrator.evaluator.evaluate_all_combinations(
        members, ensemble_size=3, max_combinations=20, strategy="random",
    )
    print(f"Best ensemble val_loss: {rankings[0]['ensemble_val_loss']:.4f}")
    print(f"Best members: {rankings[0]['member_ids']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
