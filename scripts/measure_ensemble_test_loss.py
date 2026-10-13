"""Retrofit diagnostic: measure ensemble *test* loss for a completed run.

Rationale
---------
The P0 question raised by the small-corpus ensemble literature review is:
at 10^5 tokens, does the selected ensemble beat the uniform baseline on
data neither the members nor the selector have seen?

- Training uses train.bin.
- Ensemble selection optimizes against val.bin.
- test.bin is never touched during training or selection.

If ensemble test NLL > ln(vocab_size), we're worse than predicting the
uniform distribution — the "meta-overfitting" pathology observed in
exp-001/002 (individual test NLL 10.87 vs uniform 9.68 = ln 16000).
See lab-notes/research-questions/literature-small-corpus-ensembles.md
§4 and §9 (P0-#1).

Usage
-----
    python3 scripts/measure_ensemble_test_loss.py \\
        --ensemble runs/smoke_loop_v006/results/best_ensemble.json \\
        --corpus  data/corpus_v003

``--ensemble`` defaults to ``results/best_ensemble.json`` (the last
run's artifact). Checkpoint paths inside ``best_ensemble.json`` are
resolved relative to the sieve repo root so the script runs from any
cwd below it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controller.ensemble_eval import EnsembleEvaluator
from controller.population import Member


def _load_test_split(
    corpus_dir: Path, block_size: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Read test.bin into (inputs, targets, vocab_size).

    Mirrors EnsembleEvaluator._load_val_data so CE is directly
    comparable to the selection/meta-val numbers reported during the
    run.
    """
    meta = json.loads((corpus_dir / "metadata.json").read_text())
    vocab_size = int(meta["vocab_size"])
    dtype = np.uint32 if "32" in str(meta.get("dtype", "uint16")) else np.uint16

    tokens = np.memmap(str(corpus_dir / "test.bin"), dtype=dtype, mode="r")
    n = len(tokens)
    n_chunks = max(1, (n - 1) // block_size)

    inputs_list, targets_list = [], []
    for i in range(n_chunks):
        start = i * block_size
        end = start + block_size + 1
        if end > n:
            break
        chunk = tokens[start:end]
        inputs_list.append(chunk[:-1])
        targets_list.append(chunk[1:])

    inputs = torch.tensor(np.array(inputs_list), dtype=torch.long)
    targets = torch.tensor(np.array(targets_list), dtype=torch.long)
    return inputs, targets, vocab_size


def _resolve(path: str, repo_root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (repo_root / p)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ensemble",
        default="results/best_ensemble.json",
        help="Path to best_ensemble.json (default: results/best_ensemble.json)",
    )
    parser.add_argument(
        "--corpus",
        default=None,
        help=(
            "Corpus directory containing test.bin + metadata.json. "
            "Defaults to data/corpus_v003."
        ),
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=256,
        help="Context window for chunking test.bin (default: 256).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Forward-pass batch size (default: 32).",
    )
    parser.add_argument(
        "--aggregator",
        default="logit_mean_smoothed",
        choices=("logit_mean", "logit_mean_smoothed", "prob_mean"),
        help="Ensemble aggregator (default matches the orchestrator).",
    )
    parser.add_argument(
        "--smoothing-eps",
        type=float,
        default=0.05,
        help="ε for the smoothed-logit-mean aggregator (default: 0.05).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for forward passes.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    spec_path = _resolve(args.ensemble, repo_root)
    spec = json.loads(spec_path.read_text())
    corpus_dir = _resolve(
        args.corpus or "data/corpus_v003", repo_root,
    )

    print(f"Ensemble spec:     {spec_path}")
    print(f"Corpus dir:        {corpus_dir}")
    print(f"Members:           {spec['ensemble_size']}")
    print(f"Selection val NLL: {spec.get('val_loss'):.4f}")
    print(f"Aggregator:        {args.aggregator} (eps={args.smoothing_eps})")
    print()

    # Load test split + uniform baseline.
    inputs, targets, vocab_size = _load_test_split(corpus_dir, args.block_size)
    uniform_baseline = math.log(vocab_size)
    print(f"Test chunks:       {inputs.shape[0]}  (block_size={args.block_size})")
    print(f"Vocab size:        {vocab_size}")
    print(f"Uniform baseline:  {uniform_baseline:.4f}  (= ln V)")
    print()

    # Instantiate the evaluator purely for its _get_logits_for_checkpoint +
    # _compute_ensemble_loss helpers. data_dir is irrelevant since we pass
    # inputs/targets directly; meta_val_fraction=0 keeps _load_val_data a
    # no-op if it were invoked.
    evaluator = EnsembleEvaluator(
        checkpoint_dir=str(repo_root),
        data_dir=str(corpus_dir),
        device=args.device,
        batch_size=args.batch_size,
        block_size=args.block_size,
        aggregator=args.aggregator,
        smoothing_eps=args.smoothing_eps,
        meta_val_fraction=0.0,
    )

    member_logits: list[torch.Tensor] = []
    print(f"{'member':<12}{'test NLL':>12}  checkpoint")
    print("-" * 80)
    for m in spec["members"]:
        ckpt = _resolve(m["checkpoint"], repo_root)
        if not ckpt.exists():
            print(f"{m['id']:<12}{'MISSING':>12}  {ckpt}")
            continue
        logits = evaluator._get_logits_for_checkpoint(str(ckpt), inputs)
        member_logits.append(logits)
        # Per-member NLL via the evaluator's own loss path so aggregation
        # semantics are identical to the ensemble calculation below.
        solo_loss = evaluator._compute_ensemble_loss([logits], targets)
        print(f"{m['id']:<12}{solo_loss:>12.4f}  {ckpt.relative_to(repo_root)}")

    if not member_logits:
        print("\nNo checkpoints loaded; cannot compute ensemble loss.", file=sys.stderr)
        return 1

    ensemble_loss = evaluator._compute_ensemble_loss(member_logits, targets)
    print("-" * 80)
    print(f"{'ensemble':<12}{ensemble_loss:>12.4f}")
    print()

    # The P0 diagnostic: compare to uniform baseline. This is what
    # tells us whether the selected ensemble is actually a predictor
    # or just noise relative to a trivial model.
    gap = ensemble_loss - uniform_baseline
    verdict = (
        "BEATS uniform (ensemble is a predictor)"
        if gap < 0
        else "WORSE than uniform (meta-overfitting — cluster run premature)"
    )
    print(f"ensemble − uniform: {gap:+.4f}  →  {verdict}")

    if spec.get("val_loss") is not None:
        sel_gap = ensemble_loss - float(spec["val_loss"])
        print(
            f"ensemble test − selection val: {sel_gap:+.4f}  "
            f"(positive means test > val → selection bias)"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
