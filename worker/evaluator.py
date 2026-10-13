"""Ensemble evaluation for SIEVE.

Forward-pass evaluation of individual models and ensembles (logit averaging).
Memory-efficient: processes batches one at a time, accumulates loss without
storing all logits in memory simultaneously.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single model evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_single(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device | str,
) -> dict[str, Any]:
    """Run forward pass on a dataloader, return val metrics.

    Parameters
    ----------
    model : a GPT model in eval mode.
    dataloader : yields batches with input_ids and labels.
    device : device to run inference on.

    Returns
    -------
    dict with val_loss, perplexity, and total tokens evaluated.
    """
    device = torch.device(device) if isinstance(device, str) else device
    use_amp = device.type == "cuda"
    dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in dataloader:
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=dtype, enabled=use_amp):
            _logits, loss = model(x, targets=y)

        n_tokens = y.numel()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / max(1, total_tokens)
    perplexity = min(float("inf"), torch.exp(torch.tensor(avg_loss)).item())

    return {
        "val_loss": avg_loss,
        "perplexity": perplexity,
        "total_tokens": total_tokens,
    }


# ---------------------------------------------------------------------------
# Ensemble evaluation (logit averaging)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_ensemble(
    checkpoint_paths: list[str],
    data_dir: str,
    block_size: int,
    batch_size: int = 128,
    device: str = "cuda",
) -> dict[str, Any]:
    """Evaluate an ensemble by averaging logits across member models.

    Memory-efficient implementation: for each batch, loads logits from all
    members, averages them, computes loss, then discards the batch logits
    before moving to the next batch.

    Parameters
    ----------
    checkpoint_paths : list of paths to model checkpoint files.
    data_dir : path to directory containing val.bin.
    block_size : context window size (must match model configs).
    batch_size : batch size for evaluation.
    device : torch device string.

    Returns
    -------
    dict with ensemble_val_loss, ensemble_perplexity, individual_losses,
    and diversity_score.
    """
    from data.loader import SieveDataset, create_dataloader
    from worker.checkpoint import load_model_from_checkpoint

    if not checkpoint_paths:
        raise ValueError("No checkpoint paths provided")

    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    use_amp = dev.type == "cuda"
    dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    n_members = len(checkpoint_paths)

    logger.info("Evaluating ensemble of %d members on %s", n_members, dev)

    # Load validation dataset
    from pathlib import Path as _Path
    import json as _json

    val_path = _Path(data_dir) / "val.bin"
    meta_path = _Path(data_dir) / "metadata.json"
    data_dtype = "uint16"
    if meta_path.exists():
        meta = _json.loads(meta_path.read_text())
        if "32" in meta.get("dtype", "uint16"):
            data_dtype = "uint32"

    val_loader, _ = create_dataloader(
        data_path=str(val_path),
        block_size=block_size,
        batch_size=batch_size,
        shuffle=False,
        dtype=data_dtype,
    )

    # Load all member models
    models: list[nn.Module] = []
    for path in checkpoint_paths:
        model = load_model_from_checkpoint(path, device=str(dev))
        model.eval()
        models.append(model)
        logger.info("Loaded member: %s", path)

    # Evaluate individual models and ensemble simultaneously
    individual_losses = [0.0] * n_members
    individual_tokens = [0] * n_members
    ensemble_loss_sum = 0.0
    ensemble_tokens = 0

    # For diversity: accumulate pairwise KL stats per batch
    diversity_kl_sum = 0.0
    diversity_kl_count = 0

    for batch in val_loader:
        x, y = batch
        x = x.to(dev, non_blocking=True)
        y = y.to(dev, non_blocking=True)
        n_tokens = y.numel()

        # Collect logits from all members for this batch
        member_logits: list[torch.Tensor] = []
        for i, m in enumerate(models):
            with torch.amp.autocast("cuda", dtype=dtype, enabled=use_amp):
                logits, loss = m(x, targets=y)
            individual_losses[i] += loss.item() * n_tokens
            individual_tokens[i] += n_tokens
            # Keep logits in float32 for accurate averaging
            member_logits.append(logits.float())

        # Average logits across members
        # Stack: (n_members, batch, seq, vocab) -> mean over dim 0
        stacked = torch.stack(member_logits, dim=0)
        avg_logits = stacked.mean(dim=0)  # (batch, seq, vocab)

        # Compute cross-entropy on averaged logits
        # Reshape to (batch*seq, vocab) and (batch*seq,)
        B, T, V = avg_logits.shape
        ensemble_ce = F.cross_entropy(
            avg_logits.reshape(-1, V),
            y.reshape(-1),
            reduction="sum",
        )
        ensemble_loss_sum += ensemble_ce.item()
        ensemble_tokens += n_tokens

        # Compute pairwise KL divergence for diversity scoring
        if n_members > 1:
            # Use log_softmax for numerical stability
            log_probs = F.log_softmax(stacked, dim=-1)  # (n_members, B, T, V)
            probs = F.softmax(stacked, dim=-1)

            # Sum pairwise KL: KL(p_i || p_j) for all i != j
            for i in range(n_members):
                for j in range(i + 1, n_members):
                    # KL(p_i || p_j) = sum(p_i * (log_p_i - log_p_j))
                    kl_ij = F.kl_div(
                        log_probs[j].reshape(-1, V),
                        probs[i].reshape(-1, V),
                        reduction="batchmean",
                        log_target=False,
                    )
                    kl_ji = F.kl_div(
                        log_probs[i].reshape(-1, V),
                        probs[j].reshape(-1, V),
                        reduction="batchmean",
                        log_target=False,
                    )
                    # Symmetric KL
                    diversity_kl_sum += (kl_ij.item() + kl_ji.item()) / 2.0
                    diversity_kl_count += 1

        # Free batch logits immediately
        del member_logits, stacked, avg_logits

    # Finalize individual losses
    individual_results = []
    for i in range(n_members):
        avg = individual_losses[i] / max(1, individual_tokens[i])
        ppl = min(float("inf"), torch.exp(torch.tensor(avg)).item())
        individual_results.append({
            "checkpoint": checkpoint_paths[i],
            "val_loss": avg,
            "perplexity": ppl,
        })

    # Finalize ensemble metrics
    ensemble_avg_loss = ensemble_loss_sum / max(1, ensemble_tokens)
    ensemble_ppl = min(float("inf"), torch.exp(torch.tensor(ensemble_avg_loss)).item())

    # Diversity score: average symmetric KL divergence across all pairs
    diversity = diversity_kl_sum / max(1, diversity_kl_count) if diversity_kl_count > 0 else 0.0

    # Cleanup
    del models
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    result = {
        "ensemble_val_loss": ensemble_avg_loss,
        "ensemble_perplexity": ensemble_ppl,
        "individual_losses": individual_results,
        "diversity_score": diversity,
        "n_members": n_members,
        "total_tokens": ensemble_tokens,
    }

    logger.info(
        "Ensemble eval: loss=%.4f  ppl=%.2f  diversity=%.4f  members=%d",
        ensemble_avg_loss, ensemble_ppl, diversity, n_members,
    )
    return result


# ---------------------------------------------------------------------------
# Diversity measurement
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_diversity(logits_list: list[torch.Tensor]) -> float:
    """Measure disagreement among ensemble members via average pairwise KL divergence.

    Parameters
    ----------
    logits_list : list of logit tensors, each of shape (N, vocab_size) where N
        is the total number of token predictions. All tensors must have the
        same shape.

    Returns
    -------
    float : average symmetric KL divergence across all member pairs.
        Higher values indicate more diversity (more disagreement).
        Returns 0.0 if fewer than 2 members.
    """
    n = len(logits_list)
    if n < 2:
        return 0.0

    # Compute softmax/log_softmax once for each member
    log_probs = [F.log_softmax(lg.float(), dim=-1) for lg in logits_list]
    probs = [F.softmax(lg.float(), dim=-1) for lg in logits_list]

    total_kl = 0.0
    n_pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            kl_ij = F.kl_div(log_probs[j], probs[i], reduction="batchmean", log_target=False)
            kl_ji = F.kl_div(log_probs[i], probs[j], reduction="batchmean", log_target=False)
            total_kl += (kl_ij.item() + kl_ji.item()) / 2.0
            n_pairs += 1

    return total_kl / n_pairs


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="SIEVE ensemble evaluator")
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True,
                        help="Paths to member checkpoint files")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to prepared data directory")
    parser.add_argument("--block-size", type=int, default=512,
                        help="Context window size")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Evaluation batch size")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run evaluation on")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to write result JSON")
    args = parser.parse_args()

    result = evaluate_ensemble(
        checkpoint_paths=args.checkpoints,
        data_dir=args.data_dir,
        block_size=args.block_size,
        batch_size=args.batch_size,
        device=args.device,
    )

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(f"Result written to {args.output}")
    else:
        json.dump(result, sys.stdout, indent=2)
        print()
