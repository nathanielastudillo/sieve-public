"""Single model training loop for SIEVE.

Trains one GPT model to completion and returns metrics + checkpoint paths.
Designed to run standalone on a GB10 worker node, invokable via HTTP or CLI.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from model.config import ModelConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Training hyperparameters for a single model run."""

    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.3
    warmup_fraction: float = 0.05
    lr_schedule: str = "cosine"  # cosine | linear | constant_with_warmup
    optimizer_name: str = "muon"  # muon | adamw
    gradient_clip: float = 1.0
    shuffle_per_epoch: bool = True
    seed: int = 42
    device: str = "cuda"
    checkpoint_dir: str = "./checkpoints"
    checkpoint_every: int = 10  # save every N epochs
    # Stochastic Weight Averaging: keep a running average of model weights
    # over the last `swa_tail_fraction` of training epochs. Saved as a
    # second checkpoint (`*.swa.pt`) alongside the best-val checkpoint,
    # so the ensemble evaluator can use whichever is better.
    # References:
    #   Izmailov et al. 2018 — SWA (arxiv.org/abs/1803.05407): weight
    #     averaging along the SGD trajectory finds wider minima.
    #   Sanyal et al. 2023 — LAWA (arxiv.org/abs/2306.03241): on GPT-2
    #     small + OpenWebText (70K steps), val loss baseline 2.963 →
    #     SWA 2.952 → LAWA 2.917. LM pretraining-specific validation.
    # Near-zero compute overhead (one extra param-copy per end-of-epoch).
    # See lab-notes/research-questions/literature-small-corpus-ensembles.md §6.
    swa_enabled: bool = True
    swa_tail_fraction: float = 0.2  # average over last 20% of epochs

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainConfig:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


@dataclass
class TrainResult:
    """Outcome of a completed training run."""

    final_loss: float
    best_loss: float
    checkpoint_paths: list[str]
    model_config: dict[str, Any]
    train_config: dict[str, Any]
    wallclock_time: float  # seconds
    tokens_seen: int
    loss_history: list[float] = field(default_factory=list)
    # Held-out validation metrics (on val.bin). best_val_loss is what should
    # drive ensemble fitness; best_loss is training loss and overfits fast.
    final_val_loss: float = float("inf")
    best_val_loss: float = float("inf")
    val_loss_history: list[float] = field(default_factory=list)
    best_val_checkpoint: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Learning rate scheduler
# ---------------------------------------------------------------------------

def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    schedule_name: str,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """Create a LambdaLR scheduler supporting cosine, linear, and constant_with_warmup.

    Parameters
    ----------
    optimizer : the torch optimizer to schedule.
    schedule_name : one of "cosine", "linear", "constant_with_warmup".
    warmup_steps : number of linear warmup steps from 0 to base LR.
    total_steps : total number of training steps (warmup included).
    """
    if total_steps < 1:
        raise ValueError(f"total_steps must be >= 1, got {total_steps}")
    warmup_steps = max(0, min(warmup_steps, total_steps))

    def _cosine(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def _linear(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)

    def _constant_with_warmup(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0

    schedules = {
        "cosine": _cosine,
        "linear": _linear,
        "constant_with_warmup": _constant_with_warmup,
    }
    if schedule_name not in schedules:
        raise ValueError(
            f"Unknown schedule '{schedule_name}'. Choose from {list(schedules)}"
        )
    return LambdaLR(optimizer, schedules[schedule_name])


# ---------------------------------------------------------------------------
# Optimizer construction
# ---------------------------------------------------------------------------

def _build_optimizer(
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Build an optimizer, handling muon's split parameter groups.

    For "muon": weight matrices (2D params in attention/MLP) go to Muon;
    embeddings, biases, layernorms, and 1D params go to AdamW.

    For "adamw": all parameters go to AdamW with standard weight-decay
    separation (no decay for biases and layernorm weights).
    """
    if optimizer_name == "muon":
        from model.muon import Muon

        muon_params = []
        adamw_params_decay = []
        adamw_params_nodecay = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # 2D weight matrices in transformer blocks -> Muon
            # Exclude embedding weights and lm_head
            is_embedding = "wte" in name or "wpe" in name or "lm_head" in name
            is_layernorm = "ln" in name or "norm" in name
            is_bias = name.endswith(".bias")

            if param.dim() >= 2 and not is_embedding and not is_layernorm:
                muon_params.append(param)
            elif is_bias or is_layernorm:
                adamw_params_nodecay.append(param)
            else:
                adamw_params_decay.append(param)

        param_groups = [
            {"params": muon_params, "muon": True},
            {"params": adamw_params_decay, "muon": False},
            {"params": adamw_params_nodecay, "muon": False, "adamw_wd": 0.0},
        ]
        optimizer = Muon(
            param_groups,
            lr=learning_rate,
            momentum=0.95,
            adamw_lr=learning_rate,
            adamw_betas=(0.9, 0.95),
            adamw_wd=weight_decay,
        )
    elif optimizer_name == "adamw":
        decay_params = []
        nodecay_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            is_layernorm = "ln" in name or "norm" in name
            is_bias = name.endswith(".bias")
            if is_bias or is_layernorm:
                nodecay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "lr": learning_rate, "weight_decay": weight_decay},
                {"params": nodecay_params, "lr": learning_rate, "weight_decay": 0.0},
            ],
            betas=(0.9, 0.95),
        )
    else:
        raise ValueError(f"Unknown optimizer '{optimizer_name}'. Choose 'muon' or 'adamw'.")

    return optimizer


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train_model(
    model_config: dict[str, Any],
    train_config: dict[str, Any],
    data_dir: str,
) -> TrainResult:
    """Train a single GPT model end-to-end.

    Parameters
    ----------
    model_config : dict of ModelConfig fields.
    train_config : dict of TrainConfig fields.
    data_dir : path to directory containing train.bin and val.bin
        (output of ``data.prepare.prepare_corpus``).

    Returns
    -------
    TrainResult with final metrics, checkpoint paths, and timing info.
    """
    from data.loader import SieveDataset, create_dataloader
    from worker.checkpoint import save_checkpoint

    # --- parse configs ---
    mcfg = ModelConfig.from_dict(model_config)
    tcfg = TrainConfig.from_dict(train_config)

    logger.info("Training model: %s", mcfg)
    logger.info("Train config: epochs=%d, bs=%d, lr=%.2e, wd=%.2e, opt=%s, schedule=%s",
                tcfg.epochs, tcfg.batch_size, tcfg.learning_rate, tcfg.weight_decay,
                tcfg.optimizer_name, tcfg.lr_schedule)

    # --- reproducibility ---
    torch.manual_seed(tcfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(tcfg.seed)

    # --- device ---
    device = torch.device(tcfg.device if torch.cuda.is_available() or tcfg.device == "cpu" else "cpu")
    use_amp = device.type == "cuda"
    dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and dtype == torch.float16))

    # --- model ---
    from model.gpt import GPT
    model = GPT(mcfg)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %s", f"{n_params:,}")

    # --- try torch.compile ---
    compiled = False
    # Skip torch.compile — it requires triton build toolchain (Python.h)
    # which may not be available. The overhead for small models is minimal.

    # --- optimizer ---
    optimizer = _build_optimizer(model, tcfg.optimizer_name, tcfg.learning_rate, tcfg.weight_decay)

    # --- dataset ---
    data_dir_path = Path(data_dir)
    train_path = data_dir_path / "train.bin"

    # Determine dtype from metadata
    meta_path = data_dir_path / "metadata.json"
    data_dtype = "uint16"
    if meta_path.exists():
        import json as _json
        meta = _json.loads(meta_path.read_text())
        if "32" in meta.get("dtype", "uint16"):
            data_dtype = "uint32"

    # Create dataloader (returns DataLoader + EpochShuffler)
    dataloader, sampler = create_dataloader(
        data_path=str(train_path),
        block_size=mcfg.block_size,
        batch_size=tcfg.batch_size,
        seed=tcfg.seed,
        shuffle=tcfg.shuffle_per_epoch,
        dtype=data_dtype,
    )
    n_tokens_per_epoch = len(dataloader.dataset) * mcfg.block_size

    # --- load val.bin into a fixed tensor of (N, block_size+1) chunks ---
    val_path = data_dir_path / "val.bin"
    val_inputs: Optional[torch.Tensor] = None
    val_targets: Optional[torch.Tensor] = None
    if val_path.exists():
        import numpy as _np
        _np_dtype = _np.uint32 if data_dtype == "uint32" else _np.uint16
        val_mm = _np.memmap(str(val_path), dtype=_np_dtype, mode="r")
        _T = mcfg.block_size
        _n_chunks = max(1, (len(val_mm) - 1) // _T)
        _chunks = []
        for _i in range(_n_chunks):
            _s = _i * _T
            _e = _s + _T + 1
            if _e > len(val_mm):
                break
            _chunks.append(_np.array(val_mm[_s:_e], dtype=_np.int64))
        if _chunks:
            _stk = torch.from_numpy(_np.stack(_chunks)).to(torch.long)
            val_inputs = _stk[:, :-1].contiguous()
            val_targets = _stk[:, 1:].contiguous()
            logger.info("Val set: %d chunks x %d tokens (total %d tokens)",
                        val_inputs.shape[0], _T, val_inputs.numel())
    else:
        logger.warning("No val.bin at %s — skipping held-out evaluation.", val_path)

    def _eval_val_loss() -> float:
        """CE on val chunks. Returns float('inf') if no val set."""
        if val_inputs is None:
            return float("inf")
        model.eval()
        total = 0.0
        total_tokens = 0
        with torch.no_grad():
            for _s in range(0, val_inputs.shape[0], tcfg.batch_size):
                _x = val_inputs[_s:_s + tcfg.batch_size].to(device, non_blocking=True)
                _y = val_targets[_s:_s + tcfg.batch_size].to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=dtype, enabled=use_amp):
                    _, _loss = model(_x, targets=_y)
                _n = _y.numel()
                total += float(_loss.item()) * _n
                total_tokens += _n
        model.train()
        return total / max(1, total_tokens)

    # Estimate total steps for scheduler
    steps_per_epoch = len(dataloader)
    total_steps = steps_per_epoch * tcfg.epochs
    warmup_steps = int(total_steps * tcfg.warmup_fraction)

    scheduler = get_lr_scheduler(optimizer, tcfg.lr_schedule, warmup_steps, total_steps)

    # --- checkpoint dir ---
    ckpt_dir = Path(tcfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- SWA (Stochastic Weight Averaging) ---
    # Izmailov 2018 / LAWA 2023. Average weights over the last
    # `swa_tail_fraction` of epochs. Saved as a sibling checkpoint to
    # `best.pt` so the ensemble evaluator can choose per-member.
    # We update *at the end of each tail epoch* rather than per-step:
    # simpler, cheaper, and matches the SWA paper's recommended schedule.
    swa_start_epoch = (
        int((1.0 - tcfg.swa_tail_fraction) * tcfg.epochs)
        if tcfg.swa_enabled else tcfg.epochs + 1
    )
    swa_model = None
    if tcfg.swa_enabled and swa_start_epoch < tcfg.epochs:
        from torch.optim.swa_utils import AveragedModel
        swa_model = AveragedModel(model)
        logger.info(
            "SWA enabled: averaging weights from epoch %d onward (last %.0f%% of training)",
            swa_start_epoch + 1, 100 * tcfg.swa_tail_fraction,
        )

    # --- training loop ---
    best_loss = float("inf")
    best_val_loss = float("inf")
    best_val_ckpt: Optional[str] = None
    val_loss_history: list[float] = []
    checkpoint_paths: list[str] = []
    loss_history: list[float] = []
    tokens_seen = 0
    global_step = 0
    t_start = time.perf_counter()

    for epoch in range(tcfg.epochs):
        # Set epoch for shuffle seeding
        sampler.set_epoch(epoch)

        model.train()
        epoch_loss_sum = 0.0
        epoch_steps = 0
        epoch_tokens = 0
        t_epoch = time.perf_counter()

        for batch in dataloader:
            x, y = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=dtype, enabled=use_amp):
                _logits, loss = model(x, targets=y)

            scaler.scale(loss).backward()

            # Gradient clipping (unscale first for accurate norms)
            if tcfg.gradient_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), tcfg.gradient_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            batch_loss = loss.item()
            batch_tokens = x.numel()
            epoch_loss_sum += batch_loss * batch_tokens
            epoch_tokens += batch_tokens
            epoch_steps += 1
            global_step += 1
            tokens_seen += batch_tokens

        # Epoch summary
        epoch_avg_loss = epoch_loss_sum / max(1, epoch_tokens)
        loss_history.append(epoch_avg_loss)
        elapsed_epoch = time.perf_counter() - t_epoch
        tokens_per_sec = epoch_tokens / max(1e-6, elapsed_epoch)
        current_lr = scheduler.get_last_lr()[0]

        if epoch_avg_loss < best_loss:
            best_loss = epoch_avg_loss

        # --- SWA: fold this epoch's weights into the running average ---
        # Only in the tail `swa_tail_fraction` of epochs. Per Izmailov
        # 2018 §3.1: early-training weights are noisy and poison the
        # average; restricting to the tail is standard.
        if swa_model is not None and epoch >= swa_start_epoch:
            swa_model.update_parameters(model)

        # --- held-out val evaluation ---
        val_loss = _eval_val_loss()
        val_loss_history.append(val_loss)
        improved_val = val_loss < best_val_loss
        if improved_val:
            best_val_loss = val_loss

        logger.info(
            "epoch=%d/%d  train=%.4f  val=%.4f  best_val=%.4f  lr=%.2e  tok/s=%.0f  step=%d",
            epoch + 1, tcfg.epochs, epoch_avg_loss, val_loss, best_val_loss,
            current_lr, tokens_per_sec, global_step,
        )

        # Periodic training checkpoint (every N epochs and at final epoch)
        is_final_epoch = (epoch + 1) == tcfg.epochs
        want_periodic = (epoch + 1) % tcfg.checkpoint_every == 0 or is_final_epoch
        if want_periodic or improved_val:
            ckpt_path = ckpt_dir / f"step_{global_step:06d}_epoch_{epoch + 1:04d}.pt"
            metrics = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "train_loss": epoch_avg_loss,
                "best_loss": best_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "tokens_seen": tokens_seen,
                "learning_rate": current_lr,
            }
            # Unwrap compiled model if needed
            raw_model = model._orig_mod if compiled and hasattr(model, "_orig_mod") else model
            save_checkpoint(raw_model, optimizer, mcfg.to_dict(), metrics, str(ckpt_path))
            if want_periodic:
                checkpoint_paths.append(str(ckpt_path))
                logger.info("Saved checkpoint: %s", ckpt_path)
            if improved_val:
                best_val_ckpt = str(ckpt_path)
                logger.info("  ↑ new best val_loss=%.4f at %s", best_val_loss, ckpt_path)

    # --- SWA: save the averaged model alongside the best-val checkpoint ---
    # The SWA checkpoint is a sibling: `*.swa.pt`. Stored in the same
    # format as the regular checkpoints (via save_checkpoint) so the
    # existing ensemble loader handles it unchanged. Evaluating both
    # variants lets Caruana greedy pick whichever member produces a
    # better contribution — sometimes SWA wins, sometimes raw wins,
    # letting the selector decide costs one extra forward-pass per
    # member at ensemble-eval time and is well worth it.
    swa_ckpt_path: Optional[str] = None
    if swa_model is not None and best_val_ckpt is not None:
        try:
            # Sibling path: same stem as the best-val checkpoint but with
            # the ".swa.pt" suffix. update_bn would normally be called
            # here to recompute BatchNorm stats, but GPT uses only
            # LayerNorm (stateless), so we can skip it — saves one pass
            # over the training set.
            swa_ckpt_path = best_val_ckpt.replace(".pt", ".swa.pt")

            # AveragedModel wraps the underlying module in `.module`.
            raw_swa = swa_model.module
            raw_swa_unwrapped = (
                raw_swa._orig_mod if compiled and hasattr(raw_swa, "_orig_mod")
                else raw_swa
            )
            swa_metrics = {
                "source_checkpoint": best_val_ckpt,
                "swa_tail_fraction": tcfg.swa_tail_fraction,
                "swa_start_epoch": swa_start_epoch + 1,
                "epochs_averaged": tcfg.epochs - swa_start_epoch,
            }
            save_checkpoint(
                raw_swa_unwrapped, optimizer, mcfg.to_dict(),
                swa_metrics, swa_ckpt_path,
            )
            checkpoint_paths.append(swa_ckpt_path)
            logger.info("Saved SWA checkpoint: %s", swa_ckpt_path)
        except Exception as e:
            # SWA is an optional improvement; if anything goes wrong with
            # the save, log it and continue — the non-SWA checkpoint is
            # still valid.
            logger.warning("SWA save failed (non-fatal): %s", e)
            swa_ckpt_path = None

    wallclock = time.perf_counter() - t_start
    final_val = val_loss_history[-1] if val_loss_history else float("inf")
    logger.info(
        "Training complete: final_train=%.4f best_train=%.4f final_val=%.4f best_val=%.4f time=%.1fs tokens=%d",
        loss_history[-1] if loss_history else float("inf"),
        best_loss, final_val, best_val_loss, wallclock, tokens_seen,
    )

    # Ensure the best-val checkpoint is included in the returned list so
    # callers (orchestrator) can pick it up deterministically.
    if best_val_ckpt is not None and best_val_ckpt not in checkpoint_paths:
        checkpoint_paths.append(best_val_ckpt)

    return TrainResult(
        final_loss=loss_history[-1] if loss_history else float("inf"),
        best_loss=best_loss,
        checkpoint_paths=checkpoint_paths,
        model_config=mcfg.to_dict(),
        train_config=tcfg.to_dict(),
        wallclock_time=wallclock,
        tokens_seen=tokens_seen,
        loss_history=loss_history,
        final_val_loss=final_val,
        best_val_loss=best_val_loss,
        val_loss_history=val_loss_history,
        best_val_checkpoint=best_val_ckpt,
    )


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

    parser = argparse.ArgumentParser(description="SIEVE single model trainer")
    parser.add_argument("--model-config", type=str, required=True,
                        help="Path to model config JSON file")
    parser.add_argument("--train-config", type=str, required=True,
                        help="Path to training config JSON file")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to prepared data directory (train.bin, val.bin)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to write result JSON")
    args = parser.parse_args()

    with open(args.model_config) as f:
        mcfg = json.load(f)
    with open(args.train_config) as f:
        tcfg = json.load(f)

    result = train_model(mcfg, tcfg, args.data_dir)

    if args.output:
        Path(args.output).write_text(json.dumps(result.to_dict(), indent=2))
        print(f"Result written to {args.output}")
    else:
        json.dump(result.to_dict(), sys.stdout, indent=2)
        print()
