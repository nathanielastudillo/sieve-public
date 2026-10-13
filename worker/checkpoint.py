"""Checkpoint save/load/transfer for SIEVE.

Handles model + optimizer state persistence, metadata tracking, and
cross-node checkpoint transfers via rsync for distributed operation.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Checkpoint filename convention:
#   step_{step:06d}_epoch_{epoch:04d}.pt
#   gen_{generation:04d}_member_{member_id:04d}.pt
_CKPT_PATTERN = re.compile(r".*\.pt$")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    metrics: dict[str, Any],
    path: str | Path,
) -> None:
    """Save a training checkpoint.

    Parameters
    ----------
    model : the GPT model (use the unwrapped model if torch.compiled).
    optimizer : the optimizer (state dict will be saved).
    config : model configuration dict (ModelConfig.to_dict()).
    metrics : training metrics dict (loss, epoch, step, etc.).
    path : file path to write the checkpoint.

    The checkpoint dict contains:
        model_state_dict, optimizer_state_dict, config, metrics,
        timestamp, and pytorch_version.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "metrics": metrics,
        "timestamp": time.time(),
        "pytorch_version": torch.__version__,
    }

    # Write to a temp file first, then rename for atomicity
    tmp_path = path.with_suffix(".pt.tmp")
    torch.save(checkpoint, str(tmp_path))
    tmp_path.rename(path)

    size_mb = path.stat().st_size / (1024 * 1024)
    logger.debug("Saved checkpoint: %s (%.1f MB)", path, size_mb)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_checkpoint(
    path: str | Path,
    device: str = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint from disk.

    Parameters
    ----------
    path : path to the .pt checkpoint file.
    device : device to map tensors to (default "cpu" for safe loading).

    Returns
    -------
    dict with keys: model_state_dict, optimizer_state_dict, config,
    metrics, timestamp, pytorch_version.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(str(path), map_location=device, weights_only=False)

    # Validate expected keys
    expected = {"model_state_dict", "config"}
    missing = expected - set(checkpoint.keys())
    if missing:
        raise ValueError(f"Checkpoint {path} missing keys: {missing}")

    logger.debug("Loaded checkpoint: %s (device=%s)", path, device)
    return checkpoint


def load_model_from_checkpoint(
    path: str | Path,
    device: str = "cpu",
) -> nn.Module:
    """Load a checkpoint and return an instantiated GPT model.

    Convenience function that handles ModelConfig construction and
    state_dict loading.

    Parameters
    ----------
    path : path to the .pt checkpoint file.
    device : device to place the model on.

    Returns
    -------
    GPT model with loaded weights, in eval mode.
    """
    from model.config import ModelConfig
    from model.gpt import GPT

    ckpt = load_checkpoint(path, device=device)

    mcfg = ModelConfig.from_dict(ckpt["config"])
    model = GPT(mcfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    logger.info("Loaded model from %s (%s, %s params)", path, mcfg, f"{mcfg.n_params:,}")
    return model


# ---------------------------------------------------------------------------
# Transfer
# ---------------------------------------------------------------------------

def transfer_checkpoint(
    src_path: str | Path,
    dest_host: str,
    dest_path: str | Path,
    timeout: int = 300,
    bandwidth_limit: Optional[str] = None,
) -> None:
    """Transfer a checkpoint to another node via rsync.

    Parameters
    ----------
    src_path : local path to the checkpoint file.
    dest_host : destination hostname or IP (e.g., "gb10-worker-3" or "<node-ip>").
    dest_path : path on the destination node.
    timeout : rsync timeout in seconds.
    bandwidth_limit : optional rsync --bwlimit value (e.g., "50m" for 50MB/s).
    """
    src_path = Path(src_path)
    if not src_path.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {src_path}")

    cmd = [
        "rsync",
        "-az",  # archive + compress
        "--progress",
        f"--timeout={timeout}",
    ]
    if bandwidth_limit:
        cmd.append(f"--bwlimit={bandwidth_limit}")

    cmd.extend([
        str(src_path),
        f"{dest_host}:{dest_path}",
    ])

    logger.info("Transferring %s -> %s:%s", src_path, dest_host, dest_path)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 30,  # subprocess timeout slightly longer than rsync timeout
            check=True,
        )
        logger.info("Transfer complete: %s -> %s:%s", src_path, dest_host, dest_path)
    except subprocess.CalledProcessError as e:
        logger.error("rsync failed (exit %d): %s", e.returncode, e.stderr.strip())
        raise RuntimeError(f"Checkpoint transfer failed: {e.stderr.strip()}") from e
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"Checkpoint transfer timed out after {timeout + 30}s: "
            f"{src_path} -> {dest_host}:{dest_path}"
        )


# ---------------------------------------------------------------------------
# Listing / discovery
# ---------------------------------------------------------------------------

def list_checkpoints(checkpoint_dir: str | Path) -> list[dict[str, Any]]:
    """List all checkpoints in a directory with metadata.

    Scans for .pt files, loads their metadata (config + metrics) without
    loading the full model weights, and returns a sorted list.

    Parameters
    ----------
    checkpoint_dir : path to search for checkpoint files.

    Returns
    -------
    list of dicts, each with: path, filename, generation (if present),
    member_id (if present), val_loss (if present), train_loss (if present),
    epoch, global_step, timestamp, file_size_mb.
    Sorted by timestamp (newest first).
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return []

    results: list[dict[str, Any]] = []

    for pt_file in sorted(checkpoint_dir.rglob("*.pt")):
        if pt_file.suffix != ".pt":
            continue
        # Skip temp files
        if pt_file.name.endswith(".pt.tmp"):
            continue

        entry: dict[str, Any] = {
            "path": str(pt_file),
            "filename": pt_file.name,
            "file_size_mb": pt_file.stat().st_size / (1024 * 1024),
        }

        # Try to load metadata without loading full tensors
        try:
            # Load with map_location=cpu to avoid GPU memory
            ckpt = torch.load(str(pt_file), map_location="cpu", weights_only=False)

            # Extract metadata from metrics dict
            metrics = ckpt.get("metrics", {})
            entry["epoch"] = metrics.get("epoch")
            entry["global_step"] = metrics.get("global_step")
            entry["train_loss"] = metrics.get("train_loss")
            entry["val_loss"] = metrics.get("val_loss")
            entry["best_loss"] = metrics.get("best_loss")
            entry["tokens_seen"] = metrics.get("tokens_seen")

            # Evolution metadata (set by controller)
            entry["generation"] = metrics.get("generation")
            entry["member_id"] = metrics.get("member_id")

            # Timestamp
            entry["timestamp"] = ckpt.get("timestamp")

            # Config summary
            config = ckpt.get("config", {})
            entry["n_params"] = ModelConfig.from_dict(config).n_params if config else None

        except Exception as e:
            logger.warning("Failed to read metadata from %s: %s", pt_file, e)
            entry["error"] = str(e)

        results.append(entry)

    # Sort by timestamp, newest first
    results.sort(key=lambda x: x.get("timestamp") or 0, reverse=True)
    return results


# Need ModelConfig for list_checkpoints — import at module level is fine
# since config.py has no heavy dependencies
from model.config import ModelConfig


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="SIEVE checkpoint manager")
    sub = parser.add_subparsers(dest="command")

    # list
    ls_parser = sub.add_parser("list", help="List checkpoints in a directory")
    ls_parser.add_argument("dir", type=str, help="Checkpoint directory")

    # transfer
    tx_parser = sub.add_parser("transfer", help="Transfer a checkpoint to another node")
    tx_parser.add_argument("src", type=str, help="Source checkpoint path")
    tx_parser.add_argument("host", type=str, help="Destination host")
    tx_parser.add_argument("dest", type=str, help="Destination path")
    tx_parser.add_argument("--bwlimit", type=str, default=None, help="Bandwidth limit")

    # info
    info_parser = sub.add_parser("info", help="Show checkpoint metadata")
    info_parser.add_argument("path", type=str, help="Checkpoint file path")

    args = parser.parse_args()

    if args.command == "list":
        ckpts = list_checkpoints(args.dir)
        if not ckpts:
            print("No checkpoints found.")
        else:
            for c in ckpts:
                ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(c["timestamp"])) if c.get("timestamp") else "unknown"
                loss_str = f"loss={c['train_loss']:.4f}" if c.get("train_loss") is not None else "loss=?"
                gen_str = f"gen={c['generation']}" if c.get("generation") is not None else ""
                print(f"  {c['filename']:40s}  {loss_str:16s}  {gen_str:10s}  {c['file_size_mb']:.1f}MB  {ts}")

    elif args.command == "transfer":
        transfer_checkpoint(args.src, args.host, args.dest, bandwidth_limit=args.bwlimit)

    elif args.command == "info":
        ckpt = load_checkpoint(args.path)
        print(f"Config: {json.dumps(ckpt.get('config', {}), indent=2)}")
        print(f"Metrics: {json.dumps(ckpt.get('metrics', {}), indent=2)}")
        print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ckpt.get('timestamp', 0)))}")
        print(f"PyTorch: {ckpt.get('pytorch_version', 'unknown')}")

    else:
        parser.print_help()
