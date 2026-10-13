"""Lesson 07 — crack open a real checkpoint.

Run: python teaching/exercises/07_checkpoint.py
"""
import sys
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from model.config import ModelConfig
from model.gpt import GPT

CKPT = REPO / "runs/smoke_v003/checkpoints/step_013240_epoch_0005.pt"

# 1. It's just a pickled dict.
print(f"Opening: {CKPT.relative_to(REPO)}")
print(f"File size on disk: {CKPT.stat().st_size / 1024 / 1024:.1f} MB")
print()

ckpt = torch.load(str(CKPT), map_location="cpu", weights_only=False)

print(f"Top-level type: {type(ckpt).__name__}")
print(f"Top-level keys: {list(ckpt.keys())}")
print()

# 2. Poke at each top-level key
print(f"config (the architecture spec):")
for k, v in ckpt["config"].items():
    print(f"  {k:15s} = {v}")
print()

print(f"metrics (training state at save time):")
for k, v in ckpt["metrics"].items():
    print(f"  {k:15s} = {v}")
print()

print(f"pytorch_version: {ckpt['pytorch_version']}")
print(f"timestamp:       {ckpt['timestamp']}")
print()

# 3. The model_state_dict — every weight by name
sd = ckpt["model_state_dict"]
print(f"model_state_dict has {len(sd)} entries. First 15:")
for i, (name, tensor) in enumerate(sd.items()):
    if i >= 15: break
    print(f"  {name:45s}  {tuple(tensor.shape)}  {tensor.dtype}")
print(f"  ... ({len(sd) - 15} more)")
print()

total_params = sum(t.numel() for t in sd.values())
total_bytes  = sum(t.numel() * t.element_size() for t in sd.values())
print(f"total params in state_dict:  {total_params:,}")
print(f"total bytes (tensors only):  {total_bytes / 1024 / 1024:.1f} MB")
print()

# 4. RECONSTRUCT the model from the config + state_dict
mcfg = ModelConfig.from_dict(ckpt["config"])
model = GPT(mcfg)             # empty tensors of the right shape
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"Reconstructed model:")
print(f"  missing keys    (in class, not in ckpt): {len(missing)}  {missing if missing else ''}")
print(f"  unexpected keys (in ckpt, not in class): {len(unexpected)}  {unexpected if unexpected else ''}")
print(f"  total params in model: {sum(p.numel() for p in model.parameters()):,}")
print()

# 5. Verify weight tying survived round-trip
print(f"After load: lm_head.weight is wte.weight? {model.lm_head.weight is model.transformer.wte.weight}")
print(f"(They should be the same object — weight tying is reconstructed by the class's __init__.)")
print()

# 6. Is the checkpoint on its own enough to serve the model?
print("Summary:")
print("  - config lets us build the architecture shell")
print("  - model_state_dict fills the tensors")
print("  - optimizer_state_dict is only needed to RESUME training, not for inference")
print("  - metrics/timestamp are pure metadata")
print()
print("So the minimum to do inference: config + model_state_dict + the GPT class definition.")
