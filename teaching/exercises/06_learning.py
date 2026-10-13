"""Lesson 06 — one manual training step.

Run: python teaching/exercises/06_learning.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from model.config import ModelConfig
from model.gpt import GPT

CORPUS = REPO / "data/corpus_v008"
meta = json.loads((CORPUS / "metadata.json").read_text())
train = np.memmap(CORPUS / "train.bin", dtype="uint16", mode="r")

BLOCK, BATCH = 64, 8
mcfg = ModelConfig(
    n_layer=4, n_head=4, n_embd=256, block_size=BLOCK,
    dropout=0.0, bias=False, vocab_size=meta["vocab_size"],
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT(mcfg).to(device)
model.train()

# Simplest possible optimizer — plain AdamW
optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

# Build one batch from the real corpus
rng = np.random.default_rng(0)
starts = rng.integers(0, len(train) - BLOCK - 1, size=BATCH)
xs = np.stack([train[s : s + BLOCK].astype(np.int64)     for s in starts])
ys = np.stack([train[s + 1 : s + BLOCK + 1].astype(np.int64) for s in starts])
x = torch.from_numpy(xs).to(device)
y = torch.from_numpy(ys).to(device)

print(f"device = {device}")
print(f"batch: x {tuple(x.shape)}, y {tuple(y.shape)}")
print(f"expected untrained loss ≈ ln(vocab_size) = {math.log(mcfg.vocab_size):.4f}")
print()

# ---- one full training step, broken open ----

# 1. FORWARD
logits, loss = model(x, targets=y)
print(f"step 0 — loss BEFORE update: {loss.item():.4f}")
print(f"  logits shape: {tuple(logits.shape)}  (B, T, vocab)")

# 2. BACKWARD — compute gradients for every param
loss.backward()

# Show a few gradients so you see they're real numbers, not magic
print("  gradient norms for a few parameters:")
for name, p in list(model.named_parameters())[:5]:
    if p.grad is not None:
        print(f"    {name:40s}  grad.norm() = {p.grad.norm().item():.4f}")
print()

# Save the BEFORE weights so we can verify the update actually moved them
before = {n: p.detach().clone() for n, p in model.named_parameters() if "wte" in n or "h.0.attn.c_attn" in n}

# 3. STEP — optimizer uses grads to update params
optim.step()
optim.zero_grad()  # clear grads for next iter

# Verify params moved
print("How much did weights move?")
for n, p in model.named_parameters():
    if n in before:
        delta = (p - before[n]).abs().mean().item()
        print(f"  {n:40s}  mean |Δw| = {delta:.6f}")
print()

# 4. Look at loss after — should be slightly lower on the same batch
with torch.no_grad():
    _, loss_after = model(x, targets=y)
print(f"step 0 — loss AFTER  update: {loss_after.item():.4f}")
print(f"           change: {(loss_after - loss).item():+.4f}")
print()

# 5. Run 50 more steps to watch loss drop over a real (tiny) training run
print("Running 50 more steps on the same batch (overfitting one batch on purpose):")
for i in range(1, 51):
    logits, loss = model(x, targets=y)
    loss.backward()
    optim.step()
    optim.zero_grad()
    if i % 10 == 0:
        print(f"  step {i:3d}  loss = {loss.item():.4f}")

print()
print("Loss drops fast when the model is just memorizing one batch.")
print("Real training samples a NEW batch every step, which is harder —")
print("loss declines more slowly but the model generalizes.")
