"""Lesson 03 — the sliding window (x, y) pair.

Run: python teaching/exercises/03_window.py
"""
import json
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "data/corpus_v008"

train = np.memmap(CORPUS / "train.bin", dtype="uint16", mode="r")
meta = json.loads((CORPUS / "metadata.json").read_text())
tok = Tokenizer.from_file(str(REPO / meta["tokenizer_path"]))

BLOCK_SIZE = 32   # small on purpose, so you can eyeball every token

# 1. How many distinct windows are there?
n_tokens = len(train)
n_windows = n_tokens - BLOCK_SIZE
print(f"stream has {n_tokens} tokens")
print(f"distinct (x, y) windows at block_size={BLOCK_SIZE}: {n_windows}")
print()

# 2. Pull one window at a fixed offset so you can see the shift
i = 1000
chunk = train[i : i + BLOCK_SIZE + 1].astype(np.int64)  # block_size+1 tokens
x = chunk[:-1]     # (block_size,)
y = chunk[1:]      # (block_size,)

print(f"Window at i={i}:")
print(f"  chunk (block_size+1 = {BLOCK_SIZE+1} tokens): {list(chunk)}")
print(f"  x  (inputs):  {list(x)}")
print(f"  y  (targets): {list(y)}")
print()
print("Verify y is x shifted right by 1:")
print(f"  x[1:]  == y[:-1]  ? {list(x[1:]) == list(y[:-1])}")
print(f"  y[0]  is the token at stream[{i+1}]: {y[0]} vs stream[{i+1}]={int(train[i+1])}")
print()

# 3. Decode both so you can SEE what the model learns to predict
print("x as text:")
print(f"  {tok.decode([int(t) for t in x])!r}")
print()
print("y as text (same passage starting one token later):")
print(f"  {tok.decode([int(t) for t in y])!r}")
print()

# 4. Build a mini-batch just like the dataloader would
#    Two windows drawn from two different positions
batch_indices = [1000, 5000]
xs = np.stack([train[i : i + BLOCK_SIZE].astype(np.int64) for i in batch_indices])
ys = np.stack([train[i + 1 : i + BLOCK_SIZE + 1].astype(np.int64) for i in batch_indices])
xs_t = torch.from_numpy(xs)
ys_t = torch.from_numpy(ys)
print(f"batch xs shape: {tuple(xs_t.shape)}  (B=batch, T=block_size)")
print(f"batch ys shape: {tuple(ys_t.shape)}")
print()

# 5. Now run a forward pass on this batch — demonstrate the output shape
#    (We'll use the same config as the smoke_v003 checkpoints so it loads.)
import sys
sys.path.insert(0, str(REPO))
from model.config import ModelConfig
from model.gpt import GPT

mcfg = ModelConfig(
    n_layer=4, n_head=4, n_embd=256, block_size=BLOCK_SIZE,
    dropout=0.0, bias=False, vocab_size=meta["vocab_size"],
)
model = GPT(mcfg)  # untrained! random weights. we just want to see shapes.
model.eval()
with torch.no_grad():
    logits, loss = model(xs_t, targets=ys_t)

print(f"logits shape: {tuple(logits.shape)}  <-- (B, T, vocab_size)")
print(f"   This is T={BLOCK_SIZE} next-token distributions per sequence, per sample.")
print(f"loss (untrained, random weights): {loss.item():.4f}")
print(f"   For reference: random-guess loss on vocab={mcfg.vocab_size} is ln({mcfg.vocab_size}) = {np.log(mcfg.vocab_size):.4f}")
print()
print("An untrained model's loss ≈ ln(vocab_size) because it outputs uniform noise.")
print("Training drives this down; our smoke_v003 checkpoints ended around ~4-5.")
