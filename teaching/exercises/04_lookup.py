"""Lesson 04 — embeddings, positions, weight tying.

Run: python teaching/exercises/04_lookup.py
"""
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from model.config import ModelConfig
from model.gpt import GPT

mcfg = ModelConfig(
    n_layer=4, n_head=4, n_embd=256, block_size=32,
    dropout=0.0, bias=False, vocab_size=8000,
)
model = GPT(mcfg)
model.eval()

# 1. The embedding IS a matrix. Shape (vocab_size, n_embd).
wte = model.transformer.wte.weight
wpe = model.transformer.wpe.weight
lm_head_w = model.lm_head.weight

print("Embedding matrices:")
print(f"  wte (token embed)     shape = {tuple(wte.shape)}")
print(f"  wpe (position embed)  shape = {tuple(wpe.shape)}")
print(f"  lm_head               shape = {tuple(lm_head_w.shape)}")
print()

# 2. Weight tying — literally the same tensor
print(f"lm_head.weight IS wte.weight? {lm_head_w is wte}")
print(f"  (they share memory: data_ptr match? {lm_head_w.data_ptr() == wte.data_ptr()})")
print()
print("Demonstration: mutate wte[0, 0] and watch lm_head[0, 0] change too.")
orig = wte[0, 0].item()
with torch.no_grad():
    wte[0, 0] = 999.0
print(f"  wte[0, 0]      = {wte[0, 0].item()}")
print(f"  lm_head[0, 0]  = {lm_head_w[0, 0].item()}  <-- same!")
with torch.no_grad():
    wte[0, 0] = orig   # restore

print()

# 3. Look up a single token's embedding
token_id = 4217
vec = wte[token_id]
print(f"Embedding for token ID {token_id}:")
print(f"  shape = {tuple(vec.shape)}")
print(f"  first 8 values = {vec[:8].tolist()}")
print("  This is random at init. After training it encodes what the model 'knows' about this token.")
print()

# 4. Trace the shapes through the forward pass manually
print("Forward pass shape trace:")
idx = torch.tensor([[10, 200, 4217, 50, 1, 999, 42, 7]])   # (B=1, T=8)
print(f"  input idx                : {tuple(idx.shape)}  [integers]")

tok_emb = model.transformer.wte(idx)                        # (B, T, D)
pos = torch.arange(0, idx.size(1))
pos_emb = model.transformer.wpe(pos)                        # (T, D)
print(f"  tok_emb = wte(idx)       : {tuple(tok_emb.shape)}")
print(f"  pos_emb = wpe(arange(T)) : {tuple(pos_emb.shape)}")

x = tok_emb + pos_emb                                       # broadcast over B
print(f"  x = tok_emb + pos_emb    : {tuple(x.shape)}   <-- this is the residual stream")

# Run through blocks
with torch.no_grad():
    for i, block in enumerate(model.transformer.h):
        x_before = x.clone()
        x = block(x)
        delta = (x - x_before).abs().mean().item()
        print(f"  after block {i}         : {tuple(x.shape)}  (avg |delta| added: {delta:.4f})")

x = model.transformer.ln_f(x)
logits = model.lm_head(x)
print(f"  logits = lm_head(x)      : {tuple(logits.shape)}   <-- (B, T, vocab_size)")
print()

print("Takeaways:")
print("  - Every operation preserves (B, T) — every input position produces an output.")
print("  - Each block ADDS a small update (residual) — it doesn't replace x.")
print("  - The final matmul through lm_head is the SAME matrix that embedded tokens in.")
