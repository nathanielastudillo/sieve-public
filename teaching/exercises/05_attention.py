"""Lesson 05 — attention, residual stream, MLP.

Run: python teaching/exercises/05_attention.py
"""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from model.config import ModelConfig
from model.gpt import GPT

mcfg = ModelConfig(
    n_layer=2, n_head=4, n_embd=256, block_size=16,
    dropout=0.0, bias=False, vocab_size=8000,
)
model = GPT(mcfg)
model.eval()

B, T, D = 1, 8, mcfg.n_embd
idx = torch.randint(0, mcfg.vocab_size, (B, T))
pos = torch.arange(T)
x = model.transformer.wte(idx) + model.transformer.wpe(pos)
print(f"residual stream x: {tuple(x.shape)} (B, T, n_embd)")
print()

# 1. The Q/K/V projection
block = model.transformer.h[0]
c_attn = block.attn.c_attn  # Linear(n_embd -> 3*n_embd)
print(f"c_attn = Linear({c_attn.in_features} -> {c_attn.out_features})")
print(f"   It emits Q, K, V stacked together — one matmul instead of three.")
print(f"   weight shape: {tuple(c_attn.weight.shape)}")
print()

# Manually do one head of attention so you can SEE the weight matrix
with torch.no_grad():
    ln_x = block.ln_1(x)             # pre-norm
    qkv = c_attn(ln_x)               # (B, T, 3D)
    q, k, v = qkv.split(D, dim=2)    # each (B, T, D)

    # Take head 0 only so we get a plain (T, T) matrix to look at
    head_dim = D // mcfg.n_head
    q0 = q[:, :, :head_dim]          # (B, T, head_dim)
    k0 = k[:, :, :head_dim]
    v0 = v[:, :, :head_dim]

    # Compute scores
    scores = q0 @ k0.transpose(-2, -1) / (head_dim ** 0.5)  # (B, T, T)
    print(f"raw scores[0]  (T x T):")
    print(scores[0].round(decimals=2).tolist())
    print()

    # Apply causal mask manually
    mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
    scores_masked = scores.masked_fill(mask, float("-inf"))
    weights = F.softmax(scores_masked, dim=-1)
    print("attention weights after causal mask + softmax  (row t attends to cols 0..t):")
    for row in weights[0]:
        print("  " + " ".join(f"{w.item():.2f}" for w in row))
    print("  Notice: upper triangle is zero. Each row sums to 1.")
    print()

# 2. Show that the block ADDS to the residual stream
print("Residual structure in a block:")
with torch.no_grad():
    x0 = x.clone()
    attn_out = block.attn(block.ln_1(x0))
    x1 = x0 + attn_out
    mlp_out = block.mlp(block.ln_2(x1))
    x2 = x1 + mlp_out
print(f"  ||attn_out||  (the update)        = {attn_out.norm().item():.3f}")
print(f"  ||x before attn||                  = {x0.norm().item():.3f}")
print(f"  ||x after attn + residual||       = {x1.norm().item():.3f}")
print(f"  ||mlp_out||                       = {mlp_out.norm().item():.3f}")
print(f"  ||x after full block||            = {x2.norm().item():.3f}")
print()
print("Each sublayer writes a SMALL additive update. The stream carries forward.")
print()

# 3. SwiGLU MLP structure
mlp = block.mlp
print(f"SwiGLU MLP:")
print(f"  gate_proj: Linear({mlp.gate_proj.in_features} -> {mlp.gate_proj.out_features})")
print(f"  up_proj:   Linear({mlp.up_proj.in_features} -> {mlp.up_proj.out_features})")
print(f"  down_proj: Linear({mlp.down_proj.in_features} -> {mlp.down_proj.out_features})")
print(f"  output = down_proj( SiLU(gate_proj(x)) * up_proj(x) )")
print(f"  hidden = ceil(8 * n_embd / 3 / 64) * 64 = {mlp.gate_proj.out_features}")
