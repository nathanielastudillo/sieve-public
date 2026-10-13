"""Lesson 08 — generation from a real checkpoint.

Run: python teaching/exercises/08_generate.py
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from model.config import ModelConfig
from model.gpt import GPT

CKPT = REPO / "runs/smoke_v003/checkpoints/step_052960_epoch_0020.pt"
CORPUS = REPO / "data/corpus_v008"
meta = json.loads((CORPUS / "metadata.json").read_text())
tok = Tokenizer.from_file(str(REPO / meta["tokenizer_path"]))

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load(str(CKPT), map_location=device, weights_only=False)
mcfg = ModelConfig.from_dict(ckpt["config"])
model = GPT(mcfg)
model.load_state_dict(ckpt["model_state_dict"])
model.to(device).eval()
print(f"Loaded {CKPT.name} ({mcfg.n_params:,} params) on {device}")
print(f"Training stopped at: {ckpt['metrics']}")
print()

PROMPT = "The subject is"
ids = tok.encode(PROMPT).ids
idx = torch.tensor([ids], dtype=torch.long, device=device)
print(f"Prompt: {PROMPT!r}  →  ids: {ids}")
print()

# 1. What does the model predict for the NEXT token, as a probability distribution?
with torch.no_grad():
    # Use the training branch (pass targets=None but get full logits) via a hack:
    # run forward, get (B, 1, V) from the inference branch, that IS the next-token distn
    logits, _ = model(idx, targets=None)    # (B, 1, V)
    probs = F.softmax(logits[0, -1], dim=-1)

topk = torch.topk(probs, 10)
print("Top 10 candidates for the next token:")
for p, i in zip(topk.values.tolist(), topk.indices.tolist()):
    piece = tok.id_to_token(i)
    print(f"  id={i:5d}  p={p:.4f}  piece={piece!r}")
print()

# 2. Temperature sweep
print("=" * 60)
print("TEMPERATURE SWEEP (top_k=50, 60 new tokens, seed=0)")
print("=" * 60)
for T in [0.3, 0.7, 1.0, 1.5]:
    torch.manual_seed(0)
    out = model.generate(idx, max_new_tokens=60, temperature=T, top_k=50)
    text = tok.decode(out[0].tolist())
    print(f"\n[T={T}]\n{text}")

# 3. Top-k sweep
print()
print("=" * 60)
print("TOP-K SWEEP (T=0.8, 60 new tokens, seed=0)")
print("=" * 60)
for k in [1, 5, 40, 200]:
    torch.manual_seed(0)
    out = model.generate(idx, max_new_tokens=60, temperature=0.8, top_k=k)
    text = tok.decode(out[0].tolist())
    print(f"\n[top_k={k}]\n{text}")

# 4. Manual generation, step by step, so you see the loop
print()
print("=" * 60)
print("MANUAL GENERATION — 10 tokens, showing each step")
print("=" * 60)
torch.manual_seed(0)
idx2 = idx.clone()
for step in range(10):
    with torch.no_grad():
        logits, _ = model(idx2[:, -mcfg.block_size:], targets=None)
    logits = logits[:, -1, :] / 0.8
    topk_vals, _ = torch.topk(logits, min(40, logits.size(-1)))
    logits[logits < topk_vals[:, [-1]]] = -float("inf")
    probs = F.softmax(logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1)
    idx2 = torch.cat([idx2, next_id], dim=1)
    piece = tok.id_to_token(next_id.item())
    so_far = tok.decode(idx2[0].tolist())
    print(f"  step {step}: sampled id={next_id.item()} piece={piece!r}  →  {so_far!r}")

print()
print("Note: this model was trained for 20 epochs on 82K tokens.")
print("It's a TOY — not for production inference. Used here so you can SEE the mechanics.")
