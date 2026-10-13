"""Lesson 09 — ensemble three checkpoints.

Run: python teaching/exercises/09_ensemble.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from model.config import ModelConfig
from model.gpt import GPT

CORPUS = REPO / "data/corpus_v008"
meta = json.loads((CORPUS / "metadata.json").read_text())
val = np.memmap(CORPUS / "val.bin", dtype="uint16", mode="r")

CKPTS = [
    REPO / "runs/smoke_v003/checkpoints/step_013240_epoch_0005.pt",
    REPO / "runs/smoke_v003/checkpoints/step_039720_epoch_0015.pt",
    REPO / "runs/smoke_v003/checkpoints/step_052960_epoch_0020.pt",
]

device = "cuda" if torch.cuda.is_available() else "cpu"

models = []
for p in CKPTS:
    ckpt = torch.load(str(p), map_location=device, weights_only=False)
    mcfg = ModelConfig.from_dict(ckpt["config"])
    m = GPT(mcfg)
    m.load_state_dict(ckpt["model_state_dict"])
    m.to(device).eval()
    models.append((p.name, m, mcfg))
    print(f"loaded {p.name}: {mcfg.n_params:,} params  val_loss@save={ckpt['metrics'].get('train_loss','?'):.4f}")
print()

# Build a batch of val windows
BLOCK = models[0][2].block_size
BATCH = 32
rng = np.random.default_rng(0)
# max start so x+BLOCK+1 fits
max_start = len(val) - BLOCK - 1
starts = rng.integers(0, max_start, size=BATCH)
xs = np.stack([val[s : s + BLOCK].astype(np.int64) for s in starts])
ys = np.stack([val[s + 1 : s + BLOCK + 1].astype(np.int64) for s in starts])
x = torch.from_numpy(xs).to(device)
y = torch.from_numpy(ys).to(device)

# IMPORTANT: to get ensemble logits properly we need full logits for the whole window,
# not just the last position. So we'll call forward with targets=y (that branch returns
# full (B, T, V) logits). We ignore the loss that comes back with each one.
all_logits = []
individual_losses = []

for name, m, _ in models:
    with torch.no_grad():
        logits, loss = m(x, targets=y)
    all_logits.append(logits.float())
    individual_losses.append(loss.item())
    print(f"{name}  individual val loss: {loss.item():.4f}  (ppl={math.exp(loss.item()):.2f})")

print()

# Ensemble: average logits across members, then compute cross-entropy ONCE
stacked = torch.stack(all_logits, dim=0)   # (K, B, T, V)
avg_logits = stacked.mean(dim=0)            # (B, T, V)
V = avg_logits.size(-1)
ens_loss = F.cross_entropy(avg_logits.reshape(-1, V), y.reshape(-1))
print(f"ENSEMBLE (K={len(models)}, logit avg)  val loss: {ens_loss.item():.4f}  (ppl={math.exp(ens_loss.item()):.2f})")

best_indiv = min(individual_losses)
print()
print(f"Best individual loss: {best_indiv:.4f}")
print(f"Ensemble vs best:     {ens_loss.item() - best_indiv:+.4f}  ({'better' if ens_loss.item()<best_indiv else 'worse'})")
print()

# Pairwise symmetric KL across members (diversity score)
print("Pairwise symmetric KL (a.k.a. 'diversity') between members:")
log_probs = F.log_softmax(stacked, dim=-1)
probs = F.softmax(stacked, dim=-1)
K = len(models)
for i in range(K):
    for j in range(i + 1, K):
        kl_ij = F.kl_div(log_probs[j].reshape(-1, V), probs[i].reshape(-1, V),
                         reduction="batchmean", log_target=False)
        kl_ji = F.kl_div(log_probs[i].reshape(-1, V), probs[j].reshape(-1, V),
                         reduction="batchmean", log_target=False)
        sym = (kl_ij.item() + kl_ji.item()) / 2
        print(f"  {models[i][0]} <-> {models[j][0]}: symKL = {sym:.4f}")
print()

print("Interpretation:")
print("  - These three checkpoints are DIFFERENT EPOCHS of the SAME training run.")
print("  - So they share initialization & data path — low diversity expected.")
print("  - The ensemble helps a little, because epoch-5 and epoch-20 make different mistakes.")
print("  - In a real SIEVE run, you'd ensemble checkpoints from DIFFERENT seeds / configs,")
print("    yielding higher symKL and bigger ensemble wins.")

# Finally: compare to probability averaging (common mistake) to see it underperforms
prob_avg = F.softmax(stacked, dim=-1).mean(dim=0)      # (B, T, V)
# manual cross-entropy from probs (need log of averaged probs)
log_prob_avg = prob_avg.clamp_min(1e-12).log()
ens_loss_probavg = F.nll_loss(log_prob_avg.reshape(-1, V), y.reshape(-1))
print()
print("For comparison — averaging SOFTMAX PROBS (the tempting wrong choice):")
print(f"  prob-avg ensemble loss:    {ens_loss_probavg.item():.4f}")
print(f"  logit-avg ensemble loss:   {ens_loss.item():.4f}")
print(f"  Logit averaging usually wins. Always use logit averaging.")
