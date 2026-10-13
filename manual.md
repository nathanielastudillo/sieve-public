# SIEVE Inference Manual

Quick reference for doing inference + benchmarks on the trained ensembles.
This is the view **after** two completed evolutionary runs on corpus_v003 and
corpus_v005. Tasks #14-16 (tokenizer retrain → corpus_v006 → rerun) are
slated but not yet run.

## What we have on disk

40 transformer checkpoints across two runs, 7.4M params each (n_layer=4,
n_head=4, n_embd=256, block_size=256, vocab_size=16000, fp32).

```
runs/smoke_loop/           # v003 corpus (16 sources, 99.7K tokens)
  checkpoints/gen-{000..002}-{id}/*.pt         18 checkpoints
  results/best_ensemble.json                   val_loss 8.7315
  results/lineage.db

runs/smoke_loop_v005/      # v005 corpus (35 sources, 97.1K tokens)
  checkpoints/gen-{000..002}-{id}/*.pt         18 checkpoints
  results/best_ensemble.json                   val_loss 8.8247
  results/lineage.db
```

Baseline: uniform over vocab = `ln(16000) = 9.68` nats. Both ensembles beat
it by ~0.85-0.95 nats. Individual members overfit hard (train loss ~0.05);
ensemble gain is the SIEVE thesis in miniature.

## Key APIs

```python
# Load a single checkpoint → GPT model (config recovered from metadata)
from worker.checkpoint import load_model_from_checkpoint
model = load_model_from_checkpoint("runs/.../step_NNN_epoch_NN.pt", device="cuda")
model.eval()

# Tokenize with the custom 16K BPE
from tokenizers import Tokenizer
tok = Tokenizer.from_file("data/tokenizers/v002/tokenizer.json")
ids = tok.encode("The subject of the unconscious is structured like").ids
text = tok.decode(ids)

# Autoregressive generation (single model)
import torch
x = torch.tensor([ids], device="cuda")
y = model.generate(x, max_new_tokens=100, temperature=0.8, top_k=50)
print(tok.decode(y[0].tolist()))

# Ensemble generation: average logits across N models at each step
# (see scripts/ensemble_sample.py once you write it — not yet present)
```

## Ensemble specs

`best_ensemble.json` is the authoritative artifact per run:

```json
{
  "members": [{"checkpoint": "...", "config": {...}, "val_loss": ..., "id": "..."}],
  "ensemble_size": 3,
  "val_loss": 8.7315,              // or 8.8247 for v005
  "diversity_score": 0.42,
  "aggregation": "logit_average"
}
```

Load all 3 members, forward the same input through each, average the logits
(pre-softmax), then sample. Averaging probs is strictly worse — logit
averaging is what the SIEVE math says to do.

## Data access

```python
# Val/test bins are uint16 memmaps
import numpy as np
val = np.memmap("data/corpus_v003/val.bin", dtype=np.uint16, mode="r")
# v003: 84992 train / 9728 val / 4608 test (tokens, not bytes)
# v005: 82944 train / 9216 val /  4608 test

# Metadata has full provenance
import json
meta = json.load(open("data/corpus_v003/metadata.json"))
meta["num_sources"], meta["tokenizer_path"], meta["per_source"]
```

**test.bin is the meta-overfitting guardrail** — never used in ensemble
selection. Any real benchmark number should come from test.bin, not val.bin.

## Things worth benchmarking

1. **Single-model vs ensemble loss on test.bin** for each run.
   Hypothesis: ensemble gain on test mirrors the gain on val.
   If it doesn't, we've meta-overfit val via selection.

2. **Logit-average vs prob-average** on the same ensemble.
   Measures whether the theoretical argument for logit-averaging holds
   empirically at 7.4M params / 100K tokens.

3. **Diversity metric**: pairwise KL divergence across the 3 members'
   next-token distributions on test.bin. SIEVE claims selection pressure
   favours diversity; this is the direct measurement.

4. **Ensemble size ablation**: grow from 1 → 6 members (we have 6 gen-2
   members per run), measure loss at each size. Expect diminishing returns.
   Look for the elbow.

5. **Cross-run comparison on a shared test set**. v003.test.bin has
   different tokens than v005.test.bin (different subsample), so
   numbers aren't directly comparable. For that, tokenize a fresh
   holdout (e.g. a chapter from a source present in both) with both
   tokenizers and measure bits/char — bits/char is tokenizer-agnostic
   and the honest cross-run metric.

6. **Qualitative generation samples** at different temperatures
   (0.3, 0.8, 1.2) from both single-member and full-ensemble to
   eyeball diversity-quality tradeoffs. Small models on tiny corpora
   tend to near-memorize; inspect for verbatim recall.

## Known gotchas

- **Tokenizer is v002** (trained on the narrower 21-source pool). Both
  v003 and v005 corpora use it. Task #14 retrains to v003 on the
  37-source pool; cross-version comparisons should account for this.
- **block_size=256** everywhere. Don't feed longer contexts; `generate()`
  truncates but you'll silently lose prefix.
- Individual members are near-memorized on train (train loss < 0.1).
  Test-set numbers are the honest signal.
- `diversity_score` in `best_ensemble.json` is the selection-time KL
  metric, not a test-set measurement. Recompute on test.bin for a
  clean number.

## Suggested starting scripts

Reasonable files to add under `scripts/` when you want them:

- `scripts/ensemble_sample.py` — load best_ensemble.json, average logits
  across members at each generation step, sample from the average
- `scripts/eval_on_test.py` — forward test.bin through single + ensemble,
  dump per-sequence loss CSV
- `scripts/diversity_on_test.py` — compute pairwise KL between members'
  next-token dists on test.bin
- `scripts/compare_runs.py` — unified table across v003/v005 (and later
  v006) with test-set bits/char, loss, diversity
