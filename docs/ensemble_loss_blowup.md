# Ensemble loss blow-up — diagnosis + remediation

**Date:** 2026-04-18
**Run:** `exp006_parallel_v2_20260418_022711` (6 workers, smoke_v8000, 3 generations)

> **Update 2026-04-18 (post-lit-review):** See `docs/ensemble_lit_review.md` for the grounded version. Two things in this doc turned out wrong:
> 1. **Fix D ("switch to probability averaging")** is misframed — current logit-mean + softmax already *is* geometric-mean-of-probabilities = Product of Experts (Wood 2023). Arithmetic-mean of probabilities is a different aggregator (BMA-style), not the fix.
> 2. The *actual* first-line fixes are **ε-smoothing each member's softmax** (α≈0.05 before aggregation) to floor probabilities and kill geometric-mean blowup, and — decisively — **growing tokens/param past 100** (corpus growth or model shrinkage). No aggregator rescues 15 tokens/param.
>
> Keep reading for the original diagnosis, which is correct; treat Fix D as superseded.

## Symptom

| gen | best individual "val_loss" | best ensemble val_loss | diversity |
|-----|-----------------------------|------------------------|-----------|
| 0   | 0.0332                      | 9.8624                 | 0.29      |
| 1   | 0.0648                      | 9.9283                 | 0.23      |
| 2   | 0.0708                      | 9.9760                 | 0.24      |

Individual members are reported at near-zero loss. Logit-averaged ensembles score ~10 — which is slightly *worse* than uniform (log(8000) = 8.987). Individual scores drift up across generations; ensemble scores drift up too.

## Root causes (ordered by leverage)

### 1. The trainer's "val_loss" is not a val loss

`worker/trainer.py:304-391` never touches `val.bin`. It reports `best_loss = min(epoch_avg_training_loss)`. The daemon (`worker/daemon.py:83`) then hands that number up as `val_loss` in the `TrainResultMsg`. So what the orchestrator sees as "individual fitness" is actually the best epoch-average **training** loss.

The ensemble evaluator (`controller/ensemble_eval.py:46-85`) does load `val.bin` and computes real CE on averaged logits. That's why only the ensemble column is honest. The mismatch is entirely accounting, not algorithmic — until you realize it reveals point (2).

### 2. The individuals are memorizing

corpus_v008 has 81,920 train tokens. With block_size=256, batch_size=32, that's ~320 non-overlapping chunks. The DataLoader uses a shifted-window sampler that produces ~2,552 overlapping windows per epoch. 12 epochs × 2,552 steps = ~30K gradient updates on 5.3M-parameter models hitting the same 80K tokens repeatedly. Memorization is the only equilibrium the optimizer can find. Training loss of 0.03 on a language model is "memorized the sequence", not "learned the distribution".

### 3. Logit averaging of memorizers diverges

Logit-averaging assumes each member puts calibrated probability mass near the true next-token distribution, with complementary errors. Memorizers put all mass on the *training* next token. On a held-out val chunk:

- Member A has memorized a *different* overlapping window than member B.
- A's logits are a sharp spike at token *x*; B's at token *y*; neither is the true *z*.
- The average is a bimodal spike at {x, y}, still nowhere near *z*.
- CE on this average can exceed `log(V)` because the softmax mass is concentrated far from the true token.

Classical fix: average *probabilities* (after softmax), or use *temperature-scaled* logits. Averaging probabilities is bounded by `log(V)`. But the underlying problem — no-signal members — persists either way.

### 4. Evolutionary selection drifts up, not down

The fitness function = "best ensemble loss". Once the ensemble floor is trash (point 3), the evolutionary step optimizes against noise. Gen-over-gen drift confirms this is a dead selection signal. You need a fitness function that distinguishes *generalization*, not *memorization breadth*.

## Remediation

Ordered by cost / leverage:

### Fix A (low cost, high leverage): Trainer reports real val loss

Modify `worker/trainer.py`:
1. Load `val.bin` at training start (using `data_dir`).
2. After each epoch (or at the end), run `model.eval()` + compute CE on the full val split.
3. Keep the best-val checkpoint alongside the final checkpoint.
4. Report `val_loss` as the best val loss, not the best train loss.

This alone converts the fitness signal from memorization-score to generalization-score. Individual member "val_loss" numbers will immediately look sane (single-digit, near log(V) at start, below log(V) only if the model is actually learning).

### Fix B (cheap): Drop or cap epochs

smoke_v8000's search space allows `epochs: [6, 7, 8, 10, 12]`. On 80K tokens this is all "way too many". Cap at 3-4 epochs and watch val loss. If val loss still rises after epoch 1, the model is already memorizing.

### Fix C (medium): Regularization sweep

Slowrun found weight decay up to 16× standard helps in this regime. Current `weight_decay: [0.1, 1.5]`. Try `[0.5, 8.0]` and `dropout: [0.1, 0.4]`.

### Fix D (medium): Switch to probability averaging

In `ensemble_eval.py:_compute_ensemble_loss`:

```python
# Current — unbounded, pathological on miscalibrated members:
avg_logits = stacked.mean(dim=0)
loss = F.cross_entropy(avg_logits, targets)

# Better — bounded at log(V), robust to miscalibration:
avg_probs = F.softmax(stacked, dim=-1).mean(dim=0)
loss = F.nll_loss(avg_probs.clamp_min(1e-12).log(), targets)
```

Probability averaging is what Solomonoff actually prescribes (mixture over hypotheses), not logit averaging. Logit averaging is a heuristic that happens to work when members are calibrated (Njieutcheu Tassi et al., 2022). SIEVE's CLAUDE.md already flags this as an open question.

### Fix E (higher cost): Per-member temperature calibration

Before logit averaging, scale each member's logits by a learned `T` that minimizes val CE on a calibration split. Takes an extra forward pass and a tiny 1-parameter optimization per member.

### Fix F (structural): More data or smaller models

5.3M params on 80K tokens is ~70 tokens/param. TinyStories runs at ~10× that (10M-param models on 100M tokens). Slowrun runs at 100M tokens. The Solomonoff framing *requires* that individual members approximate the true distribution; at 70 tokens/param they can't. Options:
- Use a bigger corpus (v008 → expand beyond 35 sources).
- Shrink models aggressively (n_embd=128, n_layer=2 → ~1.3M params).
- Both.

## Prescribed next actions

1. Implement Fix A. Re-run 1-generation smoke. Confirm individual val_loss lands in `[log(V)·0.6, log(V)·1.0]` range (roughly 5-9 for V=8K). This is the "am I actually training" sanity check.
2. Apply Fix B (cap epochs at 4) concurrently.
3. If individual val losses are still >log(V) after Fix A+B, the corpus/model ratio is wrong — apply Fix F before touching ensemble logic.
4. Only after (3) is healthy does Fix D / E matter. You cannot ensemble your way out of no-signal members.

## 2026-04-18 post-exp007 addendum — second, unrelated cause of `inf` ensembles

The exp007_v009 full-cluster run surfaced a completely different class
of `inf` ensemble, unrelated to the aggregator math in this doc. It
looks identical from the orchestrator log:

```
[Generation N] Best ensemble loss: inf | Best individual: inf | Diversity: ...
```

but the mechanism is **orchestrator-level job dispatch**, not aggregation:

- `job_timeout_s=7200.0` (hardcoded, `orchestrator.py:731`) was tuned for
  smoke runs (~14 min/member). On corpus_v009 (4.3M tokens) members
  train 5 – 11 hours.
- `dispatcher.wait_for_generation` hits its deadline, fills every
  unreturned member with `status="error", error="timeout"`
  (`dispatch.py:178-185`).
- Orchestrator marks all members `val_loss=inf, checkpoint_path=None`,
  advances to the next generation. Workers keep training and publish
  late results to Redis; the orchestrator never re-reads those keys.
- Per-gen `best_individual` is therefore `inf`, `trainable` is empty,
  ensemble eval sees no members.

**Distinguishing the two failure modes:**

| Evidence | Aggregator blow-up (this doc's subject) | Dispatch timeout (exp007) |
|---|---|---|
| Per-member val_loss in orchestrator log | Near-zero (training loss leaked) | `inf` — member marked error |
| Redis `sieve:results:...` values | match orchestrator | real val_losses (3.8-4.3), orchestrator log says inf |
| `Generation complete: N/M ok` | M/M | 0/M |
| `Member .* failed: timeout` count | 0 | one per member per gen |
| Per-gen wallclock | variable | pinned at job_timeout_s |

If the orchestrator log shows "Generation complete: 0/8 ok" and
`timeout` errors, stop looking at the aggregator. The failure is
before ensemble eval.

Fix is lab-notes/experiments/004-exp007-v009.md: raise `job_timeout_s`
to match corpus size, make it a CLI flag, and admit late results.

## Prior art

- Njieutcheu Tassi et al. 2022 — logit vs probability averaging for calibration.
- Hinton distillation papers — temperature-scaled soft targets.
- Wortsman et al. 2022 (Model Soups) — weight-space averaging requires shared initialization; SIEVE lineage graph already tracks this.
- Slowrun (Q Labs 2026) — aggressive WD + dropout + epoch shuffling in data-constrained regime.
- TinyStories (Eldan & Li 2023) — small models *can* generalize on tiny corpora, but with ≥10× more tokens per param than SIEVE currently uses.
