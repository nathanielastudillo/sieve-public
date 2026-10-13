# 09 — The ensemble

## Bridge

You already understand what a single model does: takes a prompt, produces a probability distribution over the next token. An **ensemble** is just N models, each producing their own distribution, combined into one.

The "combining" step is where design choices matter. Three common options:

1. **Voting**: each model commits to its top pick, majority wins. Crude, loses probability information.
2. **Probability averaging**: average the softmax outputs, element-wise. Reasonable but loses calibration — averaging peaky distributions flattens them in ways that don't match a coherent predictive model.
3. **Logit averaging** (what SIEVE uses): average the pre-softmax logits, then softmax once at the end. Equivalent to a geometric mean of probabilities (up to renormalization) — well-calibrated, theoretically motivated.

The formula is dead simple:

```python
all_logits = stack([m(idx)[0] for m in models])       # (K, B, T, V)
avg_logits = all_logits.mean(dim=0)                    # (B, T, V)
probs      = softmax(avg_logits / T, dim=-1)
next_id    = multinomial(probs, num_samples=1)
```

One extra dimension (the ensemble dim K), averaged away before softmax. That's it.

## The Solomonoff connection

Solomonoff induction: the provably optimal predictor. Take every program (hypothesis) consistent with observed data; weight each by `2^{-length}`; predict by weighted average over programs.

Neural networks implicitly approximate this via simplicity bias — SGD finds simple (wide-basin) solutions first. A single network commits to *one* program consistent with the data. It approximates Solomonoff but with the mixture collapsed to a point.

An ensemble maintains K programs and averages them. Structurally closer to Solomonoff. Especially when the models are **diverse** — they represent different consistent programs, not copies of the same one. A diverse ensemble IS a mixture over programs; each new member that covers different territory improves the approximation.

This is why SIEVE's fitness function isn't "how good is this model?" but "how much does this model improve the best ensemble it's part of?" The selection pressure is on diversity-adjusted quality. A mediocre model that's highly complementary gets selected; a great model that's a near-duplicate of another great model gets dropped.

## Diversity, operationalized

How do you measure "how different are two models?" The evaluator uses pairwise **symmetric KL divergence** between the models' output distributions on the val set. For each val batch, for each pair `(i, j)`:

```
KL(p_i || p_j) = sum(p_i * (log p_i - log p_j))       # "how different is j from i, from i's POV"
sym_KL(i, j)   = (KL(p_i || p_j) + KL(p_j || p_i)) / 2
```

Average over all pairs and all batches. Higher = more diverse ensemble. If all K members collapse to the same distribution, symKL = 0 and the ensemble is effectively K=1 (no benefit).

The evolutionary loop rewards ensembles with *both* low val loss AND high diversity — because low-loss-high-diversity is what matters. Low-loss-low-diversity is just "K copies of the same model" and gives no ensemble benefit over a single member.

## Inference cost

K models = K forward passes per step = K× slower generation. This is the price. You can parallelize by running each model on a separate GPU (embarrassingly parallel — no communication until the logit average at the end). That's why SIEVE's compute topology is GB10 workers — each holds one or two members, forward passes happen concurrently, logits get gathered for averaging.

**Model soups** (Wortsman 2022) is the alternative: average the *weights* across fine-tunes, yielding one model of the same size. No inference cost penalty. But it only works when the members came from the same initialization and live in a single low-error basin. SIEVE's members come from diverse configs/seeds, so they're generally in different basins — logit averaging is safer.

## ❓ Predict

Before running:

1. If all K members produce the same distribution on every input, what is the ensemble's val loss relative to a single member's?
2. If the K members have diverse but equally good distributions, will the ensemble val loss be lower, equal, or higher than any single member?
3. What's the val loss of an ensemble of size K=1?

## Run

```bash
python teaching/exercises/09_ensemble.py
```

You'll see:
- Individual val loss for each of three checkpoints (epochs 5, 15, 20 — different training stages of the same smoke run)
- The ensemble's val loss (should be roughly at or below the best individual)
- The pairwise symmetric KL between members

## What it means

Checkpoints from the same training run at different epochs are *somewhat* diverse (early epoch = more underfit / different solution than late epoch) but not maximally so — they share initialization and path. The ensemble usually does ≤ the best single member but the margin is small.

In a real SIEVE run, you'd be ensembling across runs with different seeds, architectures, optimizers — those members lie in genuinely different basins, and the ensemble effect is much bigger (Slowrun showed ~0.2-0.3 nats of val loss improvement from 8-way ensembling).

## The pitfall to avoid

Don't ensemble models that used different tokenizers. The logits are indexed by vocab ID, and if vocab IDs mean different things in different models, averaging is nonsense. Tokenizer must be shared across the ensemble.

Also: **average logits, not softmax probabilities.** Probability averaging is tempting and seems equivalent but isn't. Averaging probabilities tends to flatten distributions in a way that hurts calibration; averaging logits preserves it. `lab/server.py:_generate_ensemble` averages in fp32 before softmax for the same reason — numerical precision matters when K is large.

## Check

1. With K=2 members that are identical (same weights), what does `stack.mean(dim=0)` produce? What's the ensemble's cross-entropy?
2. Why might ensembling two DIFFERENT-ARCHITECTURE models still work, even though their hidden states have nothing to do with each other?
3. In `evaluate_ensemble`, why is `member_logits.append(logits.float())` converting to fp32 before averaging, even though the forward pass was in bf16?

---

## Where this takes you

You now hold the full pipeline in your head:

```
PDF bytes → cleaned text
          → tokenizer.encode → int array (vocab IDs)
          → concat into train.bin (uint16 memmap)
          → sliding windows (x, y = x shifted by 1)
          → forward pass through GPT (embed → blocks → lm_head)
          → cross-entropy loss on all T positions
          → backprop → optimizer step
          → checkpoint .pt every N epochs
          → load_state_dict from .pt to reconstruct
          → GPT.generate: loop (forward → last-pos logits → temp/topk → multinomial → append)
          → ensemble: K forward passes → mean logits → softmax → sample
```

Re-read any lesson where a link feels fuzzy. Then try to write a minimal `train.py` and `generate.py` from scratch without looking at the SIEVE source. When you can do that — when you can say "given a directory of PDFs, I can take them all the way to text coming out of a model" — you've built the mental model at its joints.

Come back to me with any specific point that's still murky. The fastest way to solidify is to be made to produce the explanation yourself and then have it corrected.
