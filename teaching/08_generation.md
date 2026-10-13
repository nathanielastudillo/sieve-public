# 08 — Generation

## Bridge

Generation is a tiny loop wrapped around the exact forward pass you've been looking at. No new concepts — just "do the forward pass, sample one token, append it, do the forward pass again."

```python
for _ in range(max_new_tokens):
    logits = model(idx_so_far)[:, -1, :]      # only the LAST position matters
    probs  = softmax(logits / temperature)
    if top_k: mask all but the top k to zero, renormalize
    next_token = multinomial(probs, num_samples=1)
    idx_so_far = cat([idx_so_far, next_token], dim=1)
```

Five lines. Everything interesting in text generation — style, creativity, repetition, hallucination — lives in two knobs: **temperature** and **top_k**.

## Temperature

The raw logits are the model's unnormalized preferences. Dividing them by `T > 0` before softmax makes the distribution flatter or peakier:

- **T = 1.0**: use the model's raw distribution.
- **T < 1.0** (e.g. 0.3): distribution gets peakier — the model's favorite token becomes even more favored. Generation becomes more deterministic, conservative. Approaches "always pick the argmax" as T → 0.
- **T > 1.0** (e.g. 1.5): distribution flattens — low-probability tokens become more likely. Generation becomes more diverse, surprising, and often incoherent.

Mathematically: `softmax(logits / T)`. At T=1 you get the model's calibrated distribution. At T=0.5 you're squaring the probabilities and renormalizing.

## Top-k

"Only sample from the k most likely tokens; zero out everything else." This cuts off the long tail of low-probability garbage. Common values: top_k = 40 or 50.

Why does this matter? Even a well-trained model puts a small nonzero probability on every vocab token at every step. Multiply 0.0001 × lots of chances and you'll occasionally sample truly weird tokens. Top-k prevents this by construction.

There are fancier variants: **top-p (nucleus)** sampling keeps the smallest set of tokens whose total probability exceeds p (say 0.9). SIEVE currently uses top-k because it's simple and composes cleanly.

## The math

```python
logits = logits / T                            # temperature
if top_k:
    kth_value = topk(logits, k).values[-1]
    logits[logits < kth_value] = -inf          # mask non-top-k
probs = softmax(logits)                         # normalize what's left
next_id = multinomial(probs, num_samples=1)     # weighted random sample
```

`multinomial` is just "roll a weighted die."

## Autoregressive cost

Every new token requires a full forward pass over the **entire** context-so-far. Generating 200 tokens = 200 forward passes, with contexts of length T, T+1, T+2, ..., T+200. This is why long-context generation gets slow — the attention cost is quadratic in context length.

Production systems cache the K/V tensors from prior positions (KV-cache) so each new step only does O(T) work. SIEVE doesn't implement this — the models are tiny and 200-token generation completes in milliseconds regardless.

## Why only the LAST position's logits?

During training, every position's logits are used (T parallel next-token predictions). During generation, we've already appended position `t`'s best guess to the sequence; we just need position `t`'s prediction, which is the logits at position `t` (i.e. the last one in the current sequence). Any earlier positions' logits are redundant — they'd predict tokens we've already committed to.

`GPT.forward` has a little optimization for this: if `targets is None`, it only applies `lm_head` to the last position, saving one matmul:

```python
logits = self.lm_head(x[:, -1:, :])   # (B, 1, vocab_size)
```

## ❓ Predict

Before running:

1. At T = 0.001 (essentially zero), what should generation look like?
2. At T = 5.0, what happens?
3. With top_k = 1, what should generation look like, regardless of temperature?

## Run

```bash
python teaching/exercises/08_generate.py
```

You'll see three things:
1. Raw probability distribution over the top-10 tokens after a prompt
2. Generations at different temperatures
3. Generations at different top_k values

## What it means

The model is genuinely probabilistic. Every generation is a sample from its output distribution. **Same model + same prompt + same seed = same output.** Different temperature or top_k is literally reshaping the sampling distribution.

Notice: our SIEVE smoke-run model is undertrained (5 epochs on 82K tokens). Its output is not going to be Shakespeare. It'll repeat, wander, produce grammatical-ish fragments. That's normal and expected. What you're measuring is "does the machinery work end to end?" — not literary quality.

## The pitfall to avoid

Don't mistake a bad generation for a bug in the generation code. If the output is incoherent, 99% of the time it's the model's training (not enough data, not enough epochs, wrong hyperparameters). The generation code is ~10 lines; hard to get wrong.

When a trained model generates badly: check (a) temperature isn't at 0, (b) you loaded the right tokenizer (mismatched tokenizer → output is correct token IDs but decoded back to gibberish), (c) the prompt is in-distribution for what the model trained on.

## Check

1. If I generate at T=0.0001 ten different times with different seeds, how similar will the outputs be?
2. Given a decoded output like `"Lacan's object of"` what are the next ~5 candidate tokens you'd expect to have high probability? (This is a calibration exercise.)
3. Why does `model.eval()` matter before generating, and what would go wrong if you forgot it?

→ Next: [09_ensemble.md](09_ensemble.md) — the Solomonoff move: why averaging logits across models approximates the optimal predictor.
