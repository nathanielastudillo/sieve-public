# 03 — The sliding window

## Bridge

The entire objective of a language model, in one sentence: **given a prefix, predict the next token.** Nothing else. Every capability — completing text, answering questions, generating code — is emergent from this one task performed well over diverse data.

Think of it as a cloze test administered continuously. The model reads tokens `0..T-1` and must produce a probability distribution over "what comes next" at every position. At position 0 it predicts position 1. At position 1 (having seen tokens 0..1) it predicts position 2. And so on.

In one forward pass over a sequence of length `T`, the model is being trained on **`T` independent next-token predictions, simultaneously**. That parallelism is why transformers are efficient — unlike RNNs, the T predictions don't have to be sequential.

## The concrete structure: `x` and `y`

Pick any index `i` in the token stream. The training window is:

```
chunk = stream[i : i + block_size + 1]     # block_size + 1 tokens
x     = chunk[:-1]                          # (block_size,) — the inputs
y     = chunk[1:]                           # (block_size,) — the targets
```

Notice: `y` is just `x` shifted right by one position. They share `block_size - 1` of the same tokens — only the first of `x` and the last of `y` differ.

This is the entire objective:

> For each position `t` in the input, predict `x[t+1]` given `x[0..t]`.

Written differently: `y[t] = x[t+1]`.

## Why the causal mask matters

The transformer processes all `T` positions in parallel — a single matrix multiplication computes hidden states for every position at once. But there's a cheating problem: when predicting position `t+1`, the model must not see positions `t+1, t+2, ...` (the answers).

The **causal attention mask** enforces this: at position `t`, the attention mechanism can only look at positions `0..t`. Never the future. This is what makes it a *causal* (or "autoregressive") language model — it models `P(x_t | x_0, ..., x_{t-1})`.

Under the hood, `F.scaled_dot_product_attention(q, k, v, is_causal=True)` does this for you: it sets the attention weights for future positions to `-inf` before softmax, so they become zero after.

## ❓ Predict

Before running:

1. If `train.bin` has 81,920 tokens and `block_size = 512`, how many distinct sliding windows `(x, y)` can you extract? (Hint: they're allowed to overlap.)
2. Given `x = [5, 17, 42, 103, 8]`, what is `y`? (You have to look at the stream, not just `x`. Pretend the stream continues `... 8, 91, ...`.)
3. If the model's prediction at position `t` in a sequence is a probability distribution over the vocabulary, what shape does the model's *output* have for an input of shape `(batch, T)`?

Write your answers.

## Run

```bash
python teaching/exercises/03_window.py
```

## What it means

Running the exercise, you'll see:

- A concrete `(x, y)` pair drawn from `train.bin`. The integers scroll past; you can see that `y[t] == x[t+1]` for every `t`.
- Decoded back to text, `x` and `y` are almost the same passage — `y` is "the same text starting one token later."
- The logits output from the model is shape `(batch, T, vocab)`. That's **T next-token distributions**, one per input position.

**The click moment**: the model isn't trained on a special "completion" task. At every single position inside every single training window, it's making a next-token guess and being graded. A 512-token window with batch 64 = **32,768 graded predictions per forward pass**. That's the training signal density.

## The pitfall to avoid

Beginners often think: "the model takes a prompt and outputs the next word." That framing is right for *inference* but wrong for *training*. During training, the model processes the whole window in parallel and produces `T` next-token predictions in one forward pass — all graded against `y`.

Inference is actually a special case of training: you just take the last position's logits, sample, append, and repeat. The two settings run the same forward code (`GPT.forward`) with different post-processing.

## Check

1. Why is `block_size + 1` tokens needed to form one `(x, y)` pair of length `block_size`?
2. If I remove the causal mask during training, what goes wrong? (Be specific.)
3. At position `t=0` in a sequence, what context does the model have to predict `y[0]`? What does that mean for the first token's loss?

→ Next: [04_the_lookup.md](04_the_lookup.md) — how the integer `42` becomes a vector the model can compute on.
