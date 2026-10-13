# 04 — The lookup table

## Bridge

You're handing the model an integer like `4217`. It needs to compute on that integer. But neural networks are built from matrix multiplies and nonlinearities — they act on **vectors of floats**, not integers.

So the very first operation in the model is: **integer → vector of floats**. This is an "embedding lookup." It's a learned dictionary.

Think of it as a `dict[int, np.ndarray]` — except stored as one big 2-D matrix of shape `(vocab_size, n_embd)`. To look up token ID 4217, you grab row 4217. That's the whole operation.

In code:
```python
wte = nn.Embedding(vocab_size, n_embd)   # shape (8000, 256), say
vec = wte(torch.tensor([4217]))           # shape (1, 256)
```

The embedding matrix is **learned by backprop**. At initialization it's random noise. After training, tokens that appear in similar contexts have similar vectors. Your Lacanian intuition is exactly right: the meaning of a signifier is its *position in a system of differences* — in this case, literal positions in a 256-dim space where distances correspond to co-occurrence statistics.

## Positional embeddings: where am I in the sequence?

One more problem. Attention is permutation-invariant by default — it doesn't know that token at position 3 comes before token at position 7. If you only had token embeddings, the model would process `"dog bites man"` identically to `"man bites dog"`.

Solution: add a **position embedding**. There's a second table `wpe` of shape `(block_size, n_embd)`. At position `t` we add `wpe(t)` to the token embedding. Now the model knows both "which token is this" AND "where in the sequence."

```python
x = wte(idx) + wpe(arange(T))     # (B, T, n_embd)
```

These positional embeddings are *also* learned — just another (block_size, n_embd) matrix that backprop shapes over training.

## Weight tying: why the output uses the same matrix

At the end of the model, after all the transformer blocks, you have a hidden vector of dimension `n_embd` at each position. You need to turn that back into a distribution over the vocabulary — shape `n_embd → vocab_size`.

That's just a linear layer: `lm_head = nn.Linear(n_embd, vocab_size)`. Its weight matrix has shape `(vocab_size, n_embd)` — **the same shape as the input embedding!**

Observation: the input embedding maps "which vocab token" → "its vector." The output head does the inverse: "vector" → "which vocab token is this closest to." These are natural inverses. So we **tie the weights** — use one matrix for both:

```python
self.transformer.wte.weight = self.lm_head.weight
```

One fewer parameter matrix to learn. Cleaner. Standard in GPT-style models.

## The residual stream

Zoom out. Here's the abstract picture of a transformer forward pass:

```
x = embed(tokens) + embed(positions)    # (B, T, n_embd) — "the residual stream"
for block in blocks:
    x = x + attn(norm(x))               # block ADDS to x, doesn't replace
    x = x + mlp(norm(x))                # another ADD
x = norm(x)
logits = lm_head(x)
```

Every block's output is **added back** to `x` — hence "residual." The stream of hidden vectors flows through the network with each block writing small updates into it, not replacing it. Think of it as a conveyor belt: each station adds a small annotation without erasing what came before. This is what makes training stable at depth.

## ❓ Predict

Before running:

1. If `vocab_size = 8000` and `n_embd = 256`, how many parameters are in `wte`? In `wpe` (with `block_size = 512`)?
2. Because of weight tying, is `lm_head.weight` a copy of `wte.weight`, or literally the same tensor in memory?
3. What happens if I mutate `wte.weight[0, 0]` in-place? Does `lm_head.weight[0, 0]` change too?

Write down your guesses.

## Run

```bash
python teaching/exercises/04_lookup.py
```

## What it means

- You see the embedding matrix is just a `(vocab_size, n_embd)` tensor. Index by token ID = row.
- You see `lm_head.weight is model.transformer.wte.weight` — literally the same object. Mutations show up in both.
- You see the full flow: `tokens (B,T) → embed (B,T,D) → blocks (B,T,D) → lm_head (B,T,V)`. At every step the "T" dimension is preserved — every input position produces an output.

## The pitfall to avoid

People sometimes think embeddings are a fixed, pre-trained preprocessing step (like word2vec). They're not. `wte.weight` is just more parameters — initialized randomly, updated by gradient descent along with everything else. A token's embedding IS what the model has learned about that token from the training data. Nothing more.

## Check

1. Weight tying saves parameters and (empirically) helps training. Why would the same matrix work for both "map ID to vector" and "map vector to vocab score"? Think geometrically.
2. If I trained a new tokenizer and now `vocab_size = 16000` instead of `8000`, which tensors in the model change shape?
3. Position embeddings are learned up to `block_size`. What happens at inference if you feed a sequence longer than `block_size`?

→ Next: [05_attention.md](05_attention.md) — what a transformer block actually does.
