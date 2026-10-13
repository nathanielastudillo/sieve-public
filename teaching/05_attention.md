# 05 — Attention and the block

## Bridge

Every transformer block does two things, in this order:

1. **Attention**: "for each position, look at a weighted sum of all earlier positions." This is how information moves *across* positions in the sequence.
2. **MLP**: "for each position, pass its vector through a small feedforward net." This is how information is *transformed* within a position.

Think of it as an alternation: attention mixes horizontally (across positions), MLP processes vertically (within a position). Stack N of these and you get a transformer.

The residual stream from Lesson 04 is the shared workspace. Attention reads from it, writes a small update back to it. MLP reads from it, writes another small update. Over N layers, information accumulates and composes.

## Attention, demystified

For each position `t` in the sequence, the model wants to ask: "which earlier positions should I listen to, and what should I take from them?"

It computes three things for every position:
- **Query** `q_t` — "what am I looking for?"
- **Key** `k_t` — "what do I advertise myself as?"
- **Value** `v_t` — "what information do I carry?"

These are just three linear projections of the position's hidden vector. All three are shape `n_embd` (well, split across heads — we'll get to that).

Then for each position `t`, the attention score against position `s` is `q_t · k_s` — a dot product. High score = "these positions are relevant to each other." Softmax over all positions turns scores into weights. The output at position `t` is the weighted sum of all `v_s` by those weights.

```
scores[t, s]  = q_t · k_s / sqrt(d)
weights[t, s] = softmax_s(scores[t, s])
output[t]     = Σ_s weights[t, s] * v_s
```

The `1/sqrt(d)` prevents scores from getting too peaky as `n_embd` grows.

**Causal mask**: before softmax, set `scores[t, s] = -inf` whenever `s > t`. After softmax those become zero. So position `t` can only attend to positions `0..t`. This is what `F.scaled_dot_product_attention(..., is_causal=True)` does.

## Multi-head

In practice we split the `n_embd` channels into `n_head` parallel groups (e.g. 6 heads × 64 dims each = 384). Each head runs its own attention independently, then we concatenate the outputs and project back to `n_embd`.

Why? Different heads can specialize. One head might track "subject-verb agreement," another "which noun is being referred to by a pronoun." Empirically helpful. Computationally it's just a reshape — no new ops.

## The SwiGLU MLP

After attention, each position's vector goes through an MLP:

```
gate = Linear(n_embd → hidden)
up   = Linear(n_embd → hidden)
down = Linear(hidden → n_embd)

output = down( SiLU(gate(x)) * up(x) )
```

`hidden ≈ 8/3 × n_embd` rounded to a multiple of 64 for GPU alignment. The `*` is elementwise — `SiLU(gate(x))` acts as a multiplicative gate on `up(x)`. This is SwiGLU: GLU with SiLU activation. Slowrun showed it beats `ReLU²` for data efficiency. That's the whole justification.

## The pre-norm + residual pattern

Each block is wrapped:

```python
x = x + attn(ln_1(x))    # attention branch
x = x + mlp(ln_2(x))      # MLP branch
```

Three things to notice:

1. **Pre-norm**: `LayerNorm` is applied *before* the sublayer, not after. This is the modern convention; makes deep nets trainable.
2. **Residual**: the block's output is *added* to the input, not replacing it. Gradients flow directly through the `x +` path, so deep models train stably.
3. **Residual scaling**: at initialization, the output projections (`c_proj` in attention, `down_proj` in MLP) are scaled by `1/sqrt(2·n_layer)`. This keeps the variance of the residual stream constant regardless of depth. Look at `_apply_residual_scaling` in `model/gpt.py`.

## ❓ Predict

Before running:

1. If n_embd=256 and n_head=4, what is the per-head dimension?
2. The attention layer has a `c_attn: Linear(n_embd → 3*n_embd)` projection. Why 3x?
3. For a causal model, when you look at the attention weight matrix for position `t`, how many entries are nonzero?

Write your answers.

## Run

```bash
python teaching/exercises/05_attention.py
```

## What it means

- You'll see the Q/K/V projection literally being three slices of one big linear layer.
- You'll see attention weights as a matrix where the upper-triangle is zero (causal mask).
- You'll see each block adding a small update to the residual stream.

The key insight: **attention is where information moves between positions; MLP is where it gets transformed within a position.** Both are residual additions to a shared stream. The stream is the data; blocks are read-modify-write stations.

## The pitfall to avoid

Don't try to "interpret" individual attention heads too literally early on. They're messy — a single head's pattern often makes sense on some inputs and looks like noise on others. Treat attention as "soft routing between positions" and move on. Interpretability of heads is its own research area.

## Check

1. Without the causal mask, what task would this architecture be solving instead of next-token prediction?
2. Why is the MLP's `down_proj` initialized with a smaller std than other Linears?
3. If you removed all the LayerNorms, what would break? (Think about variance through many layers.)

→ Next: [06_learning.md](06_learning.md) — how the loss turns into a gradient that updates every weight.
