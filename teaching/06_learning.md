# 06 — Loss and learning

## Bridge

You have logits — shape `(B, T, V)`. You have targets `y` — shape `(B, T)`. You need a single scalar number that measures "how wrong is the model." Then you compute the gradient of that scalar with respect to every parameter, and nudge each parameter a small step in the downhill direction.

That's the entire training loop. Everything else is plumbing.

## Cross-entropy, demystified

At each position `t` in each sequence of the batch, the model has output a vector of `V` numbers — the **logits**, which are unnormalized log-probabilities. The target is a single integer `y[t]`, the correct next token.

We want to measure: "how much probability did the model put on the correct answer?"

Step 1: Turn logits into probabilities with softmax.
```
probs[v] = exp(logits[v]) / Σ_v' exp(logits[v'])
```

Step 2: Look up the probability the model assigned to the correct token, `probs[y[t]]`.

Step 3: Take the negative log. The loss at position `t` is `-log probs[y[t]]`.

Why `-log`? Because:
- If `probs[y[t]] = 1` (perfect), loss = 0.
- If `probs[y[t]] = 0.5`, loss = 0.69 (≈ ln 2).
- If `probs[y[t]] → 0` (model is confident-wrong), loss → ∞.

It's a calibrated "surprise" metric. An untrained model outputs ~uniform probability over V tokens, so its loss is `-log(1/V) = log(V)` ≈ 8.99 for V=8000. A well-trained model can get this down into the low single digits.

Average over all positions in all sequences in the batch → scalar loss.

`F.cross_entropy(logits.view(-1, V), y.view(-1))` does all of this in one call, using the log-sum-exp trick for numerical stability.

## Gradients and backprop

`loss.backward()` triggers PyTorch's autograd: every operation you did during the forward pass was recorded as a node in a computation graph, and `.backward()` walks that graph in reverse, applying the chain rule to compute `∂loss/∂θ` for every parameter `θ`.

You never write this code. You just use PyTorch ops, and gradients happen automatically. The only thing you need to know: after `loss.backward()`, every parameter `p` with `requires_grad=True` has its gradient stored in `p.grad`.

## Optimizers, minimum-viable explanation

An optimizer reads `p.grad` and updates `p`. The simplest form (SGD):

```python
p ← p - learning_rate * p.grad
```

That is literally: "move every weight a small step opposite to its gradient." Do this thousands of times → loss goes down.

**AdamW** (what you've heard of): maintains a running mean and running squared-mean of the gradient per parameter. Uses these to adapt the effective step size per parameter, and adds weight decay. 95% of ML papers use it. Safe default.

**Muon** (what SIEVE uses for 2D weight matrices): a newer optimizer designed for matrix-shaped parameters. It takes the gradient matrix and runs a Newton-Schulz iteration to produce a "spectrally normalized" version (keeps the singular values roughly uniform), then applies that. Net effect: better conditioning of the update direction, often better loss in fewer steps. But it's only defined for 2D params, so Muon runs only on attention/MLP weights; embeddings, biases, layernorms all get an AdamW sidecar. That's the split you saw in `trainer.py`.

## The mixed-precision wrinkle

On GPU, matmuls are much faster in `bfloat16` than `float32`. But gradients in bf16 can underflow to zero. So:

- Forward pass: run in bf16 (via `torch.amp.autocast(..., bf16)`). Logits are in bf16.
- Loss: computed in bf16, but we promote to fp32 for accumulation.
- Backward: grads are in bf16.
- Optimizer: on bf16-safe hardware (Ampere, Hopper, Blackwell), we skip the gradient scaler. On older hardware, `GradScaler` multiplies the loss by a big constant before backward, then unscales the grads before clipping and stepping.
- Parameters: kept in fp32 ("master weights").

The `scaler.scale(loss).backward()` / `scaler.step(optimizer)` dance in `trainer.py` handles this.

## The full loop, pseudocode

```python
for epoch in range(epochs):
    sampler.set_epoch(epoch)  # reshuffle with new seed
    for x, y in dataloader:
        x, y = x.to(gpu), y.to(gpu)
        with autocast(bf16):
            logits, loss = model(x, targets=y)       # forward
        scaler.scale(loss).backward()                  # backward
        scaler.unscale_(optimizer)
        clip_grad_norm_(params, 1.0)                   # stability
        scaler.step(optimizer)                         # update weights
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()                               # adjust LR
```

That's all of training. Everything you'd see in `worker/trainer.py` is this, plus bookkeeping (checkpointing, metrics, configuration plumbing).

## ❓ Predict

Before running:

1. An untrained model on vocab_size=8000 — what's its expected cross-entropy loss?
2. If a model perfectly memorizes the training set (putting probability 1 on every correct next-token), what's its training loss?
3. You see training loss going down but val loss going up. What's the name of this phenomenon, and what causes it?

## Run

```bash
python teaching/exercises/06_learning.py
```

This does one real training step on the real corpus — you'll see loss before, gradients computed, weights updated, loss after.

## What it means

Training is **a tight loop of: forward → loss → backward → step → reset**. Each step nudges every weight a tiny amount. Multiply by tens of thousands of steps, and the random-noise model becomes one that models your corpus.

The loss-over-time curve (lesson 08 uses real checkpoints) is your primary feedback signal. Smooth monotonic decline = healthy. Chaotic spikes = LR too high. Flat = LR too low or dead gradients. The whole SIEVE evolutionary loop is essentially *automating the choice of those hyperparameters*.

## The pitfall to avoid

"Training loss went to zero, we're done!" — no. You care about **val loss**. Training loss at zero means the model memorized the training set, which is easy and useless. Val loss (on text the model never saw) is the honest signal. In small-data regimes like SIEVE's 100K tokens, overfitting is the constant enemy — hence the heavy weight decay, dropout, and regularization.

## Check

1. Why do we multiply the loss by batch size × sequence length when accumulating per-batch losses, then divide by total tokens at the end?
2. What's `optimizer.zero_grad()` for? Why does PyTorch not auto-clear gradients between steps?
3. In `_build_optimizer`, why are LayerNorms and biases excluded from weight decay? (What would happen if you decayed them?)

→ Next: [07_the_pt_file.md](07_the_pt_file.md) — what's actually in a checkpoint, and how to load one.
