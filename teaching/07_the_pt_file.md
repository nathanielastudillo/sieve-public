# 07 — The .pt file

## Bridge

You already know what `pickle` is. `torch.save(obj, path)` is essentially `pickle.dump(obj, f)` with some special handling for tensors (stored as raw binary blobs with a manifest). `torch.load(path)` is the inverse.

A "checkpoint" in PyTorch is just **a Python dict** that someone saved with `torch.save`. There's no magic. No bespoke format. You could save a dict of `{"foo": 42, "bar": [1,2,3]}` to `.pt` and it would round-trip fine.

What makes it a *useful* checkpoint is the convention of what to put in the dict. For SIEVE, it's:

```python
{
    "model_state_dict":     model.state_dict(),        # every tensor by name
    "optimizer_state_dict": optimizer.state_dict(),    # AdamW's running means, step counts
    "config":               mcfg.to_dict(),            # n_layer, n_head, n_embd, etc.
    "metrics":              {"epoch": ..., "loss": ..., ...},
    "timestamp":            time.time(),
    "pytorch_version":      torch.__version__,
}
```

## What is a state_dict?

`model.state_dict()` returns an `OrderedDict[str, Tensor]`. Keys are the dotted-path names of every parameter (and every registered buffer) in the module tree:

```
"transformer.wte.weight"                    torch.Size([8000, 256])
"transformer.wpe.weight"                    torch.Size([512,  256])
"transformer.h.0.ln_1.weight"               torch.Size([256])
"transformer.h.0.attn.c_attn.weight"        torch.Size([768, 256])   # 3*D for QKV
"transformer.h.0.attn.c_proj.weight"        torch.Size([256, 256])
"transformer.h.0.ln_2.weight"               torch.Size([256])
"transformer.h.0.mlp.gate_proj.weight"      torch.Size([704, 256])
"transformer.h.0.mlp.up_proj.weight"        torch.Size([704, 256])
"transformer.h.0.mlp.down_proj.weight"      torch.Size([256, 704])
... (repeat for h.1, h.2, ...)
"transformer.ln_f.weight"                   torch.Size([256])
"lm_head.weight"                            torch.Size([8000, 256])  # tied with wte!
```

It's literally just a name-to-tensor map. When you call `model.load_state_dict(sd)`, PyTorch iterates keys in the dict and copies each tensor into the matching parameter by name. If a name is missing, or a shape doesn't match, it raises.

## Reconstruction, the full recipe

Given a .pt file and the class definition `GPT`:

```python
ckpt = torch.load("run1/step_XXXXXX.pt", map_location="cpu", weights_only=False)

mcfg = ModelConfig.from_dict(ckpt["config"])   # rebuild the spec
model = GPT(mcfg)                                # allocate empty tensors of right shape
model.load_state_dict(ckpt["model_state_dict"]) # fill them by name
model.to("cuda").eval()                          # put on GPU, disable dropout
```

Four lines. That's all "loading a model" is.

**The catch**: `GPT(mcfg)` has to build a module tree whose names exactly match what's in `state_dict`. If you rename a submodule, old checkpoints break. If you change the architecture (add a new layer), old checkpoints don't have weights for it — `load_state_dict` errors unless you pass `strict=False`.

## Why `weights_only=False`?

`torch.load` in recent PyTorch defaults to `weights_only=True`, which refuses to unpickle anything except tensors and a whitelist of safe types. That's a security feature — pickled data can execute arbitrary code on load, so loading an untrusted `.pt` is dangerous.

SIEVE checkpoints contain `config` (a dict) and `metrics` (a dict) — basic Python types. Still, with `weights_only=True` you'd need to whitelist them. The code currently uses `weights_only=False` because the checkpoints are locally trained and trusted. If you were loading third-party checkpoints, you'd want `weights_only=True`.

## ❓ Predict

Before running:

1. The `runs/smoke_v003/checkpoints/step_013240_epoch_0005.pt` checkpoint — roughly how big do you expect it to be in MB? (Count parameters, assume fp32.)
2. If you `torch.save({"hello": 42}, "test.pt")` and then `torch.load("test.pt")`, what do you get back?
3. Checkpoint at epoch 5 and checkpoint at epoch 20 — what's the same between them? What's different?

## Run

```bash
python teaching/exercises/07_checkpoint.py
```

## What it means

You'll see:
- A real checkpoint opened and its keys listed. Notice it's just a dict.
- The state_dict's keys match the architecture you saw in lesson 04/05.
- The file size is ~ sum of tensor sizes + tiny overhead.
- Loading a model is: read config → build empty architecture → populate tensors by name.

## The pitfall to avoid

Don't save models without saving the config. Without the config, you'd have a pile of tensors with no way to know what shape architecture they belong to. SIEVE's convention bakes config INTO the .pt file so each checkpoint is self-describing.

Also: `model.load_state_dict(ckpt)` vs `model.load_state_dict(ckpt["model_state_dict"])`. The first is a common bug — the dict is one level above the tensors. Always index into the right key.

## Check

1. If I trained a model with `n_embd=256`, can I `load_state_dict` its weights into a model built with `n_embd=384`? Why / why not?
2. What happens if `wte.weight is lm_head.weight` (weight tying) at save time — does the state_dict contain both keys? (Go check.)
3. Why do we save `optimizer.state_dict()` alongside the model? If we're just doing inference, do we need it?

→ Next: [08_generation.md](08_generation.md) — turning a loaded model into text.
