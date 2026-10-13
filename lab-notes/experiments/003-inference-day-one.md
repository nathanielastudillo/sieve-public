# Experiment 003 — Inference day one (bench + smoke-test)

- **Status**: ad-hoc, completed
- **Date**: 2026-04-17
- **Node**: gpu-01
- **Wallclock**: minutes (single model forward passes, no training)
- **Result**: **Single-member test loss on v005 is 10.87 — WORSE than uniform baseline 9.68.**

Not an experiment in the "run the evolutionary loop" sense. A first
attempt to poke the trained checkpoints with real inference and
benchmarking to see what we actually have.

## What we did

Opened a 4-pane tmux window (`sieve-bench`, all panes sharing
`/home/<user>/dev/sieve`):

| Pane | Role | What it did |
|------|------|-------------|
| 0    | sieve-inference | Text generation from gen-2 member `c884ab4b` on a prompt |
| 1    | sieve-bench     | Test-set NLL sweep on the same checkpoint |
| 2    | sieve-scratch   | Backlog / reference display |
| 3    | sieve-manual    | Opened `manual.md` for reference |

## Pane 0 — generation sample

Loaded `runs/smoke_loop_v005/checkpoints/gen-002-c884ab4b/step_018088_epoch_0007.pt`,
encoded a short prompt through the v002 tokenizer, and generated 100
tokens with temperature 0.8, top-k 40.

Output: syntactically plausible text with domain vocabulary leaking
through (clinical/psychoanalytic phrasing), but semantically incoherent
at sentence boundaries. Expected for a 7.4M-param model on 82K training
tokens.

## Pane 1 — single-member test NLL

Ran the same checkpoint over `runs/smoke_loop_v005/.../test.bin`
(4,608 tokens, uint16 memmap), block_size=256, stride=block_size, 18
non-overlapping windows.

```python
inp = batch[:, :-1].contiguous()
tgt = batch[:, 1:].contiguous()
logits, loss = model(inp, targets=tgt)
```

Required `.contiguous()` on both slices — the stride-skipping slice of a
memmap-backed tensor fails `view()` inside `GPT.forward` otherwise.
(Documented in `manual.md` under "Known gotchas".)

**Result: mean per-token NLL = 10.87 nats.**

Compare:
- Uniform over 16K vocab: `ln(16000) = 9.680` nats
- Val loss of this member (from `lineage.db`): 0.068
- Ensemble val loss: 8.82

The individual member is **1.19 nats worse than uniform** on held-out
test. Memorizing train set (val 0.068) cost it general language-modeling
capability to the point that betting on a uniform distribution would
beat it on text it's never seen.

This is the first concrete measurement of meta-overfitting in a SIEVE
run — the SIEVE thesis (evolutionary pressure for diversity produces
better ensembles than independent training) is premised on individual
members being OK-ish at LM; if they're *below* uniform, the ensemble
gain on val has to be doing a lot of calibration work.

Promoted to `findings.md` for follow-up.

## Pane 2 & 3

Diagnostic/reference panes. No experimental output.

## Immediate questions this opened

1. **Does ensemble val gain transfer to ensemble test loss?** Haven't
   yet measured `ensemble_test_loss` for the winning 3-member combo.
   If ensemble test loss is also > uniform, the SIEVE signal is
   meta-overfit to val.
2. **Is this specific to v005, or does v003 show the same?** Need to
   repeat the bench against a v003 gen-2 member on v003's test.bin.
3. **At what training step does the test loss cross the uniform
   baseline?** With epochwise checkpoints we could answer — but the
   workers only save one checkpoint per member at `best_val_loss`,
   so we don't have the trajectory.

All three are sitting in `findings.md` open questions.

## Artifacts produced

None persisted — generated text was not saved, test-loss sweep was a
one-off `python -c` block. Intentional: these are orientation runs, not
reproducible experiments. The next real experiment (ensemble test-loss
measurement, both corpora) gets its own numbered page and is worth
writing a proper script for.
