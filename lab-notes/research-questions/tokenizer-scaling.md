# Research question — tokenizer scaling vs multi-tokenizer vs pure data scaling

Drafted 2026-04-17 while exp-004 (v006 smoke on gpu-01) trains.

## The question

Given the two empirical findings from exp-001/002:
1. **Meta-overfitting** — individual member test loss (10.87 nats) worse
   than uniform-over-vocab baseline (9.68 nats) on 7.4M params trained
   on ~85K tokens
2. **Tokenizer mismatch** — v002 BPE (trained on 21-source pool) on
   v005 (35-source pool) produced worse ensemble loss than v002 on
   v003 (matched pool)

Does it make sense to:

- **A)** Train a larger BPE on more data (single, fat tokenizer)
- **B)** Build several domain-specific BPEs (clinical / theory / physics)
  and compose them (per-doc selection, or mixture-of-tokenizers)
- **C)** Ignore tokenizer choice, scale corpus + model + compute — the
  mismatch "comes out in the wash" with enough tokens

## First-principles analysis (before literature review)

### A) Single larger BPE

**What it buys us.** More vocab = lower fertility = more tokens per
unit context = each member sees more information per forward pass.
For idiosyncratic domain vocabulary (`deterritorialization`, `jouissance`)
that today are 4-5 tokens, getting them to single-token status reduces
the number of positions the model has to coordinate over.

**What it costs.** Larger vocab = larger embedding matrix (V × d_model).
Our current setup: 16K × 256 = 4.1M params = 55% of the whole model.
Doubling vocab to 32K pushes embedding params to 68% of model. Beyond
that, the tokenizer is eating the entire parameter budget.

**Direct intervention for our specific failure mode?** Partially. The
meta-overfitting problem is driven by training-tokens scarcity, not
by tokenizer fidelity. A better tokenizer slightly increases effective
training signal per token (higher entropy per step = more gradient
information), but does not fundamentally change the data/params ratio.

**Confound with our current setup.** v003 tokenizer already jumped
domain integrity from 65.7% → 75.7% simply by training on a broader
pool, same 16K vocab. The remaining 24% of missed domain terms are
long physics words (`eigenvector`, `electromagnetic`). Those appear
rarely in our corpus, so vocab slots for them would be a waste.

### B) Multi-BPE / mixture-of-tokenizers

**What it buys us.** Each tokenizer can specialize: a clinical BPE
captures `transference`, `countertransference` as single tokens; a
theory BPE captures `hegemony`, `assemblage`; a physics BPE captures
`hamiltonian`, `eigenvector`. For a given document, you use the
matching tokenizer.

**What it costs (conceptually).** This is a different model class,
not a drop-in. Options:
- **Per-document**: at train time, each document is tokenized with its
  source's BPE. Model sees a union vocabulary. This is just A with
  vocab = sum(domain vocabs) and is not really multi-tokenizer.
- **Per-member**: different ensemble members use different tokenizers.
  Then ensembling via logit averaging breaks — logits live in different
  vocab spaces. Would need a shared decoder head or logit projection.
  Significant architecture change.
- **Hierarchical**: first-stage tokenizer splits into chunks, each chunk
  routed to a domain tokenizer. Complex; hasn't been demonstrated to
  help at small scale.

**Direct intervention for our failure mode?** No, as a first-order
effect. Mixture-of-tokenizers addresses *coverage*, not *meta-overfitting*.
It would help with a different problem: when a new corpus introduces
vocabulary the base tokenizer doesn't know. We solved that in v003 by
just retraining on the expanded pool.

**Where it *might* be interesting in SIEVE specifically.** If
different ensemble members used different tokenizers, the members
would be *forced* to learn different tokenization structures, which
increases diversity in a dimension the current search doesn't explore.
This is orthogonal to meta-overfitting though — it's a diversity-gain
research direction, not a data-scarcity fix.

### C) Scale corpus + compute, accept tokenizer as "good enough"

**What it buys us.** Corpus v004 has 7.5M training tokens — 88× more
than v003/v005/v006 at 85K. At that scale:
- Each token in vocab gets enough training exposure to learn a good
  embedding, even for mid-frequency tokens.
- Memorization pressure drops dramatically: 7.4M params / 7.5M tokens
  ≈ 1 param/token, vs today's 7.4M params / 85K tokens ≈ 87 params/token.
  The latter is guaranteed to memorize; the former has to actually
  generalize.
- Whatever tokenizer we use, the model sees enough tokens through each
  vocab slot to fit them usefully.

**What it costs.** Wallclock. At v004 scale, one training run takes
hours, not minutes. Full SIEVE loop (population × gens × train time)
is the cluster run, not the laptop run.

**Direct intervention for our failure mode?** Yes, decisively. The
first-principles diagnosis of 10.87 test nats on a 7.4M-param model
trained on 85K tokens is: **catastrophic memorization of a tiny dataset**.
The model has far more parameters than it has unique examples to fit.
The fix is more examples, not a better tokenizer.

## Pre-literature opinion

**C is the dominant intervention.** A is a minor refinement, useful
once C is in place. B is a research direction that's more interesting
for diversity than for fixing the current failure. I expect the
literature to back this up — "more data" beats "better tokenizer"
whenever they compete directly in small-model regimes.

Key decisive experiment: run SIEVE at v004 scale (same architecture,
same evo params) and measure:
- Single-member test NLL: if this is < ln(V) ≈ 9.68 nats, memorization
  is no longer dominating.
- Ensemble test NLL: if lower than single-member, ensembling is adding
  value over raw scale.

If at v004 scale single-member test NLL drops to ~7-8 nats, C is
proven sufficient. If it's still > 9.68, then the architecture or
search is the bottleneck, not data — and we need to revisit A/B.

## Experimental design flow per hypothesis

### Design A — larger BPE on more data

1. **Ablation targets.** Train 3 tokenizers on the v004 source pool:
   - A1: 8K vocab (smaller; test whether our current 16K is already too big)
   - A2: 16K vocab (control; ≈ v003 tokenizer)
   - A3: 32K vocab (bigger)
   Same config otherwise (byte-level, NFKC+lowercase).
2. **Metric at tokenizer level.** Compare compression, fertility,
   domain integrity on a held-out set of 5 cleaned files never seen
   during BPE training. This catches vocab-OOD effects directly.
3. **Downstream eval.** Build 3 corpora (v007/v008/v009 — same 100K
   target, same sources, each using A1/A2/A3). Run smoke loop on each.
   Compare gen-2 ensemble val loss and gen-2 single-member test NLL.
4. **Decisive signal.** Does 32K beat 16K on ensemble loss more than
   16K beats 8K? Sub-linear returns → vocab size has saturated for
   this corpus. Super-linear → there's more room to scale.
5. **Cost.** 3 tokenizers (minutes each) + 3 smoke runs (~4h each on
   gpu-01) = ~12h. Could parallelize to 3 nodes on compute-cluster.

### Design B — multi-tokenizer ensemble

1. **Precondition.** Requires modifying `worker.evaluator` to handle
   per-member tokenizers. This is a nontrivial code change. Don't
   start B until A is complete.
2. **Tokenizer split.** Train 3 domain tokenizers on disjoint splits
   of `cleaned/`:
   - B-clinical: Fink, Ecker, Bion-Grotstein, Stavrakakis, Lacan
     seminars (~12 files)
   - B-theory: Deleuze, Guattari, DeLanda, Zizek, Kristeva (~12 files)
   - B-physics + other: Philosophy-of-Complex-Systems, Mann, Latour,
     misc (~13 files)
   Each at 8K vocab (smaller, since each sees less data).
3. **Population assignment.** Each ensemble member randomly assigned
   one of the 3 tokenizers. 33% per type.
4. **Logit alignment.** For ensemble eval: either (a) a shared dev
   tokenizer (v003) that all members' logits project into, or
   (b) convert to character-level before averaging. Start with (a)
   — project each member's logits into v003-token-space via a
   pre-computed BPE-to-BPE mapping.
5. **Decisive signal.** Does ensemble val loss decrease more with
   heterogeneous tokenizers (same member count) than with homogeneous?
   If yes, mixture adds diversity beyond what hyperparam perturbation
   alone produces.
6. **Cost.** ~300 LOC of new code + cluster run. 2-3 days work before
   any experimental numbers.

### Design C — just scale the data

1. **Single-node prereq.** Run smoke on v004 at gpu-01:
   - population=6, ensemble=3, generations=3 (minimal, to confirm
     pipeline survives 75× more data)
   - Measure single-member test NLL. This is the load-bearing number.
2. **If single-member test NLL drops below 9.68:** C is confirmed,
   move to cluster.
3. **Cluster run design:**
   - Corpus: v004 (6.4M train tokens)
   - Population: 24, ensemble_size: 8, generations: 30
   - Architecture: n_layer=6, n_embd=384 → ~20M params. Data/params
     ratio ≈ 0.3 — still memorization-prone but manageable.
   - Wallclock: estimate ~20-30h on 12 nodes
4. **Decisive signal.** (i) ensemble test NLL drops below ln(16000);
   (ii) ensemble test NLL < single-member test NLL (ensembling is
   net positive at scale); (iii) per-gen plateau shape — flat means
   we hit data limits, slow decrease means room for more gens.
5. **Cost.** Phase 1: ~1 day. Phase 2: ~1-2 days once cluster is
   wired (see strategy-cluster-run.md).

## Recommended sequence

1. **Finish exp-004** (v006 smoke, in progress). This alone is a mini
   version of A: same data as v005, different tokenizer, 16K vocab.
   The delta is informative about A's direction even before A is
   formally tested.
2. **Run exp-005** = C-phase-1: v004 smoke on gpu-01 at the same
   population/ensemble/gens as the existing smokes, just with 75× more
   training data. Cheapest informative experiment by far.
3. **If exp-005 single-member test NLL is good:** build cluster
   dispatch code, run exp-006 = C-phase-2 (full cluster run on v004).
4. **If exp-005 single-member test NLL is still poor:** move to
   Design A — 3-tokenizer ablation on v004.
5. **Design B** stays on the shelf unless A and C both saturate — at
   that point mixture-of-tokenizers becomes the natural next lever
   for ensemble diversity.

## Literature review

(Populated by the general-purpose agent, in progress as of
2026-04-17 16:48. Results will be written to
`literature-tokenizers.md` when available and cross-referenced here.)

## What this doc is NOT

- Not a commitment to run any of these. It's a menu.
- Not a claim that tokenizer choice doesn't matter — at production
  scale, mixture-of-experts / mixture-of-tokenizers work is active
  research (Gemma / DeepSeek tokenizer ablations). At *our* scale,
  with *our* failure mode, data is more limiting.
- Not ignoring that the 10.87-nats test-loss finding is preliminary —
  it's one checkpoint on one corpus. exp-004 result will either
  corroborate or invalidate that story.
