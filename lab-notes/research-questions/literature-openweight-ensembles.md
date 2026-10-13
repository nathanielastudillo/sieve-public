# Literature — evolutionary ensembles of fine-tuned open-weight models

Compiled 2026-04-18. Index document for the openweight track's research
corpus. Detailed per-topic reviews live in sibling files; this doc
consolidates the cross-cutting conclusions and decision anchors that
feed `docs/architecture-openweight.md`.

Sibling reviews (read these for depth):

- [openweight-merging.md](openweight-merging.md) — §1.1 Sakana evo-merge;
  §1.2 merging primitives (TIES, DARE, SLERP, Task Arithmetic, Model
  Soups, frankenmerging).
- [openweight-inference-arch.md](openweight-inference-arch.md) — §1.3
  ensemble-vs-merge tradeoffs; §1.4 cross-tokenizer alignment
  (DeePEn, EVA, PackLLM, LLM-Blender, MoA).
- [openweight-finetune-distill.md](openweight-finetune-distill.md) —
  §1.5 LoRA/DoRA/QLoRA + LoraHub crossover; §1.6 distillation.
- [openweight-evolution.md](openweight-evolution.md) — §1.7 diversity
  metrics; §1.8 multi-objective EA (NSGA-II/III, MAP-Elites, MOME).

Companion docs:
- [literature-small-corpus-ensembles.md](literature-small-corpus-ensembles.md)
  — the small-corpus track's review. Carries over the
  *Krogh-Vedelsby ceiling* argument, *allow-repeats greedy selection*,
  and *sequestered meta-val*; all transfer to openweight unchanged.
- `docs/architecture-openweight.md` — architecture doc consuming this
  review.

## TL;DR

The combined evidence supports the openweight track's default design:
**within-family LoRA fine-tuning as the mutation op, LoraHub-style
adapter merging as the primary crossover op, NSGA-II + MOME for
selection, and output-ensembling at inference time as the deployment
mode that the 12×GB10 fabric makes effectively free.** Cross-family
integration is staged as a later upgrade, not a first-generation
capability.

Five concrete choices fall out of the review:

1. **Mutation = LoRA/DoRA**, not full fine-tune. 8× cheaper GPU-hours
   per member; required for crossover to work at all (LoraHub needs a
   shared frozen base). Full FT is explicitly out. QLoRA only for 13B
   members that won't fit LoRA-bf16 on a single GB10.
   *(§1.5, openweight-finetune-distill.md)*
2. **Crossover = LoraHub adapter-merge** (gradient-free CMA-ES over
   coefficients, shared base). TIES / DARE / Task-Arithmetic are
   secondary merge primitives for full-weight children when we want
   cross-base variants. Plain TIES collapses past n=8 parents (Liu
   2026, 10,506-merge study); Task Arithmetic is monotonic through n=12.
   *(§1.2, §1.5)*
3. **Inference = within-family output-ensembling on the 12-node cluster.**
   The merge-gap literature is consistent: every well-studied merge
   primitive lies strictly below the separate-ensemble upper bound,
   and the CX-7 fabric makes K≤12 output-averaging ~1× wall-clock
   rather than K×. Merge into a Model Soup only when a cluster's
   CKA radius drops below threshold during survival.
   *(§1.3)*
4. **Diversity metric = composite**: 0.5·embedding-spread +
   0.3·MCQ-disagreement + 0.2·task-vector-cosine-complement, on a
   200-prompt probe set. Parameter-space signals (CKA, weight norms)
   are not trustworthy as fitness inputs per 2024–2026 work.
   *(§1.7)*
5. **Selection = NSGA-II wrapped in a MOME 2-D grid** (quality×diversity
   cells), with Hyperband-style surrogate screening (cheap MCQ proxy →
   full benchmark only for survivors). NSGA-III's reference-point
   machinery is marginal at N=16, holds as a fallback for higher-pop
   experiments. Full fitness every 3 generations; surrogate every gen.
   *(§1.8)*

Three cross-cutting caveats that the individual reviews surface
consistently:

- **Sakana's PS/DFS hybrid is specialization-with-tradeoffs**: +3.2 on
  the target, −4.3 on adjacent tasks (openweight-merging.md §1.1). SIEVE
  will see the same if Canadian-policy is the *sole* fitness criterion —
  hence §1.7 composite diversity + §1.8 multi-objective fitness as a
  counterweight.
- **Most merge methods report missing baselines.** Sakana does not
  report plain logit-averaging the 3 source models, nor
  fine-tuning-at-equivalent-compute. These are cheap baselines for
  SIEVE's first openweight run to produce.
- **Cross-family integration has no controlled 7B evaluation.** The
  tokenizer-alignment methods (DeePEn, EVA, PackLLM) are validated
  mostly at larger scales or on different tasks. Treat §1.4 as a
  Stage-2 upgrade path, not a Stage-1 design.

## Decision anchors for `docs/architecture-openweight.md`

Mapping between open questions surfaced in the architecture doc and the
sibling review that resolves them:

| Arch-doc open Q | Resolved by | Decision |
|---|---|---|
| 1. Evolve LoRAs directly vs. merged full weights? | finetune-distill §1.5 | LoRA-adapter-merge (LoraHub) is the default crossover. A/B against full-merge in cluster run 1. |
| 2. Same-family vs. cross-family staging? | inference-arch §1.4 | Stage 1 within-family only. Stage 2 introduces EVA. Stage 3 MoA text-level for benchmarks. |
| 3. NSGA-II vs. NSGA-III at pop=16? | evolution §1.8 | NSGA-II is fine at N=16; NSGA-III held for N≥32 or ≥5 objectives. |
| 4. Parallel logit-averaging latency vs. single-model? | inference-arch §1.3 | 12×GB10 CX-7 makes output-ensembling ~1× wall-clock; benchmark this directly before committing. |
| 5. How much Caruana-greedy transfers? | small-corpus + evolution §1.8 | Role shrinks to "final aggregator over the NSGA-II Pareto front"; not the main selector. |

## Open questions (carry into first cluster run)

Consolidated from the four sibling reviews; listed here only when
cross-cutting enough that no single review owns them. Single-review
questions stay in their sibling file.

1. **Sakana DFS portability to GQA / Qwen-3 / Gemma architectures.**
   Original paper validated on Mistral-7B. Open across merging +
   inference-arch reviews; flagged by openweight-merging.md as the
   likely first real research contribution SIEVE can publish.
2. **Does composite-diversity fitness actually correlate with ensemble
   test gain in the openweight regime?** The small-corpus track
   derived diversity from random init + hyperparameter variation;
   openweight derives it from LoRA rank + data shard + merge recipe.
   Whether the same JSD-disagreement signal that works in small-corpus
   transfers is an empirical question. Resolvable in cluster run 1.
3. **Is LoraHub-crossover strictly better than full-weight TIES in our
   regime, or only cheaper?** Finetune-distill §1.5 expects "better
   *and* cheaper" based on 2024–2025 results, but no controlled
   comparison at our domain. A/B candidate for cluster run 2.
4. **At what CKA radius should a cluster be Model-Souped?** Inference
   §1.3 identifies the Model-GLUE (Zhou 2024) threshold of 0.95 cosine,
   but that study was not multi-objective. We need our own threshold
   calibrated against fitness-before-merge vs. fitness-after-merge on a
   few example clusters.
5. **Ensemble-as-deployable vs. distillation.** finetune-distill §1.6
   concludes distillation is *probably not worth doing* given cheap
   cluster ensembling. Revisit only when a product constraint forces
   single-model deployment.

## What SIEVE's openweight track adds vs. prior art

Most individual primitives are off-the-shelf (mergekit, axolotl, sglang,
NSGA-II, LoraHub). The composition is where the contribution sits:

- **Generation-scale evolutionary loop over merges + fine-tunes +
  merged-finetunes** — Sakana did evo-merge as a one-shot search over
  fixed parents; GENOME+ did PBT-style evolution over fine-tunes without
  merges. No published system we found combines both inside one EA
  loop on a homogeneous hardware substrate.
- **Hardware-aware selection.** The 12×GB10 fabric turns a "K× inference
  cost" into a ~1× wall-clock cost for K≤12. Most merge-vs-ensemble
  conclusions in the literature are driven by the opposite cost
  assumption. Re-deriving the trade in this regime is novel.
- **Multi-objective fitness with diversity as a first-class axis.** Our
  NSGA-II+MOME hybrid treats diversity as a dimension to optimize, not
  just a regularizer — consistent with the evolution-review's §1.8
  recommendation but not standard in mergekit / PBT literature.

Cluster run 1 is where these three claims get their first controlled
test. See `docs/architecture-openweight.md` §"Testing strategy".
