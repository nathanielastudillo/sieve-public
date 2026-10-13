# Literature — open-weight ensemble architecture for SIEVE's 12× GB10 track

Compiled 2026-04-18. Sibling to
[literature-small-corpus-ensembles.md](literature-small-corpus-ensembles.md)
(diversity sources + aggregation at the from-scratch-small-model scale) and
[literature-tokenizers.md](literature-tokenizers.md) (tokenizer quality for
custom BPE). This review scopes the openweight track specifically:
fine-tuning 7B–13B open bases (Llama/Qwen/Gemma/Mistral) on Lever's
Canadian-policy corpus, with crossover = model merging, and evaluating on
a 12-node GB10 cluster (ARM Blackwell, ~100GB accessible VRAM each,
CX-7 200G fabric). Small-corpus findings are not repeated.

## TL;DR

SIEVE-openweight has a hardware profile that inverts the assumption in
most of the ensemble-vs-merge literature: those papers optimize for the
*single-GPU* inference regime, where running K models in parallel means
K× memory and linear latency increases. The 12×GB10 cluster makes
**output-space (logit) ensembling essentially free** at member counts
K≤12 — each member fits on one node with headroom, CX-7 200G collapses
the cross-node logit-sum to single-digit milliseconds, and wall-clock
latency is roughly `max_i T_i` rather than `Σ T_i`. This changes the
ranking of the four architectural options:

1. **Within-tokenizer-family logit averaging is the default.** Pick a
   single tokenizer family per run (Llama-3 family: Llama-3-8B +
   fine-tunes; Qwen family: Qwen2.5-7B + variants; Gemma family;
   Mistral family). Token-ID indices align trivially. Inference cost is
   1× per-node, 12× aggregate (but the cluster *has* 12 nodes). Ambiguity
   decomposition still applies; the merge-gap is avoided entirely.
2. **Weight-space merging (TIES / Task-Arithmetic / DARE / Sakana-
   Evolutionary) is the crossover operator.** It is *not* the deployment
   path. Use it to produce children during evolution (cheap: one forward
   pass to validate, no training), keep the parents around, let the
   evolutionary loop decide whether the merged child or its parents
   survive into the next generation's ensemble.
3. **MoE upcycling (Branch-Train-MiX) is a P2 experiment.** It collapses
   K experts into one routed model and matches a K-ensemble's accuracy
   on in-expertise tasks with 2×-active-param inference — worth it only
   if you need to deploy outside the 12-GB10 cluster. BTX requires
   **80B–160B tokens of post-merge MoE finetuning** (Sukhbaatar et al.
   2024), so this is a project-scale investment, not a casual option.
4. **Cross-family ensembling (Llama + Qwen + Gemma together) is only
   viable at text level** (Mixture-of-Agents, Cool-Fusion) or with
   heavy logit-alignment machinery (DeePEn, EVA, PackLLM). Token-level
   methods add 10-20% latency overhead plus per-run calibration; text-
   level methods add multiple full-length generations. **Defer until
   within-family evolution plateaus.**

The practical recommendation: **run evolution within one tokenizer
family per corpus-version.** Merging (TIES/Task-Arithmetic) as the
crossover operator, logit averaging as the aggregator, rank-weighted
Caruana-greedy for ensemble composition. Cross-family combination is a
later experimental arm and is dominated by within-family gains at the
member counts (≤12) the hardware supports.

## §1.3 Ensemble inference vs. weight-space merging

The central trade-off: you can collapse K fine-tuned models into a
single model (1× inference cost, some accuracy loss = "merge gap"), or
you can keep all K and combine at inference (K× cost, full diversity
preserved). Four modes worth distinguishing.

### Mode A — Weight-space merging (deploy as 1 model)

Represents fine-tuned models as task vectors τ_i = θ_i − θ_base and
combines: θ_merged = θ_base + Σ λ_i · aggregate(τ_i). The merge-gap is
the accuracy delta between this single merged model and the K
separately-evaluated fine-tunes.

- **Task Arithmetic** (Ilharco et al. 2023, arXiv:2212.04089). The
  simplest baseline: θ_merged = θ_base + Σ λ_i τ_i. *Still the most
  reliable method at scale.*

- **TIES-Merging** (Yadav et al. 2023, arXiv:2306.01708). Drop low-
  magnitude delta entries, resolve sign conflicts, merge only the
  dominant-sign parameters. *Wins on 2-4 models; degrades past ~8.*

- **DARE** (Yu et al. 2023, arXiv:2311.03099). Randomly drop 90-99% of
  delta entries and rescale the rest. *Often combined with TIES;
  survives extreme dropout because deltas are redundant.*

- **Sakana Evolutionary Merge** (Akiba et al. 2024, arXiv:2403.13187).
  CMA-ES over merging recipes in parameter space + dataflow space.
  Produced SOTA Japanese-math 7B from 3 source models. *This is the
  crossover operator we want — searches the recipe, not just applies
  a fixed one.*

- **Model-GLUE** (Zhou et al. 2024, arXiv:2410.05357). Cluster the zoo
  by weight-cosine-similarity (threshold 0.95), merge within clusters,
  ensemble across clusters ("model-level mixture"). **This is the
  cluster-and-merge hybrid** we asked about — and it exists, though
  single-paper. +5.61% avg over Llama-2 zoo, no training.

- **A Systematic Study of Model Merging** (Liu et al. 2026,
  arXiv:2511.21437). 10,506 merges across 0.5B-72B base sizes × 9
  domains × 4 methods. **Numbers worth quoting:**
  - Task Arithmetic: +1.02 avg gain at n=12 merged models; monotonic
    improvement with n.
  - TIES-Merging: **-4.32** avg drop at n=12 (catastrophic past n=8).
  - Only Task Arithmetic produces consistent "constructive
    interference" across n.
  - All merges are **strictly below the ensemble of the same
    fine-tuned sources** ("all merging methods lie between the
    pretrained model and the individually fine-tuned models"). No
    merge method closes the merge-gap entirely.

**Expected merge gap at SIEVE scale:** for small N (2-4 same-family
same-init fine-tunes), task-arithmetic gaps are ~0.3-1.0 accuracy
points vs the separate-ensemble upper bound. Past N=8, TIES catastrophe
territory; task-arithmetic continues to extract value but at reduced
marginal rate. Cross-family merging (Llama-3 + Qwen) is effectively
**infeasible without neuron-alignment machinery** (OT-Fusion,
Git-Re-Basin) — the Liu 2026 study explicitly scopes this out.

**Takeaway.** Weight merging is right as a **crossover operator** in
SIEVE's evolutionary loop (cheap child-generation from elite parents),
but **not as the deployment path** when 12 GB10s are available to
serve the ensemble directly.

### Mode B — Output-space ensembling (deploy as K models)

Run K models, combine at the output.

- **Logit averaging vs probability averaging.** Njieutcheu Tassi et al.
  2022 (ceur-ws:Vol-3215/19.pdf) — the classic comparison. Logit
  averaging reduces expected calibration error from 12.03% → 5.44% on
  CIFAR-10 with MC-Dropout ensembles; but can *increase* false-positive
  confidence (51.31% → 94.58% on MNIST) when members are overconfident.
  **For LLMs specifically**, logit averaging is the near-universal
  default in the 2024 ensemble literature (LLM-Blender, DeePEn, EVA,
  PackLLM, Cool-Fusion) because softmax-space averaging over-weights
  confident-but-wrong members. Probability averaging is only preferred
  in safety-critical regimes where separability-of-confidence matters
  more than NLL.

- **Plurality / majority voting.** Used when logits aren't available
  (API ensembles). Loses all gradient information; worse NLL than both
  logit and probability averaging on per-token tasks. Only viable at
  full-sequence granularity (agree on final answer) — the
  Mixture-of-Agents regime, not the per-token-decoding regime.

- **Expected gain.** At the same-family, same-base, K=4-8 fine-tuned
  scale, logit-averaging ensembles beat the best single member by
  **1-3 points** on most benchmarks (consistent across DeePEn, EVA,
  PackLLM reports at K=2-4). DeePEn specifically reports +1.43 MMLU to
  +2.72 PIQA over best-of-top-2. The gains are smaller than merge-gap
  losses on Task Arithmetic at small n, so **the ensemble beats the
  merge by ~0.5-2 points net** in the SIEVE target regime.

- **Inference latency at 7B scale.** A Llama-3-8B at bf16 is ~16GB VRAM
  and decodes at ~40-80 tok/s on a GB10 (comparable Blackwell
  characterization). Running K models **on K separate nodes**:
  per-token latency is `max_i(T_i) + cross-node logit-sum latency`.
  Over CX-7 200G, a 128k-vocab logit vector is 128k × 2 bytes × 12
  nodes = ~3MB to gather, ~0.1ms at saturated RDMA. Per-token
  *effective latency ≈ single-member latency*, with the logit-sum
  overhead noise. This is the key hardware insight: **on 12×GB10, the
  output ensemble is ~1× latency, not K×.**

- **Memory footprint.** Each node independently holds one 7B member +
  KV cache for the inference batch. At bf16: ~16GB weights + ~8-20GB
  KV cache = ~25-40GB / node, well under GB10's ~100GB. We could
  actually serve two 7B members per node, unlocking K up to ~24 at the
  cost of halved per-member batch size.

- **Training overhead.** Zero beyond training each member individually
  (which SIEVE does anyway).

**The 12-GB10 hardware inversion.** Most ensemble-vs-merge papers
(including the Liu 2026 systematic study) score the comparison on
**single-GPU inference**, where K=4 ensembling means ~4× wall-clock.
That's where the "merge is nearly-as-good and much cheaper" story
comes from. On our hardware, K=12 ensembling is ~1× wall-clock (each
member runs on its own node in parallel; interconnect handles the
logit-sum). **The merge-gap isn't being traded against inference cost;
it's being paid for nothing.** This flips the default: ensemble first,
merge only as an evolutionary operator or for off-cluster deployment.

### Mode C — Hybrid cluster-and-merge

K-means on parameter deltas → merge within each cluster → ensemble
across clusters.

- **Model-GLUE** (Zhou et al. 2024, arXiv:2410.05357) implements
  exactly this. Pre-clusters by weight-cosine-similarity ≥0.95, merges
  each cluster with TIES/task-arithmetic, ensembles the cluster
  representatives. +5.61% on Llama-2 zoo.

- **Channel Merging** (Yu et al. 2025, various surveys). Within-expert
  channel clustering before per-channel parameter merging. Reduces
  within-cluster interference further.

- **Twin-Merging / Weight-Ensembling MoE.** Dynamic routing between a
  merged backbone and per-task residuals. Hybrid of Mode A and Mode D.

**Does anyone actually run cluster-and-merge at scale?** Only
Model-GLUE, in a benchmark study. It's **under-explored given the
hardware inversion described above** — clustering isn't needed to save
inference cost when 12 nodes are available, but clustering IS useful
to group SIEVE's lineage tree: siblings within a lineage are soupable,
models from different lineage branches may not be. We could use
cluster-and-merge as:

- Crossover within a lineage-cluster (cheap, well-supported by Wortsman
  model-soups at shared-init);
- Output-ensembling *across* lineage-clusters (preserves the cross-
  branch diversity).

Treat this as the **default evolutionary-loop architecture for the
openweight track**: merge within lineage, ensemble across lineages.
This matches SIEVE's existing architecture (crossover, selection,
ensemble-level fitness) 1-to-1.

### Mode D — MoE upcycling (collapse K experts → 1 routed model)

Per-token routing between K expert feedforward layers.

- **Sparse Upcycling** (Komatsuzaki et al. 2022, arXiv:2212.05055). Take
  a dense checkpoint, duplicate its MLP into K experts, initialize a
  router, finetune. Upcycled T5 Base/Large/XL match dense at ~50% of
  initial pretraining compute; outperform sparse-from-scratch at 100%
  of dense compute. Key baseline result.

- **Branch-Train-MiX (BTX)** (Sukhbaatar et al. 2024, arXiv:2403.07816).
  (1) Branch: copy the seed K times. (2) Train: each copy specializes
  on its data slice (math / code / wiki / generalist). (3) Mix: take
  the experts' FFN params as K-way MoE experts, average the shared
  backbone, re-finetune the router + backbone for 80B-160B tokens. On
  Llama-2-7B seed with 4 experts: **avg score 47.9 vs 40.7 baseline,
  beating sparse-upcycling (46.3) and Branch-Train-Merge weight-avg
  (43.4).** One-line takeaway: BTX is the best way to deploy K SFT
  experts as a single model *if you can afford the 80B-160B-token MoE
  finetune*.

- **Drop-Upcycling** (Nakamura et al. 2025, arXiv:2502.19261). Partial
  re-initialization of duplicated MLP weights breaks symmetry faster
  and fixes the "sparse upcycling trains slower than from-scratch in
  the long run" problem. Minor refinement of Komatsuzaki.

- **Recycling Pretrained Checkpoints** (Zhou et al. 2025,
  arXiv:2510.08008). Orthogonal-growth upcycling for efficient LLM
  pretraining — different problem framing but same infrastructure
  (expand a dense checkpoint into an MoE).

**Compute cost to upcycle N SFT models into a MoE**, in SIEVE-
relevant units: **80-160B tokens of finetuning** (BTX reported).
Lever's Canadian-policy corpus is currently ~400M tokens (lex-corpus
estimated); at 160B tokens of MoE finetune this is **400 epochs of
the whole corpus**, which is well past any reasonable repetition
budget (Muennighoff 2023 RD*≈15). MoE upcycling on our corpus scale
is **not practical** unless we either (a) use a much smaller
finetuning budget and accept the post-MoE performance drop, or
(b) mix open-domain data into the MoE-finetune stage.

**Inference latency and memory at 7B scale.** BTX-style Top-2 routing
on Llama-2-7B: ~8GB additional per expert beyond the shared backbone,
2 experts active per token → ~40% extra compute vs the seed. On a
single GB10 the 4-expert BTX fits in ~40GB. Faster than 4-way output
ensembling on a single GB10 (which would be 4× single-model latency).
**But on 12×GB10 with one expert per node, output ensembling is
faster.** BTX wins for *off-cluster deployment*; output ensembling
wins for *on-cluster evaluation/production*.

### Summary table — four modes at the SIEVE openweight scale

| Mode | Inference cost (12-GB10) | Memory | Train overhead | Expected gain | When to use |
|------|--------------------------|--------|----------------|---------------|-------------|
| **A** Weight merging (TIES/Task-Arith/Sakana-Evo) | 1× | 1 model | 0 (merge is ~1hr) | +0 to +1 pts vs best single; **-1 to -2 pts vs output ensemble** | Crossover operator; off-cluster deployment |
| **B** Output (logit) ensembling | ~1× (parallel) | K nodes | 0 | +1-3 pts vs best single member | **Default deploy + eval on cluster** |
| **C** Cluster-and-merge (Model-GLUE-style) | ~1× (parallel, fewer branches) | C nodes (C<K clusters) | 0 | +5.61% reported (Llama-2 zoo) | Evolutionary loop with lineage structure |
| **D** MoE upcycling (BTX) | ~1.4× dense | 1 node (larger) | **80-160B tokens** | +7.2 pts vs seed (BTX reported) | Only for off-cluster single-model deployment, and only if finetune budget is available |

**Cross-cutting ceiling.** All four modes are upper-bounded by the
same ambiguity decomposition inequality — ensemble/merged accuracy is
at most `avg member accuracy + diversity bonus`. Openweight 7B-13B
bases start at much higher single-member accuracy than from-scratch
7.4M, so the headroom is smaller in absolute nats but substantial in
pass-rate benchmarks where current Llama-3.1-8B on policy-QA is
~70-78%. Gains of +1-3 points are realistic; gains of +10 points are
not (unless members are complementary across *disjoint* domains,
which is the BTX regime).

## §1.4 The tokenizer alignment problem

Ensembling (Mode B) assumes the logit vectors from different members
are index-compatible. For same-tokenizer ensembles (all Llama-3-8B
fine-tunes) this is trivially true. For **cross-family** ensembles
(Llama-3 + Qwen2.5 + Gemma-2 + Mistral-Nemo), the vocabularies
disagree — different token IDs map to different strings, different
BPE merges, different special tokens. Direct logit averaging across
these models is undefined.

The cross-tokenizer alignment literature has exploded since mid-2024.
Three families of solution; all have sharp cost-quality trade-offs.

### §1.4.1 Approaches to align logits across tokenizers

**Vocabulary-overlap projection** (early 2024 approach).
Find tokens present in both vocabularies → restrict logit averaging to
the intersection. Works for reading off ensemble disagreement but
throws away most of the probability mass (cross-family overlap is
typically 10-20% of either vocab). Only useful as a **diversity
measurement** tool.

**Minimum-Edit-Distance mapping — PackLLM** (Mavromatis et al. 2024,
arXiv:2404.11531). For each token in a reference tokenizer, find the
textually-closest token in each foreign vocabulary via edit distance.
Use this mapping to rewrite inputs for foreign models, get their
logits, map back. Adds test-time perplexity-minimization to weight
each member. +1.89 accuracy over test-time-fusion baselines; +3.92 to
+11.94 over learning-based fusion. **One-line takeaway:** MinED-based
fusion is cheap (no training, no alignment step) but lossy where
tokenizers segment differently — especially bad for rare/technical
terms that segment to different sub-word sequences.

**Relative-representation projection — DeePEn** (Huang et al. 2024,
arXiv:2404.12715; NeurIPS 2024 spotlight). Map each model's token
embeddings into a "relative representation" space defined by cosine
similarity to a set of anchor tokens shared across vocabularies.
Average in relative space, then gradient-descent back to the main
model's token space. **+1.43 MMLU to +2.72 PIQA over best single
member.** Latency overhead: **~17% per-token, ~2% per-sentence** (most
tokens don't need fusion). Works across 7B-20B heterogeneous LLMs
(Llama-2, Mistral, InternLM, Yi, Skywork, Tigerbot). **One-line
takeaway:** DeePEn is the strongest cross-tokenizer logit-ensemble
method as of 2024, but the 17% per-token latency is substantial when
you're already running on a cluster.

**Learned vocab projection — EVA** (Xu et al. 2024,
arXiv:2404.09492, NAACL 2024). Learn projection matrices between each
pair of LLM vocabularies using overlapping tokens as supervision. At
inference, use the learned matrix to project foreign logits into the
reference vocab. Adds a filter to drop "unfaithful" tokens per
generation step. Validated on commonsense reasoning, arithmetic,
translation, data-to-text. **One-line takeaway:** higher-quality
alignment than MinED at the cost of a one-time training step per
(model-pair) — reasonable in an evolutionary loop where the
population is somewhat stable across generations.

**Generation-as-Classification — GaC** (Yao et al. 2024,
arXiv:2406.12585, EMNLP-Findings). Treat next-token generation as a
classification problem over the union vocabulary. At each step, each
LLM produces a distribution over its vocab; project all into a shared
**union** space via tokenization-based matrices, aggregate there.
Requires no training, just two matrix ops per step. One-line
takeaway: the lightest-weight cross-tokenizer ensemble — minimal
compute, minimal setup, but quality tracks the vocab-overlap ceiling.

**FusionRoute** (2026, arXiv:2601.05106). A lightweight router both
picks the best expert per token AND contributes a corrective logit.
Beats sequence-level and token-level collab plus merging plus fine-
tuning across Llama-3 / Gemma-2 families. *Outside our current
evaluation window (published 2026), but flag for v2 of this review.*

**Cool-Fusion** (Liu et al. 2024, arXiv:2407.19807). Text-level fusion
— LLMs rerank each other's *text* outputs at different granularities,
no logit-space alignment needed. +17.4% on GSM8K from three strong
LLMs. **One-line takeaway:** solves the cross-tokenizer problem by
avoiding it; pay the cost in K× generation latency and O(K²) reranking.

**Universal Logit Distillation** (Boizard et al. 2024,
arXiv:2402.12030). Optimal-transport-based loss for cross-tokenizer
**distillation** (training-time), not inference-time ensembling. Uses
Wasserstein distance to align distributions over mismatched vocabs.
**Relevant only if the SIEVE endpoint is a distilled single-model
deployment**, not a live ensemble. Follow-ups include Cross-Tokenizer
Distillation via Approximate Likelihood Matching (arXiv:2503.20083)
and byte-level interface distillation (arXiv:2604.07466).

### §1.4.2 Text-level cross-family ensembling

**LLM-Blender** (Jiang et al. 2023, arXiv:2306.02561). Two stages:
PairRanker (cross-attention over pairs of candidate outputs → rank)
plus GenFuser (a learned seq2seq model that rewrites the top-K ranked
outputs into a merged response). **Latency cost:** GenFuser is a
forward pass of a fine-tuned fusion model *on top of* K full-length
generations from members; overall latency is roughly (K+1) × single-
member + O(K²) pairwise ranker comparisons. For GSM8K-style
step-by-step tasks, this is typically 3-10× the single-member
latency. **One-line takeaway:** strong quality ceiling (strongest
reported for cross-vocab ensembles in 2023), but the pairwise O(K²)
scaling kills it past K≈4-6. Not tractable at K=12 on our cluster.

**Mixture-of-Agents (MoA)** (Wang et al. 2024, arXiv:2406.04692).
Stacked layers of LLMs, each layer synthesizes outputs from the
previous. Default: 3 layers × 6 LLMs = 18 full-length generations.
**Latency:** 3× the slowest single-member, assuming parallel within-
layer. **Accuracy:** 65.1% AlpacaEval 2.0 vs GPT-4 Omni 57.5% (+7.6
points). Ensembled: Qwen1.5-110B-Chat, Qwen1.5-72B-Chat, WizardLM-8×
22B, Llama-3-70B-Instruct, Mixtral-8×22B, dbrx-instruct. **One-line
takeaway:** state-of-the-art text-level ensemble; dominates when you
have time to wait 3× longer and members are 70B-class. **For SIEVE at
7B-class members on open-Canadian-policy this is overkill at
generation time** — use as a final-stage aggregator for policy-QA
benchmarks where latency doesn't matter.

**When does text-level ensembling (MoA/Cool-Fusion) dominate vs lose
to merge-based approaches?**

- **Dominates** when: member tokenizers span ≥2 families and per-token
  alignment would lose too much mass; tasks are open-ended generation
  (policy drafts, briefs) where output quality >> inference cost; K is
  large (≥6) and pairwise alignment (LLM-Blender) gets O(K²).

- **Loses** when: members share a tokenizer (just logit-average, much
  cheaper); tasks are short-form / multiple-choice / closed (MMLU,
  policy classification) where 3× latency buys little; deployment is
  latency-constrained (end-user-facing).

- **SIEVE-Lever likely needs both.** Policy-QA classification + fact-
  retrieval: within-family logit ensembling. Long-form policy-brief
  generation: MoA as a final-stage aggregator over top-K ensembles.

### §1.4.3 Token-level alignment — 2024-2025 landscape summary

| Method | Cost | Cross-family? | Setup | Quality |
|--------|------|---------------|-------|---------|
| Vocab-overlap projection | Cheap | Measurement only | None | Low |
| PackLLM (MinED) | Cheap | Yes | None | Medium; bad on rare terms |
| DeePEn (relative-rep) | +17%/token | Yes | Pick anchors | **Strongest 2024** |
| EVA (learned projection) | Cheap at run time | Yes | Train per-pair matrix | High (matches DeePEn quality with lower runtime overhead, higher setup cost) |
| GaC (union-vocab classification) | Cheapest | Yes | None | Medium |
| LLM-Blender (text+rerank) | O(K²) rerank | Yes | Train fusion model | High but blows up at K≥6 |
| MoA (text, multi-round) | K×3 layers | Yes | None | **SOTA text-level** |
| Cool-Fusion (text rerank) | K× gen + K² rerank | Yes | None | High on math/reasoning |
| ULD (distillation, not ensemble) | Train-time | Yes | Training | Distillation endpoint |

### §1.4.4 Practical recommendation: staged strategy

The claim to validate: **stay within one tokenizer family early,
introduce cross-family combination (expensive text-level) only after
within-family evolution plateaus.**

**Status: validated in spirit by multiple 2024 studies, with the
caveat that the cross-family step is optional, not mandatory.**

- Within-family gains (same-tokenizer logit ensembling, same-base
  merging) are **cheap** (0 alignment overhead), **empirically robust**
  (well-tested across all four model families), and extract the bulk of
  ensemble gain (~1-3 points over best single member).
- Cross-family gains on top of within-family are **smaller** (DeePEn
  reports +1.43-2.72, not +5-10) because member outputs become
  partially redundant once each family is already extracting
  within-family diversity.
- Cross-family cost is **disproportionate**: token-level methods add
  17% per-token latency plus per-run calibration; text-level methods
  add 3× generation time.
- The **two tokenizer families that are most complementary on
  English-Canadian policy text** are Llama-3 (tiktoken-family, 128k
  vocab, general-purpose) and Qwen2.5 (custom vocab optimized for
  code + zh/en, 152k vocab). Gemma-2 (256k vocab, SentencePiece) and
  Mistral-Nemo (131k Tekken) are third/fourth tier for our corpus.

**Concrete staging for SIEVE-openweight:**

1. **Stage 1 (generations 0-50)**: within-family only. Pick Llama-3-8B
   (or Qwen2.5-7B) as the seed. Run the evolutionary loop with logit
   averaging as the aggregator, TIES / Task-Arithmetic as the
   crossover operator, Caruana-greedy as the ensemble-selection
   algorithm. Expected ceiling: +1-3 points over best-single-fine-tune
   on Canadian-policy benchmarks.
2. **Stage 2 (generations 50-100, only if Stage 1 plateaus)**:
   introduce a second family. Use **EVA** as the alignment method
   (lowest runtime overhead, one-time per-family-pair training cost).
   Expect marginal gain of +0.5-1.5 points on top of Stage 1.
3. **Stage 3 (only for benchmark-reporting runs)**: add **MoA-style
   text-level aggregation** as a final pass for long-form outputs.
   3× latency, ~+2-5 points on open-ended evals.

**Do not** start with cross-family ensembling. Every cross-family
paper shows same-family + same-base is the bulk of the gain;
cross-family is the icing.

**Do not** start with MoE upcycling unless the deployment target is a
single node without the 12-GB10 cluster available (e.g., open-weight
release of a merged SIEVE model to the HuggingFace Canadian-policy
community).

**Do** track cross-tokenizer diversity as a *signal*: even during
Stage 1, periodically compute a Llama-3 × Qwen2.5 ensemble with EVA
or DeePEn and log its val loss. This tells us whether cross-family
gains would be large (worth Stage 2) or small (skip Stage 2).

## Open questions

1. **What fraction of SIEVE's evolutionary gain comes from cross-
   lineage output-ensembling vs within-lineage weight-merging?**
   Directly testable: run the same budget twice — (A) pure merging
   (Mode A only, 1× deploy), (B) pure output-ensembling (Mode B only,
   K× deploy). Measure Canadian-policy-QA accuracy delta. The
   Model-GLUE result (+5.61%) suggests the hybrid wins, but at 7B
   scale on a policy corpus this has not been empirically tested.

2. **At what K does logit ensembling on 12× GB10 stop being "free"?**
   CX-7 200G interconnect should handle 128k-vocab × 12-node logit
   sums in ~0.1ms. But at K=24 (2 members per node) the KV-cache
   fight gets real; at K=48 (4/node) contexts >2k tokens start
   spilling. **Benchmark before deciding the default population
   size.** Hypothesis: K=12 is sweet spot (one-per-node, no contention);
   K=24 is viable for short-context; K≥36 not useful.

3. **Does Sakana evolutionary-merge find better recipes than TIES +
   CMA-ES on coefficient?** Sakana searches both parameter-space and
   dataflow-space; TIES only does weight-merging. On Canadian-policy
   the dataflow-search advantage is unclear — tasks are less
   structurally heterogeneous than Japanese-math. Worth an A/B at
   Stage 1.

4. **Can we use MoE upcycling as a *compression* step at the end of
   evolution, rather than during?** I.e. run SIEVE as ensemble-of-12
   for research, then BTX-collapse the winning ensemble into a single
   MoE for deployment. Sukhbaatar 2024 shows the 80-160B-token
   finetune budget; on Lever's ~400M-token corpus this is infeasible
   without open-domain mixing. **Open: whether a short (~10B token)
   BTX-finetune preserves enough ensemble gain to be worth the
   deployment simplification.**

5. **Does within-family tokenizer *version* matter?** Llama-3 ships
   with 128k vocab; Llama-3.1 and Llama-3.2 share it; Llama-3.3 may
   extend it. Fine-tunes across minor-version bumps are usually
   logit-compatible but the literature doesn't validate this
   explicitly for ensembles. Cheap sanity check.

6. **How do Canadian-policy-specific tokens segment differently across
   families?** Legal terms ("notwithstanding clause", "Charter",
   "règlement"), French passages, acronyms (CSIS, CRTC). Unknown
   whether Llama-3 tokenizes these more compactly than Qwen2.5 —
   which if true would weight Llama-3 toward preferred family. **30
   minutes of work to answer.** Cross-reference with
   [literature-tokenizers.md](literature-tokenizers.md) domain-
   integrity metric.

7. **Is there an evolutionary-crossover operator that's cheaper than
   TIES and better than Task-Arithmetic?** Liu 2026 finds only Task-
   Arithmetic reliably constructive at n≥8. **DARE+TIES might fit but
   requires validation at the SIEVE population scale (12-24 members).**

8. **How does Mixture-of-Agents scale when the members have already
   been fine-tuned on overlapping data?** MoA's diversity assumption is
   models-from-different-providers. SIEVE fine-tunes from one base on
   one corpus — members are *less* diverse. Unclear whether MoA's
   +7.6-point gain transfers or whether we'd see much smaller lift.
   Testable at Stage 3.

## Bibliography

- Akiba, Shing, Tang, Sun & Ha 2024 — Evolutionary Optimization of Model
  Merging Recipes — https://arxiv.org/abs/2403.13187
- Boizard, El Haddad, Hudelot & Colombo 2024 — Towards Cross-Tokenizer
  Distillation: the Universal Logit Distillation Loss for LLMs —
  https://arxiv.org/abs/2402.12030
- Huang, Yu, Luan, Luo, Meng & Huang 2024 — Ensemble Learning for
  Heterogeneous LLMs with Deep Parallel Collaboration (DeePEn) —
  https://arxiv.org/abs/2404.12715
- Ilharco, Ribeiro, Wortsman, Gururangan, Schmidt, Hajishirzi & Farhadi
  2023 — Task Arithmetic — https://arxiv.org/abs/2212.04089
- Jiang, Ren & Lin 2023 — LLM-Blender: Ensembling Large Language Models
  with Pairwise Ranking and Generative Fusion —
  https://arxiv.org/abs/2306.02561
- Komatsuzaki, Puigcerver, Lee-Thorp, Ruiz, Mustafa, Ainslie, Tay,
  Dehghani & Houlsby 2022 — Sparse Upcycling: Training Mixture-of-
  Experts from Dense Checkpoints — https://arxiv.org/abs/2212.05055
- Liu, Chen, Rawat, Xu et al. 2026 — A Systematic Study of Model
  Merging Techniques in Large Language Models —
  https://arxiv.org/html/2511.21437v1
- Liu, Wang, Xu, Lin, Gao & Shang 2024 — Cool-Fusion: Fuse Large
  Language Models without Training — https://arxiv.org/abs/2407.19807
- Mavromatis, Karypis & Karypis 2024 — Pack of LLMs: Model Fusion at
  Test-Time via Perplexity Optimization (PackLLM) —
  https://arxiv.org/abs/2404.11531
- Nakamura, Sakai, Kumagai & Yamada 2025 — Drop-Upcycling: Training
  Sparse Mixture of Experts with Partial Re-initialization —
  https://arxiv.org/abs/2502.19261
- Njieutcheu Tassi et al. 2022 — The impact of averaging logits over
  probabilities on ensembles of neural networks —
  https://ceur-ws.org/Vol-3215/19.pdf
- Sukhbaatar, Golovneva, Sharma, Xu, Lin, Rozière, Kahn, Li, Yih,
  Weston & Li 2024 — Branch-Train-MiX: Mixing Expert LLMs into a
  Mixture-of-Experts LLM (BTX) — https://arxiv.org/abs/2403.07816
- Wang, Wang, Athiwaratkun, Zhang & Zou 2024 — Mixture-of-Agents
  Enhances Large Language Model Capabilities —
  https://arxiv.org/abs/2406.04692
- Wortsman et al. 2022 — Model Soups —
  https://proceedings.mlr.press/v162/wortsman22a.html
- Xu, Lu & Zhang 2024 — Bridging the Gap between Different Vocabularies
  for LLM Ensemble (EVA) — https://arxiv.org/abs/2404.09492
- Yadav, Tam, Choshen, Raffel & Bansal 2023 — TIES-Merging: Resolving
  Interference When Merging Models — https://arxiv.org/abs/2306.01708
- Yao, Li, Hong, Wang, Xu, Wan & Liu 2024 — Breaking the Ceiling of
  the LLM Community by Treating Token Generation as a Classification
  for Ensembling (GaC) — https://arxiv.org/abs/2406.12585
- Yu, Yu, Yu, Xiang, Zhao & Li 2023 — Language Models are Super Mario:
  Absorbing Abilities from Homologous Models as a Free Lunch (DARE) —
  https://arxiv.org/abs/2311.03099
- Zhou et al. 2024 — Model-GLUE: Democratized LLM Scaling for a Large
  Model Zoo in the Wild — https://arxiv.org/abs/2410.05357
