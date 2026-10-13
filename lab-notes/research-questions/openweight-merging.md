# Literature — evolutionary merging on open-weight 7B–13B bases

Compiled 2026-04-18. Sibling note to
[literature-small-corpus-ensembles.md](literature-small-corpus-ensembles.md)
(which covered from-scratch ensembling on the 10⁵-token corpus) and
[literature-tokenizers.md](literature-tokenizers.md). This review is
scoped to SIEVE's **openweight track**: evolutionary search over
**merging recipes** on 7B–13B open-weight bases (Llama / Qwen / Gemma /
Mistral family), fine-tuned on Lever's Canadian-policy + polling
corpus, with inference parallelized across a 12-node GB10 cluster
(ARM Blackwell SBCs on CX-7 200G RDMA fabric). Small-corpus material
is not repeated here.

## TL;DR

The core move is Sakana's (Akiba et al. 2024, arXiv:2403.13187):
**treat merging recipes as a hyperparameter vector and run CMA-ES over
them, evaluating each candidate by forward-pass fitness** — no gradient
steps, no training compute on the merge itself. The paper is the
load-bearing reference for the openweight track. Two findings carry
most of the weight:

1. **Parameter-space (PS) merging** — evolving DARE-TIES coefficients
   per-layer-per-model — is cheap (~6 coefficients for a 3-model,
   simplified-per-layer setup), converges in ~1000 trials of a ~10-
   member CMA-ES, and **beat hand-crafted recipes** on MGSM-JA
   (9.6–30.0% → 52.0%, best 70B JP baseline 30.0%).
2. **Data-flow-space (DFS) merging** — evolving a permuted
   layer-sequence through the union of source models, with a scaling
   matrix W on skip-connections — is higher-dimensional (~192-slot
   indicator array + 64×64 W for M=64 layers, r=3 repeats), takes
   population=128 × 100 generations, and is most useful as a
   **complement** to PS. The hybrid (PS-then-DFS, with the PS-merged
   model added to the DFS source pool) was the winning recipe:
   55.2% MGSM-JA at 10B params.

For SIEVE-openweight, the crossover operator is the merge itself. The
**decision tree** at §1.2 recommends:

- **2 models, same init / same family** → SLERP (default t=0.5) or
  linear — reliable, 1 hyperparameter.
- **N≥3 models, same base** → DARE-TIES (density 0.53, weights summing
  to ~1.0, `normalize=true`) — community default for the leaderboard.
- **N≥3 models, fitness-validated** → CMA-ES over DARE-TIES per-layer
  density+weight (Sakana PS recipe) — +1–2 evaluation-budget orders
  of magnitude, but dominates hand-tuned recipes when there is a
  fitness signal to optimize.
- **Same family, different depths / capability-stacking** →
  passthrough/frankenmerge via mergekit layer-range slices; highly
  experimental, no theoretical grounding, wide variance in outcomes.
- **Cross-family / different tokenizers** → SLERP/TIES/DARE all fail
  without tokenizer surgery. Use `mergekit-tokensurgeon` to align
  vocabularies first; expect capability loss. No clean winner here —
  needs A/B.
- **Shared base, many fine-tunes, no fitness signal** → Model Soups
  greedy (Wortsman 2022, arXiv:2203.05482); zero-hyperparameter, free.
- **Shared base + closed-form coefficient solve** → RegMean (Jin 2023,
  arXiv:2212.09849) or Fisher-weighted; used when you have a small
  unlabeled dev set and want a one-shot merge.

**Cost order of magnitude on a 12-node GB10 cluster** (each node ~128
GB unified memory, 4× 200G CX-7 fabric, ARM Blackwell, ~130 TFLOPS
FP16): a single 7B merge via mergekit takes ~5–15 minutes on one node
(mostly I/O and tensor math, trivial FLOPs). With 12 nodes running
merges in parallel, we can evaluate ~60–100 candidate merges/hour
assuming each fitness forward-pass on a ~1K-example benchmark subset
takes ~60 s at 7B. A 100-generation × 128-population CMA-ES run
(Sakana DFS budget) = 12,800 evaluations = ~5–10 days wallclock —
which is why Sakana only ran PS+DFS at this budget once, not per-
experiment. SIEVE should use the 12-node cluster to **evaluate
fitness in parallel**, not to multiply budget.

Key open questions (flagged inline and collected at the end):

- **Cross-family merging** — can we meaningfully merge a Llama-3-8B
  fine-tune with a Qwen-2.5-7B fine-tune? Tokenizer/embedding
  mismatch is the dominant failure mode; mergekit-tokensurgeon is
  the only documented path, and its quality-loss on 7B is unquantified
  in the published literature. **Needs A/B.**
- **Saturation beyond 6 experts** — Mergenetic (Minetto et al. 2025,
  arXiv:2505.11427) reports merge quality plateaus past ~6 source
  models. SIEVE's population is 32–128; only a small subset can be
  merge-participants per generation. **Selection strategy needs
  design.**
- **Fitness estimators under evaluation cost** — Mergenetic emphasizes
  "lightweight fitness estimators to reduce evaluation costs." At 7B
  on a 12-node cluster, full-benchmark eval per candidate is
  infeasible. Subset / proxy-metric choice is an open design problem.
- **Multi-objective fitness with Canadian-policy benchmarks** — we
  don't have a Canadian-policy "MGSM" yet. The benchmark design
  problem is upstream of the merging method. **Open until Lever's
  benchmark suite is specified.**
- **DFS search on Llama-3 / Qwen-3 architectures** — Sakana DFS was
  validated on Mistral-7B (homogeneous 32-layer blocks). Llama-3's
  GQA, Qwen-3's thinking tokens, and Gemma's logit softcap introduce
  architecture-specific interactions with layer permutation that
  the Sakana paper does not address. **Unknown portability; flag
  as open.**

## Scope

- Regime of interest: merging 7B–13B open-weight base models
  (Llama-3, Qwen-2.5/3, Gemma-2, Mistral, Phi-3) that have each been
  separately fine-tuned on Lever's Canadian-policy + polling corpus
  (or are already community fine-tunes of relevant capability —
  e.g. policy-adjacent: instruct, long-context, math, multilingual
  FR-EN, JSON/structured-output variants).
- Hardware: 12× GB10 (ARM Blackwell SBC, 128 GB unified memory, ~130
  TFLOPS FP16) on CX-7 200G RDMA fabric. Per-node: single 7B fits
  comfortably in bf16; 13B fits in 4/8-bit.
- Fitness: **multi-objective benchmark suite** — Canadian-policy QA,
  polling-aggregate prediction, structured-output fidelity, honesty/
  hedging under out-of-distribution queries, French-English
  bilingual coverage. Exact metrics TBD and out of scope here.
- Crossover = **merging** (not parent blending). Mutation =
  fine-tuning noise / DARE dropout / layer-slice perturbation /
  merge-coefficient perturbation.
- Out of scope: from-scratch pretraining (covered by sibling review),
  RLHF, distillation of the merged ensemble, fine-tuning recipe
  design.

---

## §1.1 Evolutionary Model Merging (Sakana — Akiba et al. 2024, arXiv:2403.13187)

**Published**: arXiv 2024-03; Nature Machine Intelligence
2025-01. Authors: Takuya Akiba, Makoto Shing, Yujin Tang, Qi Sun,
David Ha. Code: [SakanaAI/evolutionary-model-merge](https://github.com/SakanaAI/evolutionary-model-merge).

**One-line takeaway**: CMA-ES over merge coefficients (parameter
space) + CMA-ES over layer-permutations-with-scaling (data flow
space) produces state-of-the-art 7B Japanese LLMs and 10B
Japanese VLMs without any additional gradient training — the
merging recipe IS the learned artifact.

### §1.1.1 Two search spaces: PS vs DFS

The paper's central claim is that model merging has two orthogonal
search axes that have historically been conflated:

**Parameter space (PS)** — "which weights from which source get
mixed, and how." In the Sakana setup the primitive is **DARE-TIES**
(TIES-Merging with DARE dropout applied first, as proposed in the
DARE paper), parameterized per-layer-per-model by two coefficients:
a **density** (what fraction of the delta params to keep) and a
**weight** (how much this model contributes to the merge).
Optimization variable is therefore a vector of length `2 × N × M`
where N = number of merged layers, M = number of source models.

In the simplified variant the paper actually runs for its 3-model
math-Japanese merge, "each source model is treated as a singular
layer" — collapsing N to 1 — giving a compact **6-dimensional**
search (2 coefficients × 3 models). This is what makes PS cheap.

**Data flow space (DFS)** — "in what order do activations flow
through the layers of the source models, and how do we compensate
for distribution shift at the seams." The primitive here is an
**indicator array ℐ** of length T = M · r, where M is the total
layer count across all source models and r is the number of
"repetitions" of the source pool (Sakana uses M = 64, r = 3 → T =
192). Each entry ℐᵢ is binary-gated: if ℐᵢ > 0, the layer at source
index (i mod M) is included at position i in the inference path.
This encodes a **permutation with allowed repeats and gaps** over
the source layer set.

Because naively threading a layer from model A into a position
where model B's representation was expected introduces distribution
shift, DFS also evolves a **scaling matrix W of shape M × M**: when
an activation produced by source-index i flows into source-index j,
it is multiplied by Wᵢⱼ. W absorbs basis mismatch. (In the abstract
terms of the Krogh-Vedelsby decomposition, W is a learned "linear
compatibility" between the residual streams of the two source
models.)

Total DFS search dimensionality: T = 192 indicator entries + M² =
4096 scaling entries = **~4288 params**. Three orders of magnitude
larger than PS. This is why PS converges in ~1000 trials and DFS
needs 128 × 100 = 12,800.

### §1.1.2 CMA-ES setup

Per the paper:

- **PS optimizer**: CMA-ES via Optuna. Population size `λ = 4 + ⌊3
  ln(n)⌋` (the CMA-ES default heuristic), which for n=6 gives λ≈9.
  1000 total trials. Initial mean = 0.5 for every coefficient (each
  coefficient is in [0,1]), initial σ = 1/6. Selection criterion:
  training-set accuracy (not val).
- **DFS optimizer**: CMA-ES via EvoJAX at default hyperparameters.
  Population size 128, 100 generations → 12,800 evaluations. Same
  initial σ regime.
- **Wallclock**: the paper does not publish wallclock explicitly,
  but the Sakana blog reports "hours to days on a single node" for
  PS and "order-of-week on a multi-GPU node" for DFS. No FLOP
  breakdown.

One-line takeaway: **CMA-ES is the default evolutionary optimizer
for any moderately low-dimensional continuous merge problem; the
dimensionality gap between PS and DFS is what dictates wallclock.**

### §1.1.3 Fitness function

**Japanese-math LLM (EvoLLM-JP)**: fitness = accuracy on a held-out
subset of MGSM-JA (Multilingual Grade School Math, Japanese split;
Shi et al. 2022, arXiv:2210.03057) drawn from the full 1069-item
MGSM pool. Single-objective.

**Japanese VLM (EvoVLM-JP)**: fitness = ROUGE-L on JA-VG-VQA-500
(Japanese Visual Genome VQA subset) plus JA-VLM-Bench-In-the-Wild
score. Presented single-objective per run; not a weighted composite.

**Multi-objective**: the paper claims multi-objective handling via
running separate single-objective CMA-ES instances per objective
and selecting per target, rather than a Pareto-frontier optimizer
(no NSGA-II). This is a **gap for SIEVE**: Lever's benchmark suite
is intrinsically multi-objective (policy QA + polling accuracy +
bilingual coverage + honesty), and running per-objective CMA-ES
loses the ability to discover Pareto-frontier candidates. Flag.

**Normalization**: the paper reports raw benchmark scores per
dimension, no cross-objective normalization shown. The decision on
whether this is defensible depends on whether the benchmarks share
scale — MGSM and JA-VLM-Bench are both 0–100 accuracies, which
sidesteps the problem. SIEVE's benchmark suite likely won't, so **a
normalization scheme (z-score vs rank vs min-max) will need
calibration.**

### §1.1.4 Merge primitives

Sakana uses **DARE-TIES** as the primitive inside PS and
**identity with scaling** (no primitive, just concatenation + W)
inside DFS. The choice of DARE-TIES is not deeply defended in the
paper; the community consensus circa late 2023 when Sakana ran the
experiments was that DARE-TIES was the best multi-model primitive
(community leaderboard evidence, not controlled ablations). The
paper does not run an ablation of DARE-TIES vs plain TIES vs plain
Task Arithmetic as the primitive; this is a **missed comparison**
that would be worth ~2 days of compute on the GB10 cluster.

One-line takeaway: **DARE-TIES is the default PS primitive by
community convention, not by controlled comparison within the
Sakana paper. Re-running PS on plain TIES vs DARE-TIES is low-
hanging empirical work.**

### §1.1.5 Hybrid PS + DFS

Hybrid setup:

1. Run PS on source models {A, B, C} → get PS-merged model M_PS.
2. Add M_PS to the source pool: {A, B, C, M_PS}.
3. Run DFS over the extended pool.

Claimed gain: **MGSM-JA 52.0% (PS only, 7B) → 55.2% (PS+DFS, 10B).**
The JP-LMEH average actually *degrades* from 70.5 → 66.2 in the
hybrid, which the paper interprets as "DFS specializes harder on
the PS target objective." The 10B param count comes from the DFS
graph including the PS-merged model + extra layers from a Japanese
base.

The hybrid finding is **mixed evidence**: +3.2 points on target,
-4.3 points on adjacent capabilities. This is a canonical specialization
tradeoff, and the paper is not critical about it. For SIEVE, if
Canadian-policy fitness is the only selection criterion, expect the
same dynamic — bilingual coverage or math reasoning will drift.

One-line takeaway: **Hybrid PS+DFS is +3 on target, -4 on adjacent;
budget dominated by DFS. Not obviously worth it unless target
objective is worth the collateral.**

### §1.1.6 Results and critique

Reported headline numbers:

- **EvoLLM-JP PS-only (7B)**: MGSM-JA 52.0 (source models: Shisa-
  Gamma 9.6, WizardMath 18.4, Abel 30.0; 70B baselines: Japanese
  StableLM 30.0, Qwen-72B 36.4). **+22 points over best 7B source**,
  **+16 over best 70B Japanese baseline**. JP-LMEH average 70.5.
- **EvoLLM-JP PS+DFS (10B)**: MGSM-JA 55.2. JP-LMEH 66.2.
- **EvoVLM-JP (7B)**: JA-VG-VQA-500 ROUGE-L 19.7 (vs LLaVA-1.6
  Mistral-7B baseline 14.3); JA-VLM-Bench-In-the-Wild 51.2 (vs
  baselines 40.5–41.1).

**Critical reading** — what comparisons are with-vs-without fine-
tuning budget held equal?

- **None of the source models were further fine-tuned by Sakana.**
  The paper's pitch is "no training compute on the merge itself" —
  true in the narrow sense that CMA-ES only does forward passes,
  but the underlying source models embed *many GPU-years* of
  fine-tuning by their original authors (WizardMath, Shisa-Gamma,
  Abel). The gain is therefore partly **meta-learned by the
  ecosystem**, not purely emergent from merging.
- **The "7B beats 70B" comparison is framed favorably.** The 70B
  baselines (Japanese StableLM, Qwen-72B) are general-purpose
  models evaluated in zero-shot on MGSM-JA. The 7B EvoLLM-JP
  combines a Japanese-specialist (Shisa-Gamma) and two math-
  specialists (WizardMath, Abel) — it is a specialist merge. A
  fairer comparison would be "fine-tune a 70B model on MGSM-JA for
  equivalent compute," which the paper does not do.
- **No held-out cross-validation.** Fitness is MGSM-JA training
  accuracy; the reported score is on the *same benchmark*. MGSM-JA
  total pool is 1069 examples; the paper does not clearly describe
  a train/test split on the benchmark. **This is a val-set meta-
  overfitting risk equivalent to the one flagged in the small-
  corpus review §8.2.** The generalization claim relies on the
  separate JP-LMEH average, which does show lift.
- **No ensemble comparison.** A trivial baseline — logit-averaging
  the three source models — is not reported. Given the Krogh-
  Vedelsby bound (ensemble ≥ avg member), merging 3 specialists and
  getting +22 over the best single member is not shocking; the
  question is whether merging beats logit averaging. **Open
  question.** (Merging wins on inference cost — 1× vs 3× — but not
  necessarily on fitness.)

### §1.1.7 Open questions the Sakana paper leaves

For each, our best current answer + the evidence gap.

**Q1: Does merging generalize across model families (different
tokenizers, different architectures)?** The paper uses only
Mistral-7B-v0.1-derived sources — same base, same tokenizer, same
architecture. Cross-family is uncharted.
*Best answer*: probably requires mergekit-tokensurgeon or
embedding-matrix alignment first; expected quality loss is large.
*Gap*: no published 7B-scale cross-family merge with controlled
evaluation.

**Q2: Does PS+DFS scale beyond 6 source models?** Sakana uses 3–4.
Mergenetic (2025, arXiv:2505.11427) reports saturation past ~6.
*Best answer*: saturation is probably because the search space
grows as O(2M·N) for PS and O((M·r)^T · M²) for DFS, and CMA-ES
with fixed budget can't cover it.
*Gap*: no paper runs PS at >6 sources with scaled evolutionary
budget; nobody has tested whether saturation is algorithmic or
fundamental.

**Q3: Is DFS portable to architectures with GQA / MoE / thinking
tokens?** Sakana tests on Mistral-7B (dense, MHA, 32 uniform
layers). Llama-3 uses GQA, Qwen-3 has explicit thinking tokens,
Gemma-2 has logit softcaps.
*Best answer*: GQA breaks the clean per-layer abstraction because
KV projections have different shapes; MoE merging is its own
subfield (not DARE-TIES-compatible); thinking tokens affect the
input distribution seen by scaling matrix W.
*Gap*: no published DFS on these architectures. **High priority
for SIEVE** — probably the first real open-research contribution
this track could make.

**Q4: What is the CMA-ES budget scaling law?** Paper uses
ad-hoc budgets (1000 PS trials, 12,800 DFS evaluations). No scaling
law is reported.
*Best answer*: for CMA-ES, wall-budget scales roughly as O(n²) in
dimensionality (covariance adaptation). PS at n=6 is cheap; DFS at
n=4288 needs the 2 orders of magnitude Sakana uses.
*Gap*: no paper empirically verifies CMA-ES budget-vs-quality curves
for model merging. Worth a sweep (e.g. 100, 500, 1000, 5000 trials
on PS) to know the elbow.

**Q5: How much does the fitness set leak into the merge?** MGSM-JA
has 1069 items; Sakana's trial budget is 1000 (PS) or 12,800 (DFS).
*Best answer*: at 12,800 evaluations on a 1069-item benchmark,
each item is effectively queried ~12× across the search. The
returned W and ℐ are certainly fitted to the specific items.
*Gap*: no leave-one-out or cross-validation reported. Held-out
generalization was shown on JP-LMEH (different benchmark), which is
re-assuring, but does not rule out benchmark-specific overfitting
on MGSM-JA itself.

**Q6: Can PS and DFS be jointly optimized rather than sequentially?**
Sakana only does PS-then-DFS.
*Best answer*: joint optimization would be n ≈ 4294 dimensional (PS
6 + DFS 4288) — dominated by DFS. Sequential likely close to
optimum.
*Gap*: no test; hard to justify the compute.

---

## §1.2 Merging primitives — decision tree for crossover

Each primitive gets: (a) mechanism, (b) failure modes, (c) key
hyperparameters, (d) best-use regime, (e) compute cost per 7B
merge. Compute costs are for a single merge operation on a GB10-
class node (128GB unified memory, one merge process, no parallel);
cited as "GPU-seconds" when a GPU is involved and "CPU-minutes"
when the method is CPU-dominated. All costs are for mergekit
implementations as of 2026-04 unless noted otherwise.

### §1.2.1 Task Arithmetic / Task Vectors (Ilharco et al. 2022, arXiv:2212.04089)

**Mechanism**. Define a task vector τᵢ = θᵢ_ft − θ_pre (fine-tuned
weights minus pre-trained weights). Merge by addition with a
scaling coefficient: θ_merged = θ_pre + λ · Σᵢ τᵢ. Subtraction
("negation") removes a capability; analogical compositions
(τ_A − τ_B + τ_C) are claimed but brittle in practice.

**Failure modes**. (1) Sign conflicts: different tasks push the
same weight in opposite directions; naïve summation cancels both.
TIES fixes this. (2) Magnitude drift: task vectors from different
training-run lengths have different norms; one task dominates. (3)
Out-of-tangent-space behavior: task-arithmetic assumptions hold best
in the linear (NTK/tangent) regime of fine-tuning; strong fine-tunes
(many epochs, high LR) drift enough that the linearity breaks (Ortiz-
Jiménez et al. 2023, arXiv:2305.12827).

**Hyperparameters**. One coefficient λ per task vector (or a single
global λ). Community default: λ ≈ 0.3–0.5 (Ilharco finds average
peak near 0.3); grid-search on validation is the standard.

**Best-use regime**. Small number (2–4) of fine-tunes from shared
base; task-level specialization; when you want surgical
capability-editing including *negation* (unlearn a task).

**Compute cost**. ~30 CPU-seconds for a 7B merge (mergekit `linear`
or `task_arithmetic`); I/O bound. No GPU required.

One-line takeaway: **Task arithmetic is the minimum-viable merge —
it's what every other method builds on and every other method
beats.**

### §1.2.2 TIES-Merging (Yadav et al. 2023, arXiv:2306.01708)

**Mechanism**. Three steps applied to each task vector before
summation:

1. **Trim**: keep top-k% by magnitude, zero out the rest.
2. **Elect**: for each parameter, pick the sign (+/−) that has the
   larger total magnitude across tasks.
3. **Merge**: sum only the kept, sign-aligned values; divide by the
   count of non-zero contributors (to avoid over-weighting dense
   regions).

**Failure modes**. (1) Density k too low → loses capabilities (at
k=0.1, 7B TIES merges routinely lose instruct formatting). (2) k
too high → sign conflicts resurface. (3) Equal-magnitude opposing
signs: the elected sign is arbitrary; small perturbations can flip
it. (4) Scaling imbalance between tasks (from different fine-tune
regimes): one task's kept parameters dominate.

**Hyperparameters**. `density` (k, fraction kept; community default
0.5), `weight` per model (community default 1.0 each, normalized),
`normalize` (true recommended).

**Best-use regime**. 3–8 fine-tunes from shared base with partially
overlapping capabilities. The community-leaderboard default for
multi-model merges.

**Compute cost**. ~60 CPU-seconds for a 7B merge (mergekit
`ties`); additional memory for the magnitude-sort but still fits
on 128GB unified memory. No GPU required.

One-line takeaway: **TIES is the conflict-resolution step that
makes task arithmetic actually work at N>2.**

### §1.2.3 DARE (Yu et al. 2023, arXiv:2311.03099)

**Mechanism**. Randomly drop delta parameters at rate p, then
rescale surviving deltas by 1/(1−p). Preserves expectation of the
merged output. Composable with other primitives: `dare_linear` = DARE
+ task arithmetic; `dare_ties` = DARE + TIES.

**Failure modes**. (1) High drop rate (p > 0.9) with low fine-tuning
magnitude → rescaling amplifies noise. DARE's own claims go up to
p=0.99 but the failure regime isn't characterized. (2) Variance:
different random drop masks give different merges; not all are
equivalent. (3) Interaction with TIES: DARE-then-TIES is the usual
composition, but the inverse (TIES-then-DARE) is not studied.

**Hyperparameters**. `density` = 1 − p (fraction kept; community
default 0.53, higher than TIES because DARE is less destructive);
`weight` per model (same as TIES).

**Best-use regime**. Same as TIES, but when you want to reduce
interference between many (N ≥ 4) highly-similar fine-tunes. The
"magnitude-cleaning" interpretation.

**Compute cost**. ~75 CPU-seconds for a 7B merge (mergekit
`dare_ties`); marginally above TIES due to the random mask.

One-line takeaway: **DARE is a regularizer on top of task
arithmetic / TIES; use `dare_ties` as the default multi-model
primitive.**

### §1.2.4 Model Soups (Wortsman et al. 2022, arXiv:2203.05482)

**Mechanism**. Given N fine-tunes from a shared init, take a
weighted average. Three variants:

- **Uniform soup**: θ = (1/N) Σ θᵢ.
- **Greedy soup**: sort by val accuracy; add each to the running
  average only if it improves val. The robust variant; recommended.
- **Learned soup**: optimize the per-model coefficients on val;
  approaches linear merge with learned coefficients.

**Failure modes**. (1) Uniform soup is dragged down by any bad
ingredient. (2) Greedy soup still requires the ingredients to share
a low-error basin; independent random inits break this (Fort 2019).
(3) On 7B bases: when fine-tunes drift far from the base (high LR,
many epochs), soups degenerate — each fine-tune is in a separate
post-fine-tune basin.

**Hyperparameters**. None for uniform. Greedy: just val ranking.
Learned: coefficient vector dimensionality N.

**Best-use regime**. N identical runs differing only in
hyperparameters (seed, LR, warmup). In SIEVE-openweight: the
output of a population-based fine-tune on Lever corpus where
children share init with parents.

**Compute cost**. ~10 CPU-seconds for a 7B soup (just weight
averaging); essentially I/O.

One-line takeaway: **Soup early in a lineage, merge late — soups
are the zero-cost thing you should do whenever siblings share an
init.**

### §1.2.5 SLERP (spherical linear interpolation)

**Mechanism**. Given two weight tensors A and B, compute the angle
Ω between them; interpolate along the great-circle arc: slerp(A,
B, t) = [sin((1−t)Ω) · A + sin(tΩ) · B] / sin(Ω). As Ω → 0, degrades
to linear. Originally from Shoemake 1985 (computer graphics).

**Failure modes**. (1) Only **two** models. Hierarchical stacking
(A+B → SLERP; (A+B)+C → SLERP) is possible but the order matters
and isn't commutative. (2) Can amplify high-frequency weight
components relative to linear interpolation. (3) No mechanism for
sign conflicts; equivalent to task arithmetic in that failure mode.

**Hyperparameters**. Single `t ∈ [0, 1]`; can be specified
per-layer-type (self_attn, mlp, embed) in mergekit for finer
control.

**Best-use regime**. Two fine-tunes from same base, quick crossover
in an evolutionary loop. Widely used in the Open LLM Leaderboard
community.

**Compute cost**. ~20 CPU-seconds for 7B SLERP (mergekit `slerp`);
dominated by trig ops on weight tensors.

One-line takeaway: **SLERP is the safe, single-hyperparameter
two-parent crossover; for N>2 hierarchically compose or switch to
TIES.**

### §1.2.6 Passthrough / frankenmerging / depth scaling

**Mechanism**. Rather than blending weights, *concatenate* layers.
mergekit `passthrough` takes `layer_range: [start, end]` slices
from each source and stacks them into a longer transformer. E.g.
"layers 0–24 from A + layers 8–32 from B" = 48-layer model with
8-layer overlap from B grafted onto A's bottom. Inspired by
SOLAR-10.7B's depth-upscaling recipe (Goddard et al. 2024,
arXiv:2403.13257 / SOLAR paper arXiv:2312.15166).

**Failure modes**. (1) Seam shock: the activation distribution at
the A→B boundary is generally mismatched; without fine-tuning,
downstream layers see out-of-distribution inputs. (2) No theoretical
grounding for which layer ranges work. (3) Parameter count grows
without additional capability in many cases (empirically, on the
Leaderboard, ~60% of frankenmerges scored below their best source).
(4) Cannot be used with different base-model architectures (layer
shapes must match).

**Hyperparameters**. Layer-range slices per source; number of
passes. No continuous tunables that CMA-ES would operate on
naturally — discrete optimization territory.

**Best-use regime**. **Experimental only.** Unless followed by
fine-tuning to heal the seams (SOLAR does this for ~3B tokens of
healing), passthrough results are highly variable. Depth-scaling
from a single base (e.g. "double the middle layers") is more
predictable than cross-model franken.

**Compute cost**. ~30 CPU-seconds for the merge itself; but the
**real** cost is seam-healing fine-tuning, which for a 10.7B model
was ~1000 GPU-hours (SOLAR). On our 12-node GB10 cluster: ~80
GPU-hours per franken, which dominates everything else.

One-line takeaway: **Frankenmerge is not really a merge — it's an
architecture mutation that requires seam-healing fine-tuning to be
useful. Treat as a separate evolutionary operator with its own
compute budget.**

### §1.2.7 Linear merge with learned coefficients (RegMean; Jin et al. 2023, arXiv:2212.09849)

**Mechanism**. RegMean formulates merging as a closed-form least-
squares problem: given per-model inner-product matrices (XᵢᵀXᵢ)
computed from a small unlabeled dev set, solve for the merge
weights that minimize per-model prediction drift:

  θ_merged = (Σᵢ XᵢᵀXᵢ)⁻¹ (Σᵢ XᵢᵀXᵢ θᵢ)

Related: **Fisher-weighted averaging** (Matena & Raffel 2022,
arXiv:2111.09832) uses the Fisher information matrix instead of
inner-products as the weighting; more principled but more
expensive.

**Failure modes**. (1) Requires access to an unlabeled dev set on
the data distribution you care about (SIEVE has this — Lever
corpus). (2) Inner-product matrices can be huge for 7B (layer-wise
sizes up to 4096²); needs out-of-core computation. (3) Assumes
fine-tunes are close enough that linearization is valid (same
tangent-space assumption as task arithmetic).

**Hyperparameters**. Regularization strength for the inverse
(Tikhonov); mergekit exposes this. Closed-form solve — no search.

**Best-use regime**. When you have a target distribution to merge
*onto* (as opposed to averaging across general capability). For
SIEVE: **compute an unlabeled representation of Lever's Canadian-
policy corpus once; solve RegMean across N fine-tunes per
generation.** Zero-evolution fallback path for the openweight
track.

**Compute cost**. ~5–10 GPU-minutes on a 7B merge with a 1K-example
dev set (forward pass to compute XᵢᵀXᵢ dominates); much higher than
TIES but still << fine-tuning. One-shot per merge — no search.

One-line takeaway: **RegMean is the closed-form answer to "given a
target distribution, what's the optimal linear merge?" Use it as
a baseline and as a fast fallback when CMA-ES is too expensive.**

### §1.2.8 Pointer back: Evolutionary Model Merge (Sakana) as the meta-method

All the primitives above can be wrapped as the inner step of an
evolutionary outer loop (§1.1). The **meta-method**:

- PS: evolutionary search over the hyperparameters of one primitive
  (density, weight, λ, t, drop rate) per-layer-per-model.
- DFS: evolutionary search over the composition graph (which layers
  connect to which).

The Sakana paper uses DARE-TIES as the inner primitive for PS.
Nothing prevents replacing it with SLERP (continuous t per layer is
natural), TIES (density + weight), or RegMean (λ regularization).
This is the **first experimental axis** SIEVE should sweep on the
openweight track: **which inner primitive does CMA-ES most benefit
from wrapping?** No published answer.

One-line takeaway: **Evolutionary merge is a wrapper, not a
primitive; the inner choice (DARE-TIES by default) is a
hyperparameter of the meta-method and an open empirical question.**

### §1.2.9 Decision tree — which merge method to reach for first

Given two (or more) candidate parents A and B in SIEVE's
population, the orchestrator selects a crossover recipe by the
following branching:

```
┌─ Same base init? ──────────────────────────────────────────────┐
│                                                                 │
│  YES                                          NO                │
│   │                                            │                │
│   │                                            │                │
│   ▼                                            ▼                │
│  Same tokenizer?                         Same tokenizer?        │
│   │                                            │                │
│   ├── YES (homogeneous family)            ├── YES (rare —       │
│   │   │                                   │    e.g. Llama-2      │
│   │   │                                   │    re-tokenized to   │
│   │   │                                   │    Llama-3)          │
│   │   │                                   │                      │
│   │   ▼                                   ▼                      │
│   │  Only 2 parents?                    mergekit-tokensurgeon    │
│   │   │                                 to align; then treat as  │
│   │   ├── YES → SLERP (t=0.5, tune      "homogeneous" branch.    │
│   │   │        per-layer if needed)     Expect quality loss.     │
│   │   │        [§1.2.5]                 OPEN QUESTION — A/B.     │
│   │   │                                                          │
│   │   └── NO (N ≥ 3)                                             │
│   │       │                                                      │
│   │       ├── Fitness signal available?                          │
│   │       │   │                                                  │
│   │       │   ├── YES  → CMA-ES over DARE-TIES per-layer        │
│   │       │   │         (Sakana PS; §1.1.1)                      │
│   │       │   │         Budget: ~1000 evals                      │
│   │       │   │                                                  │
│   │       │   └── NO   → DARE-TIES fixed (density 0.53,         │
│   │       │             weights normalized) [§1.2.3]             │
│   │       │                                                      │
│   │       └── Small unlabeled dev set available? (we do: Lever) │
│   │           │                                                  │
│   │           └── YES → also try RegMean as a baseline           │
│   │                     [§1.2.7]                                 │
│   │                                                              │
│   └── NO (cross-tokenizer but shared base — edge case)           │
│       → Tokenizer surgery (mergekit-tokensurgeon), fall through  │
│                                                                  │
└── NO base-init  ────────────────────────────────────────────────┘
    │
    ├── Homogeneous layer count + shapes? (e.g. both 32-layer,
    │   same hidden dim)
    │   │
    │   ├── YES → DFS layer-permutation (Sakana DFS; §1.1.1)
    │   │         OR passthrough/frankenmerge with planned
    │   │         seam-healing fine-tune [§1.2.6]
    │   │         WARNING: cross-family seam distributions differ
    │   │         more than within-family; expect higher
    │   │         evolutionary budget or lower success rate.
    │   │
    │   └── NO (different depths or widths)
    │       → Distillation (out of scope) or capability-specific
    │         depth-upscaling per family, merge intra-family first.
    │
    └── Different architecture families (Llama vs Qwen vs Gemma)
        → OPEN QUESTION. Literature has no clean answer at 7B
          scale with controlled evaluation. Flag and defer to
          an A/B experiment.

Additional: when all parents share an init AND siblings (children
of a common elite), always soup first (greedy; Wortsman), THEN
propose merges. Free diversity-preserving aggregation.
```

**Defaults where literature supports them** (green light):
- 2 models, same base → **SLERP, t=0.5**.
- 3–6 models, same base, fitness signal → **CMA-ES over DARE-TIES,
  density ∈ [0.4, 0.6], weight ∈ [0.5, 1.5] normalized**, ~1000
  evaluations.
- 3–6 models, same base, no fitness signal → **DARE-TIES, density
  0.53, normalize=true**.
- Siblings with shared init → **Greedy soup**.
- Unlabeled dev set available → **RegMean as a baseline**.

**No clear winner — needs A/B** (red flag):
- Cross-family merging (Llama ↔ Qwen ↔ Gemma ↔ Mistral).
- DFS on non-Mistral architectures (GQA, thinking tokens, logit
  softcap).
- Inner primitive for the evolutionary wrapper (DARE-TIES vs TIES
  vs SLERP vs RegMean).
- Fitness-estimator choice under a 12-node cluster budget
  constraint (subset sampling, proxy metrics, early-stopping).
- Multi-objective fitness aggregation (per-objective CMA-ES vs
  NSGA-II vs weighted composite).

## Open questions (collected)

Carried inline above; repeated here for the log.

1. **Cross-family merging quality-loss.** Tokenizer surgery (mergekit-
   tokensurgeon) is the only published path; no controlled 7B study
   quantifies degradation. **Propose A/B**: Llama-3-8B-instruct ↔
   Qwen-2.5-7B-instruct via tokensurgeon, compare MGSM / MMLU /
   policy-QA post-merge to the stronger source.
2. **Saturation at N ≈ 6 sources.** Mergenetic reports this; nobody
   has tested whether it's algorithmic (CMA-ES budget) or
   fundamental (parameter-space geometry).
3. **DFS portability beyond Mistral-7B.** GQA layer shapes, MoE
   experts, and thinking tokens are not addressed in Sakana.
   Potential SIEVE contribution: first DFS run on Llama-3 and
   Qwen-3.
4. **CMA-ES budget scaling.** No published budget-vs-quality curve
   for model merging. Cheap sweep (100/500/1000/5000 PS trials).
5. **Fitness-set leakage.** At 12,800 DFS evaluations on a 1069-item
   benchmark, items are queried ~12× each. Sakana doesn't report
   leave-one-out generalization. **Required for SIEVE**: rotating
   fitness splits + sequestered meta-val (same as §8.2 in sibling
   review).
6. **Multi-objective merging.** Sakana runs single-objective per
   merge. Lever's fitness is multi-objective by construction;
   NSGA-II vs per-objective CMA-ES vs composite weighting is an
   open design question.
7. **Inner-primitive choice.** Sakana uses DARE-TIES as the PS
   primitive by convention, not by ablation. Whether SLERP / TIES /
   RegMean-wrapped CMA-ES outperform remains untested.
8. **PS vs ensemble-of-sources baseline.** Sakana never reports
   "logit-average the 3 source models" as a baseline. Plausible it
   beats the merge at N=3, loses at N=8 (merging compresses into 1×
   inference, ensemble is N×).
9. **Seam-healing fine-tuning budget for passthrough.** SOLAR used
   ~1000 GPU-hours per 10.7B franken. On GB10: ~80 GPU-hours. Scales
   linearly with param count? Linearly with number of seams? Unstudied.
10. **Canadian-policy benchmark design.** The fitness function is
    upstream of everything here. Until Lever's benchmark suite is
    specified, the merge method choice is speculative.

---

## Bibliography

- Akiba, Shing, Tang, Sun & Ha 2024 — Evolutionary Optimization of
  Model Merging Recipes — arXiv:2403.13187 —
  https://arxiv.org/abs/2403.13187 (also Nature MI
  https://www.nature.com/articles/s42256-024-00975-8)
- Goddard, Siriwardhana, Ehghaghi, Meyers, Karpukhin, Benedict,
  McQuade, Solawetz 2024 — MergeKit: A Toolkit for Merging Large
  Language Models — arXiv:2403.13257 —
  https://arxiv.org/abs/2403.13257
- Ilharco, Ribeiro, Wortsman, Schmidt, Hajishirzi, Farhadi 2022 —
  Editing Models with Task Arithmetic — arXiv:2212.04089 —
  https://arxiv.org/abs/2212.04089
- Jin, Ren, Preoțiuc-Pietro, Cheng 2023 (presented 2022) — Dataless
  Knowledge Fusion by Merging Weights of Language Models (RegMean)
  — arXiv:2212.09849 — https://arxiv.org/abs/2212.09849
- Kim, Kim, Kim, Kim, Hwang, Lee, Kim, Kim, Kim, Kim 2023 — SOLAR
  10.7B: Scaling Large Language Models with Simple yet Effective
  Depth Up-Scaling — arXiv:2312.15166 —
  https://arxiv.org/abs/2312.15166
- Matena & Raffel 2022 — Merging Models with Fisher-Weighted
  Averaging — arXiv:2111.09832 — https://arxiv.org/abs/2111.09832
- Minetto, Mancino, Grazzi, Coppini, Conti, Frontera, Pellegrini,
  Lucas 2025 — Mergenetic: a Simple Evolutionary Model Merging
  Library — arXiv:2505.11427 — https://arxiv.org/abs/2505.11427
- Ortiz-Jiménez, Favero, Frossard 2023 — Task Arithmetic in the
  Tangent Space: Improved Editing of Pre-Trained Models —
  arXiv:2305.12827 — https://arxiv.org/abs/2305.12827
- Shi, Suzgun, Freitag, Wang, Srivats, Vosoughi, Chung, Tay, Ruder,
  Zhou, Das, Wei 2022 — Language Models are Multilingual Chain-of-
  Thought Reasoners (MGSM) — arXiv:2210.03057 —
  https://arxiv.org/abs/2210.03057
- Shoemake 1985 — Animating Rotation with Quaternion Curves (SLERP)
  — SIGGRAPH 1985 (no arXiv; original computer-graphics source).
- Wang et al. 2025 — GENOME / GENOME+ — arXiv:2503.01155 —
  https://arxiv.org/abs/2503.01155 (cited in sibling review; repeated
  here as evolutionary LLM framework)
- Wortsman, Ilharco, Gadre, Roelofs, Gontijo-Lopes, Morcos,
  Namkoong, Farhadi, Carmon, Kornblith, Schmidt 2022 — Model Soups:
  averaging weights of multiple fine-tuned models improves accuracy
  without increasing inference time — arXiv:2203.05482 —
  https://arxiv.org/abs/2203.05482 (ICML)
- Yadav, Tam, Choshen, Raffel, Bansal 2023 — TIES-Merging: Resolving
  Interference When Merging Models — arXiv:2306.01708 —
  https://arxiv.org/abs/2306.01708
- Yu, Yu, Zhang, Meng, Xu, Zhang, Cao, Huang, Mao 2023 — Language
  Models are Super Mario: Absorbing Abilities from Homologous Models
  as a Free Lunch (DARE) — arXiv:2311.03099 —
  https://arxiv.org/abs/2311.03099
