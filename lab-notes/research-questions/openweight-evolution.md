# Literature — diversity metrics and multi-objective selection for
# fine-tuned open-weight LLM ensembles

Compiled 2026-04-18. Sibling note to
[literature-small-corpus-ensembles.md](literature-small-corpus-ensembles.md).
Scope of that review: from-scratch transformer pretraining on 10⁵-token
corpora, diversity sources for deep ensembles of tiny models. **This
review's scope**: the *openweight track* — fine-tuning pretrained
7B–13B bases (Llama / Qwen / Gemma / Mistral) on Lever's Canadian-
policy + polling corpus, where "diversity" is measured on generated
text and the fitness vector has 5 objectives that conflict.

Cross-refs:
- **§3 of the sibling review** already surveys output-disagreement,
  cosine-on-representations, and classical ensemble-diversity metrics
  for small-model regimes. Not rehashed.
- §1.1–§1.6 of the pretraining-side literature notes (init diversity,
  hyper diversity, soups, checkpoint ensembling) also transfer to
  fine-tuned LLMs with caveats; quoted where relevant.

## TL;DR

For a 16-member population of LoRA-fine-tuned 7B bases, parameter-
space distance is **cheap but unreliable** as a behavioral-diversity
signal — use it as a coarse filter, not as selection pressure. The
signal that *actually* predicts ensemble gain on open-ended generation
is **embedding-space spread of generated responses on a held-out probe
set**, with **MCQ disagreement rate** as a cheap secondary that
captures discriminative-task diversity. Token-level JS divergence only
works within a single tokenizer family (so: Llama-only populations, or
Qwen-only populations — not mixed). Recommend a **composite**:
`0.5 · embedding-spread + 0.3 · MCQ-disagreement + 0.2 · task-vector-
cosine-complement`, evaluated on a fixed 200-prompt probe set per
generation. Per-generation cost on a single GB10: ~15 minutes for 16
members. MAP-Elites-over-LoRA has one published instantiation
(CycleQD, ICLR 2025) — promising but early.

For selection, **NSGA-II is the right default** — 5 objectives is at
the exact boundary where NSGA-III's reference-point machinery starts
paying off, and the user's prior NSGA-II experience tips the tradeoff
toward the simpler algorithm. Do layer in **Hyperband-style progressive
evaluation** as a surrogate: cheap benchmarks (MMLU-subset, 500 samples)
first, full benchmark (lm-eval-harness + Lever custom) only on the
top-25% survivors. Full fitness every 3 generations; surrogate every
generation. At 16 members × 30 generations × 5 objectives, this pulls
wallclock from ~2000 GB10-hours down to ~400.

---

## §1.7 Population diversity metrics for LLM ensembles

### 1.7.1 The measurement problem restated for fine-tuning

The sibling review settled the from-scratch case: members trained from
different seeds occupy different loss-landscape modes (Fort et al. 2019,
[arXiv:1912.02757](https://arxiv.org/abs/1912.02757)) and logit-averaged
ensembles reap Krogh-Vedelsby ambiguity gain.

The fine-tuning regime is different:

1. All members share the **same pretrained initialization**. They're in
   the *same basin* (Fort 2019's premise). Parameter-space distances
   between fine-tuned members are tiny relative to the base model.
2. Members emit **free-form text** during most useful inference. Token-
   level logit averaging only works when members share a tokenizer —
   and rarely produces a coherent sample when members have merged into
   different solution regions (the text is averaged *per token*, but
   the sequence-level trajectories diverge).
3. We care about **task behavior**: does this member handle questions
   about Saskatchewan municipal finance differently from that one? That
   is a property of *generated completions*, not of weight vectors or
   token-level logits.

So our diversity measurement has to operate mostly at the output/text
level, with parameter-space signals serving only as cheap proxies.

### 1.7.2 Parameter-space distance — cheap, low-fidelity

#### Task vectors (Ilharco et al. 2023)

**Editing Models with Task Arithmetic**
[arXiv:2212.04089](https://arxiv.org/abs/2212.04089). Defines the task
vector τ = θ_finetuned − θ_base. Task vectors can be added, subtracted,
and linearly combined meaningfully. Cosine distance between two task
vectors correlates (weakly) with behavioral difference on the
fine-tuning target task.
- *Takeaway*: Task-vector cosine is the **cheapest** diversity signal
  available — computed once per checkpoint, O(params) time. Use it as
  a first-pass coarse filter, not as the primary signal.

#### CKA on activations (Kornblith et al. 2019)

**Similarity of Neural Network Representations Revisited**
[arXiv:1905.00414](https://arxiv.org/abs/1905.00414). Centered Kernel
Alignment measures similarity of internal representations across
networks. Needs a probe dataset and a forward pass per member per
layer — more expensive than task-vector cosine but independent of the
specific output task.

**Critical limitation for our purposes**: 2024–2026 work (Estimating
Neural Representation Alignment,
[arXiv:2502.15104](https://arxiv.org/abs/2502.15104); the TRepLiNa
analysis [arXiv:2510.06249](https://arxiv.org/abs/2510.06249)) shows
**high CKA does not guarantee functional similarity, and low CKA does
not guarantee behavioral divergence** — linear CKA disproportionately
emphasizes high-variance principal directions, so two members with
similar "loud" circuits but different decision-relevant low-variance
circuits will score as similar. For instruction-tuned LLMs where the
relevant behavioral differences often live in small directions of
residual-stream space, CKA understates real behavioral diversity.
- *Takeaway*: CKA is **not trustworthy** as a standalone diversity
  metric for fine-tuned LLMs. Valuable only as a layer-wise diagnostic
  once anomalies show up elsewhere.

#### Weight-space L2 / Frobenius norm on deltas

Even weaker. Two members can have nearly identical delta norms while
making *opposite* predictions on a held-out distribution. Cited in many
model-soup papers (Wortsman 2022
[ICML](https://proceedings.mlr.press/v162/wortsman22a.html)) as a
sanity check, not as primary evidence.
- *Takeaway*: Don't use as selection pressure. Useful only as a
  lineage-tracking / drift-detection indicator.

**Net on parameter-space metrics**: cheap, cheap, cheap — compute
them all per member at nearly zero cost — but treat as coarse coarse-
to-fine filters, not fitness terms.

### 1.7.3 Output-space diversity

#### Disagreement rate on held-out MCQ

For multiple-choice tasks (MMLU, BBH-multiple-choice, HellaSwag, etc.),
pairwise disagreement rate between members is the simplest well-behaved
signal. For N members on M questions: count the fraction of (member-i,
member-j, question-k) triples where predictions disagree. Averaged over
all pairs, this is the *ensemble disagreement rate*.

Classical references: Kuncheva & Whitaker 2003 "Measures of Diversity
in Classifier Ensembles" — 10 diversity metrics analyzed, pairwise
disagreement (the "Q statistic" and simple disagreement rate) is the
most interpretable and correlates with ensemble gain when paired with
majority voting.

Recent LLM uses: LLM-TOPLA (Jiang et al. 2024, EMNLP findings,
[ACL](https://aclanthology.org/2024.findings-emnlp.698.pdf))
explicitly maximizes disagreement diversity for LLM ensembles;
D-BAT (Pagliardini et al. 2023) uses disagreement as a
regularizer at training time.
- *Takeaway*: MCQ disagreement is **cheap, behavior-level, and
  predictive** — but only covers the discriminative part of the
  behavior. For open-ended generation (drafting, summarization, RAG
  synthesis) it is blind.

#### JS / KL divergence on output distributions (token-level)

For members that share a tokenizer, next-token distributions p_i(·|x)
can be compared via JS divergence. Per-position ensemble diversity =
mean pairwise JS. Ortega et al. 2022 [PMLR](https://proceedings.mlr.press/v151/ortega22a/ortega22a.pdf)
(cited in sibling review) uses this in the cross-entropy ambiguity
decomposition: ensemble NLL = avg member NLL − mean KL(q_i ‖ q̄).

Strong for pretraining / next-token perplexity. **Weak for
instruction-following LLMs** because:
1. Population must share tokenizer (Llama-2-7B and Qwen-7B have
   different vocabularies — comparing their softmaxes is meaningless).
2. Identical next-token distributions can produce wildly different
   multi-step completions via amplification of small differences
   (trajectory divergence).
3. High JS at token 1 but identical semantic outcome by end of
   generation = "fake" diversity.
- *Takeaway*: Useful **only within a single tokenizer family** (so:
  all-Llama or all-Qwen populations), and then only as a
  *complementary* signal to embedding-spread. Zero when members don't
  share a tokenizer.

#### ROUGE / BLEU disagreement across generations

Generate responses to a fixed probe set from each member, then compute
pairwise ROUGE-L or BLEU. Low pairwise overlap → high lexical
diversity. Classical in NMT ensemble work (see Fadaee et al. 2017
for prior on BPE + ensembles).

**Known pitfall**: two members that produce semantically identical
answers with different wording score as "diverse" under ROUGE but are
redundant for ensemble voting. Conversely, two members with opposite
claims phrased similarly score as "similar" but are actually the
ensemble's most valuable pair.
- *Takeaway*: ROUGE-disagreement is a **surface-level signal** — cheap
  to compute, misleading on semantic diversity. Useful as a tiebreaker
  when embedding-spread is inconclusive.

#### Embedding-space clustering of generations (recommended)

Generate K responses per member on a fixed probe set. Embed each
response via a frozen high-quality sentence embedder (Sentence-BERT,
all-MiniLM-L6, or a domain-specific embedder). For each probe prompt,
compute the **spread** of its N member-responses in embedding space
(mean pairwise cosine distance, or determinant of the Gram matrix =
"volume"). Average over probe prompts.

Prior art: Text Clustering with LLM Embeddings (Petukhova et al. 2024,
[arXiv:2403.15112](https://arxiv.org/abs/2403.15112)) validates that
sentence embeddings from LLM-sized encoders produce clusters that
match human-judged semantic categories. Ensemble Embedding Approach
for Semantic Caching (2025,
[arXiv:2507.07061](https://arxiv.org/html/2507.07061v1)) applies
embedding-space similarity to ensemble selection.

**Why this is the right primary signal** for our regime:
1. Works across *any* member tokenizer — the embedding space is
   determined by the fixed external embedder, not by the member's
   output format.
2. Captures *semantic* differences (the members saying different things)
   rather than *lexical* ones (different wording for the same thing).
3. Cheap enough at our scale: 200 probe prompts × 16 members × 1
   generation each = 3,200 generations. At 5 tok/s average on a GB10
   for 7B-class models with 256-token outputs, this is ~45
   GB10-minutes per generation (parallelizable across members).
4. Embedding computation itself is cheap (all-MiniLM at ~10k
   embeddings/sec on CPU).

- *Takeaway*: **Embedding-spread of generations on a held-out probe
  set** is the dominant diversity signal for open-ended instruction-
  following LLMs. This is where SIEVE's openweight track should put
  its primary selection pressure.

#### Semantic-preserving "effective" diversity

**Evaluating the Diversity and Quality of LLM Generated Content**
(Kirchenbauer et al. 2024,
[OpenReview](https://openreview.net/forum?id=O7bF6nlSOD)) — framework
for **effective semantic diversity**: diversity *among outputs that
meet quality thresholds*. Raw diversity counts members that produce
garbage as "diverse"; effective diversity gates on a quality filter
first.
- *Takeaway*: When composing the diversity term with the quality term,
  clamp diversity to zero for members below a minimum quality
  threshold. Otherwise the evolution rewards noisy junk-producers.

### 1.7.4 Behavioral diversity via targeted probes

The rising 2024–2025 pattern: instead of generic diversity, probe
*specific capabilities* across members.

**Mixture of Complementary Agents for Robust LLM Ensemble** (NeurIPS
2025, [abstract](https://neurips.cc/virtual/2025/133207)) —
complementarity scored as the *marginal ensemble improvement* from
adding a member, computed on a capability-specific probe suite (math,
code, factuality, long-context, instruction-following). A member with
*lower* individual scores can rank higher in complementarity if its
profile covers a gap.
- *Takeaway*: This is the right abstraction for Lever-specific
  evolution — build a probe suite that *reflects actual operating
  conditions* (policy QA, polling-data interpretation, constraint
  translation for lex) rather than generic benchmarks.

**LLM-Blender** (Jiang et al. 2023 ACL,
[arXiv:2306.02561](https://arxiv.org/abs/2306.02561)) — PairRanker
+ GenFuser. Pairwise comparison determines which candidate output is
better per-example, then a fusion model synthesizes the winners.
- *Takeaway*: LLM-Blender's PairRanker is essentially a *learned
  ensemble aggregator*, orthogonal to the diversity metric problem.
  Worth revisiting once diversity selection is settled; PairRanker's
  model-agnosticism makes it cheap to bolt on at inference time.

**How Diversely Can Language Models Solve Problems?** (Kim et al. 2025,
[arXiv:2503.00691](https://arxiv.org/html/2503.00691v2)) — introduces
an *algorithmic diversity* metric for code generation: members are
scored by whether they produce solutions using structurally different
algorithms (measured via AST clustering and runtime profiling).
- *Takeaway*: For code-adjacent tasks, AST-level diversity is a
  cleaner signal than embedding-space spread. Not directly applicable
  to our policy-analysis use case but a template for domain-specific
  structural diversity.

**Post-training Large Language Models for Diverse High-Quality
Responses** (2026, [arXiv:2509.04784](https://arxiv.org/html/2509.04784))
— explicit diversity objective during fine-tuning that preserves
quality. Confirms what Kirchenbauer 2024 said above: quality-gated
diversity is the right frame.

### 1.7.5 Quality-Diversity (QD) algorithms applied to LLMs

#### MAP-Elites canon

**Mouret & Clune 2015** — *Illuminating search spaces by mapping
elites* [arXiv:1504.04909](https://arxiv.org/abs/1504.04909). Core
idea: define a *behavior descriptor* (low-dim feature vector that
describes what the solution *does*, not how good it is), tessellate
the descriptor space into cells, keep the highest-fitness solution per
cell. Produces a *grid* of diverse-but-competent solutions rather than
a single optimum.
- *Takeaway*: This is the QD canon. The descriptor-space framing is
  philosophically aligned with SIEVE — find complementary solutions
  across a designed "niche" space, not just one best model.

**Novelty Search** (Lehman & Stanley 2008) — precursor. Selects
purely on novelty (distance to archive in behavior space), ignoring
objective fitness. Used as an auxiliary drive in later QD variants.
- *Takeaway*: Pure novelty search is too undirected for our setting;
  MAP-Elites' "keep the best per niche" is the better frame.

#### QD on LLMs — the 2024–2025 state of the art

**CycleQD** (Agent Skill Acquisition for LLMs, ICLR 2025,
[PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/755acd0c7c07180d78959b6d89768207-Paper-Conference.pdf))
applies MAP-Elites to LLM adapters with a *cyclic* adaptation: each
agentic skill's fitness is optimized in isolation while the others
become the behavior descriptors. **Crossover is implemented as model
merging** (adapter-weight interpolation), not random parameter swap.
On agentic benchmarks, CycleQD-produced LLMs match specialized
experts while being broadly competent.
- *Takeaway*: **This is the paper most directly relevant to SIEVE's
  openweight track.** Validates: (a) MAP-Elites can be run over LoRA
  adapters, (b) model-merging crossover works as a QD variation
  operator, (c) QD produces usable ensembles (the grid *is* the
  ensemble). Budget and compute pattern look compatible with our
  cluster.

**Evolving Language Models without Labels** (2025,
[arXiv:2509.15194](https://arxiv.org/html/2509.15194v1)) —
majority-vote selection + QD-style diversity preservation, no labels
needed. Grounds LLM evolution in "variation creates, selection keeps"
biological analogy. Explicitly cites MAP-Elites as preventing
premature convergence.
- *Takeaway*: Supports the thesis that *selection alone is insufficient*
  — even for LLM populations, explicit diversity preservation
  (MAP-Elites-style) is needed to avoid collapse. Reinforces SIEVE's
  existing concern about diversity collapse across generations
  (sibling review §8.3).

**Sakana CycleQD page** ([sakana.ai/cycleqd](https://sakana.ai/cycleqd/))
— "Population-based Model Merging via Quality Diversity". Frames the
whole merging problem as QD. This is a natural extension of Sakana's
earlier evolutionary-merge work
([arXiv:2403.13187](https://arxiv.org/abs/2403.13187)) and uses
NSGA-II under the hood for the actual selection step.
- *Takeaway*: The state-of-the-art practitioners are converging on
  **MAP-Elites-style QD + NSGA-style Pareto selection + model-merging
  operators**. SIEVE's openweight track should adopt this pattern
  directly.

**Multi-Objective Quality-Diversity** (Pierrot et al. 2022,
[arXiv:2202.03057](https://arxiv.org/abs/2202.03057)) — MOME:
each MAP-Elites cell holds a *local Pareto front* rather than a single
elite. Natural fit with our 5-objective fitness.
- *Takeaway*: **MOME is the exact algorithmic primitive we want** if
  we commit to QD — per-niche Pareto front = per-niche NSGA-II. The
  grid gives diversity, NSGA-II gives multi-objective selection within
  each niche. Combines cleanly.

**MOME-PGX** (2023, follow-up to MOME) — policy-gradient version
that scales to higher-dimensional descriptor spaces via RL-style
policy learning.
- *Takeaway*: Overkill for our scale. Plain MOME is right-sized.

**CMA-MAP-Annealing for multi-objective** (2025,
[arXiv:2505.20712](https://arxiv.org/html/2505.20712)) — recent
MO-QD work using CMA-ES as the variation operator. Good reference for
state of the algorithm family but CMA-ES is awkward over LoRA weight
spaces.

### 1.7.6 Composite diversity metric — recommendation for SIEVE openweight

**Proposed composite**:

```
D(population) = 0.5 · D_embed + 0.3 · D_mcq + 0.2 · D_taskvec
```

where:

- **D_embed**: mean pairwise cosine distance of Sentence-BERT
  embeddings of member generations on a fixed 200-prompt policy-domain
  probe set. Quality-gated: zero contribution from members below a
  15% MMLU-policy-subset threshold.
- **D_mcq**: pairwise disagreement rate on a 500-item MCQ probe
  (MMLU-lever-subset + Canadian-policy MCQ constructed from Lever
  corpus).
- **D_taskvec**: 1 − mean pairwise cosine similarity of LoRA task
  vectors (the Ilharco 2023 quantity).

Weights justified:

1. **D_embed = 0.5 (dominant)** — it's the only signal that tracks
   semantic diversity on open-ended generation, which is the actual
   ensemble use case.
2. **D_mcq = 0.3 (secondary)** — directly comparable across
   tokenizers, catches discriminative-task diversity that D_embed
   might miss for short-form questions.
3. **D_taskvec = 0.2 (tiebreaker)** — nearly free, coarse proxy,
   useful as a sanity check. Downweighted because of its known
   unreliability (§1.7.2).

**Redundant / excluded**: token-level JS divergence (collapses to zero
across tokenizers), ROUGE-disagreement (surface-only, swamped by
D_embed), CKA (high cost, untrustworthy signal per 2024–2026 work),
raw weight-norm distance (covered by task-vector cosine).

**Per-generation wallclock on a single GB10 for 16 members**:

| Signal | Cost | Notes |
|--------|------|-------|
| D_taskvec | ~30s | Vector ops on LoRA deltas. Tiny. |
| D_mcq | ~5 min | 500 MCQ × 16 members, batched. Short answers. |
| D_embed generation | ~10 min | 200 prompts × 16 members × 256 tokens at 5–10 tok/s (7B), batched across devices. |
| D_embed embedding + cosine | ~30s | all-MiniLM is fast; matrix ops trivial. |
| **Total** | **~15 min/gen** | Scalable: parallelize across worker nodes, ~1 min/gen on 16-node cluster. |

At 30 generations × 15 min = 7.5 GB10-hours for diversity tracking
alone. Fine.

---

## §1.8 Multi-objective optimization for SIEVE openweight

### 1.8.1 The fitness vector

Five objectives:

1. **f_bench** — aggregated benchmark score from lm-eval-harness (MMLU,
   HellaSwag, ARC, TruthfulQA subsets) + Lever custom (policy-QA,
   polling-interp, constraint-translation). Maximize.
2. **f_div** — composite diversity from §1.7.6, computed *over the
   current population* (so this is a population-level, not individual-
   level, objective — see §1.8.5 on how to handle this). Maximize.
3. **f_latency** — wallclock ms/token at 7B with standard decoding
   (vLLM, bs=1, fp16). Minimize.
4. **f_mem** — peak GPU memory at inference (GB). Minimize.
5. **f_ece** — expected calibration error on MMLU-style probe.
   Minimize.

### 1.8.2 NSGA-II vs. NSGA-III vs. MOEA/D — algorithm landscape

#### NSGA-II (Deb et al. 2002)

[Deb 2002, IEEE TEVC](https://dl.acm.org/doi/10.1109/4235.996017) (no
arXiv — the canonical reference; widely summarized, e.g.,
[arXiv:2407.17687](https://arxiv.org/html/2407.17687v2) discusses
crowding-distance improvements). Fast non-dominated sort + crowding
distance for diversity preservation.
- *Takeaway*: Battle-tested for 2–3 objectives. Crowding distance is
  known-weak for many-objective problems (≥4) because in high-dim
  space nearly every point is non-dominated, and the crowding metric
  degenerates.

#### NSGA-III (Deb & Jain 2013)

[Deb 2013, IEEE TEVC](https://ieeexplore.ieee.org/document/6600851)
(no arXiv for original; modern runtime analyses on
[arXiv:2404.11433](https://arxiv.org/abs/2404.11433),
[arXiv:2505.01256](https://arxiv.org/abs/2505.01256)). Replaces
crowding with **reference points**: pre-specified directions in
objective space that guide the population to fill the Pareto front
uniformly. Dominates NSGA-II for ≥4 objectives on standard benchmarks.

GPU-accelerated tensorized NSGA-III (2025,
[arXiv:2504.06067](https://arxiv.org/html/2504.06067v1)) shows NSGA-III
can scale to large populations on modern hardware.
- *Takeaway*: For 5 objectives, NSGA-III is *theoretically* the better
  choice. But the practical gain is small at population size 16 —
  the reference-point machinery is designed for populations of 100+
  where front-filling uniformity matters. **At N=16, NSGA-II's
  crowding is still workable.**

#### MOEA/D (Zhang & Li 2007)

Decomposes the multi-objective problem into N scalar subproblems,
each with its own weight vector, and solves them jointly with
neighborhood cooperation. Recent benchmark:
[MOEA/D vs. NSGA-II, 2023 ACM TODAES](https://dl.acm.org/doi/full/10.1145/3626096)
— MOEA/D wins on *complicated nonlinear Pareto sets*; NSGA-II wins on
*homogeneous fronts and highly non-linear problems*.
- *Takeaway*: MOEA/D is academically well-regarded, rarely deployed.
  Wins when the Pareto front is known to be complicated (non-convex
  with discontinuities) — we have no reason to expect that in our
  fitness landscape. Skip for our setting.

#### Where the "many-objective" threshold actually sits

Literature consensus (multiple sources, e.g., Deb 2013 original,
recent runtime analyses): **NSGA-II degrades noticeably at ≥4
objectives**; **NSGA-III designed for ≥3 and dominant at ≥5**. Our
5-objective problem is at the tipping point.

**Practical call for SIEVE**: **NSGA-II with crowding distance**.
Reasons:
1. Population size 16 is small enough that NSGA-II's crowding
   degeneracy is mitigated (the degeneracy is a large-population
   phenomenon).
2. User's prior experience with NSGA-II (sluice trading system)
   removes integration / debugging risk.
3. NSGA-III's reference-point benefit only kicks in when you want
   uniform front coverage — with only 16 members, we don't have
   enough solutions to actually cover a 5-D front uniformly anyway.
4. If post-hoc analysis shows poor front coverage, the upgrade path
   to NSGA-III is well-established (just swap the selection step;
   rest of pipeline unchanged).

### 1.8.3 Weighted-sum vs. Pareto — devil's advocate

The user leans Pareto. Playing devil's advocate for weighted-sum:

**For weighted-sum**:
1. Single scalar objective → easier to reason about, easier to stop on
   (no ambiguity about "which front solution is the winner").
2. Easier to integrate with surrogate models (most BO libraries
   default to scalar).
3. If the objectives have *obvious* relative priorities (benchmark
   score matters 3× more than latency, etc.), weighted-sum directly
   encodes them.
4. At selection time, a weighted-sum lets you pick a single winner
   for deployment without needing a second-stage tiebreaker.

**Against weighted-sum**:
1. **Cannot recover non-convex portions of the Pareto front** —
   fundamental limitation. Any solution on a concave portion is
   unreachable by any weighting of a linear scalarization.
2. Weights must be *guessed up front*. Inevitably wrong. Tuning them
   is a meta-optimization problem that Pareto avoids.
3. No single "winner" is a feature not a bug: SIEVE's output is an
   *ensemble*, and the Pareto set directly gives us candidate ensemble
   members varying across the trade-off.

**Verdict**: Pareto wins for this use case. The output *is* an
ensemble — we want the trade-off curve, not a point. User's instinct
is correct. Devil's-advocate concessions: (a) *log* the weighted-sum
score with a default weighting (e.g. 0.4 · bench + 0.2 · div +
0.15 · (1−latency/max) + 0.1 · (1−mem/max) + 0.15 · (1−ece)) as a
secondary scalar for monitoring and stopping criteria, even if
selection is Pareto.

### 1.8.4 Surrogate-assisted MOEA — critical for us

Each full fitness evaluation (lm-eval-harness + Lever custom
benchmarks on a 7B model) is ~30–120 GB10-minutes. At 16 members × 30
generations = 480 evaluations × 60 min avg = 480 GB10-hours minimum.
Without surrogates this is the rate-limiting step.

**Core idea**: train a cheap proxy (subset of benchmark, smaller
model, or learned surrogate) that correlates with full fitness, use
it to pre-screen candidates, spend the full evaluation budget only on
promising ones.

#### Successive Halving / Hyperband (Li et al. 2017)

[JMLR](https://jmlr.org/papers/volume18/16-558/16-558.pdf). Start many
candidates with small budget, halve the survivors and double the
budget, repeat. Non-Bayesian, simple, effective. Recent LLM-specific
extension: **EcoTune** (EMNLP 2025,
[PDF](https://aclanthology.org/2025.emnlp-main.394.pdf)) — token-aware
SH/Hyperband for LLM HPO.
- *Takeaway*: Hyperband is the **right first surrogate** for SIEVE
  openweight. Concrete mapping: initial screen on 500-item MMLU
  subset (~2 min) + 100-item Lever-custom quick probe (~3 min).
  Top-50% graduate to the 2000-item full MMLU + full Lever probe
  (~15 min). Top-25% graduate to the full lm-eval-harness run
  (~60 min). Expected speedup: 4–6×.

#### Surrogate-assisted NSGA variants

**Two-stage dominance-based SAEA** (Nature Sci Reports 2023,
[link](https://www.nature.com/articles/s41598-023-40019-6)) — RBF
surrogate per objective, two-stage screening (dominance-based in
latent space, then real-fitness on survivors).
- *Takeaway*: Right architectural pattern but RBF surrogates are
  weak for LLM-benchmark landscapes. Use as a structural reference,
  not a drop-in implementation.

**Surrogate-assisted evolutionary algorithms: survey** (Springer
2024, [link](https://link.springer.com/article/10.1007/s40747-024-01465-5))
— broad survey. Key classification: *offline* surrogates (trained
once on historical data, used throughout) vs. *online* surrogates
(updated each generation).
- *Takeaway*: **Online surrogate with periodic ground-truth
  recalibration** is the right mode for our setting. Every 3
  generations, do full ground-truth evaluation on current Pareto
  front; use the resulting (config, full-fitness) pairs to refit the
  cheap-benchmark → full-benchmark regression.

#### BoTorch qLogNEHVI — Bayesian multi-objective

**BoTorch** ([docs](https://botorch.org/docs/multi_objective/)):
qLogNoisyExpectedHypervolumeImprovement (qLogNEHVI) is the current
recommended acquisition function for MO-BO — replaces qEHVI with
better numerics
([arXiv:2310.20708](https://arxiv.org/abs/2310.20708)).
- *Takeaway*: Useful for the *screening* step of new candidate
  configs before we commit GPU time. Train a GP on (config-features,
  cheap-proxy-scores) pairs, let qLogNEHVI propose the next batch to
  evaluate. Population size 16 per generation is within the comfort
  zone of GP-based BO. **Use BoTorch qLogNEHVI as the
  candidate-proposal step** inside the NSGA loop; NSGA itself
  remains the top-level selection algorithm.

#### Racing algorithms (F-race / irace)

**F-race / Iterated F-race**
([mlopez-ibanez.github.io/irace](https://mlopez-ibanez.github.io/irace/))
— eliminate inferior candidates as soon as statistical evidence
accumulates (Friedman rank test). Originally for metaheuristic
parameter tuning.
- *Takeaway*: Structurally similar to Hyperband but statistically
  principled. Overkill for our use case *unless* we discover that
  benchmark noise is high relative to inter-candidate differences.
  Flag as a fallback.

### 1.8.5 Quality-Diversity + Multi-Objective — how MAP-Elites fits with NSGA

The tidy picture:

- **MAP-Elites / MOME** (§1.7.5) manages the *descriptor space*:
  partition LLM population into behavior-descriptor cells (e.g.,
  "math-heavy", "long-context-focused", "policy-jargon-native",
  "concise vs. verbose"), keep per-cell Pareto fronts.
- **NSGA-II** (§1.8.2) operates *within* each cell (or across the
  whole grid if you flatten) as the selection algorithm on the
  multi-objective fitness.
- **Surrogate** (§1.8.4) gates which candidates even get full
  fitness evaluation.

This is the architecture used implicitly by CycleQD and explicitly by
MOME. **Recommended concrete stack for SIEVE openweight**:

1. **Behavior descriptors** (2-D): (math-accuracy axis, policy-
   domain-specificity axis) — both measured on small probe sets (~50
   items each) during cheap-proxy evaluation. 8×8 = 64 cell grid,
   initially mostly empty.
2. **Per-cell**: keep 2-member NSGA-II Pareto front on the 5-objective
   fitness.
3. **Variation operators**: LoRA-merge crossover (CycleQD-style) +
   hyperparameter mutation (Gaussian perturbation of lr, alpha, rank,
   epochs, WD).
4. **Selection**: sample parents proportional to cell sparsity
   (favor under-represented regions) × per-cell Pareto rank.

### 1.8.6 Expensive fitness — strategies summary

| Strategy | Speedup | Implementation cost | When to use |
|----------|---------|---------------------|-------------|
| **Hyperband (SH-style)** | 3–6× | Low (standard lib) | Every generation, as first-pass filter |
| **BoTorch qLogNEHVI** | 2–3× (on *proposal*) | Medium (GP training each gen) | Candidate proposal step |
| **Archival / memoization** | Free, variable | Trivial | Always — hash configs, don't re-evaluate duplicates |
| **Progressive evaluation** | 2–4× | Low | Always — cheap benchmarks first, full last |
| **Racing (F-race)** | 2–5× when benchmark noise is high | Medium | Fallback if benchmark noise dominates |
| **Full ground-truth every K gens** | N/A (anchor) | Low | K=3 recommended |

Combine: Hyperband filters → BoTorch proposes → archive dedupes →
full-fitness every 3 gens. Total speedup target: 8–12×. Pulls
480 GB10-hours down to ~50 GB10-hours. Comfortable on 4 GB10s × 12h.

### 1.8.7 Concrete selection-pressure recipe for SIEVE openweight

For a 16-member population with 5-objective fitness:

**Algorithm**: NSGA-II with Pareto selection, with an outer
MOME (Multi-Objective MAP-Elites) layer keeping a 2-D behavior grid.

**Parameters**:
- Population size: 16 (one per cell for 4×4 grid, or 2 per cell for
  an 8×8 grid if compute allows).
- Mutation rate: 0.3 (Gaussian σ = 0.1 on continuous hypers, flip
  prob 0.05 on discrete).
- Crossover: LoRA-merge with random interpolation ratio ∈ [0.3, 0.7]
  (CycleQD-style).
- Tournament size: 2 (standard NSGA-II).
- Immigration fraction: 0.15 (fresh random samples each generation
  to maintain diversity; matches upward-revised sibling review recommendation of 0.25 but lower here because QD grid provides additional diversity pressure).

**Pre-screen / surrogate**:
- Generation 1–2: Hyperband (successive halving) with 3 rungs — 500-
  item MMLU subset → 2000-item MMLU + 100-item Lever → full
  benchmark. Top-25% survive to full eval.
- Generation 3+: Add BoTorch qLogNEHVI for *new candidate proposal*
  (not selection — selection stays NSGA-II). GP trained on (config,
  cheap-proxy) pairs, recalibrated every 3 gens on full-fitness
  evals.

**Full fitness evaluation cadence**:
- Every generation: cheap proxy (MMLU-subset + quick Lever probe,
  ~5 min/candidate).
- Every 3 generations: full lm-eval-harness + full Lever (~60
  min/candidate) on current Pareto front members only.

**Termination**:
- Hypervolume improvement <1% over 5 generations, OR
- 30 generations max, OR
- Compute budget exhausted (set per-cluster).

**Expected budget** (on 4-GB10 subcluster):
- 30 gens × 16 members × 5 min cheap-proxy = 40 GB10-hours (at 4-way
  parallelism: 10 wallclock hours).
- 10 full evals × 60 min × 10 rounds = 100 GB10-hours (25 wallclock
  hours).
- Plus diversity probe (§1.7.6): 30 × 15 min = 7.5 GB10-hours.
- **Total: ~150 GB10-hours = ~38 wallclock hours on 4-GB10
  subcluster**, well inside a long weekend run.

Justification for the recipe:

1. **NSGA-II over NSGA-III**: see §1.8.2 — marginal benefit at N=16,
   user comfort dominates.
2. **MOME wrapper**: §1.7.5 + §1.8.5 — prevents diversity collapse
   which the sibling review (§8.3) already identified as a real
   failure mode in SIEVE exp-001/002.
3. **Hyperband surrogate**: §1.8.4 — the single highest-ROI move.
   Benchmark evaluation is the clear bottleneck.
4. **LoRA-merge crossover**: CycleQD (ICLR 2025) demonstrates this
   actually works, beats random mutation + classical crossover.
5. **3-generation full-eval cadence**: balances surrogate drift
   (refitted regularly) against full-eval cost.

---

## Open questions

Flagged for resolution after first run; no clear winner in the
literature yet.

1. **Does composite D_embed + D_mcq + D_taskvec actually predict
   ensemble gain on Lever's corpus?** Literature supports each piece
   separately; the composite is our construction. After first run,
   validate by regressing ensemble improvement against individual
   diversity terms.

2. **Is MOME with 2-D behavior descriptors enough for LLM
   diversity?** MAP-Elites benefit grows with descriptor
   dimensionality up to a point; past that, cells become sparse. 2-D
   is a guess; 3-D (adding "reasoning-chain-length") might be
   better. Empirical.

3. **Hyperband cutoff fidelity**: what fraction of true top-25%
   candidates does our cheap proxy actually retain? Literature
   suggests 80–90% retention is the usual range; if ours is worse,
   the surrogate is lossy and NSGA-II may drift. Measure after first
   run.

4. **Do CycleQD's LoRA-merge crossover gains transfer to our
   domain?** CycleQD validated on agent-skill benchmarks with Gemma-
   2B base. Our bases are 7–13B and the task is policy analysis —
   close but not identical. A/B against random parameter perturbation
   on gen-1 to validate.

5. **Population size 16 vs. 32 at this scale**: CycleQD used smaller
   populations (4–8); MOEA/D comparisons typically use 100+. Our 16
   is a middle-ground guess. May want to sweep 8 / 16 / 32 in the
   first study.

6. **Calibration (ECE) under LoRA fine-tuning**: recent work
   (Calibration Across Layers, 2025
   [arXiv:2511.00280](https://arxiv.org/html/2511.00280v1); In-Context
   Learning calibration, NAACL 2024
   [ACL](https://aclanthology.org/2024.naacl-long.340.pdf)) shows
   fine-tuning typically *worsens* ECE even while improving accuracy.
   Our fitness includes ECE as a separate objective — what if the
   Pareto front ends up being dominated by unfine-tuned models that
   win on ECE? Set a minimum benchmark-score threshold (hard
   constraint) to avoid "best by doing nothing".

7. **Interaction of QD and multi-objective Pareto**: MOME maintains
   per-cell fronts; in principle the global Pareto front is the
   union across cells. Does this produce a healthy full-grid Pareto
   or fragmented cell-local optima? Open theoretical question (MOME
   paper [arXiv:2202.03057](https://arxiv.org/abs/2202.03057)
   discusses but doesn't fully resolve).

8. **Does tokenizer-family heterogeneity help or hurt?** Mixed-family
   populations (Llama + Qwen + Gemma) forbid token-level JS (§1.7.3)
   but potentially bring genuine representational diversity. Does
   embedding-spread see it? Untested at our scale — one-family vs.
   mixed-family A/B after baseline stabilizes.

9. **Surrogate drift**: online surrogates can drift if the fitness
   landscape of the current population is different from training
   data. Recalibration every 3 gens is a guess; could need to be
   every 1–2 gens. Watch the surrogate-vs-truth residuals.

10. **Can we skip full benchmarks entirely after generation 5?**
    Once the Pareto front has stabilized, cheap-proxy + final
    confirmation on the winner only may be enough. Not tested —
    needs empirical validation that cheap-proxy rank correlation
    with full-benchmark rank stays high throughout evolution.

---

## Bibliography

- Akiba et al. 2024 — Evolutionary Optimization of Model Merging
  Recipes — [arXiv:2403.13187](https://arxiv.org/abs/2403.13187)
- Bischl et al. 2023 — Hyperparameter optimization survey — WIREs
  [Wiley](https://wires.onlinelibrary.wiley.com/doi/full/10.1002/widm.1484)
- Calibration Across Layers 2025 —
  [arXiv:2511.00280](https://arxiv.org/html/2511.00280v1)
- CycleQD / Agent Skill Acquisition for LLMs (ICLR 2025) —
  [PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/755acd0c7c07180d78959b6d89768207-Paper-Conference.pdf)
- Daulton et al. 2023 — qLogNEHVI, numerical fixes to qEHVI —
  [arXiv:2310.20708](https://arxiv.org/abs/2310.20708)
- Deb et al. 2002 — NSGA-II, IEEE TEVC 6(2) — no arXiv; modern
  crowding-distance analysis in
  [arXiv:2407.17687](https://arxiv.org/html/2407.17687v2)
- Deb & Jain 2013 — NSGA-III — IEEE TEVC; runtime analyses in
  [arXiv:2404.11433](https://arxiv.org/abs/2404.11433),
  [arXiv:2505.01256](https://arxiv.org/abs/2505.01256)
- EcoTune 2025 — token-efficient multi-fidelity HPO for LLMs —
  [ACL](https://aclanthology.org/2025.emnlp-main.394.pdf)
- EnsembleEmbedding 2025 — semantic caching via ensemble embeddings —
  [arXiv:2507.07061](https://arxiv.org/html/2507.07061v1)
- Evolving LMs without Labels 2025 —
  [arXiv:2509.15194](https://arxiv.org/html/2509.15194v1)
- Fort, Hu & Lakshminarayanan 2019 — Deep Ensembles Loss Landscape —
  [arXiv:1912.02757](https://arxiv.org/abs/1912.02757)
- GPU-accelerated NSGA-III 2025 —
  [arXiv:2504.06067](https://arxiv.org/html/2504.06067v1)
- Ilharco et al. 2023 — Task Arithmetic —
  [arXiv:2212.04089](https://arxiv.org/abs/2212.04089)
- In-Context Learning Calibration 2024 (NAACL) —
  [ACL](https://aclanthology.org/2024.naacl-long.340.pdf)
- Jiang et al. 2023 — LLM-Blender —
  [arXiv:2306.02561](https://arxiv.org/abs/2306.02561)
- Jiang et al. 2024 — LLM-TOPLA (EMNLP findings) —
  [ACL](https://aclanthology.org/2024.findings-emnlp.698.pdf)
- Kim et al. 2025 — Algorithmic diversity of LLM code generation —
  [arXiv:2503.00691](https://arxiv.org/html/2503.00691v2)
- Kirchenbauer et al. 2024 — Effective Semantic Diversity —
  [OpenReview](https://openreview.net/forum?id=O7bF6nlSOD)
- Kornblith et al. 2019 — CKA revisited —
  [arXiv:1905.00414](https://arxiv.org/abs/1905.00414)
- Kuncheva & Whitaker 2003 — Diversity measures for classifier
  ensembles — Machine Learning 51(2); no arXiv
- Li et al. 2017 — Hyperband —
  [JMLR](https://jmlr.org/papers/volume18/16-558/16-558.pdf)
- López-Ibáñez et al. — irace package —
  [irace site](https://mlopez-ibanez.github.io/irace/)
- Mouret & Clune 2015 — MAP-Elites —
  [arXiv:1504.04909](https://arxiv.org/abs/1504.04909)
- Multi-Objective Quality-Diversity Unstructured 2025 —
  [arXiv:2504.03715](https://arxiv.org/html/2504.03715v1)
- Multi-Objective CMA-MAP-Annealing 2025 —
  [arXiv:2505.20712](https://arxiv.org/html/2505.20712)
- NeurIPS 2025 — Mixture of Complementary Agents —
  [abstract](https://neurips.cc/virtual/2025/133207)
- Ortega, Cabañas & Masegosa 2022 — Diversity and Generalization —
  [PMLR](https://proceedings.mlr.press/v151/ortega22a/ortega22a.pdf)
- Pagliardini et al. 2023 — D-BAT diversity regularizer —
  via [Logit Disagreement 2025, arXiv:2502.15648](https://arxiv.org/abs/2502.15648)
- Petukhova et al. 2024 — Text Clustering with LLM Embeddings —
  [arXiv:2403.15112](https://arxiv.org/abs/2403.15112)
- Pierrot et al. 2022 — Multi-Objective Quality Diversity (MOME) —
  [arXiv:2202.03057](https://arxiv.org/abs/2202.03057)
- Post-training LLMs for Diverse Responses 2026 —
  [arXiv:2509.04784](https://arxiv.org/html/2509.04784)
- Sakana CycleQD — [sakana.ai/cycleqd](https://sakana.ai/cycleqd/)
- SAEA Survey 2024 — [Springer](https://link.springer.com/article/10.1007/s40747-024-01465-5)
- TRepLiNa: CKA+REPINA 2025 —
  [arXiv:2510.06249](https://arxiv.org/pdf/2510.06249)
- Two-stage SAEA 2023 —
  [Nature Sci Reports](https://www.nature.com/articles/s41598-023-40019-6)
- Wortsman et al. 2022 — Model Soups —
  [ICML](https://proceedings.mlr.press/v162/wortsman22a.html)
- Zhang & Li 2007 — MOEA/D — IEEE TEVC 11(6); modern benchmark in
  [ACM TODAES 2023](https://dl.acm.org/doi/full/10.1145/3626096)
