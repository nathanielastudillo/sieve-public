# Literature — fine-tuning-as-mutation and ensemble distillation (openweight track)

Compiled 2026-04-18. Sibling note to
[literature-small-corpus-ensembles.md](literature-small-corpus-ensembles.md)
and [literature-tokenizers.md](literature-tokenizers.md). Those two cover
SIEVE's from-scratch micro-LM track. This one covers the new **openweight
track**: fine-tuning open-weight 7B–13B bases (Llama-3.1, Qwen-2.5,
Gemma-2, Mistral-NeMo) on Lever's Canadian-policy + polling corpus
(50–200M tokens), where **mutation = a fine-tune run** and the final
step is **distillation of the surviving ensemble into a single
deployable model**.

Scope of this review:

- §1.5 — the fine-tuning methods that are candidate mutation operators,
  with 2024–2026 rank/quality numbers.
- §1.6 — distillation strategies for converting the converged ensemble
  into one deployable model.

Out of scope (already covered elsewhere): small-corpus pretraining
dynamics; tokenizer design; diversity metrics; ensemble-level fitness
vs individual-level fitness; greedy ensemble selection (Caruana 2004 —
still the right inference-time aggregator here).

## TL;DR

For the mutation operator: **LoRA with per-member rank ∈ {8, 16, 32, 64}
and per-member learning rate ∈ [1e-5, 5e-4] is the right default**.
LoRA adapters are the evolutionary unit. DoRA is a drop-in upgrade at
matched rank (+3–5 points on commonsense, same VRAM) and should be
tried in a second generation. QLoRA is only worth it if we can't fit
full LoRA on a GB10 (we can — 7B LoRA fits in ~28 GB; 13B LoRA fits in
~56 GB across two nodes). PiSSA and LoRA+ are mature enough to include
as perturbation points in the search space; AdaLoRA's extra machinery
isn't justified at our scale. **Full fine-tuning is not a mutation
operator** in SIEVE — it's ~8× the GPU-hours of LoRA at comparable
quality for domain adaptation, and the produced weights can't be
cheaply merged or evolved. Keep it as an occasional "elite re-seed"
operation, not the default. **LoraHub-style evolutionary composition
of adapter weights** (Huang et al. 2024, arXiv:2307.13269) is the
closest literature analogue to what SIEVE is doing, and its
gradient-free coefficient search becomes our crossover operator.
Preference optimization (DPO/IPO/KTO) is **not** a mutation operator
for a corpus task — skip it unless we pivot to producing stylistically
distinct members.

For distillation: **on-policy distillation** (student generates, teacher
scores, per-token KL supervision) is the current state of the art, and
**ensemble-as-teacher** distillation gets a real but modest gain over
single best teacher when teachers disagree (the "dark knowledge"
regime). When the ensemble has converged to consensus, distillation
caps at single-teacher performance. The right distillation target is
**7B → 7B** (compressing ensemble behaviour into one member-sized
model at inference cost 1/K) or **13B → 7B** if we find that 13B bases
were necessary. DistillKit (Arcee, 2024) + on-policy tweaks from
Thinking Machines Lab (2026) is the implementation path. **Do not
distill** if deployment stays on the 12-node GB10 cluster: logit-
averaging K members at inference is ~free on that hardware and retains
all ensemble gain.

## §1.5 Fine-tuning methods for the mutation step

### 1.5.1 Full fine-tuning — when does it dominate LoRA?

**Dettmers, Pagnoni, Holtzman & Zettlemoyer 2023** —
*QLoRA: Efficient Finetuning of Quantized LLMs*
(arXiv:[2305.14314](https://arxiv.org/abs/2305.14314)). Report that
LoRA at rank 64 on all linear layers **matches** full fine-tuning on
instruction-following benchmarks for Llama-7B, reaching 99.3% of
ChatGPT's Vicuna score.
*Takeaway: LoRA-on-all-linear-layers closes the gap with full FT in
the post-2023 recipe. The old "LoRA only on qkv" defaults are obsolete.*

The 2024–2026 follow-ups (e.g. *Learning Rate Matters: Vanilla LoRA
May Suffice for LLM Fine-tuning*, arXiv:2602.04998) confirm this —
the residual gap to full FT usually turns out to be a tuning gap, not
a capacity gap. Where full FT still clearly wins:

1. **Distribution shift** larger than a domain corpus — e.g. pretraining
   a base from scratch on a new language (Saul-141B, Portuguese LLaMAs).
   Our corpus is English Canadian policy text; shift is small.
2. **Very small adapter ranks** (r ≤ 4) on downstream tasks that need
   lots of new knowledge — domain-heavy QA with unfamiliar named
   entities. Fix: just use higher rank.
3. **Training signal changes the conditional structure heavily** —
   task-format changes (completion → chat) that LoRA can't cheaply
   reshape without affecting identity. Not our case; Lever's task is
   domain-conditional continuation.

**Memory (LoRA r=64 on all linear, FP16 weights + BF16 optim in
bitsandbytes 8-bit):**

| Base | Full FT (FP16 w + AdamW, zero3) | LoRA r=64 all linear | QLoRA r=64 NF4 |
|------|----------------------------------|----------------------|-----------------|
| 7B   | ~100–120 GB peak (needs H100/B200 or FSDP) | ~28 GB (single GB10) | ~9–12 GB (single GB10) |
| 13B  | ~200–240 GB (multi-GPU) | ~56 GB (2× GB10 FSDP) | ~18–22 GB (single GB10) |

At 128 GB unified memory per GB10, **7B LoRA fits with headroom, 13B
LoRA needs FSDP across 2 nodes, 7B full FT is possible with FSDP
across 4–6 nodes but brutal**. The cluster has 12 GB10 nodes; a full-FT
mutation burns half the cluster for one child. A LoRA mutation burns
one node for one child. The 12-to-1 throughput ratio is the decisive
argument: **mutation = LoRA**, not full FT, for population economics.

*Decision for SIEVE: LoRA (or DoRA) is the mutation operator. Full FT
is an occasional operation — once per ~10 generations on the single
best ensemble member, as a re-seed against adapter rank saturation.*

### 1.5.2 LoRA rank selection — 2024+ ablations

**Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Chen 2021** — *LoRA:
Low-Rank Adaptation of Large Language Models*
(arXiv:[2106.09685](https://arxiv.org/abs/2106.09685)). Original paper;
recommended r ∈ {1, 2, 4, 8} on GPT-3 175B attention-only. The
low-rank-suffices hypothesis is preserved for narrow adaptations.
*Takeaway: The paper's own numbers do not justify rank defaults for
modern 7B–13B domain adaptation — wrong model scale, wrong task type.*

**Biderman, Portes, Ortiz, Paul, Greengard et al. 2024** — *LoRA
Learns Less and Forgets Less*
(arXiv:[2405.09673](https://arxiv.org/abs/2405.09673)). Head-to-head
LoRA vs full FT on Llama-2-7B across code and math domains.
*Takeaway: LoRA underperforms full FT on continued-pretraining
workloads (code, ~20B tokens) but is competitive on instruction
tuning; it also forgets less of the base distribution. For our
~100M-token domain corpus, this puts us in the "instruction-tuning
competitive" regime, not the "continued pretraining, LoRA lags"
regime.*

**Zhang, Wang, Roberts, Hooker & Raffel 2024+** and the **NLP/R 2024**
community rank sweeps converge on:

- **r = 8**: sufficient for pure-style domain adaptation (output
  formatting, terminology).
- **r = 16–32**: sweet spot for semantic domain adaptation on 7B–13B
  with 10M–200M domain tokens. Default in most modern recipes
  (HuggingFace Alignment Handbook, Axolotl defaults).
- **r = 64+**: closes to within 0.5–1 point of full FT on most
  benchmarks; diminishing returns past r = 128.
- **Target modules**: q_proj, k_proj, v_proj, o_proj, gate_proj,
  up_proj, down_proj (all linear). MLP projections matter more than
  attention for knowledge, attention more than MLP for style.

Trainable parameters at r=32, all-linear, Llama-3-8B: ~84 M (≈1% of
base). At r=64: ~168 M (≈2%). Still cheap.

**Shuttleworth, Hou et al. 2024** — *LoRA vs Full Fine-tuning: An
Illusion of Equivalence*
(arXiv:[2410.21228](https://arxiv.org/abs/2410.21228)). LoRA and full
FT produce different spectral perturbations even at matched quality.
*Takeaway: Two LoRA children with different ranks will live in
different regions of weight space → useful ensemble diversity source.
Rank variation is **itself** a mutation dimension, not just a hyper.*

*For SIEVE's search space:*

```python
"lora_rank":    [8, 16, 32, 64],
"lora_alpha":   lambda r: 2*r,  # standard "alpha = 2r" heuristic (stable)
"target_modules": "all-linear", # fixed; modules is not a productive mutation axis
```

### 1.5.3 DoRA — weight-decomposed LoRA

**Liu, Wang, Chen, Chen, Molchanov & Cheng 2024** — *DoRA:
Weight-Decomposed Low-Rank Adaptation*
(arXiv:[2402.09353](https://arxiv.org/abs/2402.09353), ICML 2024 Oral).
Decomposes the fine-tuning update into magnitude + direction; applies
LoRA only to direction. Direct comparison on Llama-7B / Llama-2-7B
commonsense reasoning: +3.7 to +4.4 pts over LoRA at r=16; at r=4,
+22.4 pts; at r=8, +37.2 pts (DoRA shines at very low rank). Training
overhead: +~10% wallclock, no extra inference cost after merge.
*Takeaway: DoRA is strictly better than LoRA at every rank tested.
At matched quality, DoRA can halve the rank. Drop-in replacement in
the peft library (`use_dora=True`).*

**QDoRA** (same group, 2024) combines DoRA with 4-bit NF4 weights and
on Orca-Math fine-tuning of Llama-2-7B **beats full FT** — a surprising
result that suggests the DoRA parameterization is well-suited to
domain adaptation, not a pure efficiency hack.

*For SIEVE: include `method ∈ {lora, dora}` as a mutation dimension.
First generation: 50/50. Later generations: evolution will pick.
Expected: DoRA dominates in low-rank members, LoRA dominates in
high-rank members (hypothesis — worth tracking).*

### 1.5.4 QLoRA — 4-bit backbone

**Dettmers, Pagnoni, Holtzman & Zettlemoyer 2023** — *QLoRA*
(arXiv:[2305.14314](https://arxiv.org/abs/2305.14314)). NF4 quantization
of base weights + LoRA on top. Memory reduction: 4× on the frozen
weights → 7B fits in ~12 GB of VRAM. Quality cost on Guanaco/Vicuna:
within 1% of 16-bit LoRA.
*Takeaway: QLoRA is the lever for fitting bigger bases or bigger batches
on a single GB10. Quality regression is small but non-zero. Not free.*

**When QLoRA is still worthwhile in 2026**:

1. **13B bases on a single GB10**: 13B LoRA needs FSDP across 2 nodes
   (56 GB); 13B QLoRA fits on one node (~20 GB). One node per child
   = one child per generation per node, which we need for a 16-member
   population on 12 nodes with some headroom.
2. **Batch-size-limited quality** — larger effective batch from freeing
   VRAM. Can matter for policy corpora where long contexts (4–8K) are
   common.

**When QLoRA is not worthwhile**:

1. When full 16-bit LoRA fits: **always prefer it**. QLoRA's NF4
   quantization introduces 0.5–1 pt regression per PiSSA/DoRA paper
   comparisons; no reason to eat that if VRAM allows.
2. When merging adapters back for deployment — the merged model's base
   is NF4, quality compounds.

*For SIEVE: 7B default = LoRA (no Q). 13B default = QLoRA. Mix them
in the population so evolution can pick if QLoRA-13B > LoRA-7B on
ensemble contribution.*

### 1.5.5 LoRA+ / AdaLoRA / PiSSA — are they worth it?

**Hayou, Ghosh & Yu 2024** — *LoRA+: Efficient Low Rank Adaptation of
Large Models* (arXiv:[2402.12354](https://arxiv.org/abs/2402.12354)).
A and B matrices should have **different learning rates** (B learns
~16× faster than A to account for init asymmetry). +1–2 pts and
~2× faster convergence, zero extra compute, one-line code change.
*Takeaway: Free lunch. Turn on.*

**Zhang, Chen, Bukharin et al. 2023** — *AdaLoRA*
(arXiv:[2303.10512](https://arxiv.org/abs/2303.10512)). Dynamically
reallocates rank budget across layers during training via SVD-based
pruning. Small quality gain (+0.5–1 pt) over fixed-rank LoRA at matched
parameter budget; adds training complexity (rank-schedule, importance
scoring, periodic budget pruning).
*Takeaway: Real gain but implementation overhead is non-trivial, and
the extra machinery interacts badly with adapter-space evolution
(different AdaLoRA children end up with different rank allocations
across layers, blocking LoraHub-style composition). Skip for SIEVE.*

**Meng, Wang, Zhang 2024** — *PiSSA: Principal Singular values and
Singular vectors Adaptation*
(arXiv:[2404.02948](https://arxiv.org/abs/2404.02948), NeurIPS 2024).
Initializes LoRA's A,B from the top-r SVD components of the **base
weights**, rather than random. Trains those principal components;
freezes the "noise" residual. Gemma-7B on GSM8K: 77.7% (PiSSA) vs
74.5% (LoRA). Mistral-7B on GSM8K: 72.9% (PiSSA) vs 67.7% (LoRA).
*Takeaway: PiSSA consistently beats LoRA at matched rank by 3–5 pts;
initialization is the whole trick. Implementation cost is a one-time
SVD of each target matrix (~5 minutes on 7B). Include in the mutation
space.*

**Survey caveat**: *Learning Rate Matters* (arXiv:2602.04998, 2026)
argues that many LoRA-variant gains disappear once learning rates are
tuned per-variant. The 10× LR gap between PiSSA's and vanilla LoRA's
optimal LRs is a confound in published comparisons. *For SIEVE, this
is fine — evolution will tune LR per-member anyway.*

*For SIEVE's mutation space:*

```python
"method":     ["lora", "dora", "pissa"],
"lora_plus":  [False, True],    # B-matrix LR multiplier ∈ {1, 16}
"rank":       [8, 16, 32, 64],
"alpha":      lambda r: 2*r,
```

LoRA+, DoRA, PiSSA are all one-line flag changes in the `peft` library
(≥0.12.0). No custom training code. Evolution explores; we don't
pre-commit to a winner.

### 1.5.6 Preference fine-tuning — DPO / IPO / KTO

**Rafailov, Sharma, Mitchell, Manning, Ermon & Finn 2023** —
*Direct Preference Optimization* (arXiv:[2305.18290](https://arxiv.org/abs/2305.18290)).
Eliminates the RLHF reward model by training directly on preference
pairs with a classification-style loss.
*Takeaway: Gold standard for alignment fine-tuning. Not a domain-
adaptation method.*

**Azar, Rowland, Piot, Guo, Calandriello, Valko & Munos 2023** —
*IPO* (arXiv:[2310.12036](https://arxiv.org/abs/2310.12036)). IPO fixes
DPO's overfitting on deterministic preferences by replacing the BT
model with a regularized identity objective.

**Ethayarajh, Xu, Muennighoff, Jurafsky & Kiela 2024** — *KTO*
(arXiv:[2402.01306](https://arxiv.org/abs/2402.01306)). Uses
Kahneman–Tversky prospect theory; accepts unpaired thumbs-up /
thumbs-down signal instead of pairs. Better when preference pairs are
scarce.

*For SIEVE (openweight track on Canadian policy corpus)*: **skip**.
Preference fine-tuning requires preference data, which Lever does not
have in bulk for Canadian policy. Synthesizing preferences (pairwise
scoring by a judge LLM) is possible but the resulting "diversity"
across members is *stylistic*, not *domain-informational* — two DPO
children with different preference data learn to produce different
tones on the same question. That's useful diversity for a chatbot
ensemble but not for a policy-parsing / fact-grounded ensemble.

*Conditional use case*: if Lever later spawns a policy-Q&A product
where response **style** matters (formal vs. plainspoken explanations,
academic vs. journalistic framing), preference fine-tuning with
synthetically generated style preferences becomes a valid diversity
axis — each member tuned toward a different reading style. Revisit at
that point.

### 1.5.7 Continued pretraining vs. instruction fine-tuning for domain adaptation

**Cheng, Huang & Wei 2023** — *Adapting Large Language Models via
Reading Comprehension* / **AdaptLLM**
(arXiv:[2309.09530](https://arxiv.org/abs/2309.09530), ICLR 2024).
Raw continued pretraining (CPT) on domain text injects knowledge but
**destroys prompting ability**. Fix: reformat raw corpus into
reading-comprehension QA pairs, then CPT. Biomedicine, finance, law
domains: matches or beats pure-CPT while retaining downstream
instruction-following.
*Takeaway: For a knowledge-heavy domain like Canadian policy, the
corpus must be preprocessed into (context, question, answer) triples
or (summary, expansion) pairs before fine-tuning. Raw token prediction
on policy PDFs will give a model that "knows" policy but can't be
prompted.*

**Ovadia, Brief, Mishaeli & Elisha 2024** — *Fine-Tuning or Retrieval?
Comparing Knowledge Injection in LLMs*
(arXiv:[2312.05934](https://arxiv.org/abs/2312.05934)). For factual
knowledge injection, continued pretraining + RAG beats fine-tuning +
RAG; for style/format adaptation, fine-tuning wins.
*Takeaway: If Lever's deployment includes RAG over Canadian legislation
(it should), fine-tuning should target **reasoning style and format**
on policy text, not fact memorization. Facts come from retrieval.*

**Fujii, Nakamura, Loem, Iizuka, Hattori, Hirai, Mizuki, Yokota &
Okazaki 2024** — *Continual Pre-Training for Cross-Lingual LLM*
(arXiv:[2404.17790](https://arxiv.org/abs/2404.17790)). On English→
Japanese continued pretraining of Llama-2, 100B tokens of CPT gave
+20 points on Japanese benchmarks.
*Takeaway: CPT scales with domain-corpus size. At 50–200M tokens we
are 500–2000× below this regime. LoRA-based light-touch CPT is right;
heavy-weight CPT is not.*

*For SIEVE*: The corpus should be preprocessed into a mixture of:
(a) raw-chunk next-token prediction (~50%), (b) summarization pairs
(document → summary; ~20%), (c) reading-comprehension QA pairs
(~20%), (d) structured-extraction pairs (document → JSON of entities/
dates/amounts; ~10%). Generate (b)-(d) with a strong teacher (Kimi
K2.5 via OpenRouter, or a self-bootstrapping loop using the strongest
surviving ensemble member).

### 1.5.8 LoraHub and evolving adapters directly — **this is the key section**

**Huang, Liu, Wan, Wei, Li, Xie, Luo, Feng, He & Zhou 2024** —
*LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA
Composition* (arXiv:[2307.13269](https://arxiv.org/abs/2307.13269),
COLM 2024). Given N task-specific LoRA adapters {A_1, …, A_N}, compose
a new adapter Â = Σ w_i · A_i by searching for scalar weights {w_i}
using **gradient-free optimization** (CMA-ES) against a handful of
examples from a target task. No extra training, no gradients through
the base.
*Takeaway: **Adapter weights can be evolved as a small, cheap,
continuous parameter space.** This is crossover for SIEVE.*

**Follow-ups worth knowing about:**

- **Wu, Huang, Xie et al. 2024** — *LoRA-Composition*
  (arXiv:[2404.18065](https://arxiv.org/abs/2404.18065)). Merging
  concept LoRAs for image generation; same compositional principle,
  different modality.
- **Jang, Yun & Hu 2024** — *Model Stock*
  (arXiv:[2403.19522](https://arxiv.org/abs/2403.19522), ECCV 2024
  oral). Only **two** fine-tuned models are enough to find near-
  optimal layer-wise merge coefficients by exploiting a geometric
  fact: the optimal merge is near the center of fine-tuned-weights
  space. 87.8% ImageNet top-1 from 2-model averaging; beats Model
  Soup's many-model averaging. *Takeaway: merge coefficients are
  low-dimensional and smooth — evolution will find them fast.*
- **Akiba, Shing, Tang, Sun & Ha 2024** — *Evolutionary Optimization
  of Model Merging Recipes*
  (arXiv:[2403.13187](https://arxiv.org/abs/2403.13187), Nature MI).
  CMA-ES over both parameter-space merge coefficients and dataflow
  (layer-routing) space. Produces SOTA Japanese LLMs with zero extra
  training. **Direct validation** that evolutionary search in weight-
  merge space works for LLMs at 7B scale.

**Why this is load-bearing for SIEVE's openweight track:**

The classical SIEVE loop on from-scratch small models had
**mutation = retrain from perturbed hypers** (minutes) and
**crossover = ill-defined** (different inits → different basins →
weight averaging unsafe; Fort 2019).

For openweight LoRA fine-tuning, both collapse to something much
cheaper:

| Operator | Classical SIEVE | LoraHub-style SIEVE |
|----------|-----------------|---------------------|
| Mutation | Retrain (minutes–hours per child) | Retrain LoRA OR perturb coefficients (free) |
| Crossover | Not well-defined | **Weighted sum of parent LoRA weights** (free) |
| Fitness  | Ensemble val loss | Ensemble val loss — same |

Because parent LoRAs all sit on the **same frozen base**, they live in
the same basin; Wortsman-soup-style merging is **safe by construction**.
The search space for crossover is just {w_1, …, w_K} ∈ ℝ^K, which is
directly addressable by CMA-ES / DE / simple perturbation. Per-
generation compute collapses: mutation budget ~80% on LoRA training
runs, ~20% on near-free crossover-and-evaluate cycles.

*For SIEVE: build the crossover operator on top of peft's
`add_weighted_adapter` API (available in peft ≥0.8). One line:*

```python
# peft >= 0.8
model.add_weighted_adapter(
    adapters=["parent_A", "parent_B", "parent_C"],
    weights=[w_A, w_B, w_C],     # found by CMA-ES on held-out val
    adapter_name="child",
    combination_type="linear",   # or "ties", "dare_linear", "svd"
)
```

Combination types to explore as mutation sub-operators: **linear**
(simplest, LoraHub default), **ties** (Yadav et al. 2023,
arXiv:2306.01708 — sign-consensus resolution), **dare_linear** (Yu et
al. 2023, arXiv:2311.03099 — delta dropout). Evolution picks.

### 1.5.9 Recommendation — default mutation operator for first cluster run

**Base model**: Llama-3.1-8B (default). Rationale: 8B fits LoRA on a
single GB10 with headroom; 128K native context handles full policy
documents; permissive license; strong English + passable French
(Canadian bilingual corpora); mature in the ecosystem. Alternates for
the second-gen diversity injection: Qwen-2.5-7B (different tokenizer,
different pretraining mix → different base direction in weight space),
Gemma-2-9B (different arch → different inductive biases). Mistral-NeMo-
12B as an elite re-seed.

**Population: 16 members per generation**, one per GB10 node. Generation
0 = 16 independent LoRA fine-tunes with sampled hypers. Later
generations = mix of new fine-tunes (from mutated parents) + LoraHub
crossovers.

**Default mutation config (generation 0 sampling distribution):**

| Parameter | Distribution | Default |
|-----------|--------------|---------|
| Method | Uniform over {LoRA, DoRA, PiSSA} | 1/3 each |
| Rank | Log-uniform over {8, 16, 32, 64} | log-mean 16 |
| Alpha | Deterministic, 2×rank | — |
| Target modules | Fixed: all-linear | — |
| LoRA+ (B-matrix LR mult) | {1, 16} | 16 |
| LR | Log-uniform [1e-5, 5e-4] | 2e-4 |
| Batch size | Fixed 64 (grad-accum to fit) | — |
| Epochs | {1, 2, 3} | 2 |
| Warmup fraction | {0.02, 0.05, 0.1} | 0.05 |
| Schedule | {cosine, linear} | cosine |
| Weight decay | [0.0, 0.1] | 0.0 |
| Dropout (LoRA) | {0.0, 0.05, 0.1} | 0.05 |
| Data sampling | Per-member bootstrap of 70% of 50–200M tokens, stratified by source | — |
| Data mixing ratio | raw 50% / summary 20% / RC 20% / extraction 10% | as listed |
| Quantization | {none, NF4} | none (NF4 only if 13B) |
| Seed | Random per member | — |

**Per-member compute budget**: ~4–8 GB10-hours for a LoRA fine-tune
on 100M tokens at 2 epochs (back-of-envelope: Llama-3.1-8B, seq 4K,
bs 64, ~3000 steps, ~40 steps/min on GB10 SM121 with flash-attn — call
it 75 min, add checkpointing/eval overhead → ~2 hours. Multiply by
2–3 epochs + margin → ~4–6 hours). 16 members × 6 hours = 96 GB10-
hours per generation; 12 nodes in parallel → **8 wallclock hours per
generation**. One generation per cluster-day is comfortable.

**Regularizer of last resort**: LoRA alpha/rank ratio is the
regularization knob — higher alpha = less regularization. Weight decay
on LoRA parameters has near-zero effect in practice (the pretrained
base dominates). Use early stopping on a held-out slice of the corpus
(see §8.2 of the sibling review on val-set meta-overfitting).

**Justifications:**

- **Rank log-uniform {8…64}**: spans the frontier where quality
  plateaus (r=64 ≈ full FT) down to the "very cheap" regime (r=8, fits
  with 5× batch size). Gives evolution a chance to find the
  diversity-cost trade-off.
- **Method ∈ {LoRA, DoRA, PiSSA}**: all one-flag changes; non-trivial
  quality differences; DoRA's low-rank edge means the rank=8 members
  will likely all be DoRA. PiSSA's principal-component init means its
  children will explore a different region of adapter space than LoRA's.
- **LoRA+ on by default**: free 1–2 pt gain, 2× convergence speedup,
  no risk.
- **LR log-uniform [1e-5, 5e-4]**: spans the full reasonable range
  for LoRA on 7B. Defaults cluster around 2e-4 (standard for LoRA on
  Llama-7B from the Axolotl/peft community).
- **Epochs ∈ {1, 2, 3}**: multi-epoch (≥4) on 100M tokens with a 7B
  starts memorizing and doesn't help domain adaptation. Muennighoff et
  al. 2023 (arXiv:[2305.16264](https://arxiv.org/abs/2305.16264))
  argues up to 4 epochs ≈ fresh data on sub-1B-token corpora, but with
  a 7B the memorization onset is faster.
- **Bootstrap 70% per member**: Breiman-bagging gives useful decorrelation
  at this corpus size (100M tokens → ~70M tokens per bootstrap ≫
  Bostrom-Durrett 300-tokens-per-vocab-entry floor, unlike the 10⁵-
  token track). Diversity source; low cost.
- **Data mixing ratio**: per §1.5.7. Include in the mutation space as a
  lower-priority axis after generation 2.

## §1.6 Knowledge distillation of the ensemble

### 1.6.1 Classical logit distillation

**Hinton, Vinyals & Dean 2015** — *Distilling the Knowledge in a
Neural Network* (arXiv:[1503.02531](https://arxiv.org/abs/1503.02531)).
Train the student to match the teacher's softmax-with-temperature
output distribution. The "dark knowledge" is in the non-argmax
probabilities.
*Takeaway: Foundational. Alone, not enough for modern LLMs — the
teacher's output distribution on typical corpus tokens is already
close to one-hot (trained for low perplexity); the dark knowledge is
concentrated at disagreement points, which classical KD doesn't
upweight.*

**Why classical KD is insufficient for LLMs specifically:**

1. **Exposure bias**: Student only sees teacher-generated or corpus
   tokens at training, then generates its own sequence at inference.
   Errors compound auto-regressively. This is the same failure mode
   that on-policy RL fixes.
2. **Distribution mismatch between pretraining corpus and student's own
   generations**: The tokens the student will actually generate at
   inference differ statistically from the training corpus after
   domain adaptation. Teacher signal on corpus tokens is off-policy.
3. **Vocab mismatch** between teacher and student (different
   tokenizers): much of the literature's early ensemble-distillation
   work assumes shared vocabs; across different open-weight families
   we must either use **universal logit distillation** (match on a
   harmonized vocab subset; Boizard et al. 2024
   arXiv:[2402.12030](https://arxiv.org/abs/2402.12030)) or stick to
   single-family ensembles. *Takeaway: if SIEVE's ensemble mixes Llama
   + Qwen + Gemma, vocab alignment becomes a first-order problem.
   Prefer single-family ensembles where possible.*

### 1.6.2 On-policy distillation

**Agarwal, Vieillard, Zhou, Stanczyk, Ramos, Geist & Bachem 2024** —
*On-Policy Distillation of Language Models: Learning from Self-
Generated Mistakes* / **GKD**
(arXiv:[2306.13649](https://arxiv.org/abs/2306.13649), ICLR 2024).
Student generates sequences; teacher scores; KL or JS divergence on
per-token distributions as the training signal. Fixes exposure bias.
Beats off-policy KD by +5–10 points on summarization and translation.
*Takeaway: Current SOTA for LLM distillation. Expensive
(student inference in the training loop) but well worth it.*

**Gu, Dong, Wei, Wang & Huang 2023** — *MiniLLM: On-Policy Distillation
of Large Language Models*
(arXiv:[2306.08543](https://arxiv.org/abs/2306.08543)). Reverse-KL
variant. The reverse direction avoids the student placing probability
mass on low-probability teacher tokens.
*Takeaway: Use reverse KL (student as q, teacher as p in KL(q || p))
for distillation into a much smaller student; forward KL for same-size
student.*

**Thinking Machines Lab 2026** — *On-Policy Distillation* blog post,
corresponding empirical study
(arXiv:[2603.25562](https://arxiv.org/abs/2603.25562) *Revisiting On-
Policy Distillation: Empirical Failure Modes and Simple Fixes*).
Reports failure modes at scale: mode collapse under too-aggressive
on-policy sampling, instability when teacher and student are far apart
in capability. Simple fixes: warm-up with off-policy supervised first,
then switch to on-policy; clip student entropy; mix in pretraining
loss.
*Takeaway: Standard recipe for open-weight distillation has stabilized
by 2026 — off-policy warmup → on-policy with entropy clip → done.*

**Industry evidence (2024–2026):**

- Llama-3.2-1B / 3B distilled from Llama-3.1-8B using on-policy +
  structured-data signals (Meta's Llama-3.2 report, 2024).
- Qwen-3 reports distillation from Qwen-3-235B teacher with "on-
  policy rollout distillation" as the final post-training stage.
- GLM-5 and MiMo-V2-Flash reports explicitly name OPD as post-training.
- PyTorch Torchtune added Llama-3.1-8B → 1B distillation recipes in
  2024 (https://pytorch.org/blog/llama-into-torchtune/).

*Takeaway: On-policy distillation is **the** post-training step for
modern open-weight releases. SIEVE's distillation phase should follow
the same recipe.*

### 1.6.3 Ensemble-as-teacher vs single-best-teacher

**Allen-Zhu & Li 2020** — *Towards Understanding Ensemble, Knowledge
Distillation and Self-Distillation in Deep Learning*
(arXiv:[2012.09816](https://arxiv.org/abs/2012.09816)). Argues (with
empirical backing on vision) that the "dark knowledge" in an ensemble
is richer than in any single member because ensemble members encode
**disjoint** feature subsets. Distilling from the ensemble gives the
student access to all subsets; distilling from the best single
teacher gives it only that teacher's subset.
*Takeaway: Ensemble distillation has a real ceiling advantage over
single-teacher distillation **if and only if the ensemble members are
diverse**. Convergent ensembles lose this gain.*

**Fukuda, Suzuki, Kurata, Thomas, Cui & Ramabhadran 2017** —
*Efficient Knowledge Distillation from an Ensemble of Teachers*
(INTERSPEECH). Classical result: on speech recognition, ensemble-
distilled student approaches ensemble teacher performance; single-best
distillation caps at single-best teacher.
*Takeaway: older result, CV/speech domain, but the pattern holds.*

**Beyer, Zhai, Royer, Markeeva, Anil & Kolesnikov 2021** —
*Knowledge distillation: A good teacher is patient and consistent*
(arXiv:[2106.05237](https://arxiv.org/abs/2106.05237)). Long training
+ consistent augmentation matters more than teacher choice. Applies
to our setting: the distillation run needs to be ~10× longer than
typical fine-tuning runs.

**2024+ LLM-specific ensemble distillation:**

- **Tian, Zhang, Liu et al. 2024** — *Exploring Knowledge Purification
  in Multi-Teacher KD for LLMs*
  (arXiv:[2502.01064](https://arxiv.org/abs/2502.01064)). Multi-teacher
  rationale aggregation and conflict resolution. Non-trivial extra
  machinery; +1–3 pts on BBH over single-teacher distillation.
- **NeurIPS 2024 poster: Multi-Teacher Distillation: Ensemble-Then-
  Distill**. Two-stage pipeline: ensemble-output into a soft label,
  then standard KD. Pragmatic, simple, robust.
- **Vijayakumar et al. 2024** — *B-DISTIL: Boosted Multi-Teacher
  Distillation*. Residual boosting + log-barrier regularization +
  intermediate connections; provable O(1/√T) convergence.

*Takeaway: Ensemble-then-distill works. The gain over distilling from
the single-best ensemble member is positive but modest (roughly 1–3
points on downstream benchmarks); it scales with ensemble diversity
and caps out when the ensemble has converged.*

### 1.6.4 DistillKit and practical toolkits

**Arcee AI 2024** — *DistillKit*
([github.com/arcee-ai/DistillKit](https://github.com/arcee-ai/DistillKit),
blog post). Production-grade open-source LLM distillation toolkit.
Supports logit-based distillation (with a compressed top-k logit
format) and hidden-state distillation. Offline logit caching with
polynomial approximation keeps storage tractable on billions of
training tokens. Benchmarks: +2–6 pts on BBH, MMLU-PRO, MUSR over SFT-
only baselines.
*Takeaway: Use this. No reason to build distillation infra from
scratch in 2026.*

**EasyDistill 2025** (arXiv:[2505.20888](https://arxiv.org/abs/2505.20888))
— broader coverage including reasoning distillation and reward model
distillation.

**torchtune** (PyTorch, 2024) — native distillation recipes with
FSDP support. Llama-3.1-8B → 1B path is reference-implemented.

*For SIEVE: start with DistillKit for logit-based distillation of the
ensemble, extend with on-policy rollout loop (Thinking Machines recipe
— warm up with DistillKit's offline KD, switch to on-policy after ~30%
of steps). Implementation budget: ~2 engineer-weeks.*

### 1.6.5 When distillation helps vs plateaus

Two regimes from the literature:

1. **Dark-knowledge regime**: Ensemble members disagree on non-trivial
   fraction of tokens. Ensemble soft-target carries more signal than
   argmax. Distillation transfers this richer signal → student
   approaches ensemble performance, exceeds best single teacher by
   1–3 points on downstream tasks. SIEVE's diverse-by-construction
   ensemble should live here in early generations.

2. **Plateau regime**: Ensemble has converged to near-consensus on all
   but ambiguous tokens. Soft-target is essentially argmax-of-agreed.
   Distillation caps at single-teacher performance; ensemble gain is
   gone.

**Diagnostic for which regime we're in**: measure inter-member
disagreement on held-out val at time of distillation.

```
disagreement = E_token[ 1 - max_v ( (1/K) Σ_i [argmax p_i = v] ) ]
```

- disagreement > 0.05 → dark-knowledge regime, distillation helps.
- disagreement < 0.01 → plateau regime, distillation only compresses.

Also monitor pairwise KL among members — if median pairwise KL < 0.1
nats/token, the ensemble is redundant.

### 1.6.6 Recommendation — when and how to distill in SIEVE

**When to distill**:

- **Not before generation 20–30.** Early ensembles have high
  disagreement but their members are individually weak; distilling
  would just lock in the weakness. Wait until the ensemble's
  individual members are each >90% of single-best-full-FT quality.
- **Trigger: ensemble val loss plateaus for ≥5 generations AND
  disagreement remains >0.03.** This is the sweet spot: members are
  good, they still disagree, distillation captures the dark knowledge
  at its peak.

**What size student**:

- **If 7B base for ensemble members**: distill 7B-ensemble → 7B
  student. The student gets ensemble quality at 1/K inference cost.
  *This is the default recommendation.*
- **If 13B base**: distill 13B-ensemble → 7B student. Gives a
  smaller model with meaningfully-close quality; literature says 7B-
  from-13B-ensemble recovers 95–98% of ensemble performance.
- **Mobile/edge (1.5B / 3B)**: only if explicitly needed. Expect
  larger gap to ensemble (85–92%). Llama-3.2-1B case study
  (Torchtune) is the reference.

**Is distillation worth it on the 12-node cluster?**

Probably **not**, if deployment stays on the same hardware as training.
Logit-averaging K=8 members at inference costs ~K× FLOPs but ~1×
wallclock when members run in parallel across the cluster. Latency
and throughput both look like one-member-cost. Memory is the only real
cost: K members in VRAM across K nodes. At 8× 7B = 56 GB across 8
GB10s, that's <10% of cluster memory.

Distillation's payoff comes when:

1. We want to deploy *off* the cluster (cloud API, embedded in a
   product, running on a customer's single GPU). Then distilling to a
   single 7B is a large engineering win.
2. We want inference latency to not scale with K as we grow to
   K=32+ members.
3. We want to ship the ensemble's behavior to researchers/collaborators
   as a single model artifact for evaluation simplicity.

**Default recommendation**: run ensemble-as-deployable until a product
requirement forces distillation. When forced, use on-policy
DistillKit-based distillation as described above.

**Distillation recipe (for when we do it):**

1. Data: teacher-generated completions on ~1M prompts sampled from
   Lever's corpus (continuations, summaries, RC Q&A, extractions).
   ~200M tokens.
2. Warm-up: 30% of steps, offline logit KD from ensemble (DistillKit,
   top-K=64 logit compression, KL forward).
3. On-policy: 60% of steps, student generates completions, ensemble
   scores, per-token reverse KL (MiniLLM-style).
4. Cooldown: 10% of steps, SFT on hand-curated high-quality domain
   examples.
5. Student initialization: best single ensemble member's weights, not
   random from base. 2–3× faster convergence.
6. Budget: ~48 GB10-hours (1 node × 2 days) for a 7B student on 200M
   tokens of on-policy distillation.

## Open questions

1. **LoraHub-style crossover at the adapter level vs full-retrain
   mutation: which produces more useful diversity?** Open empirical
   question. The SIEVE cluster run should include both operators and
   measure per-operator contribution to ensemble fitness. Budget:
   one full 30-generation run, ablating the crossover operator.

2. **Does the `{LoRA, DoRA, PiSSA}` method split produce useful
   diversity, or do all three converge to the same effective behavior
   at matched quality?** Untested in the literature. A small study —
   train 4 LoRA + 4 DoRA + 4 PiSSA members on the same corpus with
   matched hypers, measure pairwise disagreement on held-out — would
   settle this in one cluster-day.

3. **Is a heterogeneous ensemble (Llama + Qwen + Gemma) worth the
   vocab-alignment complexity?** Universal Logit Distillation (Boizard
   2024) says yes in principle; no one has empirically verified this
   across four modern bases on a domain corpus. Punt unless ensemble
   plateaus motivate it.

4. **Does domain-adapted LoRA on Llama-3.1-8B forget less than QLoRA on
   Llama-3.1-8B on the Lever corpus?** Biderman et al. 2024 shows LoRA
   < forgetting than full FT; the LoRA-vs-QLoRA forgetting comparison
   is less clear. Forgetting matters for us because the model still
   needs to do general reasoning outside the policy domain at
   inference.

5. **At what rank does the "different LoRA children in different
   weight regions" property break down?** Shuttleworth et al. 2024
   show spectral differences between LoRA and full FT; the equivalent
   between two LoRA runs at r=8 vs r=64 is unstudied. If both ranks
   converge to similar spectral perturbations, rank variation is a
   weak diversity source.

6. **Is preference optimization (DPO/IPO/KTO) ever a productive
   mutation operator for a policy-domain ensemble?** We argued no in
   §1.5.6 but the argument assumes Lever's downstream task is fact-
   grounded. If the actual deployment is more explainer/summarization
   where style matters, the answer flips.

7. **Distillation-from-ensemble of LoRA-merged members: does the dark-
   knowledge gain survive the merge operation?** If the ensemble is
   already N LoRAs on one base, and we merge them via LoraHub into one
   adapter, there's no "ensemble" left — just one merged model. The
   right distillation target is ambiguous: the merged model (cheap) or
   the uncompressed K-member ensemble (expensive but preserves dark
   knowledge). Empirical question.

8. **What's the minimum useful ensemble size for distillation?** Model
   Stock's 2-model result suggests K=2 is enough for *merging*; the
   dark-knowledge literature suggests K>~4 is needed for useful
   distillation soft targets. No direct comparison. Worth a sweep
   over K ∈ {2, 4, 8, 16} on a single cluster run.

9. **On-policy vs off-policy distillation when the student shares
   architecture with the teacher (7B→7B)**: Thinking Machines Lab 2026
   shows OPD wins on student-ask/teacher-answer tasks with big
   capability gaps (8B → 1B). For same-scale distillation, the OPD
   premium may be smaller. Measure.

## Bibliography

- Agarwal, Vieillard, Zhou, Stanczyk, Ramos, Geist & Bachem 2024 —
  GKD: On-Policy Distillation of Language Models —
  https://arxiv.org/abs/2306.13649
- Akiba, Shing, Tang, Sun & Ha 2024 — Evolutionary Optimization of
  Model Merging Recipes — https://arxiv.org/abs/2403.13187
- Allen-Zhu & Li 2020 — Towards Understanding Ensemble, Knowledge
  Distillation and Self-Distillation —
  https://arxiv.org/abs/2012.09816
- Arcee AI 2024 — DistillKit — https://github.com/arcee-ai/DistillKit
- Azar, Rowland, Piot, Guo, Calandriello, Valko & Munos 2023 — IPO —
  https://arxiv.org/abs/2310.12036
- Beyer, Zhai, Royer, Markeeva, Anil & Kolesnikov 2021 — A Good
  Teacher Is Patient And Consistent —
  https://arxiv.org/abs/2106.05237
- Biderman, Portes, Ortiz et al. 2024 — LoRA Learns Less and Forgets
  Less — https://arxiv.org/abs/2405.09673
- Boizard, Haddad, Hudelot & Colombo 2024 — Universal Logit
  Distillation — https://arxiv.org/abs/2402.12030
- Cheng, Huang & Wei 2024 — AdaptLLM (Adapting LLMs via Reading
  Comprehension) — https://arxiv.org/abs/2309.09530
- Dettmers, Pagnoni, Holtzman & Zettlemoyer 2023 — QLoRA —
  https://arxiv.org/abs/2305.14314
- EasyDistill 2025 — https://arxiv.org/abs/2505.20888
- Ethayarajh, Xu, Muennighoff, Jurafsky & Kiela 2024 — KTO —
  https://arxiv.org/abs/2402.01306
- Fujii et al. 2024 — Continual Pre-Training for Cross-Lingual LLM
  Adaptation — https://arxiv.org/abs/2404.17790
- Fukuda, Suzuki, Kurata et al. 2017 — Efficient Knowledge Distillation
  from an Ensemble of Teachers — INTERSPEECH
- Gu, Dong, Wei, Wang & Huang 2023 — MiniLLM —
  https://arxiv.org/abs/2306.08543
- Hayou, Ghosh & Yu 2024 — LoRA+ —
  https://arxiv.org/abs/2402.12354
- Hinton, Vinyals & Dean 2015 — Distilling the Knowledge in a Neural
  Network — https://arxiv.org/abs/1503.02531
- Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Chen 2021 — LoRA —
  https://arxiv.org/abs/2106.09685
- Huang, Liu, Wan et al. 2024 — LoraHub —
  https://arxiv.org/abs/2307.13269
- Jang, Yun & Hu 2024 — Model Stock —
  https://arxiv.org/abs/2403.19522
- Liu, Wang, Chen, Chen, Molchanov & Cheng 2024 — DoRA —
  https://arxiv.org/abs/2402.09353
- Meng, Wang, Zhang 2024 — PiSSA —
  https://arxiv.org/abs/2404.02948
- Muennighoff et al. 2023 — Scaling Data-Constrained LMs —
  https://arxiv.org/abs/2305.16264
- NeurIPS 2024 — Multi-Teacher Distillation: Ensemble-Then-Distill —
  https://neurips.cc/virtual/2024/106613
- Ovadia, Brief, Mishaeli & Elisha 2024 — Fine-Tuning or Retrieval? —
  https://arxiv.org/abs/2312.05934
- Rafailov, Sharma, Mitchell, Manning, Ermon & Finn 2023 — DPO —
  https://arxiv.org/abs/2305.18290
- Shuttleworth, Hou et al. 2024 — LoRA vs Full Fine-tuning: An
  Illusion of Equivalence — https://arxiv.org/abs/2410.21228
- Thinking Machines Lab 2026 — Revisiting On-Policy Distillation —
  https://arxiv.org/abs/2603.25562 +
  https://thinkingmachines.ai/blog/on-policy-distillation/
- Tian, Zhang, Liu et al. 2024 — Multi-Teacher KD for LLMs (Knowledge
  Purification) — https://arxiv.org/abs/2502.01064
- Wortsman et al. 2022 — Model Soups —
  https://arxiv.org/abs/2203.05482
- Wu, Huang, Xie et al. 2024 — LoRA-Composition —
  https://arxiv.org/abs/2404.18065
- Yadav, Tam, Choshen, Raffel & Bansal 2023 — TIES-Merging —
  https://arxiv.org/abs/2306.01708
- Yu, Yu, Ouyang, Xu et al. 2023 — DARE —
  https://arxiv.org/abs/2311.03099
- Zhang, Chen, Bukharin et al. 2023 — AdaLoRA —
  https://arxiv.org/abs/2303.10512
