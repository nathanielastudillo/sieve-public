# Literature — training effective ensembles on small corpora

Compiled 2026-04-18. Sibling note to
[literature-tokenizers.md](literature-tokenizers.md) — that review covered
the tokenizer side of SIEVE's failure modes; this one covers the ensemble
side. Tokenizer-specific findings are not repeated here.

## TL;DR

The literature supports the SIEVE direction — ensembles do beat single
models, and diversity is the quantity that matters — but offers only
weak support for SIEVE's most distinctive commitment (evolutionary
pressure on *portfolio composition itself*). Every well-validated
result on "what produces ensemble diversity" points to two sources that
are already in the SIEVE search space: **random initialization** (Fort
et al. 2019) and **hyperparameter variation** (Wortsman et al. 2022,
Wenzel et al. 2020). Evolutionary ensemble *selection* from a library
of independently trained models has ~15 years of results behind it
(Zhou et al. 2002, Caruana et al. 2004); evolutionary *pressure on
hyperparameter configs* has 8 years (PBT; Jaderberg 2017). What SIEVE
adds on top — re-selecting on ensemble-level fitness to drive the
*next generation* of training — is not obviously novel given GENOME+
(2025), which also uses ensemble operators but measures fitness on
individuals; nor has anyone we found empirically validated that the
ensemble-level-fitness twist produces more gain than the standard
"train with diverse hypers, then greedy-select" pipeline.

At our specific regime (10⁵ tokens, 7.4M params, 7.4M-param members),
**the ceiling on ensemble gain is bounded by individual-member quality**
(Krogh-Vedelsby ambiguity decomposition: ensemble error ≥ average
member error − diversity). Since exp-003 showed single-member test NLL
(10.87) *worse than uniform* (9.68), the ensemble is currently running
against a ceiling set by broken tokenization and catastrophic
memorization, not by the ensembling algorithm. Fix the base learners
first. The concrete actions for SIEVE's next run are:

1. Keep the tokenizer changes from literature-tokenizers.md.
2. Swap the evolutionary loop's default aggregation from logit
   averaging to **greedy ensemble selection (Caruana 2004) with
   allow-repeats**, evaluated on val + a sequestered meta-val rotation.
3. **Add weight-averaging branches** where lineage permits (Wortsman
   model soups): children of the same elite should be soupable even if
   models from unrelated lineages cannot.
4. Keep random-init + hyperparameter diversity as the primary diversity
   sources. Do *not* waste compute on bootstrap-bagging at 10⁵ tokens
   (Breiman's 63.2% unique = 53K tokens per member, below every
   quality floor cited in Bostrom & Durrett 2020).
5. Measure ensemble **test** loss (not just val) every generation,
   with a rotating held-out split, to detect val-set meta-overfitting
   before it accumulates over 30-50 gens.

## Scope

- Regime of interest: 10⁵–10⁸ training tokens; transformer LMs
  1M–100M params; ensemble sizes 2–32; population sizes 6–64.
- Compute: 1–12× commodity GPUs (GB10-class), minutes-to-hours per run.
- Out of scope: tokenizer design (see
  [literature-tokenizers.md](literature-tokenizers.md)), curriculum
  learning, RLHF, instruction tuning, mixture-of-experts routing
  within a single model.

## Section 1 — Small-corpus LM pretraining baselines

| Paper | Regime | Key finding relevant to SIEVE |
|-------|--------|-------------------------------|
| Muennighoff et al. 2023 [arXiv](https://arxiv.org/abs/2305.16264) | 100M–900B unique tokens, up to 9B params | Up to **4 epochs of repetition ≈ fresh data**; degrades sharply past ~16 epochs (their RD*≈15). Smallest corpus tested was 100M tokens — 1000× ours. Does **not** study sub-1B-token regimes or ensembles. |
| Geiping & Goldstein 2022 (Cramming) [arXiv](https://arxiv.org/abs/2212.14034) | Single A6000 × 24h, BERT-scale | Tokenizer/architecture choices are dominated by **data curation + optimizer** at <10¹⁹ FLOPs. Recommends against vocab-size tuning in this regime. Directly relevant. |
| Charpentier & Samuel 2024 (BabyLM 2 winner; GPT-BERT) [ACL](https://aclanthology.org/2024.conll-babylm.1/) | 10M and 100M words | Hybrid causal+masked LM architectures won. 10M-word track is the closest public benchmark to SIEVE's 10⁵-token smoke regime. Best submissions varied training objective + architecture, not ensemble. |
| Eldan & Li 2023 (TinyStories) [arXiv](https://arxiv.org/abs/2305.07759) | <10M params, constrained-vocab synthetic corpus | Fluent generation possible at 10M params if vocabulary is curated. Our 7.4M-param target is in range. Confirms the base feasibility; says nothing about ensembling. |
| Gunasekar et al. 2023 (phi-1) [arXiv](https://arxiv.org/abs/2306.11644) | ~1B tokens curated textbook data | Curation quality >> raw quantity. Our corpus has similar density intent; the *volume* (100K) is still 4 OoM below phi-1. |
| Zhou et al. 2023 (LIMA) [arXiv](https://arxiv.org/abs/2305.11206) | 1K examples | 1K curated > 52K uncurated on LLaMA-65B fine-tuning. **But: this is fine-tuning, not pretraining.** Frequently misapplied — the LIMA result does not transfer to from-scratch LM training. |

**Regime transfer caveat.** None of the listed papers tests *from-scratch
transformer pretraining* at 10⁵ tokens. BabyLM's 10M-word track is
closest (100× ours). SIEVE is operating an order of magnitude below any
published baseline for pretraining. Expect findings that assume even
~10M training tokens to decay as we scale down. The "quality over
quantity" (LIMA, phi) story *never* held at from-scratch pretraining —
it's a fine-tuning result. Do not use LIMA to justify the 100K regime.

## Section 2 — Ensemble diversity sources (what actually produces gain)

### 2.1 — The load-bearing result: random initialization beats everything else

**Fort, Hu & Lakshminarayanan 2019** — *Deep Ensembles: A Loss Landscape
Perspective* [arXiv](https://arxiv.org/abs/1912.02757). Load-bearing
claim: functions obtained by *different random initializations* cluster
in *different modes* of the loss landscape; functions obtained by
subspace sampling around a single trained model (MC Dropout, SWAG,
SWA) stay in *one mode*. Different modes produce decorrelated
predictions; same-mode points don't. Implication: **the cheapest way
to get a useful ensemble is to retrain from fresh random seeds**. You
cannot simulate this with cheap subspace tricks around a single run.

At our scale this is the single most important finding. Training a
7.4M-param model on 85K tokens takes minutes on a GB10. We can afford
N=8-32 full independent runs, which is what Fort 2019 says actually
produces diversity. Snapshot/SWA tricks (Section 6) are a fallback for
when you *can't* afford full re-runs — we can.

### 2.2 — Hyperparameter variation on top of init: additive gain

**Wenzel, Snoek, Tran & Jenatton 2020** — *Hyperparameter Ensembles for
Robustness and Uncertainty Quantification*
[arXiv](https://arxiv.org/abs/2006.13570). Hyper-deep ensembles =
different inits × different hyperparameters (stratified). Beats plain
deep ensembles (same hypers, different inits) on in-dist accuracy and
shift robustness. Gains from varying lr, dropout, weight decay are
*complementary* to gains from varying seed. Not tested at sub-1M-param
scale but the compositional structure of the claim is architecture-
independent.

SIEVE already does this — the search space varies dropout, weight decay,
lr, warmup fraction, schedule, epoch count. This is correct. The
evolutionary component is layered on top of an already-good diversity
substrate.

### 2.3 — Model Soups: weight averaging from shared init

**Wortsman et al. 2022** — *Model Soups*
[ICML](https://proceedings.mlr.press/v162/wortsman22a.html). Averaging
*weights* (not predictions) of N fine-tuned models from a *shared
initialization* matches or beats the best single model, at 1× inference
cost. Two flavors:

- **Uniform soup**: mean of all models. Breaks when any ingredient has
  large validation-loss degeneracy (bad lr sweeps drag the soup down).
- **Greedy soup**: sort models by val perf, iteratively add each to the
  running average only if it improves val. On ImageNet CLIP ViT-G/14
  fine-tuning, greedy soup consistently beat best single model.

**Critical constraint**: soups work only within a single "low-error
basin" — i.e., models that share an init and haven't drifted too far.
For SIEVE-from-scratch with independent random inits, models are in
*different* basins (Fort 2019), so weight averaging is **not** safe as
a general aggregation. Soups become available *within a lineage branch*
(parent+children in the evolution) but not across the whole population.

### 2.4 — Data-order diversity (epoch shuffling)

Slowrun (Q Labs 2026) reports "outsized impact" of per-epoch shuffling
on data-constrained training. This is one of SIEVE's existing diversity
sources (`shuffle_per_epoch: True` in the search space). Cross-
referenced by Fort 2019's analysis: data-order variation on a fixed
init stays in the same mode. Therefore data-order alone is a *weak*
diversity source; it works when composed with seed variation, not
alone.

### 2.5 — Architecture variation (NES)

**Zaidi et al. 2020** — *Neural Ensemble Search*
[arXiv](https://arxiv.org/abs/2006.08573). Ensembles of *different
architectures* (varying depth/width/cell choice within a NAS search
space) beat same-architecture deep ensembles on accuracy + calibration
+ shift robustness, *even when individual members are weaker*. Uses
regularized evolution (NES-RE) as the search method.

Relevant: SIEVE's search space already includes architecture
dimensions (n_layer, n_head, n_embd). Zaidi's finding suggests
architecture variation is worth keeping in the evolution even though
it's coarser to tune than continuous hypers.

**Regime caveat**: Zaidi tests CIFAR/ImageNet classification. LM
pretraining at 10⁵ tokens has not been tested in this line.

### 2.6 — What does *not* produce diversity at small scale

- **MC Dropout alone** — same-mode subspace samples; low functional
  diversity (Fort 2019; Ovadia et al. 2019
  [arXiv](https://arxiv.org/abs/1906.02530)).
- **Bootstrap bagging of the training data** — at 100K tokens, each
  bootstrap sample contains ~63.2% unique tokens = ~63K tokens per
  member (Breiman 1996;
  [Wikipedia](https://en.wikipedia.org/wiki/Bootstrap_aggregating)).
  Bostrom & Durrett 2020's floor (>300 tokens per vocab entry) already
  fails at 100K/16K=6.25 before bagging; bagging makes it catastrophic.
  This is *specifically broken at our scale* — do not do it.
- **Adversarial training** (Lakshminarayanan et al. 2017
  [arXiv](https://arxiv.org/abs/1612.01474)) — adds adversarial
  perturbations at train time, improves calibration. Doesn't transfer
  to the pure-text small-corpus LM setting cleanly; skip.

## Section 3 — Aggregation methods compared

| Method | Cost (inference) | Works across different inits? | Known failure modes | Best use case |
|--------|------------------|-------------------------------|---------------------|---------------|
| **Logit averaging** | N× | Yes | N× memory + latency; overconfident members dominate | Default for independent-seed ensembles |
| **Probability averaging** | N× | Yes | Poor calibration on underconfident members; worse NLL than logit avg | Safety-critical apps (Njieutcheu Tassi et al. 2022) |
| **Weight averaging (uniform soup)** | 1× | **No** — only from shared init | Breaks if any ingredient is bad | Fine-tuning from same base; SIEVE-within-lineage |
| **Weight averaging (greedy soup)** | 1× | **No** | Same as uniform but safer; still needs shared init | Sibling models in a lineage branch |
| **SWA / trajectory averaging** | 1× | **No** — single trajectory | Needs LR schedule support; modest gain | Cheap add-on to every training run |
| **SWAG** | K× samples | **No** | Same as SWA + Gaussian variance estimate; calibration only | Uncertainty quantification, not accuracy |
| **FGE** (Garipov 2018) | M× | Partly (same trajectory, cyclical LR) | Needs cyclical LR schedule | Cheaper than full deep ensemble |
| **TIES** (Yadav 2023) | 1× | **Partly** — merges deltas | Needs shared init model; scales poorly past ~8 sources | Merging pre-finetuned LLMs |
| **DARE + TIES** | 1× | **Partly** | As TIES + delta dropout | Merging many finetuned LLMs |
| **EMR-Merging** | 1× | **Partly** | Tuning-free merging (2024); strongest in the merge family per the Systematic Study survey | Alternative to TIES |
| **Evolutionary merge (Sakana 2024)** | 1× | **Partly** | Parameter-space + dataflow-space search; beat hand-crafted recipes | Automating recipe search across pretrained LLMs |
| **Knowledge distillation (ensemble → single)** | 1× | Yes | Requires extra train pass; loses some ensemble gain | Compressing ensemble into a single model for deployment |

### Ranked empirical win rate at **our** regime (10⁵ tokens, from-scratch)

1. **Logit averaging with greedy ensemble selection** — safe default,
   proven at all scales, no shared-init assumption. This is the
   SIEVE baseline.
2. **Weight averaging within a lineage branch only** — when parent-
   child lineage shares init, opportunistic cheap ensembling.
3. **SWA trajectory averaging on top of individual training** — cheap
   add-on per member; tiny gain but free.
4. **TIES/DARE and the broader model-merging family** — not applicable
   to from-scratch-trained models with independent inits; becomes
   applicable only if we pivot to a "pretrained-on-OpenWebText, then
   fine-tune-on-corpus-v00X" recipe (worth considering).

### Load-bearing numbers

- Logit vs probability averaging (Njieutcheu Tassi et al. 2022,
  [PDF](https://ceur-ws.org/Vol-3215/19.pdf)): on MNIST+noise, logit
  averaging drops false-positive confidence from 94.58% to 51.31%
  (closer to correct). Logits reduce poor-classifier influence
  automatically; probabilities amplify it.
- Greedy soup vs best single model (Wortsman 2022): consistent
  win on ImageNet; the degree varies by task but sign is stable.
- Model merging survey (Liu et al. 2026
  [arXiv](https://arxiv.org/html/2511.21437v1)): TIES and DARE+TIES
  beat naive averaging by 1-4 points on multi-task merging. Less
  relevant to our from-scratch setup.

## Section 4 — PBT, evolutionary ensembles, ensemble-level selection

### 4.1 — Population Based Training (Jaderberg et al. 2017)

[arXiv](https://arxiv.org/abs/1711.09846). The canonical reference for
evolutionary-style training. Core loop:

1. Train a population of N workers with different hyperparameter
   configs in parallel.
2. Periodically: evaluate; **exploit** (low performers copy weights
   from top performers); **explore** (perturb hyperparameters).

PBT discovers *hyperparameter schedules*, not fixed settings. Validated
on DeepMind RL, supervised learning (machine translation), and GAN
training. Has become infrastructure (Ray Tune, many derivatives).

**What PBT does not do**: PBT's fitness function is individual-level
(each worker evaluated alone). The "ensemble" emerges implicitly from
the population's final state if you keep all checkpoints, but this is
not what PBT is optimizing.

### 4.2 — GENOME and GENOME+ (2025)

[arXiv](https://arxiv.org/abs/2503.01155). Four operators: crossover
(weight merging of parent models), mutation (random weight
perturbation), selection (fitness-based), succession (transfer
learned state from parent to offspring). GENOME+ adds an **ensemble**
operator that integrates top-K outputs at inference.

Load-bearing numbers (from the paper, 200-sample per-task adaptation,
population=10, max 10 generations, Gemma-2-2b-it foundation):

- +54.8% on DROP vs best single, +25.6% on GSM8K vs best single,
  +24.06% average vs best single across 12 datasets.
- Ablation: removing the ensemble operator costs −11.27% average —
  a large contribution, but **smaller than removing selection
  (−14.68%)**. So selection is more load-bearing than explicit
  ensemble aggregation in GENOME+.
- Crossover removal: −3.65%; mutation removal: −4.72%. Crossover
  (merging weights of parent models) carries less weight than
  expected — consistent with model-merging survey findings that
  merging works well when deltas are small (finetuning regime), less
  well when deltas are large (independent training).
- Scales to population=40.

**Critical caveat for SIEVE**: GENOME+ operates on *pre-fine-tuned*
LLMs (merging pretrained experts), not from-scratch training.
**Fitness is measured per-individual** on 200-sample validation, not
per-ensemble. So GENOME+ does not actually validate the SIEVE thesis
that ensemble-level fitness beats individual-level fitness; it just
shows that evolutionary operators work when composed with an ensemble
output. SIEVE's distinctive move — selecting on ensemble-level fitness
— remains empirically untested in GENOME+.

### 4.3 — Sakana Evolutionary Model Merge (Akiba et al. 2024)

[arXiv](https://arxiv.org/abs/2403.13187),
[Nature MI](https://www.nature.com/articles/s42256-024-00975-8).
Evolutionary search over *model merging recipes* in two spaces:
parameter space (which slices to average) and dataflow space (how to
route activations through merged blocks). Produces state-of-the-art
Japanese LLMs without additional training compute.

Same caveats as GENOME+: merges pretrained experts, not from-scratch.
Validates the *use* of evolution for LLM assembly but not the SIEVE
twist.

### 4.4 — Neural Ensemble Search (Zaidi et al. 2020)

[arXiv](https://arxiv.org/abs/2006.08573). NES-RE uses regularized
evolution *explicitly* to search an architecture space for diverse
ensemble members. **Its fitness function is ensemble-level** (ensemble
val loss, not individual). Closest to SIEVE's thesis in spirit of all
papers surveyed.

Reports: ensembles with varying architectures outperform deep
ensembles even when individual members are weaker. The ensemble-level
fitness is what enables this — selecting weak-but-complementary
members requires the ensemble framing.

**Limitation for SIEVE transfer**: CIFAR-10/100 classification, not
LM pretraining. NES-RE uses 64 members × hundreds of generations.
Compute assumption larger than SIEVE's 30-50 generations × 12-32
members budget.

### 4.5 — Greedy Ensemble Selection (Caruana et al. 2004)

[PDF](https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf).
Given a *library* of trained models (thousands), build an ensemble by
forward-stepwise greedy selection: repeatedly add the model that most
improves ensemble val loss, **allowing re-selection** (a good model
can be added multiple times, effectively weighting it). This is still
the state of the art in AutoML competitions (Purucker et al. 2023,
[arXiv](https://arxiv.org/abs/2307.00286)).

For SIEVE's ensemble-eval step, greedy selection is *strictly dominant*
over brute-force C(N, K) enumeration at any N>10:

- Brute force C(32, 8) = 10.5M combinations; Caruana produces near-
  optimal in O(N·K) model-evals.
- Allow-repeat version gives soft weighting for free.
- Has 22 years of empirical validation.

SIEVE's orchestrator currently brute-forces. **Replace with Caruana
greedy.**

### 4.6 — Many Could Be Better Than All (Zhou, Wu & Tang 2002)

[PDF](https://www.lamda.nju.edu.cn/publication/aij02.pdf). First proof
that *pruning* an ensemble can improve it — not all available models
should be included, even with free compute. GASEN uses a GA to select
which subset of trained ANNs to include. This is the theoretical
backing for SIEVE's "kill dead weight" step.

### 4.7 — Is the SIEVE thesis validated?

Taking stock: of the ten claims the SIEVE CLAUDE.md makes,

| SIEVE claim | Literature status |
|-------------|-------------------|
| Ensembles beat single models | Validated (everyone) |
| Diversity > individual quality | Validated (Zaidi 2020, NES-RE) |
| Random init + hyper variation = diversity | Validated (Fort 2019, Wenzel 2020) |
| Evolutionary selection on trained models | Validated (Zhou 2002, Caruana 2004, Sakana 2024) |
| **Evolutionary pressure on ensemble composition** | *Partly* validated (NES-RE; Caruana's greedy is already this at inference time but not at training time) |
| **Ensemble-level fitness drives population evolution** | **Not empirically tested at our scale**. NES-RE does it at 64-member CIFAR; nobody does it for LM pretraining. |
| Approximates Solomonoff induction | Metaphor only (see §5) |
| Works at 10⁵ tokens | **Open question.** The closest public result is BabyLM's 10M-word track. Nothing tests 100K. |
| 30-50 generations converges | Plausible from NES-RE and PBT scaling, no direct evidence |
| 8-16 ensemble size is right | Consistent with Slowrun's N=8; no formal study in our regime |

Net: SIEVE's component parts are literature-validated; the full
composition (ensemble-level fitness on LM pretraining at 10⁵ tokens)
is empirically novel. This is fine — it is genuinely new research —
but the novelty should be *priced in* when interpreting results. We
should not expect GENOME+-sized gains (+24-55%) because GENOME+ merges
already-strong pretrained experts; our members start weak.

## Section 5 — Theoretical ceiling: how much can ensembles gain?

### 5.1 — Ambiguity decomposition (Krogh & Vedelsby 1994)

For squared loss, on a fixed input x:

```
E_ensemble(x) = (1/N) Σ E_i(x) − (1/N) Σ [f_i(x) − f_bar(x)]²
                ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
                avg member error   ambiguity (diversity)
```

Ensemble error = average member error − diversity. Diversity is
*non-negative*, so ensemble ≥ single-model-avg performance. Extends to
any twice-differentiable convex loss (generalized ambiguity
decomposition).

**Operational implication for SIEVE**: ensemble gain is upper-bounded
by the *average* member loss. If single-member test NLL is 10.87 (exp-
003), the ensemble test NLL cannot be better than 10.87 + diversity
bonus, and the diversity bonus is bounded above by the variance of
member predictions. Our observed ensemble val loss 8.73 is either
(a) possible because val loss is low on members too (val is trivially
memorized, val NLL ≈ 0.05), meaning the ensemble is doing almost no
work on val, or (b) a val-specific calibration artifact. The
10.87-nats test loss hasn't been matched with an ensemble test loss
measurement yet; this is a must-do.

### 5.2 — Cross-entropy ambiguity decomposition

For cross-entropy loss (what we actually measure), the decomposition
is analogous but involves KL rather than variance (see Ortega et al.
2022 [PMLR](https://proceedings.mlr.press/v151/ortega22a/ortega22a.pdf)):

```
H(p, q_bar) = (1/N) Σ H(p, q_i) − (1/N) Σ KL(q_i || q_bar)
```

Same structure: ensemble NLL = avg member NLL − mean KL between each
member and the ensemble. Diversity → cheaper ensemble NLL.

### 5.3 — Information-theoretic ceiling (entropy of test distribution)

The irreducible NLL on our test set is the cross-entropy `H(p_true,
p_true) = H(p_true)` — the entropy of the "true" token distribution
for our domain. We don't know this but we can bound it:

- A trained LM on in-domain data at sufficient scale: GPT-2 small
  reports ~3.0 nats on OpenWebText.
- Our unigram Laplace-smoothed baseline on v003 test: 7.19 nats
  (findings.md, 2026-04-17).
- Ensemble val loss 8.73.

The *gap between 7.19 (zero-param unigram)* and *3.0 (GPT-2 small)*
is ~4 nats. That's the headroom available to ensembling if and only
if individual members actually learn. At present they don't — they
memorize and collapse on test. The theoretical ceiling is wide open;
the practical ceiling is blocked by member quality.

### 5.4 — Solomonoff framing: honest assessment

**Claim in SIEVE CLAUDE.md**: ensembles approximate Solomonoff
induction by maintaining a weighted mixture over programs.

**Literature status**: metaphor with thin empirical grounding.

Recent papers that frame neural nets as Solomonoff approximations:
- Learning Universal Predictors (Grau-Moya et al. 2024
  [arXiv](https://arxiv.org/html/2401.14953v1)) — meta-train on data
  generated by universal Turing machines.
- Transformers As Approximations of Solomonoff Induction
  (2024, [arXiv](https://arxiv.org/html/2408.12065v1)).
- LLMs as Computable Approximations to Solomonoff Induction (2025,
  [arXiv](https://arxiv.org/html/2505.15784v1)).

These are *single-model* approximations, not ensembles. The argument
that "an ensemble of N nets is closer to Solomonoff because Solomonoff
is a weighted mixture" is *structurally correct* — Solomonoff's prior
is 2^(−K(program)) weighted sum — but operationally, an N-model
ensemble with N∈{8,16,32} and random seeds does not come
meaningfully closer to enumerating computable hypotheses than a
single net does. Both are epsilon-away from Solomonoff.

**Recommendation**: treat "Solomonoff framing" as motivation and
poetic backing, not as a quantitative prediction. The literature does
not support specific numerical claims about how fast N-model
ensembles converge to Solomonoff-optimal. If the SIEVE README needs
to keep the framing for PR purposes, that's fine; don't use it to
set priors on experimental outcomes.

## Section 6 — Snapshot / SWA / checkpoint ensembles

### 6.1 — Snapshot Ensembles (Huang et al. 2017)

[arXiv](https://arxiv.org/abs/1704.00109). Cyclical LR schedule
produces M snapshots from a single training run. On CIFAR-100,
DenseNet snapshot ensembles: 17.4% error (vs 17.9% single model) at
the same training compute. So snapshot < full deep ensemble (~16%
error) but >> single model, and *free* in wallclock.

### 6.2 — SWA (Izmailov et al. 2018)

[arXiv](https://arxiv.org/abs/1803.05407). Average weights along SGD
trajectory with cyclic/constant LR. Finds wider minima than SGD, with
better generalization. Built into `torch.optim.swa_utils`. Near-zero
overhead.

### 6.3 — FGE (Garipov et al. 2018)

[arXiv](https://arxiv.org/abs/1802.10026). Discovers that optima are
connected by low-loss curves. Fast Geometric Ensembling traverses
these curves and averages. On CIFAR-10/100, beats Snapshot Ensembles at
equal training compute.

### 6.4 — SWAG (Maddox et al. 2019)

[arXiv](https://arxiv.org/abs/1902.02476). SWA + Gaussian covariance
estimate from SGD iterates. Bayesian model averaging. Best for
calibration/uncertainty; accuracy gains smaller than full deep
ensembles.

### 6.5 — LAWA / Early Weight Averaging (Sanyal et al. 2023)

[arXiv](https://arxiv.org/abs/2306.03241). Weight averaging *during*
training (early), at high LR, with spaced-out checkpoints. On
OpenWebText 70K steps, GPT-2 small validation loss: baseline 2.963,
EMA 2.949, SWA 2.952, **LAWA 2.917** — best of the trajectory-
averaging family for LM pretraining specifically.

### 6.6 — When trajectory ensembles beat vs match independent training

| Metric | Trajectory ensemble (SWA/FGE/snapshot) | Deep ensemble (independent) |
|--------|----------------------------------------|---------------------------|
| Accuracy | Match at modest compute; lose at large | Win |
| Calibration | Good (within-mode) | Best (cross-mode) |
| Compute | 1× training | N× training |
| Deploy cost | 1× (weight avg) or M× (snapshot) | N× |
| Diversity | Low (same mode; Fort 2019) | High (different modes) |

At our scale (minutes per run, plenty of workers), **do both**. Each
member's training run can produce both (a) a final checkpoint for
deep-ensemble logit averaging and (b) an SWA weight average over its
last 20% of steps at ~zero cost. SWA is free insurance.

### 6.7 — Recommendation for SIEVE

Add SWA to every training run:

```python
# in worker/trainer.py
from torch.optim.swa_utils import AveragedModel, update_bn
swa_model = AveragedModel(model)
# ... during last 20% of epochs:
if step > 0.8 * total_steps: swa_model.update_parameters(model)
# at end:
update_bn(train_loader, swa_model)  # recompute BN stats
save(swa_model.state_dict(), f"{ckpt_path}.swa.pt")
```

Store both `final.pt` and `final.swa.pt`. At ensemble eval time, allow
the evolutionary selector to pick either version. Almost certainly the
SWA version is a drop-in improvement with no hyperparameter
sensitivity.

## Section 7 — Regularization × ensembling on small data

### 7.1 — Does heavy regularization help or hurt ensemble diversity?

Two competing effects:

**(a)** Heavy regularization (high weight decay, high dropout) shrinks
member solutions toward a common "regularized" point → *less*
diversity.

**(b)** Heavy regularization prevents each member from memorizing →
higher-quality base learners → the ambiguity decomposition improves
(avg member loss goes down faster than diversity falls).

The empirical question is which effect dominates. Slowrun's finding
(WD up to 16× standard + dropout) is on individual-model quality, not
ensemble composition. No published study of "WD × ensemble diversity"
at our scale, but Wenzel et al. 2020 (hyper-deep ensembles) shows that
*varying* WD across members beats fixed-WD deep ensembles, suggesting
the right answer is to *keep WD as an evolutionary dimension* (which
SIEVE does) rather than crank it uniformly.

**Prediction for SIEVE**: evolution will converge to a mix of WD
values — some low-WD members (memorize hard, useful for train-adjacent
test tokens) and some high-WD members (smoother, useful for novel
tokens). Diversity across WD is the mechanism.

### 7.2 — BPE-Dropout (Provilkov et al. 2020)

[arXiv](https://arxiv.org/abs/1910.13267). Stochastic corruption of
BPE merges during training exposes the LM to multiple valid
segmentations of the same text. +2.3 BLEU on translation.

**For SIEVE**: two uses.

1. **Per-member diversity**: train each member with a *different* BPE-
   dropout rate (0, 0.05, 0.1, 0.15). Each member sees a different
   tokenized view of the same corpus. Cheap data-level diversity that
   doesn't cost any tokens.
2. **Data augmentation at 10⁵ scale**: a paper-level claim is that
   BPE-dropout is an implicit data augmentation worth 1.5-2× effective
   corpus. At 85K tokens this is very attractive.

Add BPE-dropout to the search space. Rate ∈ {0, 0.05, 0.1, 0.15}.

### 7.3 — Mixup / input perturbation

Token-level mixup / input perturbations are less well studied for LMs
than for vision. Consistency regularization (Xie et al. 2020,
[NeurIPS](https://papers.neurips.cc/paper/2020/file/44feb0096faa8326192570788b38c1d1-Paper.pdf))
produces gains with augmented-input training but typically requires
larger unlabeled corpora (UDA is semi-supervised). Skip for SIEVE's
current regime.

### 7.4 — Label smoothing

Standard cross-entropy label smoothing ε=0.1 is a cheap diversity
booster — each member trained with smoothed labels produces a slightly
different softmax peak. Worth adding if not already. Not covered in
SIEVE's current search space.

## Section 8 — What breaks at 10⁵ tokens

### 8.1 — Memorization

At 7.4M params / 85K tokens, capacity >> data. Train loss → 0 is not
learning; it's memorizing. exp-001/002 show this (individual val
loss 0.04-0.10, test loss 10.87).

**Mitigations**:
- Heavy regularization (but see §7.1 — don't crank it uniformly).
- Multi-epoch with shuffling (Slowrun finding).
- Early stopping on val (SIEVE's current `best_val_loss` checkpoint
  already does this).
- **Avoid checkpoint at train-loss-zero**: save the *argmin val loss*
  checkpoint, not the final. Already done.

### 8.2 — Val-set meta-overfitting

Evolutionary selection on val over 30-50 generations is 30-50 rounds
of "which ensemble fits val best?" Classical multiple-testing: the
winning ensemble's val loss is a biased estimator of its test loss by
O(sqrt(log(#combinations_evaluated) / N_val)). At 9K val tokens and
~500 combinations per gen × 50 gens = 25K selection operations,
the selection bias is meaningful.

**Mitigations (SIEVE should implement)**:

1. **Rotating val splits**: per generation, use a different bootstrap
   sample of the val pool (e.g. 8K of 9K tokens), so no single
   ensemble gets optimized end-to-end on the same slice.
2. **Sequestered meta-val**: reserve 1K tokens of val *never touched
   by the selector*. Monitor the winning ensemble on this slice each
   generation. If meta-val loss diverges from val loss, the selector
   is overfitting.
3. **Test never touched until the final report**.

SIEVE's design already mentions (3); implement (1) and (2) before the
cluster run.

### 8.3 — Diversity collapse

Over generations, if elites propagate and mutation is weak, the
population converges to a single basin. Gen-by-gen diversity score
should be monitored.

- exp-001: 0.259 → 0.234 → 0.224 (descending — mild collapse)
- exp-002: 0.290 → 0.232 → 0.245 (also descending, partial recovery)

Both runs show small-population collapse. Fort 2019 says the highest-
diversity source is fresh random init — SIEVE's `immigration_fraction`
(0.17) is supposed to counter this but may be too low. Literature
precedent (Zhang et al. 2020 Diversified Ensemble
[PDF](https://papers.nips.cc/paper/2020/file/b86e8d03fe992d1b0e19656875ee557c-Paper.pdf))
trains members jointly with an explicit diversity loss; too intrusive
for SIEVE, but suggests immigration_fraction ≥ 0.25 is closer to
healthy.

### 8.4 — Catastrophic forgetting across multi-epoch training

Multi-epoch on 85K tokens means 100+ passes over the same data
(Muennighoff's RD*≈15 says this is past the repetition-useful limit).
Members will plateau and then start specializing to train-set
idiosyncrasies. SIEVE's per-member epoch count is in the search
space; evolution should find that lower-epoch members are preferable
for diversity, higher-epoch members for in-distribution fit.

**Multi-epoch × ensemble diversity (open question)**: Muennighoff's
paper doesn't test whether more-epoched members have less diversity
than less-epoched. Plausible that over-epoched members all converge
to the same memorization, losing diversity. **Add to open questions
in §10.**

## Section 9 — Specific recommendations for SIEVE

Numbered, prioritized by expected ROI given current state (smokes
done, cluster run pending, tokenizer fix pending).

### P0 — Blocking the cluster run

1. **Fix tokenizer first.** Do not run 12× GB10 SIEVE until the
   3-tokenizer ablation from [literature-tokenizers.md](literature-tokenizers.md)
   is complete. Ensemble gain is bounded by member quality; current
   members are below uniform on test.

2. **Measure ensemble test loss** for exp-001 and exp-002 winning
   ensembles *before* launching anything new. 30 minutes of work.
   If ensemble test > uniform, the entire SIEVE loop is val-
   overfitting and the cluster run is premature.

3. **Switch ensemble eval from brute-force C(N,K) to Caruana greedy
   selection** with allow-repeats (Caruana et al. 2004). Enables
   larger populations (>10 members becomes tractable) and gives soft
   weighting for free. ~50 lines of Python.

### P1 — Cheap wins to add to the cluster profile

4. **Add SWA to every training run** (§6.7). Store `final.pt` and
   `final.swa.pt`; let the selector choose either. ~20 lines in
   `worker/trainer.py`. Near-zero compute overhead, literature-
   validated free gain.

5. **Add BPE-dropout to the search space** at rates {0, 0.05, 0.10,
   0.15} per-member (§7.2). Lets the evolution find its own
   augmentation level. Drop-in if using HuggingFace tokenizers
   (`add_special_tokens=False` + dropout param on BPE decoder).

6. **Rotating val splits + sequestered meta-val** (§8.2). Prevents
   the selector from silently meta-overfitting val across 30-50 gens.

7. **Increase `immigration_fraction` to 0.25** (§8.3). Current 0.17
   is too low given observed gen-over-gen diversity collapse.

8. **Track `best_ensemble_loss_ever` across generations and write it
   to `best_ensemble.json`**, not last-gen's best (bug identified in
   exp-002, findings.md). Separate fix; already noted.

### P2 — Thesis-level experiments worth running

9. **Ensemble-level vs individual-level fitness A/B.** Run the same
   cluster budget twice:
   - (A) Standard PBT: fitness = individual val loss.
   - (B) SIEVE: fitness = marginal improvement to best ensemble
     containing this member.

   The empirical question SIEVE is built around has never been
   answered at our scale. Without this A/B, SIEVE's distinctive
   claim is unvalidated. Budget: 2× cluster runs = ~2-3 days on
   12× GB10.

10. **Weight averaging within lineage branches (intra-lineage soups).**
    Siblings (children of the same elite with same init) can be
    averaged à la Wortsman 2022. Free inference-cost reduction when
    it works. Add as an experimental aggregation option.

11. **Architecture diversity.** Reintroduce n_layer / n_embd as
    evolutionary dimensions (currently pinned in smoke runs). Zaidi
    2020's NES-RE shows this helps. Discrete mutation: step by ±1
    layer, ±64 embed dim, reject impossible combos.

### P3 — Things to *not* do

12. **Do not bootstrap-bag the training data.** 63% unique on 100K
    tokens is catastrophic at our vocab (§2.6).

13. **Do not rely on MC Dropout as a diversity source.** Same-mode,
    no real diversity (Fort 2019). Dropout at train time is fine as
    regularization; *ensemble-of-dropout-forward-passes is not a
    substitute for a deep ensemble*.

14. **Do not chase TIES/DARE model merging** unless the pivot is to
    "pretrained-on-OpenWebText + finetune-per-member". At from-
    scratch training with independent inits, cross-model weight
    merging is unsafe (different basins).

15. **Do not expect GENOME+-sized gains.** GENOME+ merges strong
    pretrained experts; SIEVE members are weak. Set expectations at
    20-40% of the literature's reported evolution-based gains, not
    face-value 24-55%.

## Section 10 — Open questions this review did NOT resolve

1. **Does ensemble-level-fitness evolution beat individual-level-
   fitness evolution on LM pretraining at our scale?** No one has
   tested. Recommendation P2-#9 would answer.

2. **Multi-epoch × ensemble diversity.** Do more-epoched members
   collapse to the same memorization, losing diversity? Muennighoff
   2023 doesn't say. Testable by running members with {10, 40, 100}
   epochs and measuring pairwise prediction disagreement on val.

3. **What's the right ensemble size at 10⁵ tokens?** Slowrun uses 8
   at 100M tokens; nothing published at 100K. Empirical question.
   Sweep 4-16 as part of the cluster run.

4. **Val-set selection bias at 25K+ selection operations.** The
   multiple-testing correction for 30 gens × 500 combos × 50-member
   population is unformalized. Needs a simulation study or a proper
   correction (e.g. split into multiple val folds, Bonferroni-style).

5. **How does LAWA (early weight averaging during training) combine
   with ensembling?** LAWA beats SWA/EMA on LM pretraining; whether
   it also beats SWA as an ensemble-member preparation step is
   unstudied.

6. **Is there a Solomonoff-style bound on ensemble gain as N→∞ at
   fixed data?** Unclear. The existing universal-prediction papers
   (Grau-Moya 2024 et al.) address single-predictor Solomonoff
   approximation; N-predictor convergence is untreated.

7. **Does the BabyLM architecture winner (GPT-BERT causal+masked)
   transfer to SIEVE?** BabyLM's 10M-word track is closest to us.
   The hybrid objective might dominate random-seed ensembles of pure-
   CLM models. Worth prototyping as a P3/experimental arm.

8. **Distillation of the final SIEVE ensemble into a single model for
   cheap inference.** Standard practice (Hinton 2015;
   [arXiv](https://arxiv.org/abs/1503.02531)), unstudied in SIEVE
   context. Needed only if inference cost becomes a constraint.

## Bibliography

- Akiba et al. 2024 — Evolutionary Optimization of Model Merging Recipes
  — https://arxiv.org/abs/2403.13187
- Bostrom & Durrett 2020 — BPE Is Suboptimal for LM Pretraining —
  https://arxiv.org/abs/2004.03720 (cited in literature-tokenizers.md)
- Breiman 1996 — Bagging Predictors — summarized at
  https://en.wikipedia.org/wiki/Bootstrap_aggregating
- Caruana, Niculescu-Mizil, Crew & Ksikes 2004 — Ensemble Selection
  from Libraries of Models —
  https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf
- Charpentier & Samuel 2024 — GPT-BERT (BabyLM 2 winner) —
  https://aclanthology.org/2024.conll-babylm.1/
- Eldan & Li 2023 — TinyStories — https://arxiv.org/abs/2305.07759
- Fort, Hu & Lakshminarayanan 2019 — Deep Ensembles: A Loss Landscape
  Perspective — https://arxiv.org/abs/1912.02757
- Garipov, Izmailov, Podoprikhin, Vetrov & Wilson 2018 — Loss Surfaces,
  Mode Connectivity, and Fast Ensembling — https://arxiv.org/abs/1802.10026
- Geiping & Goldstein 2022 — Cramming — https://arxiv.org/abs/2212.14034
- Grau-Moya et al. 2024 — Learning Universal Predictors —
  https://arxiv.org/abs/2401.14953
- Gunasekar et al. 2023 — Textbooks Are All You Need (phi-1) —
  https://arxiv.org/abs/2306.11644
- Guo, Pleiss, Sun & Weinberger 2017 — On Calibration of Modern Neural
  Networks — https://arxiv.org/abs/1706.04599
- Hinton, Vinyals & Dean 2015 — Distilling the Knowledge in a Neural
  Network — https://arxiv.org/abs/1503.02531
- Huang, Li, Pleiss, Liu, Hopcroft & Weinberger 2017 — Snapshot
  Ensembles — https://arxiv.org/abs/1704.00109
- Izmailov, Podoprikhin, Garipov, Vetrov & Wilson 2018 — SWA —
  https://arxiv.org/abs/1803.05407
- Jaderberg et al. 2017 — Population Based Training —
  https://arxiv.org/abs/1711.09846
- Krogh & Vedelsby 1994 — Neural Network Ensembles, Cross Validation,
  and Active Learning — (via ambiguity decomposition discussion in
  Ortega et al. 2022)
- Lakshminarayanan, Pritzel & Blundell 2017 — Simple and Scalable
  Predictive Uncertainty using Deep Ensembles —
  https://arxiv.org/abs/1612.01474
- Liu et al. 2026 (Systematic Study of Model Merging) —
  https://arxiv.org/html/2511.21437v1
- Maddox, Izmailov, Garipov, Vetrov & Wilson 2019 — SWAG —
  https://arxiv.org/abs/1902.02476
- Muennighoff et al. 2023 — Scaling Data-Constrained LMs —
  https://arxiv.org/abs/2305.16264
- Njieutcheu Tassi et al. 2022 — The impact of averaging logits over
  probabilities — https://ceur-ws.org/Vol-3215/19.pdf
- Ortega, Cabañas & Masegosa 2022 — Diversity and Generalization in
  Neural Network Ensembles —
  https://proceedings.mlr.press/v151/ortega22a/ortega22a.pdf
- Ovadia et al. 2019 — Can You Trust Your Model's Uncertainty? —
  https://arxiv.org/abs/1906.02530
- Provilkov, Emelianenko & Voita 2020 — BPE-Dropout —
  https://arxiv.org/abs/1910.13267
- Purucker et al. 2023 — CMA-ES for Post Hoc Ensembling in AutoML —
  https://arxiv.org/abs/2307.00286
- Sanyal et al. 2023 — Early Weight Averaging (LAWA) —
  https://arxiv.org/abs/2306.03241
- Wang et al. 2025 — GENOME / GENOME+ —
  https://arxiv.org/abs/2503.01155
- Wenzel, Snoek, Tran & Jenatton 2020 — Hyperparameter Ensembles —
  https://arxiv.org/abs/2006.13570
- Wortsman et al. 2022 — Model Soups —
  https://proceedings.mlr.press/v162/wortsman22a.html
- Xie et al. 2020 — Unsupervised Data Augmentation for Consistency
  Training —
  https://papers.neurips.cc/paper/2020/file/44feb0096faa8326192570788b38c1d1-Paper.pdf
- Yadav et al. 2023 — TIES-Merging — https://arxiv.org/abs/2306.01708
- Zaidi, Zela, Elsken, Holmes, Hutter & Teh 2020 — Neural Ensemble
  Search — https://arxiv.org/abs/2006.08573
- Zhang, Liu & Yan 2020 — The Diversified Ensemble Neural Network —
  https://papers.nips.cc/paper/2020/file/b86e8d03fe992d1b0e19656875ee557c-Paper.pdf
- Zhou, Wu & Tang 2002 — Ensembling Neural Networks: Many Could Be
  Better Than All — https://www.lamda.nju.edu.cn/publication/aij02.pdf
- Zhou et al. 2023 — LIMA — https://arxiv.org/abs/2305.11206
