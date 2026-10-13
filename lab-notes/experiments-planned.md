# Experiments planned (Bucket B: "probably good to try")

Drafted 2026-04-18, after the ensemble literature pass
([research-questions/literature-small-corpus-ensembles.md](research-questions/literature-small-corpus-ensembles.md)).

The lit review produced ~15 actionable recommendations. The ones that
are *almost certainly good* (literature-validated, cheap, low-risk)
are being implemented directly — see the "Bucket A" checklist at the
end of this page. The ones below are *probably good but need empirical
validation first*: each is written up as a planned experiment with
design, expected signal, cost, and the decision it feeds.

Ordering is by expected ROI given current state (smokes done,
tokenizer fix pending, cluster run pending).

## Priority ordering

| # | Experiment | Answers | Cost | Unblocks |
|---|------------|---------|------|----------|
| B1 | BPE-dropout in search space | Is token-level augmentation the right diversity lever at 10⁵ tokens? | 1 smoke run | Including BPE-dropout in cluster run |
| B2 | Rotating val splits across generations | Does meta-val drift reveal selection bias at 30-50 gens? | 1 cluster run + reporting | Trusting val-based selection at scale |
| B3 | Ensemble-level vs individual-level fitness A/B | Is SIEVE's distinctive thesis actually worth the complexity? | 2 cluster runs | SIEVE scaling decision |
| B4 | Intra-lineage weight-averaging (soups) | Can siblings within a lineage branch be soup-merged for free? | 1 cluster run + aggregator plumbing | Cheaper ensemble inference |
| B5 | Architecture diversity (unpin n_layer/n_embd) | Does Zaidi 2020 (NES) carry over to LM pretraining at 10⁵ tokens? | 1 cluster run with wider search | Search-space expansion |
| B6 | Multi-epoch × diversity study | Do over-epoched members collapse to the same memorization? | 1 smoke sweep | Picking epoch range for cluster runs |
| B7 | LAWA vs SWA ablation | Is LAWA's early-in-training weight averaging better than SWA on LMs at our scale? | 1 smoke sweep | Which trajectory-averaging to default |
| B8 | GPT-BERT hybrid objective (BabyLM winner) | Does causal+masked pretraining improve ensemble-member quality at small scale? | 1 smoke run with model patch | Architecture/objective choice |
| B9 | Distillation into single-model | Can we compress the final SIEVE ensemble to 1× inference cost without killing gain? | 1 distillation run post cluster | Deployment / Lever use cases |

---

## B1 — BPE-dropout in the search space

**Question.** Does per-member BPE-dropout produce useful diversity at
10⁵ tokens, where every token is already load-bearing?

**Citations.** Provilkov, Emelianenko & Voita 2020
([arXiv:1910.13267](https://arxiv.org/abs/1910.13267)) — +2.3 BLEU on
translation, implicit ~1.5-2× data augmentation. Wenzel et al. 2020
([arXiv:2006.13570](https://arxiv.org/abs/2006.13570)) — hyper-deep
ensembles: varying hypers (including augmentation rate) across members
is additive with random-seed diversity.

**Design.**

- Add `bpe_dropout_rate ∈ {0.0, 0.05, 0.10, 0.15}` to search_space.
- Load-time change: tokenizer receives `--dropout-rate` so each
  training pass re-tokenizes stochastically. HuggingFace `tokenizers`
  BPE supports this via `BPE(dropout=...)`.
- Population=8, ensemble=4, gens=3 on corpus_v006 (or whichever
  tokenizer survives the Magikarp ablation).

**Expected signal.**

- If ensemble val loss beats the same-population no-dropout baseline
  by ≥0.05 nats: token-level augmentation is net positive. Keep it as
  a dimension.
- If gain is ≤0.02 nats and diversity score is flat: augmentation isn't
  compensating for data scarcity at our scale. Drop it.

**Cost.** ~1 smoke run (~4h on one GB10). Plumbing: `data/loader.py`
needs to accept a BPE-dropout rate, tokenizer needs to be reloaded
per-member. ~100 LOC.

**Risk.** Retokenizing per member multiplies I/O; may slow each
training run by 10-20%. Mitigate by caching tokenized views by
dropout-rate bucket.

---

## B2 — Rotating val splits across generations

**Question.** Does the winning ensemble's val loss drift away from its
true held-out loss as generations progress? If yes, how fast?

**Citations.** Classical multiple-testing; see §8.2 of the lit review
for the selection-bias bound (`O(√(log(#combos)/N_val))`). No specific
SIEVE-scale paper; this is an engineering hygiene experiment.

**Design.**

- Implementation: each generation, sub-sample `f=0.85` of the
  selection-val pool via a generation-seeded permutation. Ensemble
  eval runs on that slice; the remaining 15% goes into a
  "gen-specific meta-val" pool that's only measured, never selected.
- Already implemented (Bucket A): a sequestered meta-val slice (10%
  of val.bin, fixed once) that's reported each generation but never
  used for selection.
- Rotating-val adds a second signal: per-generation drift between
  selection-val and gen-specific meta-val.

**Expected signal.**

- At 30-50 generations: if the winning ensemble's selection-val loss
  improves while its sequestered-meta-val loss plateaus or worsens,
  selection is overfitting.
- The rotation amplifies this: rotating prevents any single slice
  from getting optimized end-to-end.

**Cost.** ~50 LOC in `ensemble_eval.py` + one cluster run to
demonstrate divergence (if it appears). Only meaningful at 30+ gens.

**Decision.** If divergence appears at our scale, rotating-val
becomes load-bearing for the cluster run. If no divergence, the
simpler fixed-meta-val stays sufficient and rotation is dead weight.

---

## B3 — Ensemble-level vs individual-level fitness A/B

**The load-bearing SIEVE experiment.**

**Question.** Does selecting population members based on their
contribution to ensemble loss (SIEVE) beat selecting on their
individual val loss (standard PBT) at 10⁵ tokens?

**Citations.** Jaderberg et al. 2017 (PBT,
[arXiv:1711.09846](https://arxiv.org/abs/1711.09846)) — individual
fitness. Zaidi et al. 2020 (NES-RE,
[arXiv:2006.08573](https://arxiv.org/abs/2006.08573)) — ensemble
fitness on CIFAR image classification (not LM). Wang et al. 2025
(GENOME+, [arXiv:2503.01155](https://arxiv.org/abs/2503.01155)) —
evolves LLM population but with per-individual fitness + ensemble
*aggregation*, not ensemble *selection*. **No published experiment
tests this for LM pretraining at our scale.** SIEVE's novelty lives
here.

**Design.**

- Run two otherwise-identical cluster runs:
  - **Arm A (PBT)**: `fitness = member.val_loss`. Standard individual
    selection.
  - **Arm B (SIEVE)**: `fitness = marginal_ensemble_improvement`
    (i.e. count of appearances in top-K ensembles × how much each
    appearance reduced ensemble loss).
- Same corpus, same architecture, same population size (12-24),
  same search space, same seed for initial population, same compute
  budget (30 generations).

**Expected signal.**

- If SIEVE's ensemble test loss is ≥0.2 nats better than PBT's at
  gen-30: SIEVE thesis validated at our scale. Publishable.
- If indistinguishable (<0.05 nats): ensemble-level selection is not
  worth the complexity at 10⁵ tokens. Fall back to standard PBT with
  greedy ensemble selection at the end.
- If SIEVE is *worse*: the ensemble-level signal has too much variance
  to drive selection at small population sizes. Interesting but kills
  the distinctive SIEVE hypothesis.

**Cost.** 2 cluster runs × ~1 day each on 12× GB10 = 2-3 wallclock
days. Code: add `fitness_mode ∈ {individual, ensemble}` to
orchestrator + Population.select().

**This is the experiment SIEVE is built around. Do it second
(immediately after B2), before any further elaboration.**

---

## B4 — Intra-lineage weight averaging (soups within branch)

**Question.** Can parent + children-from-same-parent be averaged
(model soup, Wortsman 2022) to produce a cheaper-to-ship ensemble
member?

**Citations.** Wortsman et al. 2022 (Model Soups,
[ICML](https://proceedings.mlr.press/v162/wortsman22a.html)) — weight
averaging works *within one low-error basin*; Fort, Hu &
Lakshminarayanan 2019 ([arXiv:1912.02757](https://arxiv.org/abs/1912.02757))
— independent inits are in *different* basins, so cross-population
soups are unsafe, but siblings with shared init may be soupable.

**Design.**

- Add an aggregator option: `weight_average_within_lineage`.
- For each elite in the current generation, take the elite's weights
  and uniformly average them with its children (same init, diverged
  under mutation).
- Evaluate the merged model's val loss; add it to the ensemble-eval
  pool as a synthetic member if it beats the average of its
  ingredients.
- Greedy-soup variant: only merge children whose inclusion strictly
  improves the merged model's val.

**Expected signal.**

- If merged model ≥ best individual child: free ensemble member at 1×
  inference cost.
- If ≤ best individual child: merging has already hit cross-basin
  territory; stop.

**Cost.** ~100 LOC in controller/population.py (track `parent_id`
and `init_seed` rigorously) + aggregator in ensemble_eval.py. 1
cluster run to demonstrate; can be overlaid on a run that's happening
for another reason.

**Risk.** Model-merging introduces synthetic "members" not in the
lineage DB as trained runs — need clean accounting so the lineage
tree doesn't lie about provenance.

---

## B5 — Architecture diversity (unpin n_layer / n_embd)

**Question.** Does varying architecture across population members
produce more ensemble gain than varying only hyperparameters, at 10⁵
tokens?

**Citations.** Zaidi et al. 2020 (NES-RE,
[arXiv:2006.08573](https://arxiv.org/abs/2006.08573)) — architecture
ensembles beat deep ensembles on CIFAR. Regime caveat: no LM-at-our-
scale validation.

**Design.**

- Unpin search_space: `n_layer ∈ {4, 6, 8}`, `n_embd ∈ {192, 256, 384}`,
  `n_head ∈ {4, 6, 8}` with the compatibility constraint
  (n_embd divisible by n_head).
- Discrete mutation: step ±1 layer, ±one level of embd; reject
  incompatible combos.
- Run one cluster gen-30 with arch-diverse vs arch-pinned population
  at same total compute.

**Expected signal.**

- If arch-diverse wins by ≥0.1 nats on ensemble test: Zaidi's NES
  finding transfers to our regime. Keep it.
- If flat or worse: member-quality variance from architecture
  overwhelms diversity gain at our training-token scarcity. Pin.

**Cost.** 1 cluster run. Needs proper config-space validation (reject
invalid combos pre-train).

---

## B6 — Multi-epoch × diversity study

**Question.** Do members trained for many epochs collapse to similar
memorizations (low diversity) compared to fewer-epoch members?

**Citations.** Muennighoff et al. 2023
([arXiv:2305.16264](https://arxiv.org/abs/2305.16264)) — repetition
decays past ~4 epochs on data-constrained training, but says nothing
about ensemble diversity as a function of repetition.

**Design.**

- Train 8 members at each of {10, 40, 100, 200} epochs with different
  seeds, same other hypers.
- Measure (a) individual val loss, (b) pairwise prediction
  disagreement on val (Jensen-Shannon divergence), (c) ensemble loss
  of the 8-member group at each epoch level.
- Same corpus_v004 (6.4M tokens) to get cleaner multi-epoch behavior
  than v003/v005 (~85K).

**Expected signal.**

- Monotonic diversity collapse as epochs grow: cap epoch count in the
  search space.
- Diversity rebound at high epochs (members "specialize" differently):
  explore further.

**Cost.** 32 training runs at varying epoch counts. ~1 day of smoke
compute on one GB10 per epoch bracket. Can overlap with another
cluster run.

---

## B7 — LAWA vs SWA ablation

**Question.** On transformer LM pretraining at 10⁵ tokens, does LAWA
(early weight averaging during training) beat SWA (late SGD trajectory
averaging)?

**Citations.** Sanyal et al. 2023 (LAWA,
[arXiv:2306.03241](https://arxiv.org/abs/2306.03241)) — on
OpenWebText 70K steps, GPT-2 small val loss: baseline 2.963, EMA
2.949, SWA 2.952, **LAWA 2.917**. Izmailov et al. 2018 (SWA,
[arXiv:1803.05407](https://arxiv.org/abs/1803.05407)) — the
established baseline.

**Design.**

- Fork `worker/trainer.py` to produce three checkpoints per run:
  `.pt` (best val), `.swa.pt` (last 20% average), `.lawa.pt` (early
  average, spaced checkpoints over first 50% of training).
- Run 8-member smoke; ensemble-eval all three variants separately;
  compare which variant produces the best ensemble.

**Expected signal.**

- If LAWA ensemble < SWA ensemble by ≥0.02 nats: switch default to
  LAWA.
- If indistinguishable: keep SWA (simpler, already in `torch.optim`).

**Cost.** One smoke run; ~50 LOC change in trainer.py.

---

## B8 — GPT-BERT hybrid objective (BabyLM 2 winner)

**Question.** Does a causal + masked joint pretraining objective
produce better ensemble members at 10⁵ tokens than pure causal LM?

**Citations.** Charpentier & Samuel 2024 (GPT-BERT,
[ACL](https://aclanthology.org/2024.conll-babylm.1/)) — BabyLM 10M /
100M word winner. Hybrid objective beats pure causal at small scale.

**Design.**

- Fork model/gpt.py to add a `hybrid_objective_ratio ∈ [0, 1]`
  parameter. 0 = pure causal; 1 = pure masked; 0.5 = alternating
  batches.
- Train 8 members with same config, varying only this ratio.
- Measure individual val loss, ensemble val loss.

**Expected signal.**

- If hybrid beats pure causal as individual: rewrite base model.
- If ensemble benefits specifically (via member-diversity on
  objective): add `hybrid_objective_ratio` to search_space.

**Cost.** ~300 LOC — new masking machinery in loader + forward pass.
1 smoke run. Only worth doing after B3 clarifies the direction.

---

## B9 — Distillation into single-model

**Question.** After a cluster run produces a winning ensemble, can we
distill it into a single model at 1× inference cost without losing
the ensemble gain?

**Citations.** Hinton, Vinyals & Dean 2015
([arXiv:1503.02531](https://arxiv.org/abs/1503.02531)) — standard
ensemble → single distillation. Transfers to small-data regimes;
exact loss-gain tradeoff is regime-specific.

**Design.**

- After cluster run, use the winning ensemble as teacher.
- Train a new 7.4M-param model with cross-entropy + KL(student || teacher)
  on a 1M-token corpus (corpus_v004 subsample), teacher providing
  soft targets.
- Measure student val loss + test loss vs (a) best single member,
  (b) ensemble.

**Expected signal.**

- Student ≥ best single member by ≥0.1 nats: distillation recovers
  ensemble gain for free at inference.
- Student ≤ best single member: ensemble gain is in the diversity, not
  distillable into one parameter budget.

**Cost.** 1 distillation run (~2h). Mostly a Lever/deployment concern;
not urgent for SIEVE-as-research.

---

## Bucket A — Implementation checklist (tracked here for cross-ref)

The below are *not* experiments. They are corrections + drop-in
improvements that the literature says to do right now. Tracked in
the conversation's task list and committed separately.

- [x] **Research note** — `literature-small-corpus-ensembles.md` written
- [x] **A1. Global-best ensemble tracking** — `controller/orchestrator.py`
  now compares each gen's #1 to the running-best `val_loss` and only
  overwrites on improvement. Fixes the exp-002 8.68→8.82 reporting bug.
- [x] **A2. Immigration fraction → 0.25** — bumped in `config/default.yaml`
  and all four smoke profiles with inline Fort 2019 rationale.
- [x] **A3. Caruana 2004 greedy-with-repeats** —
  `_caruana_greedy_combinations` added to `ensemble_eval.py` and wired as
  the default strategy via `strategy="caruana_greedy"` in the orchestrator.
- [x] **A4. SWA in trainer** — `worker/trainer.py` saves `*.swa.pt`
  alongside the best-val checkpoint (default tail fraction 0.2). Skipped
  `update_bn` since GPT uses only LayerNorm (stateless).
- [x] **A5. Sequestered meta-val slice** — `ensemble_eval.py` carves 10%
  of val.bin via a fixed-seed (1729) chunk-level permutation; orchestrator
  calls `compute_meta_val_loss` each gen, logs it, and writes it to the
  lineage DB (new `meta_val_loss` column, auto-migrated).
- [x] **A6. Script to measure ensemble test loss** —
  `scripts/measure_ensemble_test_loss.py` takes a `best_ensemble.json`,
  loads test.bin, reports per-member + ensemble test NLL, and compares
  to ln(V). Point it at `runs/smoke_loop_v006/results/best_ensemble.json`
  to get the P0 answer on exp-002 before the cluster run.

Each item links to a git commit when implemented.
