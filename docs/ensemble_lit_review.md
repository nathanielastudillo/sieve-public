# Ensemble Literature Review for SIEVE

Scope: 80K–100K token transformer LM ensembles where members memorize and evolutionary fitness collapses (ensemble CE ≈ 10 > log(V) ≈ 8.99). What the literature says — ranked by leverage × cost.

---

## TL;DR — contradictions to your "obvious fixes"

1. **"Switch to probability averaging."** Misframed. For cross-entropy, the optimal combiner is the **normalized geometric mean of probabilities = arithmetic mean of logits + softmax** = Product-of-Experts (Wood et al. 2023, JMLR). So "logit averaging" already *is* geometric-mean aggregation. Plain probability mean is BMA-style but known to be vulnerable to overconfident members (Grushka-Cockayne et al. 2017). Your CE=10 implies ≥1 member is assigning ~0 probability to the true token; geometric mean is *especially* fragile to that — a single zero kills it.
2. **"Just cap epochs."** Partially wrong. LTG-BERT (BabyLM 2023) and BabyLlama-2 (2024) both train ≥20 epochs and win — because they add distillation + strong regularization. Capping epochs leaves tokens/param ≈15, still a memorization regime.
3. **"Shrink models or grow corpus."** Correct and the biggest lever. BabyLM strict-small is 10M words — 125× your corpus — and still considered a hard frontier. **At 80K tokens, no aggregator trick saves you.** Consistent across Eldan & Li 2023 (TinyStories uses ~1B synthetic tokens), Warstadt 2023, Charpentier & Samuel 2024.

---

## 1. Aggregators when members are overconfident memorizers

- Logit-mean ≡ geometric-mean-of-probabilities. Optimal for CE (Wood 2023).
- But catastrophically punished by any member giving ≈0 probability to the truth — exactly your failure.
- **Fix A — ε-smooth each member's softmax with uniform at α∈[0.01, 0.1] before aggregation.** Floors probabilities, kills geometric-mean blowup. Trivial post-hoc change.
- **Fix B — Trimmed-mean aggregation** (Grushka-Cockayne 2017): drop top/bottom-k members per token. Eval-time only.
- **Fix C — Temperature-calibrated mixture.** Per-member scalar T_i fit on held-out (Raftery et al. 2005, *MWR*, BMA calibration). Near-free and almost always helps miscalibrated members.
- **Solomonoff-faithful aggregator:** the correct log-loss mixture weights members by posterior evidence, not uniformly. Orseau et al. 2017 *Soft-Bayes: Prod for Mixtures of Experts with Log-Loss* and Hutter 2003 are the references. This is what you actually want: **weight members by cumulative per-token NLL on held-out**, not a uniform average.

## 2. Ensembling when base learners overfit hard

- **NCL (Liu & Yao 1999)** — joint training penalty that negatively correlates member errors. Caveat: Chen & Yao 2009 proved vanilla NCL itself overfits (it's equivalent to training one big unregularized ensemble). **RNCL** adds L2 over the whole ensemble to fix this. Medium leverage; high cost (breaks worker isolation, needs λ tuning).
- **Snapshot Ensembles (Huang 2017, ICLR):** cosine cyclic LR, collect M checkpoints per run. Each worker produces a free mini-ensemble — good fit for SIEVE. Cheap, solid leverage.
- **Bagging:** classical theory says it helps unstable learners (transformers on 80K tokens = maximally unstable). But bootstrap resampling leaves ~50K unique tokens per member — too little. Low leverage at this scale.
- **The winner in your regime — ensemble → distill.** BabyLlama (Timiryasov & Tastet 2023) and **BabyLlama-2 (2024)** show: a distilled student on 10M words outperforms its teachers *and* same-sized models trained on 100M words. Soft targets regularize far more than label smoothing; the student doesn't inherit teacher memorization. **If you want the ensemble's wisdom without its pathology, distill and evaluate the student, not the ensemble.**

## 3. BabyLM strict-small recipes (your regime, scaled up 125×)

At 0.08M tokens you are 125× below even BabyLM strict-small. With that caveat, what wins there:

- **GPT-BERT (Charpentier & Samuel 2024, BabyLM 2 winner both tracks):** hybrid causal+MLM objective, mask-rate linear decay 30%→15%, seq-length curriculum, ~20 epochs. Aux MLM is the anti-memorization trick.
- **BabyLlama / BabyLlama-2:** ensemble-distillation, strong WD, long schedules, small vocab (~16K), heavy dropout.
- **LTG-BERT (Samuel 2023):** BERT-style MLM + NormFormer + disentangled attention; many epochs without overfitting.

Consistent cross-winner pattern: **auxiliary objective (MLM/cloze on a causal model)** + **distillation rather than raw ensembling** + long schedules + strong WD + dropout.

## 4. Fitness-signal alternatives when val CE is noise

Your val CE 7.9 / uniform 8.99 = 0.88 ratio — that *is* signal, just compressed. Sharpen it:

- **Disagreement / Predictive Diversity Score** (Rame & Cord 2021; Jain et al. 2024, *Scalable Ensemble Diversification*): rank members by how much they diverge from peers on held-out. Robust when everyone's CE is bad because it measures *relative novelty*.
- **Train/val perplexity ratio** — punishes memorizers directly.
- **ECE on val tokens** — memorizers on OOD are maximally miscalibrated; select against them.
- **Composite fitness:** `fitness = −val_nll − λ·train_val_gap − μ·ece + ν·disagreement`. Pushes away from memorization, toward diverse generalization.

## 5. Anti-memorization tricks for sub-100K-token regimes

Cheap wins (implement today):

- **Label smoothing 0.1–0.2** — caps peak softmax, directly fixes the geometric-mean aggregator blowup.
- **DropPath / stochastic depth** (Huang 2016) — randomly skip transformer blocks. Stronger than dropout.
- **Embedding dropout** (AWD-LSTM, Merity et al. 2017) — drop whole token-embedding rows. Directly targets token memorization.
- **Weight decay 16× (Slowrun — you already know).** Keep.
- **SwitchOut** (Wang 2018, EMNLP): random token replacement ~10%; provable KL regularizer; free.
- **Span masking / infill aux objective** — forces context use over position memorization.
- **MixText / SeqMix** (Chen 2020) — hidden-state interpolation. Works on small text corpora.

Slightly more expensive:

- **Self-distillation with EMA teacher** — free smooth targets, huge regularization at this scale.
- **Gradient noise** (Neelakantan 2015) — cheap, helps small models.

## 6. When do evolutionary ensembles actually work in data-constrained regimes?

Honest answer: **the literature does not support evolutionary ensembles at sub-100K tokens.** PBT (Jaderberg 2017) wins on Transformer MT with hundreds of M tokens. GENOME+ and weight-merging work (Wortsman 2022 *Model Soups*) require shared initialization. **No published result** shows evolutionary ensembles beating single-model baselines at tokens/param ≈15. Distillation (BabyLlama-2) is the only ensemble approach with wins in sub-10M-word regimes — and it collapses to one student, not an ensemble.

---

## Ranked recommendations (leverage × cost)

| # | Action | Cost | Leverage |
|---|---|---|---|
| 1 | **Grow corpus ≥5M tokens or shrink models ≤1M params (tokens/param ≥100).** | High | Decisive — non-negotiable per literature |
| 2 | ε-smooth member softmax (α=0.05) before aggregation | Trivial | High — fixes geometric-mean blowup now |
| 3 | Composite fitness: `−val_nll − λ·gap − μ·ece + ν·disagreement` | Small | High — restores working gradient |
| 4 | Soft-Bayes / BMA-weighted aggregation (weights ∝ held-out likelihood) | Small | High — the Solomonoff-faithful choice |
| 5 | Snapshot ensembling per worker (cosine cyclic LR) | Small | Medium — free diversity |
| 6 | Regularization bag: LS 0.1, DropPath 0.1, EmbDrop 0.1, WD 0.1, SwitchOut 0.1 | Small | High — attacks memorization directly |
| 7 | Self-distillation (EMA teacher) or full ensemble→student à la BabyLlama-2 | Medium | High — proven at BabyLM scale |
| 8 | Aux MLM / span-infill objective alongside causal LM | Medium | Medium — GPT-BERT's edge |
| 9 | RNCL joint training w/ L2 over ensemble | High (breaks worker isolation) | Medium — skip unless 1–8 plateau |
| 10 | Trimmed-mean aggregation at eval | Trivial | Low-Medium — cheap band-aid |

**Blunt summary:** at 80K tokens you are ensembling memorizers and no aggregator mathematics fixes that. The literature's answer is one of: **(a)** 100× more data (or 100× smaller models), **(b)** distill the ensemble into a single student, or **(c)** use the ensemble only as a *diversity signal* for evolutionary selection rather than as a predictor. Pick one and commit — stacking marginal aggregator tricks on top of a 15-tokens-per-param regime is the wrong shape of solution.

---

### Citations

- Liu & Yao 1999, *Ensemble learning via negative correlation*, Neural Networks — <https://dl.acm.org/doi/10.1016/S0893-6080(99)00073-8>
- Chen & Yao 2009, *Regularized NCL*, TNNLS — <http://staff.ustc.edu.cn/~hchen/papers/TNNLS2009.pdf>
- Huang et al. 2017, *Snapshot Ensembles*, ICLR — <https://arxiv.org/abs/1704.00109>
- Jaderberg et al. 2017, *PBT* — <https://arxiv.org/abs/1711.09846>
- Hutter 2003, *Optimality of Universal Bayesian Sequence Prediction* — <https://www.jmlr.org/papers/volume4/hutter03a/hutter03a.pdf>
- Orseau et al. 2017, *Soft-Bayes: Prod for Mixtures of Experts with Log-Loss* — <http://proceedings.mlr.press/v76/orseau17a/orseau17a.pdf>
- Raftery et al. 2005, *BMA calibration of forecast ensembles*, MWR — <https://journals.ametsoc.org/view/journals/mwre/133/5/mwr2906.1.xml>
- Grushka-Cockayne et al. 2017, *Ensembles of Overfit and Overconfident Forecasts*, Mgmt Sci — <https://pubsonline.informs.org/doi/10.1287/mnsc.2015.2389>
- Wood et al. 2023, *A Unified Theory of Diversity in Ensemble Learning*, JMLR 24 — <https://arxiv.org/abs/2301.03962>
- Timiryasov & Tastet 2023, *BabyLlama* — <https://arxiv.org/abs/2308.02019>
- Tastet & Timiryasov 2024, *BabyLlama-2* — <https://arxiv.org/abs/2409.17312>
- Charpentier & Samuel 2024, *GPT-BERT* / BabyLM-2 findings — <https://arxiv.org/abs/2412.05149>
- Warstadt et al. 2023, *Findings of BabyLM 1* — <https://arxiv.org/abs/2504.08165>
- Eldan & Li 2023, *TinyStories* — <https://arxiv.org/abs/2305.07759>
- Merity et al. 2017, *AWD-LSTM regularization* — <https://arxiv.org/abs/1708.02182>
- Wang et al. 2018, *SwitchOut*, EMNLP — <https://arxiv.org/abs/1808.07512>
- Chen et al. 2020, *MixText* — <https://arxiv.org/abs/2004.12239>
- Jain et al. 2024, *Scalable Ensemble Diversification* — <https://arxiv.org/abs/2409.16797>
- Wortsman et al. 2022, *Model Soups* (shared-init constraint)
