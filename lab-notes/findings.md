# Findings

Running list of empirical surprises and the questions they opened.
Chronological, dated, linked to the experiment that exposed them.

## 2026-04-23 — exp006 final: bench shim, pre/post clean, SFT-on-v009 lift ≈ 0

exp006 (`experiments/006-multi-model-ensemble.md`) closed today after three
fixes landed back-to-back:

1. **`benchmarks/adapters/smoke_mcq.py` thinking-model shim.** The original
   bench set `max_tokens=4` and extracted the first A/B/C/D. Qwen3-14B and
   SmolLM3-3B emit `<think>...</think>` reasoning before the answer — with a
   4-token budget they never closed the opening tag and scored 0.05 (1/20,
   below 4-way chance). Fixed by: `MAX_TOKENS=2048`, `TIMEOUT_S=300`, strip
   complete `<think>` blocks before scoring, score a truncated opening tag
   as 0, and take the **last** word-boundaried letter (models reason "...
   so the answer is D" — the first letter is "A" from "answer"). After
   the fix both models hit 1.0 baseline. Lesson: when a bench scores a
   strong model below chance, the bench is wrong, not the model.

2. **`openweight/inference.py` launch timeout.** `ready_timeout_s` default
   of 600s is insufficient for a cold HF cache download of a 13–14B
   checkpoint (observed ~4–5 MB/s → 14B base ≈ 30 min). Bumped to 1800s;
   redeployed to the whole fleet. Even 1800s is at the edge — pre-warming
   caches fleet-wide at the start of a run would be more robust.

3. **Clean pre/post on the fixed bench:** SFT-on-v009 (500 steps, LoRA
   r=16) moved **zero** of five bases meaningfully on smoke_mcq
   (aggregate lift +0.02 across phi-4, OLMo-2-13B, granite-3.3-8B,
   SmolLM3-3B, Qwen3-14B). This was pre-registered (H2) — v009 is
   clinical/theoretical, smoke_mcq is trivia — but it's now documented
   with an un-broken bench. The real signal this run gave was:
   **smoke_mcq is saturated on strong 8–14B bases (4/5 hit ≥0.95
   pre-SFT).** Any future experiment that wants to measure SFT lift or
   ensemble gain on this class of model needs headroom — exp 007's
   `ifeval` + `mmlu_pro` wiring is now load-bearing, not decorative.

Ops surprise worth recording: **<standalone>-class workers are measurably slower
than compute at thinking-model generation.** Two post-tune evals landed on
`<standalone>-*` nodes and hit 13/20 and 15/20 per-question 300s timeouts;
re-landing the same jobs on compute nodes produced 1.0 cleanly. Either
bump per-question budget or add a worker-class affinity hint for
thinking-model evals on future runs.

## 2026-04-21 — openweight dev loop end-to-end on Qwen2.5-1.5B, and ml-intern survey

Two things landed today:

**exp005 end-to-end** (`experiments/005-openweight-smoke-qwen15b.md`).
Baseline → 1-epoch LoRA SFT on `v009.jsonl` → merge → post-SFT bench on
node-8. The plumbing is green across every code path
(`LineageDB.record_openweight_member`, `openweight.finetune.fine_tune`,
`openweight.serving_shim`, `openweight.inference.merge_lora_to_dir`,
`DashboardMetricsCallback` live loss chart). Headline delta is a small
nominal regression (ifeval prompt-level loose 0.500 → 0.425), but within
1σ of baseline on n=40 proxy — cannot reject "SFT did nothing" at this
sample size. `smoke_mcq` is non-comparable this run because baseline
used n=20 (proxy=false) and post-SFT used n=5 (proxy=true) despite both
commands passing `--proxy`. Bug lives in `benchmarks/adapters/smoke_mcq.py`
— its `--proxy` semantics drifted between the two bench commands.
Surprising empirical observation: `bs=2, seq_len=1024, rank=16, bf16` on
GB10 is attention-bound, not matmul-bound — **95% GPU util, 11% MFU,
9.3/24 GB HBM used, 60/150 W TDP**. Next-run throughput wins logged in
the experiment file: bs=4, `attn_implementation="sdpa"`, `torch.compile`.

**HF ml-intern × SIEVE audit** (`research-questions/ml-intern-synthesis.md`
+ three companion docs). Three parallel agents converged on the same
recommendation: adopt ml-intern as an **edge tool**, not inner-loop.
Cleanest graft point is upstream of
`controller/openweight_orchestrator.py:298 run_gen0` — ml-intern
proposes Gen-0 seed roster + initial `SftConfig` priors + candidate data
shards; everything downstream runs unchanged. ~300 LOC wrapper
(`scripts/ml_intern_propose_immigrants.py`) is the version worth
building. Explicitly forbid ml-intern from writing to `members` /
`fitness_evals` or from the selection path: the LLM prior is strong
enough to mode-collapse on popular bases (Llama/Qwen) and prune
ensemble-load-bearing-but-individually-mediocre members — exactly the
Krogh-Vedelsby ambiguity term SIEVE's composition-fitness is designed
to preserve. Pieces worth stealing outright: Semantic Scholar
citation-graph walker (`papers_tool.py:40-129`, with `isInfluential` +
`intents` edges), snippet search over 12M papers, ChatML
auto-format-check in `hf_inspect_dataset` (add equivalent to
`data/build_sft_shard.py`, ~20 LOC). Gated on a clean positive delta
from the openweight track before logging as Bucket B.

## 2026-04-18 (later) — Bucket A landed: six literature-driven fixes in priority order

Implemented all six "almost certainly good" recommendations from the
literature review in order of expected ROI. Each change ships with a
comment citing its source paper so future readers can audit the
rationale without going back to the review.

- **A1 · Global-best ensemble tracking** (`controller/orchestrator.py`).
  Prior code overwrote `best_ensemble_spec` with every generation's #1
  ensemble. At small pop sizes Caruana greedy is non-monotonic across
  generations, so the final `best_ensemble.json` was reporting the
  *last* gen's top-1 rather than the *best* seen — that's why exp-002
  wrote 8.82 when the real run-best was 8.68 at gen 1. Now compared
  against a running best, only overwrites on strict improvement, and
  logs both the new global-best line and "this gen did not beat" lines.
- **A2 · Immigration fraction → 0.25** (`config/default.yaml` + all four
  smoke profiles). Fort, Hu & Lakshminarayanan 2019
  ([arxiv:1912.02757](https://arxiv.org/abs/1912.02757)) show that
  random initialization is the dominant diversity source for deep
  ensembles (different inits land in different loss-landscape modes).
  exp-001/002 both exhibited gen-over-gen diversity collapse with 0.10
  – 0.17 immigration; the new 0.25 matches the fraction Fort ran
  empirically well without starting from scratch.
- **A3 · Caruana 2004 greedy-with-repeats** (`controller/ensemble_eval.py`
  → `_caruana_greedy_combinations`). Re-selection is allowed at each
  step, which (a) gives picked-twice members implicit weight 2/k in the
  logit mean and (b) is empirically more overfitting-resistant than
  subset greedy (Caruana §5.1; still SOTA in AutoML post-hoc ensembling
  per Purucker 2023, [arxiv:2307.00286](https://arxiv.org/abs/2307.00286)).
  Seeded from the best-N singletons for multi-start robustness. Wired
  as default in the orchestrator with inline rationale.
- **A4 · SWA trajectory averaging** (`worker/trainer.py`). Each run now
  saves a `*.swa.pt` alongside the best-val checkpoint — running
  average over the last 20% of epochs (Izmailov 2018; LAWA 2023 shows
  2.917 vs 2.963 baseline val on GPT-2/OpenWebText with almost no
  overhead). Skipped `update_bn()` since GPT uses only LayerNorm
  (stateless). Controlled by `swa_enabled` + `swa_tail_fraction`
  fields on `TrainConfig`.
- **A5 · Sequestered meta-val slice** (`ensemble_eval.py` +
  `orchestrator.py` + `lineage.py`). 10% of val.bin is carved off via
  a fixed-seed (1729) chunk-level permutation and never shown to the
  selector. Each generation, after Caruana picks the top ensemble, the
  orchestrator calls `compute_meta_val_loss` on a fresh forward pass
  against the reserved slice and writes the result to a new
  `meta_val_loss` column in `generations` (auto-migrated via
  `ALTER TABLE` for older DBs). At 30 – 50 generations the selection-
  val NLL becomes a biased estimator of true loss; this is the
  unbiased canary we need to catch that drift before it wastes a
  cluster run.
- **A6 · `scripts/measure_ensemble_test_loss.py`** (new). Point it at a
  `best_ensemble.json` and it loads test.bin, forward-passes each
  member, reports per-member + ensemble test NLL, compares to ln(V),
  and prints the selection-val-vs-test gap. This is the retrofit path
  that answers the P0 lit-review question for exp-001/002
  ("is ensemble test NLL below uniform?") without launching new
  training. Next step before the cluster run.

All six items are marked done in
[experiments-planned.md](experiments-planned.md). Bucket B (9
experiments) remains queued there, ordered by expected information
gain.

## 2026-04-18 — exp007_v009 dispatch timeout vs training duration

See [experiments/004-exp007-v009.md](experiments/004-exp007-v009.md).

- **Training wallclock scales roughly linearly with corpus tokens.**
  v006 smoke members finished in ~14 min; v009 (63× more tokens)
  members take 5 – 11 hours. The hardcoded `job_timeout_s=7200.0` in
  `orchestrator.py:731` was set for the smoke regime and silently
  broke the full run: all 48 members timed out, workers kept training,
  late results piled up in Redis with nothing collecting them.
- **Per-generation timeout is the wrong abstraction.** One slow node
  forces the orchestrator to abandon everyone else's in-flight work.
  The right unit is per-member deadline with a generous global
  ceiling; late results should still be admitted into the *current*
  generation's ensemble eval.
- **Redis is load-bearing as the recovery log.** Because workers
  publish with 7-day TTL, a botched orchestrator run is survivable —
  all 10 successful exp007_v009 members (val_loss 3.85-4.26) are
  recoverable from `sieve:results:...` post-hoc. This is the first
  time that property paid off.
- **Worker_id format divergence matters for checkpoint pullback.**
  compute-cluster workers publish `node-9-1868887` (bare node name);
  standalones publish `a@<node-ip>-21552` (embeds user). The
  orchestrator's rsync stage needs to handle both — hardening applied
  in the same commit as the timeout diagnosis.
- **Two standalones OOM on job intake** (<standalone>-9dba/.116,
  <standalone>-ad3f/.152). Fail in ms, immediately BLPOP the next job, burn
  through the queue at every generation boundary. 29 of 48 exp007
  jobs were lost this way. Root cause not yet investigated — smaller
  free VRAM? residual process? Exclude both until triaged.

## 2026-04-18 — Literature review on ensemble training

Dedicated review compiled at
[research-questions/literature-small-corpus-ensembles.md](research-questions/literature-small-corpus-ensembles.md).
The 3-5 most decision-relevant conclusions for the cluster run:

- **Ensemble gain is upper-bounded by average member quality** (Krogh-
  Vedelsby ambiguity decomposition). Our exp-003 single-member test
  NLL 10.87 > uniform 9.68 means the ceiling is currently blocked by
  broken tokenization + memorization, not the ensembling algorithm.
  Measure ensemble *test* loss (not just val) before scaling.
- **Random init is the #1 diversity source** (Fort, Hu & Lakshminarayanan
  2019). Subspace methods (MC Dropout, SWAG) stay in one mode. SIEVE's
  current design is correct here — do not waste compute on dropout-
  forward-pass "ensembles".
- **Replace brute-force C(N,K) ensemble eval with Caruana 2004 greedy
  ensemble selection** (allow-repeats). 22 years of empirical validation,
  dominates brute-force at any N>10, enables population>12.
- **Add SWA to every training run** (Izmailov 2018, LAWA 2023). Cheap
  trajectory averaging is a free drop-in improvement; the LAWA GPT-2
  result (2.917 vs 2.963 baseline val) is directly applicable to LM
  pretraining.
- **Bootstrap bagging is catastrophic at our scale**: 63.2% unique ×
  100K tokens = 63K/member, well below Bostrom & Durrett 2020's
  quality floor. Keep random init + hyperparameter variation as the
  diversity substrate.
- **SIEVE's distinctive claim (ensemble-level fitness > individual-
  level fitness) is empirically untested at our scale.** NES-RE (Zaidi
  2020) does it on CIFAR; GENOME+ (2025) uses ensemble *operators* but
  evaluates fitness per-individual. Recommended A/B experiment to
  validate before committing heavy compute.
- **Val-set meta-overfitting risk over 30-50 generations is real.**
  Add rotating val splits + a sequestered meta-val slice before the
  cluster run.

---

## 2026-04-17 (later) — Root cause identified: vocab-to-train-tokens ratio (Magikarp)

**Mechanism.** [Land & Bartolo 2024 — Fishing for Magikarp](https://arxiv.org/abs/2405.05417)
describes "glitch tokens": vocab entries that appear in tokenizer training
but barely in LM training, whose output embeddings are effectively random.
When a test sequence hits one, the softmax assigns near-zero mass to the
correct token, producing per-token NLL worse than uniform.

**Measurement on our corpora** (see
[research-questions/literature-tokenizers.md](research-questions/literature-tokenizers.md)
and `scripts/magikarp_check.py`):

| Corpus | Vocab | Train toks | Never-seen | ≤10 occ. | Test-tok @ ≤10 | ln(V) | Unigram NLL (test) |
|--------|------:|-----------:|-----------:|---------:|---------------:|------:|-------------------:|
| v003 (tok v002) | 16K | 85K | 40.3% | 93.4% | 28.7% | 9.68 | 7.19 |
| v005 (tok v002) | 16K | 83K | 41.9% | 93.6% | 28.4% | 9.68 | 7.10 |
| v006 (tok v003) | 16K | 83K | 40.8% | 93.8% | 28.7% | 9.68 | 7.07 |
| **v008 (tok v005)** | **8K** | 82K | **19.9%** | **77.5%** | **23.5%** | 8.99 | 6.90 |
| **v007 (tok v004)** | **4K** | 81K | **8.2%** | **53.5%** | **16.4%** | 8.29 | 6.80 |

40% of the 16K vocab is *never* seen in training, and ~29% of test-token
occurrences fall on vocab entries seen ≤10 times. Shrinking to 4K cuts
the never-seen fraction 5× and test-token Magikarp exposure ~2×.

**What this explains:**
- exp-001/002 test NLL 10.87 > uniform 9.68 (meta-overfitting finding):
  this is exactly the Magikarp pathology. The random-embedding tokens
  push test loss above `ln(V)`.
- exp-002 ensemble val 8.82 > unigram baseline 7.11 (a finding I didn't
  notice before — our 7.4M-param ensemble is **beaten by a zero-param
  Laplace-smoothed unigram**. The "evolutionary signal" we've been
  measuring has been measured against a broken baseline.)
- v005 worse than v003: the v002 tokenizer was *slightly* worse matched
  to v005 (41.9% never-seen vs 40.3%), but the dominant effect is both
  are catastrophically under-sized for 85K training tokens.
- Why v006 (v003 tokenizer) barely moved the needle: the tokenizer
  quality change affected fertility (fewer fragmented domain terms)
  but did nothing for the fundamental vocab/train-token ratio.

**Implication.** The SIEVE evolutionary loop has been optimizing on top
of broken tokenization. Everything downstream — ensemble diversity,
ensemble val loss trajectories, the v003↔v005 regression, the
meta-overfitting — is a consequence of that. The fix is upstream.

**Linked:** [research-questions/tokenizer-scaling.md](research-questions/tokenizer-scaling.md), [research-questions/literature-tokenizers.md](research-questions/literature-tokenizers.md), [scripts/magikarp_check.py](../scripts/magikarp_check.py), [research-questions/magikarp_results.json](research-questions/magikarp_results.json)

---

## 2026-04-17 — Meta-overfitting: single member test-loss exceeds uniform baseline

**What we saw:** Gen-2 member `c884ab4b` from exp-002 has val_loss 0.068
(deep memorization) but test_loss **10.87 nats** — worse than
`ln(16000) = 9.68`, the uniform-over-vocab baseline.

**Why it matters:** The SIEVE thesis assumes individual ensemble members
are at least competent language models, and the ensemble improves via
diversity+calibration on top of that competence. If the individuals are
below uniform, the ensemble's val-loss win of ~0.9 nats over baseline
is suspicious — it may be val-set calibration artefact rather than
real generalization.

**Open questions:**
- Measure ensemble **test** loss for both v003 (8.7315 val) and v005
  (8.8247 val) winning ensembles. If test >> val, the val signal is
  meta-overfit.
- At what training step does test loss cross the uniform baseline?
  (Need epochwise checkpointing — currently only `best_val_loss`
  checkpoint is saved.)
- Is the effect specific to the aggressive regularization schedule,
  or does it survive with weight_decay ∈ [0.1, 0.3]?

**Linked:** [experiments/003-inference-day-one.md](experiments/003-inference-day-one.md)

---

## 2026-04-17 — v005 is worse than v003 despite 2× more sources

**What we saw:** v005 (35 sources, 97K tokens) ensemble val_loss 8.8247
vs v003 (16 sources, 100K tokens) ensemble val_loss 8.7315. Same
architecture, same search space, same evo params, same tokenizer (v002).

**Why surprising:** v005 was built specifically to test whether source
diversity improves ensemble quality. It did not — it regressed. The
effect is not huge (0.09 nats) but the sign is wrong.

**Candidate explanations:**
1. **Tokenizer mismatch.** v002 BPE trained on pre-expansion pool;
   Deleuze/Kristeva/Latour/Schuster vocab fragments inefficiently.
2. **Tokens-per-source dilution.** 35 sources × ~2.8K tokens/source =
   too few to learn per-author statistics. v003 had ~6.2K/source.
3. **v002 BPE was specifically good at v003.** v002 was trained on
   the v003 source set, so v003 has zero OOD tax. Any new corpus
   loses.

**Discriminating experiment:** tasks #14-16 — retrain BPE on v004 pool
(37 sources, 7.5M tokens), rebuild subsample v006 with the new tok,
re-run. If v006 ensemble > v005, tokenizer was the bottleneck. If
v006 ≈ v005, dilution was.

**Linked:** [experiments/002-smoke-v005.md](experiments/002-smoke-v005.md), tasks #14-16

---

## 2026-04-17 — Evolutionary loop is non-monotonic at population_size=6

**What we saw:** v005 run gen-0 best=9.39 → gen-1 best=**8.68** →
gen-2 best=8.82. Gen 1 was a local peak; gen 2 regressed.

**Why it matters:** The final `best_ensemble.json` is written from
gen 2, not from gen 1 — so the run reported 8.82 when its actual best
was 8.68. The SIEVE loop doesn't track global-best across generations.

**Fixes (not yet applied):**
- Orchestrator: track `best_ensemble_loss_ever` and write it (plus its
  member IDs) to `best_ensemble.json`.
- Evo params: population_size=6 with elite_fraction=0.34 (2 elites) and
  mutation_strength=0.2 (±20%) lets a single bad mutation undo a
  generation. Bump to population_size=12 or drop mutation to 0.1 for
  the next profile.

**Not yet linked to a numbered experiment** — this is a config/code
issue, not a research finding per se. Fix goes in a future PR.

---

## 2026-04-17 — pymupdf4llm silently drops 95%+ of text on multi-column PDFs

**What we saw:** 7 PDFs in `data/extracted/` were under-extracted by
orders of magnitude. pymupdf4llm's markdown converter silently returned
tiny outputs (e.g. Laclau/Mouffe Hegemony: 456 chars for a 300-page book)
while raw `pymupdf.page.get_text()` recovered 591K chars (1300× more).

**Why we didn't catch it earlier:** pymupdf4llm returns *plausible*
markdown. There's no error, no warning — just nearly empty output that
passed the extraction_manifest without flagging. The `density` heuristic
in `build_corpus_subsample.py` saturates at 1.0 on small inputs, so it
didn't catch the issue at corpus-build time either.

**Mitigation in place:** `scripts/reextract_broken_pdfs.py` runs the
raw-pymupdf fallback and has fixed the 7 known bad extractions for v004.

**Still at risk:** Any future PDF added to `data/raw/` will go through
`data/extract.py` → pymupdf4llm and may silently lose content. No
guard currently catches this. Candidate check: flag any extraction
where `extracted_chars / pdf_page_count < 500` for manual review.

**Linked:** [corpora.md § Source expansion work](corpora.md)

---

## 2026-04-17 — Per-source density heuristic is fully saturated

**What we saw:** Every source in v003 and v005 reports
`min_density_kept=1.0, mean_density_kept=1.0`. The 3-gram-novelty
heuristic that's supposed to filter low-value paragraphs is not
discriminating anything.

**Why:** At ≤100 paragraphs per source, nearly every 3-gram is unique
within the source, so novelty → 1.0. The heuristic only bites at
larger scales.

**Consequence:** we're not actually curating at the paragraph level
within a source for small-corpus runs. The `stratification=sqrt`
sampling is doing most of the work; the density filter is decorative.

**Not blocking** for current experiments — but if we ever need a
sharper "which paragraphs from this book are most worth keeping"
signal, this has to be rewritten. Candidate replacement: perplexity
under a pre-trained LM, or a TF-IDF-based domain-term score.
