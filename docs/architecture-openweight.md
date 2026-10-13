# SIEVE openweight track — architecture

Drafted 2026-04-18. Companion doc to `lab-notes/research-questions/literature-openweight-ensembles.md`.

This document describes the SIEVE *openweight* track — an evolutionary
ensemble trainer over fine-tuned 7B–13B open-weight bases. It is
additive to the existing small-corpus track (`controller/`, `worker/`,
`model/`) and reuses the same lineage DB, Redis dispatcher, and
evolutionary-loop shell.

## Goal

Produce a **diverse ensemble of fine-tuned open-weight members** that,
served in parallel across the 12-node compute-cluster (GB10 ARM
Blackwell, ~100GB accessible VRAM per node, CX-7 200G fabric), is SOTA
on the Lever domain benchmarks (Canadian policy QA, polling/survey
reasoning, constraint extraction, claim-to-evidence linking).

The 12 GB10 nodes turn member-level parallelism from a cost into a
given: one 7B per node fits with headroom, so output-space ensembling
is nearly free of extra memory. This is the architectural lever the
track is built around.

## What changes vs. the small-corpus track

| Concept              | Small-corpus (existing) | Openweight (new) |
|----------------------|-------------------------|------------------|
| Member               | ~7M-param GPT from scratch | Open-weight base + LoRA adapter, OR a merge of parents |
| Training data        | 85K tokens, custom BPE  | ~50–200M tokens Lever domain, base tokenizer |
| Mutation op          | Hyperparameter perturbation + retrain | LoRA/DoRA fine-tune on perturbed config + data shard |
| Crossover op         | Config blending of parents | **Model merging** (TIES / DARE / SLERP / Task Arithmetic) via `mergekit` |
| Fitness              | Held-out val NLL         | Multi-objective: bench score, diversity, calibration, latency, memory |
| Selection            | Caruana greedy ensemble selection | NSGA-II over Pareto front + Caruana on the final survivor set |
| Inference target     | One model at a time      | 1–12 models in parallel on GB10s, logit-averaged within tokenizer family |
| Diversity source     | Random init + hyper variation | Random init + LoRA rank/data-shard variation + merge-recipe variation + cross-family |

Reused as-is: `controller/lineage.py`, `controller/dispatch.py`,
`controller/ensemble_eval.py` (once we wire a logit-alignment path for
cross-tokenizer members), and the sequestered meta-val machinery from
2026-04-18.

## Genome representation

A population member is a **tagged union** of three kinds. Flat
dataclass, serialized as JSON into the lineage DB:

```python
@dataclass
class OpenweightMember:
    id: str                           # uuid4 hex[:12]
    generation: int
    kind: Literal["sft", "merge", "lora_merge"]
    parent_ids: list[str]             # empty for immigrants
    lineage: list[int]                # generation ancestors (cheap ancestry walk)

    # Shared
    tokenizer_family: str             # "llama3" | "qwen2" | "gemma2" | "mistral"
    base_model_id: str                # HF id of the base (identical across merge's parents
                                      # for PS merging; see §1.2 of research doc)

    # Kind-specific fields — at most one is non-null
    sft_config: SftConfig | None      # LoRA/DoRA hyperparams + data shard + seed
    merge_recipe: MergeRecipe | None  # mergekit YAML dict with coefficients (post-CMA-ES)
    lora_merge_recipe: LoraMergeRecipe | None  # adapter-level merge

    # Populated after evaluation
    adapter_path: str | None          # where the fine-tuned adapter lives (for sft kind)
    merged_checkpoint_path: str | None  # where the merged model lives (for merge kind)
    fitness: FitnessVector | None
```

**Why tagged union over the brief's flat tuple**: a merged member has
no `sft_config` and no single "base_model_id" in the LoRA sense (its
base is a composite of parents). A fine-tuned member has no merge
recipe. Collapsing both into one flat tuple with nullable fields makes
serialization ambiguous and validation brittle. Tagged union makes
intent explicit: every consumer pattern-matches on `kind`.

`FitnessVector` is a flat dict of the multi-objective scores so NSGA-II
sees a uniform vector:

```python
@dataclass
class FitnessVector:
    bench_agg: float       # weighted-sum benchmark aggregate (higher = better)
    diversity: float       # contribution to population JSD (higher = more diverse)
    calibration_ece: float # expected calibration error on MCQ bench (lower = better)
    latency_ms: float      # tokens/s inverse at 7B (lower = better)
    memory_gb: float       # peak VRAM at inference (lower = better)
    # Per-benchmark breakdown kept separately for auditing
```

Per-benchmark scores live in an auxiliary `eval_results` table so the
aggregate can be recomputed with different weights without re-evaluating.

## Population storage

**Lineage DB** (existing `results/lineage.db`, SQLite, extend schema):

- `members` — adds `kind`, `tokenizer_family`, `base_model_id`,
  `adapter_path`, `merged_checkpoint_path` columns. Existing
  `config_json` holds the kind-specific config blob.
- `fitness_evals` — new table. `(member_id, benchmark_name,
  score, wallclock_s, evaluated_at)`. One row per (member,
  benchmark) pair so surrogate-assisted selection can query partials.
- `merge_recipes` — new table. `(recipe_id, yaml, cma_es_step,
  coefficients_json)`. Recipes are small enough to store inline.

**Artifact storage** (outside SQLite):

- **LoRA adapters** (~50–400 MB at rank 8–64 for 7B): `corpus-host:/data/lever-evo/adapters/{member_id}/`.
  Small, rsync-cheap, fine to replicate to any training or eval node on demand.
- **Merged checkpoints** (~15 GB at 7B, ~28 GB at 13B): kept on the
  worker-node local disk that produced them. Orchestrator tracks
  `(node, path)` provenance in `members.merged_checkpoint_path` as
  `worker_id:/abs/path/file.safetensors`. Pulled to an eval node only
  when that member is up for fitness evaluation. The 200G fabric means
  a 7B model transfers node-to-node in ~60s — acceptable overhead.
- **Shared base models**: each node maintains a local HF cache under
  `/data/hf_cache/`. First use triggers a single cluster-wide download
  (seeded from a cache-warmup script); subsequent members reuse.

**Never** store merged-model weights in basin/Redis — they're too big
and have no query pattern that benefits from a KV store.

## Generation loop

For generation *g*:

1. **Selection** (fast, controller-local).
   Input: population at generation *g−1* with fitness vectors.
   Output: parent pool for reproduction. NSGA-II picks the Pareto-
   non-dominated front; fill survivors with crowding-distance tiebreak.

2. **Reproduction** (fast, controller-local):
   - *Mutation slots*: pick elites, perturb their `sft_config` or
     `merge_recipe` coefficients, emit new `sft`/`merge` children.
   - *Crossover slots*: pick parent pairs from distinct
     lineage branches, emit a `merge` child with a mergekit recipe
     seeded from the Sakana CMA-ES schedule (§1.1 of research doc).
   - *Immigration slots*: sample fresh configs at
     `immigration_fraction=0.25` (carried over from small-corpus
     findings, Fort 2019).

3. **Execute jobs** (parallel, compute-heavy). Three job kinds go
   through the existing Redis dispatcher with distinct queues:
   - `sieve:jobs:sft` — fine-tune a base with a LoRA/DoRA config.
     Worker pulls base from HF cache, runs axolotl, saves adapter,
     returns adapter_path.
   - `sieve:jobs:merge` — run mergekit with a recipe, emit merged
     safetensors. Runs on one GPU node (can be the orchestrator node
     for small merges) or CPU if the recipe is simple (Task Arith,
     SLERP). Returns merged_checkpoint_path.
   - `sieve:jobs:eval` — evaluate a member against a named benchmark.
     Worker loads the adapter-on-base (or merged model) with sglang,
     runs the benchmark suite, returns per-benchmark scores.

   All three kinds are **embarrassingly parallel** across members.
   Within a single eval, the benchmark suite is sharded across
   benchmarks (one member × one benchmark per job), not within a
   benchmark, to keep the job granularity uniform.

4. **Fitness aggregation** (fast, controller-local). Compose the
   `FitnessVector` from fresh eval rows. Diversity is computed
   population-wide on a held-out prompt set (§1.7 of research doc) and
   attributed per-member as that member's contribution to average
   pairwise JSD.

5. **Survival** (controller-local). NSGA-II over the combined
   parents+children set; keep `population_size` survivors. Record
   generation summary to lineage.

**Blocking vs. parallelizable**:
- Fully parallel per-member: mutation, merge, eval (all step 3).
- Single-threaded controller work: selection, reproduction,
  aggregation, survival — <1 min per gen at pop=16.
- Critical path: `t_gen ≈ max(sft_wallclock, eval_wallclock)` when
  SFT+eval are pipelined across generations. Ballparks:
  - LoRA fine-tune on ~10M tokens shard, 7B base, r=16, one GB10: ~20–40 min.
  - Benchmark eval (lm-eval 5 tasks + MT-Bench + 1 custom): ~15–45 min per member with sglang.
  - Merge wallclock: 2–10 min (dominated by safetensors I/O).

At pop=16, one generation lands around **45–90 min** if SFT and eval
queues are both ~12-way parallel.

## Compute scheduling

Extend `controller/dispatch.py`:

- `make_sft_job(...)`, `make_merge_job(...)`, `make_eval_job(...)` factory functions mirroring the existing `make_job`.
- Worker daemons declare **resource tags** at registration:
  `{"cuda": true, "vram_gb": 110, "roles": ["sft", "merge", "eval"]}`.
  Orchestrator routes jobs to matching queues. Low-VRAM nodes can
  still run `eval` for small (≤7B) members; CPU-only nodes can run
  simple merges (SLERP, Task Arith).
- **Reservation policy**: at most one SFT job per node at a time
  (saturates GPU). Eval jobs share nodes with completed-SFT idle
  members if VRAM permits (sglang handles the swap).
- **Surrogate screening** (from §1.8): before committing a member to
  the full benchmark suite, run a cheap proxy (500-question MCQ
  subsample + calibration-only pass). Dropouts never reach the
  expensive evals. Proxy runs on the same queue with a `proxy=True`
  flag; workers prioritize proxies to keep the funnel flowing.

## Checkpointing and resumption

Generation state is persisted to lineage after each step so a killed
orchestrator can resume from the last completed step:

- `members` table: every member whose adapter_path or
  merged_checkpoint_path is non-null is recoverable.
- `fitness_evals` table: partial evaluations are fine-grained (one row
  per benchmark), so a resumed run picks up mid-generation.
- Redis result keys have 7-day TTL (existing policy); unread results
  are reconciled on resume by reading `sieve:results:{run_id}:*`
  before enqueuing anything new. This is the same recovery pattern
  that saved 170 GPU-hours in exp007_v009 (devlog 2026-04-18).

## Inference — parallel serving across the GB10 cluster

The final artifact is not a single model. It is an **ensemble spec**
plus a serving plan. Serving modes, in order of preference:

1. **Same-tokenizer-family, within-cluster output ensembling** — each
   member runs on its own GB10 under sglang, a thin coordinator
   broadcasts the prompt over the CX-7 fabric, collects logits per
   token, averages (or logit-mean-smoothed, matching the small-corpus
   aggregator choice), resamples. Token latency dominated by the
   slowest member + one all-reduce per token. This is the default.
2. **Merged single-model inference** — when the evolutionary loop
   stabilizes on a cluster of highly-similar members, merge them
   in-place (Model Soup style) and serve the merged model on one node.
   Drops ensemble cost; accepts the merge-gap. Decided per-cluster
   during survival (the cluster's CKA-radius below a threshold → soup
   them).
3. **Cross-family text-level combination** — once within-family
   populations plateau, introduce cross-family members via LLM-Blender-
   style GenFuser or Mixture-of-Agents multi-round combination (§1.4
   of research doc). Expensive (5–10× latency) but gives a route to
   meaningfully higher quality when the within-family ceiling is hit.

Mode 1 is the steady state. Mode 3 is an upgrade path for the final
deployment when latency budget allows.

## Integration with adjacent systems

The openweight track is designed to touch neighboring systems only at
clean interfaces:

- **Corpus (inputs)** — fine-tuning data shards are produced by an
  upstream pipeline that reads from `lex:*` (regulatory corpus),
  `pig:*` (entity/policy corpus), and Lever polling datasets, and
  writes versioned shards to `corpus-host:/data/lever-evo/corpus_v{NNN}/`.
  The openweight track consumes a shard path and does not own the
  upstream crawl/clean/tokenize pipeline. (Matches SIEVE's existing
  "corpus is an input, not an output" stance.)
- **Bus** — generation boundaries and big milestones posted to the
  coordination bus so other sessions see what's running.
- **Basin (Redis)** — already used as the dispatch queue; no new usage.
- **Horse / swarm** — out of scope for this track.
- **pig** — a downstream consumer: the winning ensemble becomes a pig
  analysis tool. Integration happens in pig, not here.

## Generation 0 — the "get SOTA seeds + fine-tune + ensemble-bench" recipe

The user-facing framing of the openweight track:

> Get SOTA open-weight models that fit on a single node, fine-tune an
> ensemble of different, best-benched models to max diversity and
> converge each on the corpus domain, evaluate the ensemble against
> the benchmark, iterate — taking advantage of the cluster parallelism.

This is exactly what Gen-0 executes, with zero evolutionary operators
yet:

1. **Pick a seed roster.** `openweight/seed_roster.py` provides two:
   `llama3_within_family` (12 Llama-3.1-tokenizer models — output-
   ensembling works with no alignment machinery) and `cross_family`
   (4 tokenizer families — higher diversity, needs EVA at inference).
   Default Gen-0 = `llama3_within_family` for the cheap-win first
   iteration; upgrade to cross-family once within-family plateaus.

2. **One SFT job per seed, fanned out across the 12 GB10s.** Each
   worker pulls its assigned base from `/data/hf_cache`, runs
   `openweight.finetune.fine_tune` with a LoRA config on the same
   Lever corpus shard (different seed + slight rank/lr perturbation
   per member for hyperparameter diversity on top of pretraining
   diversity). Adapter saved to `corpus-host:/data/lever-evo/adapters/{id}/`.

3. **Ensemble-bench.** Each member registered as a sglang endpoint
   on its node. The coordinator (mode 1, within-family output
   ensembling) averages logits per token across all K≤12 members on
   every benchmark prompt. Per-member scores are kept too (for
   diversity attribution) but the **ensemble** score is the fitness
   signal.

4. **Survival.** Even Gen-0 does NSGA-II selection over the 12 members
   to drop dead weight. A member that the ensemble never uses (zero
   Caruana weight) is culled before Gen-1.

5. **Gen-1 onwards** introduces mutation (perturb LoRA hypers + re-SFT),
   crossover (LoraHub merge of two elite adapters), and immigration
   (new seed from the roster or fresh HF leaderboard pick).

**Single-generation wallclock at this design**: 12 SFTs × ~30 min
(LoRA on a 10M-token shard, one GB10 per member, parallel) + 12×
benchmark-eval × ~30 min + ~10 min coordinator overhead ≈ **60–80 min**.
Gen-0 is the longest generation because it has no warm-cache adapters to
inherit; later generations amortize.

**What's distinctive about this design vs. "just stack LoRAs on a
mono-culture of Llama-3.1-Instruct":** the seed roster forces
pretraining-level diversity between members before any fine-tuning
happens. Twelve copies of the same base with twelve LoRA seeds
produces a mono-basin ensemble (Model-Soup-compatible but
Krogh-Vedelsby-low on ambiguity). Twelve DIFFERENT fine-tuned seeds
produces a true ensemble where members disagree for substantive,
pretraining-rooted reasons — which is where ensemble gain actually
comes from.

## Testing strategy

- **Smoke path**: one-member, one-benchmark, one-GB10. Reuse the same
  dispatcher harness that runs the small-corpus smokes; swap
  `make_job` → `make_sft_job` + `make_eval_job`.
- **Plumbing validation**: pop=4, eval_benchmarks=[ifeval, hellaswag],
  2 generations on 4 cluster nodes. End-to-end in <2h. Mirrors the
  corpus_v003 smoke pattern.
- **First real run**: pop=8–12, eval_benchmarks={ifeval, mmlu-pro,
  gpqa-diamond, mt-bench, one custom Lever bench}, 10 generations.
  Compute budget: 12-node-day. Deliverable: a Pareto front of
  ensemble candidates + a single recommended serving spec.

## Open questions (surface, not resolve — see research doc for depth)

1. Does evolving LoRA adapters directly (LoraHub style) beat evolving
   merged full weights, given cheaper search? The research doc
   §1.5 expects yes; first cluster run should A/B it.
2. Same-family-only vs. cross-family populations — what's the right
   staging? (§1.4 recommendation is within-family early, cross-family
   late, but no empirical validation at our scale.)
3. At pop=16, does NSGA-II with 5 objectives become needle-in-
   haystack? §1.8 discusses NSGA-III + reference points as the
   fallback; threshold where we switch is unclear.
4. Is the parallel-logit-averaging serving mode actually
   latency-competitive with a single-model alternative after all
   coordination overhead? This is a direct benchmark we should run
   before committing to mode 1 as the steady state.
5. How much of SIEVE's existing Caruana-greedy selector transfers?
   It currently selects on val NLL, a scalar; with a multi-objective
   fitness vector and expensive evaluations, the role it plays may
   shrink to "final aggregator over a NSGA-II-surviving population".
