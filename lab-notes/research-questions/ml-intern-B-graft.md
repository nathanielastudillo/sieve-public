# ml-intern ↔ SIEVE graft analysis (Agent B)

Drafted 2026-04-21. Companion to `ml-intern-A-*.md` (capability survey) and
`ml-intern-C-*.md` (cost analysis).

## What ml-intern actually is

An **LLM-in-the-loop planner agent** — not an evolutionary framework, not a
training framework. `agent/agent_loop.py` drives a litellm completion loop
(max 300 iters, 170k auto-compaction) over a `ToolRouter` (`agent/tools.py`)
whose headline tools are the HF MCP server (papers, datasets, repos, jobs),
GitHub code search, and sandboxed bash/file ops. One invocation = one
user-prompted task (`ml-intern "fine-tune llama on my dataset"`). Default
model is Claude Opus 4.6 per `configs/main_agent_config.json`. "Training"
means it eventually emits a single-job submission to **HF Jobs** with
approval gating; it has no notion of population, ensemble, lineage, or
cluster topology.

That framing is the load-bearing fact for everything below.

## 1. Graft point in SIEVE's architecture

ml-intern does **not** replace `controller/population.py`. That file
(`select()` at `controller/population.py:210`, `mutate_config()` at `:127`,
`_find_complementary_pairs()` at `:331`) implements PBT-style
*perturbation-within-a-fixed-search-space* driven by ensemble-appearance
counts — a different problem from "choose a base model + dataset +
hyperparameter priors from the open world."

The natural graft point is **one level upstream**: the hand-rolled
`build_sft_config` (`controller/openweight_orchestrator.py:81-87`, which
just stamps `DEFAULT_SFT_HYPERS`) and the static `pick_roster` in
`openweight/seed_roster.py:217`. ml-intern is a good fit for producing:
- the Gen-0 seed roster (which 8–12 open-weight bases to include)
- the initial `SftConfig` hyperparameter priors per seed
  (`openweight/genome.py:31`)
- candidate data shards (HF datasets to compose into the corpus shard that
  `openweight_orchestrator.py:301` currently receives as a path argument)

Everything downstream — Redis dispatch, NSGA-II, Caruana-greedy ensemble
eval, merge genetics — keeps running unchanged.

## 2. What happens to composition-fitness if we bolt ml-intern on naively

ml-intern's reward signal is "did the single model improve on the
benchmark?" — whatever the LLM decides from the HF Jobs result log.
SIEVE's signal is **ensemble-contribution**: a mediocre member that's
maximally complementary wins over a strong but redundant one
(`controller/ensemble_eval.py:494` Caruana greedy, with
`logit_mean_smoothed` aggregator at `:285`, ε=0.05).

If ml-intern owns the outer loop it will converge on "the best single
model" and kill the diversity premise. The composition-fitness loop
**doesn't break mechanically** — Caruana still runs on whatever it's
given — but it becomes a post-hoc report over a mono-cultural population.
That's the Solomonoff-insight failure mode we explicitly set up SIEVE to
avoid (project `CLAUDE.md` §"The Novel Piece").

To keep composition-fitness alive under ml-intern control you'd have to
**expose `FitnessVector.bench_agg` + diversity** (`openweight/genome.py:88`)
as ml-intern's observable reward, and teach it that its proposals are
evaluated at the ensemble level. That's a prompt-engineering job plus a
tool wrapper — feasible, but the agent no longer resembles stock ml-intern.

## 3. Minimum viable adoption

Take **only** the HF MCP server access + dataset/paper search. ~300 LOC
of wrapper:

- One `controller/research_intern.py` that calls litellm with the
  `hf-mcp-server` tool (config already documented in `main_agent_config.json`)
- Two targeted prompts: "propose seed roster for domain X" and "propose
  SFT data shards for domain X"
- Output schema: `list[SeedModel]` (matching `openweight/seed_roster.py`'s
  `SeedModel` at approximately `:1-60`) + `list[dict]` (HF dataset ids +
  sampling ratios)
- Validation: schema-check the LLM output, pin the set, hand to the
  existing `run_gen0` at `openweight_orchestrator.py:298`

No changes to `controller/`, `worker/`, or the lineage DB. The 12-node
Redis cluster stays the source of truth. ml-intern's 300-iter agent loop,
doom-loop detector, and HF Jobs submission plumbing are not needed —
we're just borrowing the HF-catalog-traversal skill.

This is the version I'd actually build.

## 4. Maximum adoption — ml-intern as outer loop

"ml-intern calls SIEVE-generate(roster, benchmarks) and iterates on the
output" — what breaks:

1. **Wallclock mismatch**. One SIEVE generation is 45–90 min
   (`docs/architecture-openweight.md` §"Blocking vs. parallelizable").
   ml-intern's 300-iter LLM loop can't usefully chaperone that —
   auto-compaction at 170k will trigger mid-generation, and the agent
   loses the run-state that isn't in Redis.
2. **Dispatch model mismatch**. ml-intern submits to **HF Jobs**
   (single-node, cloud-billed). SIEVE runs on 12 GB10s behind a Redis
   coordinator on the <lan> segment (`CLAUDE.md` §"Known stack
   gotchas"). These are incompatible backends; you'd have to write an
   "HF Jobs wrapper → sieve:jobs:sft queue" adapter and re-educate every
   ml-intern prompt that assumes HF Jobs semantics.
3. **Reward mismatch**. NSGA-II over 5 objectives
   (`docs/architecture-openweight.md` §"Generation loop" step 5)
   ≠ "LLM reads a log and says 'better'". You'd have to project the
   Pareto front into a scalar ml-intern can reason about, which defeats
   the multi-objective point.
4. **Autonomy mismatch**. ml-intern defaults to approval-gated CPU jobs
   (`yolo_mode: false` in the shipped config). SIEVE cluster runs are
   expected to be overnight-autonomous. Every approval pause stalls 12
   workers.
5. **Reproducibility mismatch**. Lineage DB (`controller/lineage.py`)
   expects bit-exact replayable configs; ml-intern's choices are
   LLM-stochastic. We'd lose the "rewind to any point" invariant that
   `CLAUDE.md` §"Versioning & Git Strategy" treats as non-negotiable.

Verdict on max adoption: not worth it.

## 5. What SIEVE has that ml-intern conspicuously lacks

- 12-node GB10 cluster + Redis multi-queue dispatch (`controller/dispatch.py`
  — `SftJob`/`MergeJob`/`EvalJob`, 649 LOC)
- Multi-role worker daemon (`worker/daemon.py` `--roles train,sft,eval,merge`)
- Caruana-greedy-with-replacement ensemble selection + ε-smoothed
  logit-mean aggregator (`controller/ensemble_eval.py:494`, `:285`) —
  this is shipped, battle-tested through exp007, and ml-intern has no
  analogue
- Model-merging as a first-class genome kind
  (`openweight/genome.py:52` `MergeRecipe`, `:69` `LoraMergeRecipe`)
  with CMA-ES slot for coefficients
- Multi-objective fitness + NSGA-II design (`FitnessVector` at
  `openweight/genome.py:88`)
- Lineage DB with per-benchmark rows (`controller/lineage.py`,
  `fitness_evals` table per `architecture-openweight.md` §"Population storage")
- Resume-safe orchestration via Redis-result-scan
  (`openweight_orchestrator.py:112-149`) — saved 170 GPU-hours in exp007
- Parallel-logit-averaging inference harness (`lab/server.py`,
  per project `CLAUDE.md`)
- Magikarp data-quality tripwire (`scripts/magikarp_check.py`) —
  empirically necessary after 2026-04-17 discovery

What ml-intern would **gain** from any of these: a real training target
beyond single-node HF Jobs. If HF ever wanted ml-intern to propose
ensembles, Caruana + logit-mean-smoothed is the library function to
borrow; it's 200 LOC in `ensemble_eval.py` and already solves the
numerically-annoying problems (the ε-smoothing was added because vanilla
logit-mean blows up when any member assigns ~0 probability to truth —
this is not obvious from scratch).

## 6. Honest verdict

**Complement, not compete, and neither is a superset.** Different layers:

- ml-intern is a **research-assistant / config-proposer**: English →
  (base model, dataset, recipe) using HF catalog knowledge.
- SIEVE is a **breeding program**: (roster, config space) → diverse
  ensemble via population-parallel training + composition-aware selection
  on a fixed cluster.

Gen-0 is the natural handoff seam. ml-intern stops where
`pick_roster()` would have started. Everything from `run_gen0`
(`openweight_orchestrator.py:298`) onward is SIEVE's job and has no
ml-intern counterpart.

The minimum viable adoption (§3) is the only one I'd actually recommend:
steal the HF MCP access for roster + shard proposal, ignore the rest.
