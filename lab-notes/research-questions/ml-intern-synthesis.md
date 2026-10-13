---
title: ml-intern × SIEVE — synthesis + recommendation
date: 2026-04-21
sources: ml-intern-A-architecture.md, ml-intern-B-graft.md, ml-intern-C-theory.md
---

# ml-intern × SIEVE — synthesis

Three parallel research agents audited HuggingFace's ml-intern
(announced 2026-04, github.com/huggingface/ml-intern) against SIEVE.
All three converged on the same recommendation. This doc is the
merged view; the companion files are the primary sources.

## What ml-intern actually is (Agent A)

- ~12K LOC Python. **LiteLLM tool-calling ReAct loop** over a
  ToolRouter, max 300 iters. Not smolagents, not LangGraph.
- **Default model**: `anthropic/claude-opus-4-6` via LiteLLM; agent
  loop is at `agent/core/agent_loop.py:461`.
- **Headline tool**: a paper-research sub-agent
  (`agent/tools/research_tool.py:457`) with Semantic Scholar
  citation-graph integration including `isInfluential`, `intents`,
  and snippet search over 12M+ papers. That's the genuinely novel
  contribution.
- **Dataset "quality judgment" doesn't exist in code.** The
  healthcare synthetic-data story is the LLM reading 3 sample rows
  from `hf_inspect_dataset` (`agent/tools/dataset_tools.py:35-145`)
  and deciding qualitatively. Any quality claim is unaudited.
- **Training launcher**: HF Jobs only, no local-GPU fallback. Thin
  wrapper over `huggingface_hub.HfApi.run_job`. Approval-gated,
  retries streaming logs up to 100 times, filters uv-install spam.
- Everything else (doom-loop detector, auto-compaction, sandbox
  via duplicated HF Space, MCP via fastmcp) is standard 2025
  agent-plumbing — ~2K LOC of assembly any competent team can do.

## Where it could graft into SIEVE (Agent B)

- **Does not replace** `controller/population.py`. PBT-style
  perturbation in a fixed search space is a different problem from
  "pick a base model + dataset from the open world."
- **Natural seam**: one level upstream at
  `controller/openweight_orchestrator.py:298` (run_gen0) — ml-intern
  supplies the Gen-0 seed roster + initial `SftConfig` priors +
  candidate data shards. Everything downstream (Redis dispatch,
  NSGA-II, Caruana, merge genetics) runs unchanged.
- **Minimum viable adoption**: ~300 LOC wrapper in
  `controller/research_intern.py` that calls LiteLLM with the
  hf-mcp-server, two prompts (propose roster / propose shards),
  schema-validates the JSON output, hands to `run_gen0`. This is
  the version worth building.
- **Maximum adoption (ml-intern as outer loop)**: breaks on five
  axes — wallclock (one SIEVE gen ≥ 45min vs 300-iter LiteLLM
  budget), dispatch (HF Jobs cloud-single-node vs our Redis-on-12
  -GB10-cluster), reward (scalar-from-log vs 5-objective
  NSGA-II Pareto), autonomy (approval-gated CPU jobs vs overnight
  cluster runs), reproducibility (LLM-stochastic vs replayable
  lineage DB). Not worth it.

## Theory fit (Agent C)

- **Different altitudes, mostly complementary**: ml-intern is a
  planner/author over the HF ecosystem; SIEVE is a parallel
  optimizer over a cluster. They compete only if you treat
  "try one LoRA config, look at one eval" as a substitute for
  population-based selection — which it isn't.
- **Strongest objection to letting ml-intern inside the selection
  loop**: the LLM's prior is much stronger than the val set's, so
  it will mode-collapse on popular bases (Llama/Qwen everywhere)
  and prune ensemble-load-bearing-but-individually-mediocre
  members. That's the Krogh-Vedelsby ambiguity term SIEVE's
  composition-fitness is specifically designed to preserve.
- **Second-order concern**: ml-intern's 300-iter/approval-gate
  cadence is incompatible with overnight-autonomous cluster runs.

## Convergent recommendation

**Adopt as an edge tool, not an inner-loop component.**

Build: `scripts/ml_intern_propose_immigrants.py` — calls ml-intern
headless, asks for K new seeds (bases or LoRA recipes) with
one-line rationales, validates JSON against the existing
`SeedModel` / `SftConfig` dataclasses, enqueues via
`controller/dispatch.py`. One afternoon of work. No changes to
the evolutionary loop.

Good use cases:
1. **Gen-0 seed roster expansion** — ml-intern reads papers, we
   can't cheaply.
2. **Immigrant proposals** between generations — feeds the
   `immigration_fraction=0.25` slot with provenance-tagged picks.
3. **Post-hoc analysis** over `lineage.db` — read-only SQL +
   natural-language report-out on the dashboard.
4. **Scaffolding** around new benchmarks / merge recipes / HF
   job submission (one-off code gen).

Explicitly forbid:
- Writing to `members` / `fitness_evals`.
- Pruning or selecting population members.
- Inside `controller/orchestrator.py`'s selection path.

## Pieces worth stealing outright

- **Semantic Scholar citation-graph walker** with `isInfluential`
  + `intents` edges (`agent/tools/papers_tool.py:40-129`). Useful
  anywhere we need to expand a literature search from a single
  anchor paper — e.g. Bucket B planning in
  `lab-notes/experiments-planned.md`.
- **Snippet search over 12M papers**
  (`agent/tools/papers_tool.py:1046-1085`). Better than arxiv
  full-text search for targeted passage retrieval.
- **ChatML format auto-check** in `hf_inspect_dataset` — catches
  `messages` vs `prompt/completion` KeyErrors before a GPU spins up.
  Currently `data/build_sft_shard.py` doesn't self-check; adding
  the same check is ~20 LOC.

## What SIEVE has that ml-intern gains nothing from us adopting

- Caruana-greedy-with-replacement + ε-smoothed logit-mean
  aggregator (`controller/ensemble_eval.py:494`, `:285`) — if HF
  ever wanted ml-intern to propose ensembles, this is the 200 LOC
  library function to borrow. The ε-smoothing (α=0.05) is not
  obvious from scratch — plain logit-mean blows up when any
  member assigns ~0 probability to truth.
- Resume-safe orchestration via Redis-result-scan
  (`openweight_orchestrator.py:112-149`) — saved 170 GPU-hours
  in exp007. ml-intern has no analogue.
- Magikarp data-quality tripwire (`scripts/magikarp_check.py`)
  — empirically necessary after 2026-04-17. ml-intern's
  qualitative 3-row dataset read would have missed this.

## Next step

Log `scripts/ml_intern_propose_immigrants.py` as a Bucket B
experiment (lab-notes/experiments-planned.md). Gate it on a
clean Gen-0 run of the openweight track (this experiment, 005)
landing a positive delta — no point proposing more seeds until
the baseline pipeline is healthy.
