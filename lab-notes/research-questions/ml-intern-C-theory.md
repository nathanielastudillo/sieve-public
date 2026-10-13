---
title: ml-intern × SIEVE — theory / philosophy fit
author: Agent C (parallel research)
date: 2026-04-21
---

# ml-intern × SIEVE — theory / philosophy fit

## 1 — core theses, one sentence each

- **SIEVE.** Ensembles approximate Solomonoff induction better than single models; therefore put evolutionary pressure on *ensemble composition* (not individuals), with explicit diversity as a fitness term, on a population of fine-tuned / merged / from-scratch members run in parallel across a cluster.
- **ml-intern.** Wrap a capable LLM (default `anthropic/claude-sonnet-4-5`, swappable to Opus 4.6) in a 300-iteration agentic loop with HuggingFace-ecosystem tools — docs, repos, datasets, papers, jobs, GitHub code search, sandbox exec — so it can *research-read-write-ship* ML code end-to-end with an approval gate on sensitive operations. It is an **autonomous ML engineer**, not an optimizer. There is no fitness function, no population, no ensemble, no explicit search — the "selection signal" is the LLM's own judgement, informed by whatever docs/papers it retrieved.

## 2 — complementary or competing?

They operate at different altitudes, so mostly complementary:

- ml-intern answers *"what experiment should I run next, and how do I wire it up?"* — it's a **planner/author** over the HF ecosystem.
- SIEVE answers *"given a hypothesis space and a cluster, which combination of members maximizes ensemble quality?"* — it's a **parallel optimizer** over the sieve lineage DB.

Reach for ml-intern when the bottleneck is *coding / boilerplate / ecosystem navigation* (e.g. "find me a non-Llama 7B base with a permissive license and wire it into `openweight/seed_roster.py`"). Reach for SIEVE when the bottleneck is *distinguishing good configurations from bad under uncertainty, at cluster scale*. They compete only if you squint and treat ml-intern's "try a LoRA config on one model, look at one eval" as a substitute for population-based selection. It isn't — it's a single draw, not a search.

## 3 — can ml-intern *propose* population members that SIEVE trains + selects on?

Yes, and this is the cleanest integration point. Mapping onto the openweight track (`docs/architecture-openweight.md` §"Gen-0"): the Gen-0 seed roster and the immigration slots at `immigration_fraction=0.25` are exactly the slot ml-intern fills. Pipeline:

1. ml-intern reads the HF leaderboard + recent papers, proposes K new seeds (bases or LoRA recipes) with a one-line rationale each.
2. Orchestrator drops them into `sieve:jobs:sft` / `sieve:jobs:merge` via existing `make_sft_job` / `make_merge_job` factories (`controller/dispatch.py`). They become immigrants in the next generation.
3. NSGA-II + Caruana cull the ones that don't pull weight. ml-intern's rationale is ignored — only `FitnessVector` survives. This is the right separation: LLM proposes, empirical signal disposes.

This beats letting ml-intern pick *the* member, because one-shot LLM judgement exhibits mode collapse on popular bases (Llama-3.1-Instruct everywhere); SIEVE's pretraining-diversity mandate (openweight doc §"What's distinctive") is specifically designed to resist that.

## 4 — can SIEVE be ml-intern's inner loop?

Also yes, with a narrower role. ml-intern's "fine-tune llama on my dataset" example is a single training run terminated by the model's own satisfied judgement. If you replace that with "submit one SIEVE generation and return the `best_ensemble.json` + Pareto front," ml-intern gets something it currently lacks: a **principled serving artifact** rather than a single checkpoint. The exchange is: ml-intern gives up its ability to declare victory after one run; it gains an ensemble spec with a defensible selection story. This is worth doing for any deliverable where robustness / calibration matters (Lever policy QA, legal statute parsing) and not worth doing for one-off code generation tasks.

## 5 — strongest objection

The LLM-driven selection signal inside ml-intern is exactly the signal SIEVE's design argues against. If ml-intern is allowed inside the evolutionary inner loop — reading partial eval results and pruning members mid-generation — it will (a) collapse to whatever the base LLM thinks is "a good model" (mode collapse on Llama/Qwen), (b) prune members that look bad individually but are ensemble-load-bearing (Krogh-Vedelsby ambiguity: a mediocre complementary member is the *point*), and (c) overfit to the small set of benchmarks it can reason about verbally, at the expense of the diversity term in `FitnessVector`. `lab-notes/findings.md` 2026-04-18 flagged this same class of risk for val-set meta-overfitting over 30–50 generations; an LLM-in-the-loop makes it worse, not better, because the LLM's prior is much stronger than the val set's.

Second-order concern: ml-intern's 300-iteration-per-session budget and human-approval gates are cadence-incompatible with a cluster run that wants to complete 10+ generations unattended overnight.

## 6 — recommendation

**Adopt as an edge tool, not an inner-loop component.** Use ml-intern for (a) Gen-0 seed-roster expansion and immigrant proposals between generations — it reads papers, we can't cheaply, (b) code scaffolding around new benchmarks / new merge recipes / HF job submission, (c) post-hoc analysis against `results/lineage.db` and the Pareto front (the dashboard at port 8096 would be a good fit — give ml-intern read-only SQL + natural-language report-out). Explicitly forbid it from inside `controller/orchestrator.py`'s selection path, from writing to `members` / `fitness_evals`, or from choosing which members to cull. The integration surface is "ml-intern authors new experiments; SIEVE runs and selects over them." Lightweight first step: a `scripts/ml_intern_propose_immigrants.py` that calls ml-intern headless, validates the returned JSON against the `OpenweightMember` dataclass, and enqueues via the existing dispatcher. One afternoon of work; no changes to the evolutionary loop.
