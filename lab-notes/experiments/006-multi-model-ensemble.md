# Experiment 006 — Multi-model ensemble, pre/post SFT comparison

**Status:** running (dispatched 2026-04-22)
**Track:** openweight
**Run ID:** see `runs/exp006_sota_q2_01/`
**Dashboard:** http://node-8:8096 (tunnel locally)

## Goal

Measure two things on one run:
1. **Does ensembling help at all?** — majority-vote ensemble score vs. best-individual score, both pre-SFT and post-SFT.
2. **How much does SFT-on-v009 move the needle?** — best-individual and ensemble deltas, pre → post.

## Why now

Previous openweight smoke (exp 005) trained a single `Qwen2.5-1.5B-Instruct` → merged LoRA + smoke_mcq. That proved the plumbing but says nothing about *ensembles*. User instruction: get an ensemble run going with pre-post bench comparison. This experiment is the first real multi-model run on the openweight track.

## Hypothesis

- **H1** (weak): a 5-model majority-vote ensemble beats the best individual on smoke_mcq (small, chance=0.25, ceiling ~0.95 for strong 8-14B bases). Expected Δ on baselines: +0.00 to +0.10 because strong bases already saturate smoke_mcq.
- **H2**: SFT lift is *comparable* or *smaller* than the ensemble gain because the corpus (v009 — clinical manuals + theory) is orthogonal to the MCQ questions. I.e. smoke_mcq is not load-bearing enough to show SFT damage-or-lift — this run is diagnostic, not definitive.
- **H3**: Cross-family ensembles (5 tokenizer families) diversify more than same-family, but majority vote caps the gain (unlike logit-averaging which needs aligned vocab).

H2/H3 imply that to get a strong signal on ensemble-vs-individual, we'll need a harder benchmark (IFEval / MMLU-Pro) wired up next. Flagged for exp 007.

## Setup

- **Roster:** `sota_2026_q2` (first 5 ungated entries)
  1. `Qwen/Qwen3-14B` (qwen3 tokenizer, 14.8B)
  2. `microsoft/phi-4` (phi4 tokenizer, 14.7B)
  3. `allenai/OLMo-2-1124-13B-Instruct` (olmo2 tokenizer, 13.7B)
  4. `HuggingFaceTB/SmolLM3-3B` (smollm tokenizer, 3.0B)
  5. `ibm-granite/granite-3.3-8b-instruct` (granite tokenizer, 8.2B)

  Five distinct tokenizer families → majority vote across *decoded answers* is unbiased; logit averaging would need EVA/DeePEn (see `docs/architecture-openweight.md` §1.4), deferred.

- **SFT config:** LoRA rank=16 on q/k/v/o, alpha=32, dropout=0.05, lr=2e-4, epochs=1, batch_size=1, max_seq_len=2048, bf16 (DEFAULT_SFT_HYPERS in `controller/openweight_orchestrator.py`).
- **Corpus shard:** `data/sft_shards/v009.jsonl` (~22K paragraph-packed rows from `data/cleaned_v009/`).
- **Benchmark:** `smoke_mcq` (20-question dev canary, `benchmarks/adapters/smoke_mcq.py`). `ifeval` and `mmlu_pro` are left for a follow-up once lm-eval is vetted on the worker stack.
- **Ensemble strategy:** `weighted_majority` — each member's vote is weighted by their individual smoke_mcq score (stronger members pull more). Ties broken lexicographically.
- **Workers:** 5 compute nodes, one model per node (node-2, node-6, node-9, node-10, node-12 confirmed reachable with `/home/<user>/sieve` installed, >150 GB free). node-8 is the coordinator (Redis + dashboard).
- **Expected wallclock:** baseline evals ~15 min each (no SFT). SFT ~1-3h each at 14B scale w/ LoRA (per earlier Qwen2.5-1.5B smoke: ~8 min; scales roughly linearly with params). Post-SFT evals ~15 min each. Total: ~3-4h end-to-end.

## Resume commands

If the orchestrator dies:
```bash
# Baseline already done? skip it
python -m scripts.run_experiment_006 \
  --redis redis://<node-ip>:6379/0 \
  --roster sota_2026_q2 --pop-size 5 \
  --corpus-shard data/sft_shards/v009.jsonl \
  --adapter-root runs/exp006_sota_q2_01/adapters \
  --lineage-db runs/exp006_sota_q2_01/lineage.db \
  --benchmark smoke_mcq \
  --strategy weighted_majority \
  --run-id exp006_sota_q2_01 \
  --skip-baseline
```

Workers (any compute node):
```bash
/home/<user>/sieve-venv/bin/python -m worker.daemon \
  --redis redis://<node-ip>:6379/0 \
  --sieve-root /home/<user>/sieve \
  --roles sft,eval
```

Posthoc ensemble re-aggregation (after a crash or to try another strategy):
```bash
/home/<user>/sieve-venv/bin/python -m controller.openweight_ensemble_eval \
  --lineage-db runs/exp006_sota_q2_01/lineage.db \
  --benchmark smoke_mcq --roster sota_2026_q2 \
  --strategy majority --generation 0
```

## Results

_To be filled in once the run completes. Template:_

```
PRE  (baseline): best_ind=?.???  ensemble=?.???  Δ=+?.???  voters=5  fams=granite,olmo2,phi4,qwen3,smollm
POST (SFT'd)   : best_ind=?.???  ensemble=?.???  Δ=+?.???  voters=5  fams=granite,olmo2,phi4,qwen3,smollm
SFT lift: best_individual +?.???  ensemble +?.???
```

## Things that would change this experiment's conclusions

- **smoke_mcq ceiling:** if every member hits 0.95+ pre-SFT, there's no headroom for either SFT lift or ensemble gain. This is the most likely failure mode; mitigation is exp 007 with `ifeval` + `mmlu_pro`.
- **Unbalanced base sizes:** SmolLM3-3B is ~4× weaker than Qwen3-14B on raw capability. Majority vote with unweighted tallies would let 4 weak members outvote 1 strong one. `weighted_majority` (score-weighted) was chosen to mitigate.
- **Corpus-benchmark alignment:** v009 is clinical/theoretical; smoke_mcq is trivia. SFT lift should be ≈ 0. Use this to calibrate expectations for exp 007.

## Run log (live)

- **2026-04-22, dispatch-0 (aborted)** — runner enqueued baselines for all 12 roster entries instead of 5. Bug: `run_baseline_evals` didn't apply `--pop-size`. Patched: now honors `pop_size`. No lineage rows written because workers timed out before completion.
- **2026-04-23 06:57** — dispatch-1 baselines timed out on every 14B model: `serving endpoint not ready after 600.0s`. First-time HF downloads of 13–14B checkpoints don't fit in 600s. SFT phase aborted seconds later: `ModuleNotFoundError: No module named 'datasets'` on every worker — the `voyage4nano` venv was missing `datasets`, `peft`, `trl`.
  - Fix A: bumped `ready_timeout_s` default 600 → 1800 in `openweight/inference.py` (both `launch_transformers_server` and `launch_sglang_server`).
  - Fix B: `pip install datasets peft trl` into `/home/<user>/venvs/voyage4nano` on all 6 workers.
  - Re-deployed `inference.py` to all 6 nodes and restarted worker daemons via `screen -dmS sieve-worker`.
- **2026-04-23 07:16** — dispatch-2 short-circuited Phase 1 (stale Redis failure results from dispatch-1 matched the stable `baseline_member_id`s and were read as already-complete). Purged the 4 Qwen3-14B/Phi-4/OLMo-2/Granite failure keys; SmolLM3-3B result kept (status=ok, score=0.050 — low, likely MCQ-format parsing, noted for verification). SFT phase advanced: 4 workers entered `sft:*` state. Baselines for the 4 purged members will be redone in a follow-up `--skip-sft` dispatch after SFT completes.
- **2026-04-23 07:52** — dispatch-3 started. Phase 1 completed at 08:14 UTC (1339s wallclock, well under the 1800s `ready_timeout_s`). HF checkpoint downloads for the 4 larger members concurrent across node-1/node-9/node-12/node-2 saturated at ~3.5 GB/13min/file; the 14B models finished their 8-file download just ahead of the timeout deadline. Baseline scores:
  - `microsoft/phi-4`            → **1.000** (20/20)
  - `allenai/OLMo-2-1124-13B-Instruct` → **0.950** (19/20)
  - `ibm-granite/granite-3.3-8b-instruct` → **0.950** (19/20)
  - `HuggingFaceTB/SmolLM3-3B`   → **0.050** (1/20) — reproduced, **confirmed format-parse bug** (below chance 0.25)
  - `Qwen/Qwen3-14B`             → **0.050** (1/20) — *new* datapoint; Qwen3-14B is a strong model — same MCQ-format parse issue. Two of the 5 tokenizer families are silently degraded.
- **2026-04-23 08:14** — Phase 2 (SFT) dispatched — 5 LoRA fine-tunes on v009 shard. All 5 jobs picked up by workers within ~30s (sieve:jobs:sft queue drained).
- **2026-04-23 09:20** — **aborted Phase 2 at ~1h mark** after discovering SFT projections were 16h (3B) / ~50h (14B) per model — full epoch over v009 with batch_size=1 and rows already at ~2k tokens each gives 22264 steps at 2.6–3s/step on 3B LoRA. SFT is a *diagnostic* on this run (H2: SFT lift should be ≈ 0 against smoke_mcq) so full-epoch fidelity is unwarranted. Fix: added `SIEVE_SFT_MAX_STEPS` env-var cap to `openweight/finetune.py` (threads into `SFTConfig(max_steps=...)` when >0). Killed all 5 in-flight SFT processes, redeployed patched finetune.py to 6 workers, restarted worker daemons with `SIEVE_SFT_MAX_STEPS=500` in their env, purged the stale job state, and re-dispatched with `--skip-baseline` (baselines still valid from 08:14 completion). New projections at 500 steps: ~22min (3B), ~50min (14B). HF checkpoints already on disk → SFT setup should be <5min vs 20+ on the cold run.
- **2026-04-23 09:21** — dispatch-3 Phase 2 re-started. New member IDs (uuid-generated): 33510817bd2f (SmolLM3), 6107eb3b320a (Qwen3-14B), 9aa84ce7cfb2 (Phi-4), c4f830f6387b (OLMo-2), ceb528db5fb9 (Granite). Est Phase 2+3 completion 11:00 UTC.
- **Cluster IP map as of now** — node-1=.203, node-2=.202, node-6=.101/.247, node-8=.85 (coordinator), node-9=.102, node-10=.68/.123, node-12=.155/.209. Nodes have two interfaces; either IP reaches the node.
- **2026-04-23 11:38** — dispatch-3 finished but **collapsed in Phases 2 and 3**. Orchestrator summary: `sft_ok: 2`, POST ensemble = `no_members`. Three distinct failure modes surfaced:
  1. **MmBackward0 meta-device** on Phi-4 (node-9) and OLMo-2 (node-10) during peft LoRA attach: "Function MmBackward0 returned an invalid gradient at index 1 - expected device meta but got cuda:0". Root cause: `device_map="auto"` under peft's single-GPU dispatch leaves some layers meta-initialized while adapters land on cuda:0. Patch: switched `openweight/finetune.py` to `device_map={"": 0}` (forces whole model on cuda:0, appropriate for single-GPU GB10). Deployed to all 6 workers — **not retried this run**.
  2. **HF xet backend flakes on node-1**: Granite SFT died with "File Reconstruction Error: Failed to send data: receiver dropped"; Qwen3-14B post-SFT eval died with "Background writer channel closed". Both from the new HF xet delivery layer. Specific to node-1 network path or a transient HF issue. Mitigation for retry: set `HF_HUB_DISABLE_XET=1` in worker env, or skip node-1 in the next dispatch.
  3. **Adapter filesystem isolation (architectural)**: SmolLM3 trained successfully on node-12 and saved to `/home/<user>/sieve/runs/exp006_sota_q2_01/adapters/33510817bd2f/` on node-12's *local* disk. Post-SFT eval was dispatched to node-2, which has no such file. Eval failed with "Can't find 'adapter_config.json' at ...". Confirmed by `ls` on all 5 workers — each node holds only the adapter dirs it trained. No shared NFS between workers. This is the blocking issue: eval cannot, in general, run on a worker other than the one that produced the adapter.
- **Dispatch-3 final scoreboard**:
  - SFT phase: 2 ok (SmolLM3 33510817bd2f, Qwen3-14B 6107eb3b320a), 3 error (Phi-4, OLMo-2, Granite)
  - Post-SFT eval phase: 0 ok (Qwen3-14B xet-flake on node-1, SmolLM3 adapter-not-found on node-2)
  - Phase 3 ensemble: `PRE best_ind=1.000 ensemble=1.000 Δ=+0.000` / `POST no_members`
- **Blockers before a retry makes sense** (documented for next session):
  - (a) Adapter visibility across workers — options: (1) pin each SFT-member to the same worker for its post-SFT eval, (2) mount a shared FS (NFS from node-8, or a single SSD used via sshfs), (3) push adapter to a node-8 blob store / HF Hub as a final SFT step and pull on eval. Option (1) is smallest-change — orchestrator-side hint to the scheduler — and worth trying first.
  - (b) Retry stragglers (Phi-4, OLMo-2, Granite SFT; Qwen3-14B eval) with the patched `finetune.py` + `HF_HUB_DISABLE_XET=1` on node-1.
  - (c) Only *then* is Phase 3 meaningful.

### Open: SmolLM3 + Qwen3 MCQ-format parse bug

Two out of five baselines scored 0.050 (1 correct out of 20) — below 4-way chance (0.25). These are the two models whose chat templates are **not** OpenAI-style instruct-tuned plain text: SmolLM3 uses a reasoning-trace-heavy chat format, and Qwen3's default template may also wrap responses in thinking tags. The smoke_mcq adapter in `benchmarks/adapters/smoke_mcq.py` expects a single-letter answer extraction via regex from the assistant's completion. Investigation task (for exp 007): dump raw completions from both models against smoke_mcq, confirm parse-failure mode, and either (a) add per-tokenizer-family extraction rules, or (b) switch to logprob-of-choice-letter scoring to make it format-agnostic. Until then, SmolLM3 and Qwen3 are effectively untested on smoke_mcq; their ensemble votes should be flagged as unreliable.

## Follow-ups

- **exp 007:** add `ifeval` + `mmlu_pro` benchmarks + CrossFamily roster B run for comparison.
- **exp 008:** logit-averaging ensemble for within-family members (the 4 qwen3 entries) — compare to majority vote.
- **Roster refresh cadence:** `sota_2026_q2` reflects the HF leaderboard top of 2026-04. Re-poll before each quarterly experiment set.

---

## Addendum — 2026-04-23 final: bench-fix + clean pre/post

**Status:** complete. All 5 SFTs landed; all 10 evals (5 baseline + 5 post-tune) are clean on the fixed bench.

### What shipped

1. **`smoke_mcq` thinking-model shim** (`benchmarks/adapters/smoke_mcq.py`) — strips complete `<think>…</think>` blocks before letter extraction, returns "" (scored 0) on a truncated opening tag, and matches the **last** word-boundaried A/B/C/D (models reason "... so the answer is D" — take the conclusion, not the first letter). Also bumped `MAX_TOKENS = 4 → 2048` and `TIMEOUT_S = 60 → 300` so 14B thinking models can actually emit an answer.
2. **`openweight/inference.py` launch timeout** — `ready_timeout_s` 600 → 1800 for both `launch_transformers_server` and `launch_sglang_server`. Cold HF downloads of 13–14B checkpoints don't fit in 600s; deploying the patched file to the whole fleet fixed silent "endpoint not ready" errors.

### Clean pre/post on smoke_mcq (500-step SFT on v009)

| model           | pre    | post   | lift    |
|-----------------|--------|--------|---------|
| phi-4 (14.7B)   | 1.000  | 1.000  | +0.000  |
| OLMo-2-13B      | 0.950  | 0.900  | -0.050  |
| granite-3.3-8B  | 1.000  | 1.000  | +0.000  |
| SmolLM3-3B      | 1.000  | 1.000  | +0.000  |
| Qwen3-14B       | 0.850  | 1.000  | +0.150  |

Average lift: +0.02. SmolLM3 + Qwen3 going from 0.05 on the broken bench to 1.00 / 1.00 on the fixed bench confirms the original "pre ≈ 0.05" scores were artifacts of `<think>` truncation — not model regressions. The Qwen3 +0.15 is within noise given the 20-question scale; read: **SFT-on-v009 did not meaningfully move any model on smoke_mcq**.

This matches H2 from the original plan: v009 (clinical/theoretical) is orthogonal to trivia MCQ, so ≈ 0 lift was expected. The bench fix doesn't change the structural conclusion — it just makes the absence-of-lift honest instead of cosmetically flattering.

### Operational notes (for future runs)

- **Worker↔model affinity matters on a mixed fleet.** Two post-tune evals landed on `<standalone>-*` nodes and hit per-question 300s timeouts (13/20 and 15/20 q's timed out). <standalone> nodes are markedly slower at thinking-model generation than compute — likely thermals or slower NVMe. Redoing those evals on compute nodes produced 1.00 / 1.00 cleanly. Fix for future dispatches: either a worker-affinity hint (pin thinking-model evals to node-class workers), or a longer per-question budget (600s).
- **Late-landing duplicate jobs overwrite results.** One Qwen3 post-tune was submitted to a cold <standalone> node at dispatch-0, spent 134 min in cold download + slow inference, and clobbered a clean 1.00 result from a later-submitted-but-faster node. Lesson: before resubmitting an eval job, either (a) wait for any prior in-flight copy to complete, or (b) add a write-guard (e.g. include `submitted_at` in the result and reject stale writes). Currently the orchestrator does neither.
- **Result-key delete + resubmit is safe only if no old job is still running.** Worker daemons emit heartbeats but don't report "currently working on job X" via the queue — check `current_job` in the heartbeat JSON (compute daemons started 2026-04-23 include it) before assuming the job is orphaned.
- **Cold HF cache downloads** on the first attempt of a fresh base routinely take 20–40 min (Qwen3-14B = ~28GB at observed ~4–5 MB/s). The 1800s `ready_timeout_s` is **right at the edge** for a cold 14B. If a node has no prior HF cache and rolls badly on HF xet throughput, it may still fail. Pre-warming via a scripted `snapshot_download` fleet-wide at the start of each experiment would eliminate the flake.
- **Bench saturation**: 4/5 baselines already hit ≥0.95 on smoke_mcq. A discriminating bench for exp 007+ needs headroom — use `ifeval` (instruction-following, unbounded) and/or `mmlu_pro` (5-shot, broad) as planned.
