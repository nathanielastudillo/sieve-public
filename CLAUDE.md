# SIEVE — Solomonoff-Inspired Evolutionary Ensemble

**Evolutionary population-based training of small transformer ensembles on curated, high-density corpora.**

The core thesis: Solomonoff induction is the provably optimal predictor — a weighted mixture over all programs consistent with the data. Neural nets approximate it via simplicity bias. Ensembles approximate it better because they maintain multiple solutions rather than committing to one. This system operationalizes that insight on commodity GPU hardware by making *ensemble composition* the evolutionary optimization target.

---

## Operational state — read this first

The git log is misleading: it ends at `7f7f8f2 infra: smoke test v003`. **Most of the working tree is uncommitted** — the entire openweight track, two live frontends (`lab/`, `dashboard/`), cluster infra, `lab-notes/`, `docs/`, `benchmarks/`, `teaching/`, `manual.md`, 6+ newer corpora, 3 newer tokenizers, 11 config profiles. `git status` is the honest picture.

**Before answering questions about history, status, or "what have we tried?" — read `lab-notes/` first.** Answering from the theoretical framing below will be wrong about a dozen concrete things.

Knowledge-base entry points:

- `lab-notes/README.md` — index
- `lab-notes/findings.md` — chronological empirical log, newest first
- `lab-notes/experiments/NNN-*.md` — per-run writeups (latest is `004-exp007-v009.md`)
- `lab-notes/experiments-planned.md` — Bucket A (applied) + Bucket B (queued)
- `lab-notes/research-questions/` — 9 lit reviews + design docs; the openweight family is `literature-openweight-ensembles.md`, `openweight-evolution.md`, `openweight-finetune-distill.md`, `openweight-inference-arch.md`, `openweight-merging.md`
- `docs/cluster.md` — fleet layout, SSH convention, Redis coordinator
- `docs/architecture-openweight.md` — open-weight fine-tune-and-ensemble track design (2026-04-18)
- `docs/ensemble_loss_blowup.md` — why `logit_mean_smoothed` (ε=0.05) is the aggregator default, and how dispatch timeouts look identical to aggregator failures
- `docs/ensemble_lit_review.md` — grounded version of the above with citations
- `docs/data_pipeline.md` — the staged pipeline in `data/pipeline/` (supersedes the `extract → clean → build_corpus` trio described below)
- `manual.md` — inference quick-reference for trained checkpoints

## Two tracks, shared plumbing

1. **Small-corpus track** — from-scratch ~7M-param GPTs on curated ~100K-token corpora. This is what the theoretical framing below describes. State: `exp007_v009` was the last cluster run (failed on dispatch timeout; 10 checkpoints recovered under `runs/exp007_v009/posthoc/checkpoints/`; post-hoc ensemble eval pending). **Flag: the track is bottlenecked on vocab/train-tokens ratio (Magikarp pathology, discovered 2026-04-17).** The 7.4M-param ensemble is beaten by a zero-param unigram on test.bin. Any new small-corpus work should start with `scripts/magikarp_check.py` on the corpus/tokenizer pair before training anything.
2. **Openweight track** — LoRA/DoRA fine-tune + merge + benchmark open-weight 7B–13B seeds on compute-cluster. Architecture specified, code partially implemented under `openweight/`, no SFT dispatched yet. See `docs/architecture-openweight.md` for the Gen-0 recipe. Reuses the controller/lineage/dispatch layer.

## Existing plumbing to reuse — don't rewrite without checking

Claim-checkable. Read these before saying a SIEVE primitive doesn't exist.

- `controller/ensemble_eval.py` — **the canonical logit-averaging implementation.** Caruana-greedy-with-replacement (the default strategy, beats brute-force at N>10), ε-smoothed geometric-mean-of-probs aggregator (`logit_mean_smoothed`, α=0.05, the default — plain `logit_mean` is catastrophic when any member assigns ~0 probability to truth), sequestered meta-val slice (fixed-seed 1729, 10% of val.bin, never used for selection), pairwise-JSD diversity. Shipped through exp007. Openweight logit-averaging should adapt this abstraction; it already solved the hard problems.
- `controller/dispatch.py` — `SftJob` / `MergeJob` / `EvalJob` dataclasses + `SftResultMsg` / `EvalResultMsg` / `MergeResultMsg`, `make_*_job` factories, queue constants, and `RedisDispatcher` `.submit_*` / `.wait_for_*` / `.publish_*_result` / `.next_openweight_job` all wired. Multi-queue BLPOP via `next_openweight_job(queues=[...])`.
- `controller/openweight_orchestrator.py` — Gen-0 driver (seed roster → SftJob × N → wait → EvalJob × (N×|benches|) → wait → FitnessVector → lineage). Resume-safe: scans Redis for published results before enqueuing. CLI entry point.
- `worker/daemon.py` — role-selectable multi-queue worker. `--roles train,sft,eval,merge` pops from any matching queue and routes.
- `worker/openweight_runner.py` — `run_sft` (calls `openweight.finetune.fine_tune`), `run_eval` (launches sglang, calls `benchmarks.adapters.run_benchmark`, tears down), `run_merge` (calls `openweight.merge.run_mergekit`).
- `openweight/inference.py::launch_sglang_server` — subprocess launcher + `/v1/models` readiness poll. LoRA-on-base via `--lora-paths`. Falls back to `python -m sglang.launch_server` when `sglang` isn't on PATH.
- `openweight/node_picker.py` + `openweight/remote_launch.py` (2026-04-27, auto-replicate added 2026-04-28, pre-launch openweight pkg sync added 2026-04-29) — cross-node dispatcher for the dashboard's openweight playground. `pick_node` ssh-probes the compute fleet (avoiding self + already-reserved nodes), enforces one-shim-per-GB10 (parallel-by-default-or-panic-with-503), checks weights presence + memory headroom. With `auto_replicate=True` (the dashboard default), if every viable peer is weight-missing but `find_weights_source` finds a cached node anywhere in the fleet (including excluded hosts), the picker returns a target plus `{"needs_replication": True, "source_name": ...}`; the caller drops the pick lock and runs `replicate_weights` (rsync from source-side, pushed to target via the shared fleet key) before launching. `launch_remote_shim` first runs `_sync_openweight_pkg` (rsync of `openweight/` from node-8 → target as Phase -1, ~200 KB incremental — needed because fleet nodes carry stale snapshots from exp006), then ssh-launches `openweight.serving_shim` on the chosen node bound to `127.0.0.1`, opens an `ssh -N -L` local-port forward to bypass the compute ufw INPUT-DROP, and returns a `RemoteServerHandle` whose `stop()` tears down both the remote pid and the local tunnel. The dashboard's `playground_ensure` writes the reservation as `status="replicating"` while rsync runs (registry exposes `replicate_from`); `_start_shim_thread` flips to `starting` once rsync completes. The controller's `worker/openweight_runner.py` does NOT use this dispatcher (playground-only for now).
- `benchmarks/adapters/` — name → module registry. `smoke_mcq` is the dev-loop canary (20-q built-in MCQ over `/v1/chat/completions`); `lm_eval_task` delegates to `lm-eval` against the `local-completions` model for `ifeval` / `hellaswag` / `mmlu_pro` / `gpqa_diamond`.
- `data/build_sft_shard.py` — `cleaned_v{NNN}/` markdown → paragraph-packed JSONL with `{"text": ...}` rows for `trl.SFTTrainer`.
- End-to-end smoke command on compute:
  ```
  python -m data.build_sft_shard --cleaned data/cleaned_v009 --out data/sft_shards/v009.jsonl
  # worker on each node:
  python -m worker.daemon --redis redis://<node-ip>:6379/0 --sieve-root /home/<user>/sieve --roles sft,eval
  # orchestrator (run once):
  python -m controller.openweight_orchestrator --redis redis://<node-ip>:6379/0 \
      --roster llama3_within_family --pop-size 1 \
      --corpus-shard data/sft_shards/v009.jsonl \
      --benchmarks smoke_mcq --proxy-eval \
      --lineage-db runs/openweight_smoke/lineage.db
  ```
- `controller/orchestrator.py` — main evolutionary loop (global-best tracking, meta-val reporting, Caruana selection wired).
- `controller/lineage.py` — SQLite schema with `generations` / `members` / `ensembles` tables; auto-migrates. `meta_val_loss` column added 2026-04-18.
- `controller/population.py` — PBT selection; immigration fraction 0.25 (Fort 2019).
- `worker/{daemon,trainer,checkpoint,evaluator,finetune}.py` — per-node daemon layout.
- `lab/server.py` (port 8099, FastAPI) — **live checkpoint-exploration + ensemble-generation UI.** Auto-discovers every `runs/*/checkpoints/*.pt`, holds models GPU-resident, fans a prompt across N models with per-member logit-contribution capture on the top-k tokens. Logit-averaging for local checkpoints is a solved problem here; the openweight equivalent is "same pattern, but members are remote sglang endpoints instead of local `.pt` files."
- `dashboard/server.py` (port 8096, FastAPI + WebSocket, React SPA under `dashboard/frontend/`) — realtime run monitor polling `lineage.db` every 2s; endpoints: `/api/status`, `/api/generations`, `/api/members`, `/api/ensembles`, `/api/lineage/{id}`, `/api/hyperparams`, `/api/population_scatter`. Also hosts the **openweight playground** (`/openweight` → `dashboard/openweight.html`): `/api/openweight/playground/{ensure,status,log,stop,chat,ensemble,card}` — picks a free GB10 via `node_picker`, ssh-launches a `serving_shim` on it through `remote_launch`, exposes the shim through an ssh -L tunnel; multi-member ensemble vote endpoint fans one prompt across N members in parallel.
- `scripts/` — 19 scripts. High-signal: `magikarp_check.py`, `measure_ensemble_test_loss.py`, `posthoc_ensemble_eval.py`, `eval_existing_gen0.py`, `deploy_sieve_daemons.sh`, `setup_cluster_redis.sh`, `discover_nodes.sh`, `watch_exp007.py`, `runs_monitor.py`, `reextract_broken_pdfs.py`.
- `teaching/01_tokens.md` → `09_ensemble.md` + matching `exercises/0N_*.py` — pedagogical walkthrough of the whole stack. Good orientation if a concept is unclear.

## Known stack gotchas (discovered empirically; not in plain `ls`)

- **axolotl is blocked** on Blackwell arm64 + torch 2.13-nightly (the compute stack). 0.16.1 pins torch 2.8. Use `peft + trl.SFTTrainer + transformers` directly — see `openweight/finetune.py`.
- **pymupdf4llm silently drops 95%+ of text on multi-column PDFs** with no error. 7 known-bad extractions were rescued by `scripts/reextract_broken_pdfs.py` (raw pymupdf fallback). No guard on new PDFs yet; flag extractions where `extracted_chars / pdf_page_count < 500`.
- **SSH fleet access** — user `<user>`, password from `$SSH_PASS` (never stored in the repo). IPs are DHCP and shift on reboot — always rediscover before a run. `scripts/discover_nodes.sh` uses `BatchMode=yes` (only finds already-authorized hosts); `avahi-browse` is the more complete sweep.
- **Redis coordinator must bind to a `<lan>` interface.** gpu-01↔<fabric>.x is one-way-routable; standalones can't reach a coordinator on gpu-01. Put Redis on a compute node.
- **Two distinct "`inf` ensemble" failure modes** — aggregator blowup (fixed by `logit_mean_smoothed`) vs dispatch timeout (exp007's failure — orchestrator sees `inf`, workers have real results in Redis). See `docs/ensemble_loss_blowup.md` §"post-exp007 addendum" for how to tell them apart from the orchestrator log.
- **Worker `val_loss` must be genuine val loss, not train loss.** The from-scratch trainer used to report best-epoch-average training loss as `val_loss`; fixed, but any new trainer wrapper must forward genuine val loss or it silently poisons the selection signal.
- **Checkpoint sizes** — from-scratch members are ~88 MB fp32 (7.4M params). Openweight LoRA adapters are ~50–400 MB at rank 8–64. Merged 7B checkpoints are ~15 GB; merged 13B are ~28 GB. Don't put merged weights in Redis.

## State you can't read off the filesystem

- **Corpora on disk**: v003, v004, v005, v006, v007, v008, v009, vfull, vbatch1/2/3. Most are subsamples; see `lab-notes/corpora.md` for provenance.
- **Tokenizers on disk**: v001, v002, v003, v004, v005. v002 (16K BPE) is the canonical one for most completed runs; v003/v004/v005 are the Magikarp-fix sweep.
- **Config profiles**: `smoke{,_v4000,_v8000,_v16000}.yaml`, `small/medium/large/full/vfull.yaml`, `exp007{,s}_v009.yaml`.
- **Run directory layout**: `runs/<run-id>/{checkpoints,logs,results,posthoc/checkpoints}`. `results/best_ensemble.json` is the artifact; `results/lineage.db` is the SQLite record. `posthoc/checkpoints/` holds rescued checkpoints (exp007 has these).
- **Run inventory**: `smoke_v003`, `smoke_loop`, `smoke_loop_v005`, `smoke_loop_v006`, `exp006_parallel`, `exp007_v009`, `exp007s_v009`, `vfull_par`.

## Editing this file

This section is the operational state; the material below is the theoretical framing. **When you discover something that would have saved time if it had been here, add it here, not in the theory section.** Keep entries short and pointer-heavy — duplicated content rots; pointers stay fresh.

---

## Theoretical Foundation

**Solomonoff induction** (1964): The optimal universal predictor. Take every computable hypothesis consistent with observed data. Weight each by 2^(-length_of_program). Predict by weighted average. Provably converges to the true distribution faster than any other method. Catch: it's uncomputable.

**Neural nets approximate it** because gradient descent on overparameterized networks has a simplicity bias: simple functions occupy wider basins in parameter space. This mirrors Solomonoff's exponential prior on program length.

**Ensembles approximate it better** because Solomonoff IS an ensemble — a weighted mixture over all programs. A single net picks one program and commits. An ensemble of N nets maintains N programs. Logit averaging is structurally analogous to Solomonoff's weighted average.

**The SIEVE insight**: standard ensembles train independently and average at the end. SIEVE applies evolutionary pressure on *ensemble composition itself* — selecting for "which *combination* produces the best ensemble" rather than "which individual is best." This selects for **diversity**, not just quality. A mediocre model that's highly complementary gets selected. Closer to what Solomonoff actually does.

## Data Pipeline

### Corpus

Source: `corpus-host` (office desktop, reachable via `ssh corpus-host` with ProxyJump through jump-host). 1545 PDFs, 2.8GB. Clinical manuals, psychoanalytic theory, social theory (Mann, Collins, DeLanda, Latour), engineering.

Pipeline: `data/raw/` (PDFs) → `data/extract.py` → `data/extracted/` (markdown) → `data/clean.py` → `data/cleaned/` → `data/build_corpus.py` → `data/corpus_v{NNN}/`

At 100K tokens, every token is load-bearing. Cleaning IS architecture.

### Corpus manifest + diff (`data/manifest.py`)

Before merging a new drive's worth of texts into `data/raw/`, build a manifest on each side and diff them. Classifies each candidate file as:

- `new` — ship it
- `dup_file` — same sha256 as something we have
- `dup_text` — same hash of (NFKC-normalized, lowercased, whitespace-collapsed) first 50K chars of extracted markdown — i.e. a different PDF rendering of the same book
- `fuzzy_name` — normalized filename collision (strips `_scanned_`, `download-site`, bare years, `Nth ed(ition)`, `(1)`/`(2)` copy markers, publisher tails). Review before copying.
- `personal` — filename matches `emergency kit|1password|tax|resume|insurance|bank statement|passport|invoice|receipt|payslip|driver.?licen[cs]e`. Review.

Raw↔extracted linking is by normalized filename (not sha prefix — `extract.py` writes `{safe_stem}.md`, no hash in filename). Current local state: 1573 raw, 1310 extracted, 1246 linked (~79%). Unlinked ~= extraction failures; they get a sha256 + fname_key but no text_hash, so dedup for them falls back to file-hash + fname-key only.

```bash
# build local manifest
python3 data/manifest.py build --out data/manifest_local.json

# on corpus-host / the new drive:
scp data/manifest.py corpus-host:/tmp/
ssh corpus-host "python3 /tmp/manifest.py build --raw /mnt/drive --extracted '' --out /tmp/manifest_tub.json"
scp corpus-host:/tmp/manifest_tub.json /tmp/

# what would corpus-host add?
python3 data/manifest.py diff \
    --self data/manifest_local.json \
    --other /tmp/manifest_tub.json \
    --out data/diff_tub.json
python3 data/manifest.py list-new --diff data/diff_tub.json --bucket new
python3 data/manifest.py list-new --diff data/diff_tub.json --bucket fuzzy_name
```

Manifests are committed; the local one regenerates in <5s on 1.5K files.

### BPE Tokenizer — Iterative, Versioned

BPE is NOT a one-shot step. It's a parameter we evolve. Each version lives in `data/tokenizers/v{NNN}/` with config, artifact, and quality metrics. Different vocab sizes × normalization × pre-tokenization = different tokenizers → different token streams → different models.

```bash
# Train BPE variants
python data/train_bpe.py --version v001 --vocab-size 8000
python data/train_bpe.py --version v002 --vocab-size 16000
python data/train_bpe.py --version v003 --vocab-size 8000 --no-lowercase
```

Domain terms (psychoanalysis, jouissance, transference, countertransference, hamiltonian, etc.) should be single tokens. The `train_bpe.py` metrics track domain_integrity_pct.

### Versioning & Git Strategy

Every parameter variation gets tracked. The evolutionary loop generates thousands of configs — must be able to rewind to any point.

**Atomic commit convention:**
```
<type>: <description>
Types: data | tok | model | train | ens | eval | infra | fix
```

**What gets versioned:**
- Corpus versions: `data/corpus_v{NNN}/metadata.json` (committed, bins gitignored)
- Tokenizer versions: `data/tokenizers/v{NNN}/` (config + metrics committed, .json artifact gitignored)
- Evolution runs: `results/lineage.db` + `results/best_ensemble.json`
- Every config that produces a training run

**Tertiary test set:** 5% of data held out and NEVER used for ensemble selection. Ground truth for meta-overfitting detection.

## References & Prior Art

- **Slowrun** (Q Labs, 2026): https://github.com/qlabs-eng/slowrun — 100M tokens, unlimited compute, data-efficient training. Key findings: ensembling (averaging logits across 8 independently trained models) dramatically lowers val loss; aggressive regularization (weight decay up to 16x standard + dropout) recovers monotonic improvements with scale; epoch shuffling has outsized impact.
- **PBT** (Jaderberg et al., DeepMind, 2017): https://arxiv.org/abs/1711.09846 — Population Based Training. Train a population in parallel, periodically evaluate, exploit (copy weights from best performers to worst) and explore (perturb hyperparameters). Discovers hyperparameter *schedules*, not just fixed settings.
- **Model Soups** (Wortsman et al., 2022): https://proceedings.mlr.press/v162/wortsman22a — Averaging *weights* (not just logits) of multiple fine-tuned models from shared initialization improves accuracy without increasing inference cost. Works when models lie in a single low-error basin.
- **GENOME+** (2025): https://arxiv.org/pdf/2503.01155 — Population-based evolutionary framework for LLMs with crossover, mutation, succession, and ensemble operations. Demonstrates scalability to 40-model populations.
- **BabyLM Challenge** (2024-2025): Training on ≤100M words (human-scale data). LTG-BERT won. Demonstrates robust syntactic/semantic generalization is achievable on small corpora.
- **nanoGPT / nanochat** (Karpathy): https://github.com/karpathy/nanoGPT — Simple, hackable transformer training. Model configs from ~10M params (4 layers, 4 heads, 256 embed) to 124M+ (GPT-2 scale).
- **Logit averaging vs probability averaging**: Averaging logits (pre-softmax) generally outperforms averaging probabilities for ensemble calibration and NLL (Njieutcheu Tassi et al., 2022).
- **Phi / "Textbooks Are All You Need"** (Gunasekar et al., 2023): https://arxiv.org/abs/2306.11644 — quality over quantity. Diverse, non-repetitive, pedagogically structured data.
- **TinyStories** (Eldan & Li, 2023): https://arxiv.org/abs/2305.07759 — <10M param models on constrained vocabulary produce fluent text. Small-data regime works.
- **LIMA** (Zhou et al., Meta, 2023): https://arxiv.org/abs/2305.11206 — 1000 carefully curated examples outperformed datasets 100x larger. Redundancy actively harmful.
- **Dedup Training Data** (Lee et al., 2021): https://arxiv.org/abs/2107.06499 — near-duplicates degrade training. MinHash LSH is standard tool.
- **SWA** (Izmailov et al., 2018): Weight averaging across training trajectory. PyTorch native `torch.optim.swa_utils`.
- **SWAG-LoRA** (2024): https://arxiv.org/html/2405.03425v1 — Bayesian ensembling via SWA-Gaussian.
- **Model Merging Survey** (2026): https://arxiv.org/html/2511.21437v1 — TIES, DARE, evolutionary merging.
- **Ensemble Bayesian Inference** (2025): https://arxiv.org/abs/2504.17685 — SLM ensembles matching LLM accuracy.

## Architecture

### What This System Does

1. **Curate** a small, high-density corpus from domain-specific sources (clinical manuals, physics textbooks, etc.)
2. **Train a population** of small transformer models (10M-50M params) on this corpus with varied hyperparameters, seeds, regularization, and data orderings
3. **Checkpoint aggressively** throughout training
4. **Evaluate all ensemble combinations** by averaging logits across subsets of checkpoints on a held-out validation set
5. **Evolve** — use ensemble-level fitness (not individual model fitness) to drive the next generation of training runs
6. **Repeat** until convergence or compute budget exhausted

### The Novel Piece

Standard PBT selects individual models. Standard ensembling averages at the end. SIEVE operates evolutionary pressure on the *portfolio* — the unit of selection is a *combination* of checkpoints, optimized for ensemble diversity and complementarity. A 16-model ensemble where each member covers different solution-space territory will beat 16 copies of the best single model.

### Compute Topology

```
┌─────────────────────────────────────────────────┐
│                   CONTROLLER                     │
│          (orchestrator node on Tailnet)           │
│                                                   │
│  - Manages population registry                    │
│  - Dispatches training jobs to workers            │
│  - Collects checkpoints                           │
│  - Runs ensemble sweep (combinatorial eval)       │
│  - Executes evolutionary selection                │
│  - Stores results + lineage graph                 │
└─────────────┬───────────────────────────────────┘
              │ Tailscale mesh
    ┌─────────┼─────────┐
    ▼         ▼         ▼
┌───────┐ ┌───────┐ ┌───────┐
│ GB10  │ │ GB10  │ │ GB10  │  ... × N_WORKERS
│ Worker│ │ Worker│ │ Worker│
│       │ │       │ │       │
│ Train │ │ Train │ │ Eval  │  (roles are dynamic)
│ model │ │ model │ │ ensemble│
└───────┘ └───────┘ └───────┘
```

## Dynamic Configuration

This system adapts to the number of available GB10 nodes. Claude Code should detect available workers on the Tailnet and configure accordingly.

### Node Detection

```bash
# Discover available GB10 worker nodes on Tailnet
# Assumes workers are tagged or named with a convention like gb10-XX
tailscale status --json | jq '.Peer[] | select(.Tags // [] | any(contains("worker"))) | .HostName'
# Or by naming convention:
tailscale status | grep gb10 | awk '{print $1}'
```

### Configuration Profiles

The system should auto-configure based on `N_WORKERS` (number of available GB10 nodes):

| N_WORKERS | Population Size | Ensemble Size | Parallel Runs | Eval Workers | Strategy |
|-----------|----------------|---------------|---------------|--------------|----------|
| 4-8       | 16             | 4-ensemble    | 2 runs × 4    | shared       | Conservative: fewer larger ensembles, longer training per model |
| 8-16      | 32             | 8-ensemble    | 4 runs × 8    | 2 dedicated  | Balanced: good diversity, fast iteration |
| 16-32     | 64             | 8-16 ensemble | 4 runs × 16   | 4 dedicated  | Aggressive: maximum diversity, fast evolutionary cycles |
| 32-64     | 128            | 16-ensemble   | 8 runs × 16   | 8 dedicated  | Full: brute-force portfolio search, sub-hour generations |

**Key formula**: With 100K tokens of training data and 10-50M param models, a single training run completes in *minutes* on a GB10. This means the evolutionary loop iterates very fast — potentially hundreds of generations per day on a 64-node cluster.

### Compute Budget Allocation

```
TOTAL_FLOPS_PER_GENERATION = N_WORKERS × TIME_PER_GENERATION

Recommended split:
  - 70% training new population members
  - 20% ensemble evaluation (logit averaging across combinations)
  - 10% overhead (checkpointing, communication, selection)

For ensemble eval with K checkpoints choosing subsets of size E:
  C(K, E) combinations to evaluate
  Each eval = forward pass on val set for E models, average logits
  Parallelize across eval workers: each worker runs multiple combos
```

## Corpus

The training corpus lives on `corpus-host` (Tailnet node). It contains clinical manuals and physics textbooks — dense, high-signal, every-token-matters material.

### Corpus Preparation Pipeline

```bash
# Mount/access corpus from corpus-host
# corpus-host is a node on the Tailnet hosting the document corpus

# 1. Extract text from source documents
# Clinical manuals: likely PDF → extract with pymupdf or pdfplumber
# Physics textbooks: likely PDF → extract, preserving equations where possible

# 2. Clean and tokenize
# - Strip headers/footers/page numbers
# - Preserve technical terminology and notation
# - Remove duplicate content across editions
# - Target: ~100K tokens of maximally dense, non-redundant text

# 3. Create train/val split
# CRITICAL: val set must be held sacrosanct
# Recommended: 90/10 split, stratified by source document
# The val set is the fitness function for ensemble evolution —
# if it leaks into training, the whole system is compromised

# 4. Tokenize with a standard BPE tokenizer
# tiktoken (GPT-2/GPT-4 tokenizer) or sentencepiece
# For domain-specific corpora, consider training a custom tokenizer
# on the corpus itself — 100K tokens is small enough that vocab
# coverage matters a lot
```

### Corpus Quality Heuristics

At 100K tokens (~75 pages), curation IS architecture. Every token is load-bearing.

- **Information density**: Prefer definitional/explanatory passages over examples/exercises
- **Non-redundancy**: Deduplicate across sources (e.g., if two textbooks explain Newton's second law, keep the better one)
- **Structural coherence**: Maintain logical flow within chunks — don't randomly shuffle paragraphs
- **Domain balance**: If mixing clinical + physics, track the ratio and ensure the ensemble can specialize

## Training Configuration

### Base Model Architecture

Following Slowrun and nanoGPT findings, start with:

```python
# Small model — good for 100K token regime
model_config = {
    "n_layer": 6,
    "n_head": 6,
    "n_embd": 384,
    "block_size": 512,     # context window
    "dropout": 0.1,        # Slowrun finding: dropout helps in data-constrained regime
    "bias": False,
    "vocab_size": 50257,   # GPT-2 tokenizer, or custom
}
# ~20M params — trains in minutes on GB10 with 100K tokens
```

### Hyperparameter Search Space for Evolution

These are the dimensions the evolutionary algorithm explores:

```python
search_space = {
    # Architecture (discrete — mutate by stepping)
    "n_layer": [4, 6, 8, 10, 12],
    "n_head": [4, 6, 8],
    "n_embd": [256, 384, 512],

    # Regularization (continuous — mutate by perturbation ×0.8 or ×1.2)
    "dropout": (0.0, 0.3),
    "weight_decay": (0.1, 3.0),  # Slowrun: optimal WD is up to 16x standard

    # Optimizer
    "learning_rate": (1e-4, 1e-2),
    "optimizer": ["adamw", "muon"],  # Slowrun: Muon outperforms AdamW

    # Training schedule
    "epochs": (10, 200),            # multi-epoch is key in data-constrained regime
    "warmup_fraction": (0.01, 0.1),
    "lr_schedule": ["cosine", "linear", "constant_with_warmup"],

    # Data ordering
    "seed": "random_per_member",     # different seed = different data order
    "shuffle_per_epoch": True,       # Slowrun: outsized impact on multi-epoch training
}
```

### Activation Function

Slowrun found SwiGLU outperforms squared ReLU for data efficiency. Use SwiGLU as default.

## Evolution Protocol

### Phase 1: Initialization (Generation 0)

```
1. Sample POPULATION_SIZE random configurations from search_space
2. Assign each to a worker node
3. Train each for INIT_EPOCHS (e.g., 20 epochs) on full training corpus
4. Checkpoint each model at completion
5. Evaluate all models individually on val set
6. Record: config, val_loss, checkpoint_path, worker_id, wallclock_time
```

### Phase 2: Ensemble Evaluation

```
1. Collect all POPULATION_SIZE checkpoints
2. For each combination of size ENSEMBLE_SIZE from the checkpoint pool:
   a. Load all E models onto eval workers
   b. Forward pass on val set
   c. Average logits across E models
   d. Compute ensemble val loss (cross-entropy on averaged logits)
3. Record: combination_id, member_ids, ensemble_val_loss
4. Rank all combinations by ensemble_val_loss

NOTE: C(N,E) grows fast. For N=32, E=8: C(32,8) = 10,518,300
Strategies to make this tractable:
  - Random sampling: evaluate M random combinations (M=1000-10000)
  - Greedy construction: start with best single model, greedily add
    the model that most improves ensemble loss, repeat
  - Diversity-aware sampling: ensure combinations span different
    hyperparameter regions
```

### Phase 3: Selection & Reproduction

```
1. From ensemble rankings, identify:
   a. ELITE models: those appearing most frequently in top-K ensembles
   b. COMPLEMENTARY pairs: models that appear together in top ensembles
      but have maximally different configs
   c. DEAD WEIGHT: models never appearing in any top-K ensemble

2. Kill dead weight members

3. Reproduce:
   a. Clone elite models with perturbed hyperparameters (PBT-style:
      multiply continuous params by 0.8 or 1.2, step discrete params)
   b. For complementary pairs: create children that interpolate
      their hyperparameter configs (crossover)
   c. Fill remaining slots with fresh random samples (immigration)
      to maintain population diversity

4. CRITICAL: The fitness function is not "how good is this model alone?"
   but "how much does this model improve the best ensemble it participates in?"
   This selects for DIVERSITY, not just individual quality.
```

### Phase 4: Iterate

```
Repeat phases 1-3 until:
  - Ensemble val loss plateaus for N_STAGNATION generations
  - Compute budget exhausted
  - Wallclock time limit reached

Typical stopping: 50-200 generations
At 100K tokens / 20M params / GB10 hardware: ~5-15 min per generation
→ 200 generations in ~1-2 days
```

## Inference

The final artifact is not a single model — it's an ensemble specification:

```json
{
    "ensemble_id": "sieve-gen-147-top1",
    "members": [
        {"checkpoint": "gen-089-member-12.pt", "config": {...}},
        {"checkpoint": "gen-102-member-07.pt", "config": {...}},
        {"checkpoint": "gen-131-member-22.pt", "config": {...}},
        ...
    ],
    "ensemble_size": 8,
    "val_loss": 1.847,
    "aggregation": "logit_average"
}
```

At inference time:
1. Load all ensemble members (one per GB10, or multiple per GB10 if small enough)
2. Forward pass input through all members in parallel
3. Average logits
4. Apply softmax to averaged logits
5. Sample or argmax from resulting distribution

## Implementation Stack

```
Language:       Python 3.11+
Framework:      PyTorch 2.x (torch.compile for speed)
Base model:     nanoGPT/nanochat (fork and modify)
Tokenizer:      tiktoken or sentencepiece
Optimizer:      Muon (primary), AdamW (baseline comparison)
Communication:  Tailscale mesh + simple HTTP/gRPC between nodes
Job dispatch:   Redis queue or simple filesystem-based job board
Checkpoints:    Shared storage on Tailnet (corpus-host or dedicated checkpoint node)
Tracking:       SQLite or DuckDB for lineage/metrics (already in Lever stack)
Visualization:  Wandb or simple matplotlib dashboards
```

## Directory Structure

```
sieve/
├── CLAUDE.md              # This file
├── README.md
├── config/
│   ├── default.yaml       # Default training config
│   └── profiles/          # Auto-generated per N_WORKERS
├── controller/
│   ├── orchestrator.py    # Main loop: dispatch, collect, evolve
│   ├── population.py      # Population management, selection, reproduction
│   ├── ensemble_eval.py   # Combinatorial ensemble evaluation
│   └── lineage.py         # Track evolutionary history
├── worker/
│   ├── trainer.py         # Single model training loop (nanoGPT-derived)
│   ├── evaluator.py       # Forward pass for ensemble eval
│   └── checkpoint.py      # Save/load/transfer checkpoints
├── model/
│   ├── gpt.py             # Transformer model definition
│   ├── config.py          # Model configuration dataclass
│   └── muon.py            # Muon optimizer implementation
├── data/
│   ├── prepare.py         # Corpus extraction and tokenization
│   ├── loader.py          # DataLoader with epoch shuffling
│   └── curate.py          # Corpus quality metrics and dedup
├── scripts/
│   ├── discover_nodes.sh  # Tailnet node discovery
│   ├── deploy_workers.sh  # Push worker code to GB10 nodes
│   └── run_sieve.sh       # Main entry point
└── analysis/
    ├── diversity.py        # Measure ensemble diversity metrics
    ├── convergence.py      # Track evolutionary convergence
    └── visualize.py        # Plot fitness landscapes, lineage trees
```

## Key Design Decisions & Open Questions

1. **Logit averaging vs weight averaging (model soups)**: Logit averaging is safer (works even when models are in different basins) but costs N× inference. Weight averaging is free at inference but only works when models share an initialization. Start with logit averaging, experiment with soups for models from the same lineage branch.

2. **Ensemble size**: Slowrun used 8. Solomonoff theory says "more is better" up to diminishing returns. Empirical question for this corpus/hardware combo. Start with 8, sweep 4-16.

3. **Custom tokenizer**: At 100K tokens, vocab coverage matters enormously. A custom BPE tokenizer trained on the corpus itself may outperform GPT-2's tokenizer, especially for clinical/physics terminology. Worth the experiment.

4. **Diversity metric**: Need a quantitative measure of "how different are two models' solutions?" Options: disagreement rate on val set predictions, cosine distance between logit distributions, CKA similarity between internal representations. Use this to explicitly maximize diversity during selection.

5. **Overfitting the val set**: The ensemble fitness function uses the val set. Over many generations, you're meta-optimizing against it. Mitigation: rotate the val set periodically, or maintain a tertiary held-out test set that's never used for selection.

6. **When to stop**: Ensemble val loss will plateau. But also track: individual member diversity (if diversity collapses, the ensemble degenerates toward a single solution — the opposite of what Solomonoff wants).

## Connection to Lever

If this works, the immediate application is training domain-specific ensembles on:
- Canadian statute and regulation (once lex crawler delivers the corpus)
- Specific policy domains (housing, immigration, tax) for constraint graph parsing
- Municipal bylaw collections for the City of Toronto prototype

The Solomonoff framing provides theoretical backing for trusting these small ensembles as parsers: they're not memorizing — they're converging toward the shortest programs that explain the legal/regulatory language, which is exactly what you need for statute → formal constraint translation.
