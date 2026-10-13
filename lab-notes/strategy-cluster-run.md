# Strategy — scaling SIEVE to compute-cluster

Drafted 2026-04-17, while exp-004 (smoke on corpus_v006) trains in
background. This is a design doc for moving from 1× GB10 smoke
experiments to a real 12× GB10 multi-node run on compute-cluster.

## Status snapshot

- **Hardware on hand**: compute-cluster = 12× GB10 nodes on CX-7 200G
  fabric, jump host at compute-cluster-8 (<node-ip>). Full 1.44 TB
  unified memory, 240 ARM cores, 12 TB NVMe.
- **Hardware in use today**: gpu-01 only. 3 smoke runs completed or
  training (v003, v005, v006).
- **Corpora ready**: v004 (7.52M tokens, full pool) and v006 (97K,
  v003-tokenized subsample). Both on gpu-01 only, not replicated.

## What "a larger run" means

Four things scale independently. In order of expected ROI:

| Dimension | Smoke (today) | Real run (target) | Multiplier |
|-----------|--------------:|------------------:|-----------:|
| Training tokens | 85K | 6.4M (v004) | 75× |
| Population size | 6 | 32–64 | 5–10× |
| Ensemble size | 3 | 8–16 | 3–5× |
| Max generations | 3 | 30–50 | 10–17× |

The fastest win is *training tokens* — meta-overfitting is the visible
failure mode of the smoke runs and more data is the most direct
intervention for it. The second-fastest is *generations* because the
evolutionary signal is what SIEVE is supposedly optimizing; 3 gens is
barely enough to see a trend.

Everything else (model size, ensemble size, population size) gates the
quality of the *search*, not the quality of individual members.

## Cluster orchestration plan

### Phase 1 — single-node bigger run (no cluster)

Before touching compute-cluster, run one "realistic" smoke on gpu-01:

- Corpus: **v004** (6.4M train tokens, 752K val)
- Population: 12
- Ensemble: 4
- Generations: 10
- Same architecture (7.4M params) — architecture scaling is orthogonal
- Expected wallclock: ~10h (12 members × ~50 min/train × 10 gens, minus
  parallelism = tight)

**Why first**: rules out whether v004's 75× more tokens + 10 generations
alone closes the meta-overfitting gap. If single-member test loss drops
below uniform baseline at 6.4M tokens, the whole cluster question shifts
from "we need distributed training to rescue this" to "we have headroom
to be ambitious with architecture/ensemble size".

### Phase 2 — true multi-node SIEVE

Current orchestrator code (`controller/orchestrator.py`) is single-node,
local-workers-only. To use compute-cluster we need:

1. **Job dispatch layer.** Options:
   - SSH + shared NFS for checkpoints. Simplest. Each node pulls its
     config JSON from a shared dir, trains, writes checkpoint back.
     Orchestrator on gpu-01 dispatches by writing job files.
   - Redis queue (basind at 6479 → 6379 redirect is available). Nodes
     BRPOP for jobs, write results via SET. Already-installed
     infrastructure; no new deps.
   - gRPC via `bus`. Overkill for this — bus is for human-visible
     coordination, not job distribution.

   **Recommendation: Redis queue.** Minimal code (~100 LOC), no new
   services, leverages the basind/Redis stack that's already running.

2. **Checkpoint shuffling.** Workers produce ~88 MB checkpoints. For
   ensemble eval we need N members co-resident on one GPU. Options:
   - NFS mount across cluster. Requires one node to serve it.
   - Redis BLOB storage (slow; 88MB×18 = 1.6 GB per gen = bearable).
   - rsync over CX-7 fabric. Fastest (200 Gbit), simplest, stateless.

   **Recommendation: rsync over CX-7** with a well-known path
   convention (`/data/sieve/runs/<run-id>/checkpoints/...`). The fabric
   can easily handle it; checkpoints live on local NVMe for training,
   get pushed to the controller node for ensemble eval.

3. **Environment replication.** Python 3.12, PyTorch + CUDA Blackwell
   build, tokenizers package. Each compute-cluster node already has
   the DGX stack. Need to verify PyTorch arm64 wheel compatibility.
   - Check: `compute-cluster-{1..12}: .venv/bin/python -c "import torch; torch.cuda.is_available()"` 

4. **Config + data replication.** Push v004 corpus (46 MB) + v003
   tokenizer (1.8 MB) to all nodes once, at `/data/sieve/`. Push
   controller code via git pull. No need for anything fancier.

### Phase 3 — topology

```
┌─────────────────────────────────────────────────────┐
│  gpu-01 (controller)                              │
│    - orchestrator.py (main loop)                    │
│    - lineage.db (SQLite)                            │
│    - results/best_ensemble.json                     │
│    - dispatches via Redis                           │
└────────────┬────────────────────────────────────────┘
             │ redis://gpu-01:6379
             │ rsync over CX-7 (checkpoints)
             │
    ┌────────┴────────┬────────┬────────┐
    ▼                 ▼        ▼        ▼
┌────────┐       ┌────────┐ ...     ┌────────┐
│ node-1  │       │ node-2  │ ...     │ node-12 │
│        │       │        │         │        │
│ worker │       │ worker │ ...     │ worker │
│ (train │       │ (train │         │ (train │
│  OR    │       │  OR    │         │  OR    │
│  eval) │       │  eval) │         │  eval) │
└────────┘       └────────┘         └────────┘
```

- Dynamic role assignment: each worker can train OR run ensemble eval
  depending on what the orchestrator puts in its queue.
- 12 GPUs × 12 members trained in parallel = population of 12 fits in
  one training pass. For population=24 we do 2 passes.
- Ensemble eval (C(12,8)=495 combos) parallelizes across all 12 nodes:
  ~42 combos per node, each combo = 8 forward passes on val set.
  Rough estimate: 10-20 min per gen for eval. Training dominates.

### Phase 4 — what to build in what order

1. **Redis job dispatch**. Controller pushes jobs, workers pull, write
   results. ~100 LOC addition to orchestrator.py + new `worker/daemon.py`.
2. **Checkpoint sync**. Trivial — workers rsync to controller on
   completion, eval workers rsync requested ensembles in.
3. **Profile for cluster runs**. `config/profiles/cluster.yaml` with
   population_size=12 or 24, ensemble_size=8, generations=30,
   larger architecture (n_layer=6, n_embd=384 ≈ 20M params).
4. **Health checks**. Ping each node before starting. Abort if any
   node's GPU is busy or offline.

None of this is hard. Redis dispatch is the only real new code; the
rest is shell glue. Two days of focused work.

## Known risks / what could block the cluster run

- **compute-cluster WiFi flakiness.** Node 8 jump host is the only
  reliably reachable node from gpu-01; other nodes are accessed via
  IPv6 link-local over the CX-7 fabric from node 8. Need to verify
  Redis traffic can reach the controller from every node — likely
  requires routing the Redis port through a node on WiFi.
- **PyTorch wheel compatibility.** compute-cluster is aarch64 + CUDA
  Blackwell. We've had issues before; verify `torch.cuda.is_available()`
  on a representative node *before* writing dispatch code.
- **Corpus contamination across nodes.** Each node must use exactly the
  same train/val/test split (from the same metadata.json). A subtle
  desync here silently invalidates the whole run. Hash-check corpus
  files post-rsync.
- **Val-set meta-overfitting at scale.** The bigger the search, the
  more selection pressure on val. Must maintain strict test-set
  isolation — no test passes during the loop, only at final report.
  This already exists in the design; just don't break it.

## Concrete next-action checklist

- [ ] Confirm exp-004 (v006 smoke on gpu-01, in progress) runs clean
- [ ] Write exp-005 page for planned "realistic single-node" run on v004
- [ ] Verify PyTorch works on compute-cluster node 8: one-off test
- [ ] Write `controller/dispatch.py` with Redis-backed job queue (est 100 LOC)
- [ ] Write `worker/daemon.py` that pulls jobs, trains, writes results (est 150 LOC)
- [ ] Write `config/profiles/cluster.yaml`
- [ ] Push corpus_v004 + tokenizer_v003 to shared path on all 12 nodes
- [ ] Dry-run with population=2 on 2 nodes before full 12-node launch
