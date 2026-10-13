# Experiment 005 — openweight smoke, Qwen2.5-1.5B-Instruct × Lever corpus v009

- **Status**: complete (2026-04-21). Dev loop ran end-to-end: baseline →
  SFT (12636 s, final train_loss 2.347) → merge → post-SFT bench. Headline
  delta is a small nominal regression on both benches, but both are within
  1σ of the baseline — see *Discussion* below for the noise analysis and
  the apples-to-oranges caveat on `smoke_mcq`.
- **Node**: node-8 (Blackwell aarch64 GB10, torch 2.12.0.dev nightly + cu130)
- **Base**: `Qwen/Qwen2.5-1.5B-Instruct` — chosen as "smartest smallest"
  (SOTA ~1.5B instruct at 2026-04 — beats TinyLlama-1.1B/OLMo-1B/Pythia-1.4B
  across every bench we care about, fits comfortably on a single GB10 at
  bf16 with headroom for training).
- **Corpus**: `data/sft_shards/v009.jsonl` — 22,264 rows, 122 MB
  (markdown-paragraph-packed from `data/cleaned_v009/`). sha256
  `6a20caa1713cf072f07c16f60136f299fd87ed03f21dbccc95087a4e7f5c3c17`.
- **Lineage DB**: `runs/openweight_smoke/lineage.db` (on node-8)
- **Goal**: prove the openweight dev loop end-to-end — baseline bench →
  LoRA SFT → merged post-SFT bench → delta visible in dashboard — so
  we can iterate on seed/hyper choices with confidence.

## Why this experiment

`docs/architecture-openweight.md` spec'd the track on 2026-04-18 but no
SFT had ever actually been dispatched through it. This is the minimum
run that exercises every code path:

- `LineageDB.record_openweight_member` (member row, kind=base/sft)
- `LineageDB.record_fitness_eval` (per-bench score)
- `openweight.finetune.fine_tune` (peft+trl LoRA)
- `openweight.inference.launch_transformers_server` (OpenAI-compat serve)
- `openweight.inference.merge_lora_to_dir` (adapter → standalone dir)
- `benchmarks.adapters.run_benchmark` (smoke_mcq + lm-eval tasks)
- dashboard openweight views (live monitor + launch panel)

Also — the user's explicit ask was an "ideal lab environment for both
monitoring and experimenting." This run is the thing the dashboard UI is
built against.

## What broke (and the fix)

### 1. `transformers serve` 5.3.0 handler-binding bug

`transformers/commands/serving.py::chat_completion(request: Request, body: dict)`
misbehaves on fastapi ≥0.100 — the `request` parameter is treated as a
missing query argument, returning HTTP 422 on every
`POST /v1/chat/completions`. Verified across fastapi 0.115.0, 0.129.2,
0.135.1. `/v1/responses` works but isn't what lm-eval expects.

**Fix**: in-house `openweight/serving_shim.py` (~120 LOC). FastAPI +
uvicorn, `@app.post("/v1/chat/completions") def(payload: dict = Body(...))`,
calls `tokenizer.apply_chat_template` + `model.generate`. Also exposes
`/v1/models`, `/v1/completions`, `/health`. `launch_transformers_server`
in `openweight/inference.py` now routes through
`python -m openweight.serving_shim` instead of the stock CLI.

### 2. sieve-venv inherits lm-eval as a library, not a script

`sieve-venv` is Python 3.12 on node-8. It inherits torch/transformers/peft
from story's venv via a `.pth` file
(`sieve-venv/lib/python3.12/site-packages/_story_inherit.pth` →
`/opt/story/.venv/lib/python3.12/site-packages`). That pulls in
`lm_eval` the library but NOT `bin/lm-eval` (entry-point scripts aren't
replayed by `.pth`). `shutil.which("lm-eval")` returned None and the
bench errored out.

**Fix**: `benchmarks/adapters/lm_eval_task.py` now falls back to
`sys.executable -m lm_eval` when the `lm-eval` script is absent but the
`lm_eval` module imports fine. Same binary, no additional install.

### 3. ifeval task has runtime deps the base lm-eval install skips

`lm-eval[api]` extras weren't installed; also ifeval itself imports
`langdetect` and `immutabledict` at task-registration time.

**Fix**: `pip install --no-deps tenacity langdetect immutabledict` into
sieve-venv. `aiohttp`, `absl-py`, `nltk` were already present via the
story inheritance.

### 4. Dashboard was unreachable from the user's workstation

node-8 runs `dashboard.server` on `0.0.0.0:8096` but has **ufw with default
INPUT DROP**; only SSH (22) is whitelisted. On top of that, node-8 lives
on the `tail1fafbc.ts.net` tailnet (owned by `xyrasinclair@`) while the
user's workstation and `gpu-01` are on the `nathanielastudillo@`
tailnet. The two tailnets are disjoint — `100.70.69.90:8096` (node-8's
Tailscale IP) is not routable from this side. `docs/cluster.md` and the
root `CLAUDE.md` talk about "the Tailnet" as if it's singular; in
practice it is not.

What actually works is a single SSH hop from gpu-01 (which can ssh
into `a@<node-ip>` on the warehouse LAN) into node-8. Bind the
forward on gpu-01's tailnet interface so any nathanielastudillo@-peer
can reach it:

```sh
# on gpu-01 — one-liner, survives re-logins while sshd keep-alives hold
nohup ssh -N -o GatewayPorts=yes -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -L 0.0.0.0:19096:localhost:8096 a@<node-ip> \
  > /tmp/tunnel-bc18.log 2>&1 </dev/null &

# from anywhere on the nathanielastudillo@ tailnet:
#   http://gpu-01:19096/openweight
```

Not running `autossh` yet; if the tunnel drops, `ps -ef | grep 19096` on
gpu-01, then re-run. Switching to autossh is a ~2-line change (see
the existing `autossh -M 0 -N ... -L 127.0.0.1:8083:127.0.0.1:8083
gpu-01` systemd-style launch that the orb user already keeps
running).

## Baseline results (pre-SFT)

**smoke_mcq** (20-q built-in MCQ over `/v1/chat/completions`):

- score: **0.95** (19/20 correct)
- wallclock: 2.2s

**ifeval** (proxy=40 items, prompt-level loose acc):

- prompt-level loose acc: **0.500** (stderr 0.080)
- prompt-level strict acc: 0.400 (stderr 0.078)
- inst-level loose acc: 0.571
- inst-level strict acc: 0.492
- wallclock: 218 s

## SFT config (queued)

- kind: LoRA, rank 16, alpha 32, dropout 0.05
- target_modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- lr: 1e-4, cosine decay, warmup_ratio 0.03, weight_decay 0.0
- epochs: 1, batch_size: 2, max_seq_len: 1024, optim: adamw_torch, bf16
- adapter out: `runs/openweight_smoke/adapters/sft-qwen25-15b-v009/`
- member_id: `sft-qwen25-15b-v009`, parent: `baseline-qwen25-15b`

## Post-SFT plan

1. `openweight.inference.merge_lora_to_dir` bakes the adapter into
   `runs/openweight_smoke/merged/sft-qwen25-15b-v009/`
2. Re-run `scripts.bench_openweight_member` with `--kind sft
   --model-path <merged_dir>` on the same bench set
3. Delta visible at `/openweight` in the dashboard (click A then B to
   select the two member rows — the compare panel opens with per-bench
   deltas + Δ%)

## Dashboard additions

- **New page** `/openweight` — vanilla-JS standalone (no React rebuild);
  launch panel (kind × base × benches × hypers), live launches table,
  members × benches grid, 2-row compare view with Δ/Δ%.
- **New APIs** `/api/openweight/{members,fitness_evals/{id},compare,runs,
  launches,launch,launch/{id},launch/{id}/log,launch/{id}/cancel}`.
- **Launch wiring** — `POST /api/openweight/launch` spawns a subprocess.
  When the dashboard runs on the compute node, that's local; when it
  runs elsewhere, point `--launch-dispatch` at
  `scripts/launch_openweight_on_node.sh` (which routes through
  `gpu-01 → → node-8`).
- **Launches DB** — side-car `launches.db` next to `lineage.db`; reap
  loop updates status every 3s.

## Reproducibility

Commands used, from `/home/<user>/sieve` on node-8 (sieve-venv activated):

```sh
# baseline smoke_mcq + ifeval (proxy)
python -m scripts.bench_openweight_member \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --member-id baseline-qwen25-15b --kind base \
  --tokenizer-family qwen25 --generation 0 \
  --benchmarks smoke_mcq,ifeval --proxy \
  --lineage-db runs/openweight_smoke/lineage.db --port 8001

# SFT
python -m scripts.sft_openweight_member \
  --base-model-id Qwen/Qwen2.5-1.5B-Instruct \
  --corpus-shard data/sft_shards/v009.jsonl \
  --member-id sft-qwen25-15b-v009 \
  --out-dir runs/openweight_smoke/adapters/sft-qwen25-15b-v009 \
  --lineage-db runs/openweight_smoke/lineage.db \
  --generation 1 --parent-ids '["baseline-qwen25-15b"]' \
  --epochs 1 --rank 16 --batch-size 2 --max-seq-len 1024

# merge + post-SFT bench
python -c "from pathlib import Path; from openweight.inference import merge_lora_to_dir; \
 merge_lora_to_dir('Qwen/Qwen2.5-1.5B-Instruct', \
 'runs/openweight_smoke/adapters/sft-qwen25-15b-v009', \
 Path('runs/openweight_smoke/merged/sft-qwen25-15b-v009'))"

python -m scripts.bench_openweight_member \
  --model-path runs/openweight_smoke/merged/sft-qwen25-15b-v009 \
  --member-id sft-qwen25-15b-v009 --kind sft \
  --base-model-id Qwen/Qwen2.5-1.5B-Instruct \
  --adapter-path runs/openweight_smoke/adapters/sft-qwen25-15b-v009 \
  --merged-checkpoint-path runs/openweight_smoke/merged/sft-qwen25-15b-v009 \
  --tokenizer-family qwen25 --generation 1 \
  --benchmarks smoke_mcq,ifeval --proxy \
  --lineage-db runs/openweight_smoke/lineage.db --port 8001

# dashboard (on node-8, browsed via local port-forward)
python -m dashboard.server --db runs/openweight_smoke/lineage.db --port 8096 --host 0.0.0.0
```

## Open questions / next

- Do we see any bench movement from 1 epoch on 22k rows? If yes, scale
  rank/epochs until diminishing returns; if no, the corpus/base-model
  fit is probably too narrow — pivot to a domain-adjacent corpus.
- Next member to try: Qwen2.5-Math-1.5B-Instruct (same family, tokenizer
  matches, specialist base — lets us test intra-family output ensembling).
- Extend `/openweight` with loss curves scraped from SFT trainer logs
  so users can watch training live (currently only final adapter shows up
  in the table). **Done 2026-04-21** — `DashboardMetricsCallback` +
  log-parse ingester + Chart.js panel. See the Dashboard additions section.

## Throughput tuning (next run)

Observed on this run: `bs=2, seq_len=1024, rank=16, bf16` on GB10 →
**1.1 s/it, ~11% MFU, 9.3 GB / ~24 GB HBM used, 60 W / ~150 W TDP, 95% util**.
GPU is pinned but tensor cores aren't saturated — attention/memory bound
at this batch size, not matmul bound. Three cheap wins for the next SFT:

1. **Bump `batch_size` to 4 (or 8 with `gradient_accumulation_steps=2`).**
   9 GB used out of ~24 GB leaves huge headroom. Doubles tokens/step,
   should roughly halve wall time per epoch.
2. **Set `attn_implementation="sdpa"` in `AutoModelForCausalLM.from_pretrained`.**
   Torch SDPA is already present in this stack; flash-attention-2 isn't
   (builds fail on aarch64 + torch-2.13-nightly, same reason axolotl is
   blocked). SDPA is the best attention path we have here and it's free.
3. **Wrap model in `torch.compile(model)` before `SFTTrainer`.** ~3 min
   of compile time amortized across ~11k steps is a clear win; typical
   HF-transformer speedup is 1.3–1.8× on decoder-only. Skip if iterating
   on a script that changes shapes often.

Wire these in `openweight/finetune.py` — all three are one-line changes
and the `SftConfig` already has `batch_size` / `optimizer` knobs to lift
them to genome-level.

## Headline delta

| benchmark | baseline | sft-v009-merged | Δ | Δ% | note |
|---|---|---|---|---|---|
| smoke_mcq | 0.95 (19/20, n=20) | 0.80 (4/5, n=5 proxy) | — | — | **not comparable**, different n |
| ifeval prompt-level loose (proxy=40) | 0.500 ± 0.080 | 0.425 ± 0.079 | −0.075 | −15% | within 1σ |
| ifeval prompt-level strict (proxy=40) | 0.400 ± 0.078 | 0.350 ± 0.076 | −0.050 | −12.5% | within 1σ |
| ifeval inst-level loose (proxy=40) | 0.571 | 0.524 | −0.048 | −8.4% | — |

## Discussion

**The dev loop works end-to-end.** That was the point of the experiment:
prove every edge of the openweight graph can carry a real job. It did:
lineage rows for base + sft members, two fitness_evals per member, merged
adapter on disk (3.0 GB), dashboard renders the delta, loss chart streamed
live. The plumbing is green.

**The nominal deltas are negative, but within noise.** ifeval prompt-level
loose moved 0.500 → 0.425 with a stderr of ~0.08 on both sides. |Δ|/σ ≈
0.94 — you can't reject the null that SFT did nothing. The strict-acc and
inst-level variants all move the same direction by similar magnitudes, so
it's not pure independent noise either; it's more consistent with "SFT
nudged the model very slightly away from ifeval's formatting prior, by an
amount we can't reliably measure with n=40 prompts." To call a real
regression we'd need the full ifeval (~540 prompts) and/or multiple seeds.

**smoke_mcq is broken as a comparison here.** Baseline ran with
`"proxy": false, n=20` and scored 19/20. Post-SFT ran with `"proxy": true,
n=5` and scored 4/5. A 4/5 binomial has stderr √(0.8·0.2/5) = 0.18 —
wider than the observed Δ. The reproducibility block at the bottom of
this file shows both commands pass `--proxy`, so the inconsistency is in
how the `smoke_mcq` adapter interprets `--proxy` between runs. Worth
checking `benchmarks/adapters/smoke_mcq.py` — either the flag was
recently added/honored, or proxy's default `n` changed. Either way: don't
trust the `smoke_mcq` line. Fix: re-bench both members with
`--benchmarks smoke_mcq --no-proxy` (n=20) before drawing conclusions.

**If the regression were real, the likely culprits are standard:**

1. *Corpus/bench mismatch.* `v009.jsonl` is 22k rows of book/paper
   paragraphs (clinical + psychoanalytic + engineering per the `corpus-host`
   corpus). ifeval tests *format following* on general instruction
   prompts — "write a haiku about X, then list three adjectives." One
   epoch of domain prose will mildly erode instruction-format priors
   before any new capability shows up on our bench set.
2. *1 epoch at rank 16 is below the "transfer" threshold.* LoRA transfer
   effects usually show up around 3-5 epochs or rank 32+; rank 16 × 1
   epoch is enough to mildly perturb the instruct-tuned weights without
   fully relearning the domain.
3. *Chat template drift.* `trl.SFTTrainer` on Qwen2.5-Instruct with
   plain `{"text": ...}` rows trains the model to continue prose, not to
   respond in chat format. The bench harness then wraps in ChatML.
   Possible that SFT shifted the logits toward "more prose continuation,
   less assistant-turn-format" — exactly the thing ifeval penalizes.

**None of the above are worth chasing right now.** The forward path is
the throughput-tuning changes in the next section (batch 4, sdpa,
torch.compile) followed by a corpus that's actually instruction-shaped —
candidate: re-pack `v009` into a ChatML instruction/response format
(simple: prompt = first paragraph, response = second paragraph; or
generate synthetic Q/A pairs from the corpus). That's a data-pipeline
change, not an infra change. Log as next experiment.

**Bench-set credibility note.** ifeval at n=40 has σ ≈ 0.08 on a score
near 0.5. We cannot distinguish a real 5% effect from noise at this
sample size. Any decision about whether SFT "helped" on a Lever-relevant
bench needs to come from the full eval set or a bench whose variance
shape we've characterized. This is the `val-set meta-overfitting` class
of risk flagged in `findings.md` 2026-04-18, running in the opposite
direction: here we'd underfit our confidence by taking the proxy at face
value.
