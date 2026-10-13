# SIEVE — Solomonoff-Inspired Evolutionary Ensemble

Population-based training of small transformer ensembles, where the unit of selection is the
**ensemble**, not the individual model.

The premise: Solomonoff induction is a weighted mixture over all programs consistent with the
data, and it is provably optimal but uncomputable. Neural networks approximate it through a
simplicity bias; ensembles approximate it better because they maintain several solutions
instead of committing to one. SIEVE turns that into a training loop by applying evolutionary
pressure to *ensemble composition* — selecting for members that improve the best ensemble they
participate in, which favours diversity and complementarity over individual quality.

## Status — read this before judging the results

This repository is a research codebase, published while the work is live. The honest summary:

- **The small-corpus track is currently blocked.** From-scratch ~7M-parameter GPTs on curated
  ~100K-token corpora are beaten by a zero-parameter unigram model on the held-out set — a
  vocabulary-to-training-tokens pathology recorded in `lab-notes/findings.md` as the
  "Magikarp pathology" (2026-04-17). `scripts/magikarp_check.py` is the guard; run it on a
  corpus/tokenizer pair before training anything new.
- **The last cluster run did not complete.** `exp007_v009` was lost to a dispatch timeout
  (distinct from the aggregator blow-up documented in `docs/ensemble_loss_blowup.md`); 10
  checkpoints were recovered under `runs/exp007_v009/posthoc/checkpoints/` and a post-hoc
  ensemble evaluation is still pending.
- **The open-weight track is specified and partly implemented, not yet run.** LoRA/DoRA
  fine-tune → merge → benchmark for 7B–13B seeds (`docs/architecture-openweight.md`,
  `openweight/`). No SFT has been dispatched.
- **No ensemble results are published here.** `results/` is gitignored, so the numbers live in
  `lab-notes/` (including negative results such as v005 regressing against v003). Docs that
  describe the architecture are design records, not claims of measured performance.

`CLAUDE.md` is the operational state and the orientation file; it is honest about what is
uncommitted. Read `lab-notes/findings.md` before asking what has been tried.

## Layout

```
sieve/
├── CLAUDE.md            operational state + full technical framing (start here)
├── controller/          evolutionary loop, population, ensemble eval, lineage, dispatch
├── worker/              per-node daemon, trainer, evaluator, checkpoint, openweight runner
├── model/               transformer definition, config, Muon optimizer
├── data/                extraction, cleaning, BPE, manifest diff, corpus builds
├── openweight/          LoRA fine-tune / merge / inference for the open-weight track
├── benchmarks/          adapter registry (smoke MCQ canary, lm-eval tasks)
├── dashboard/           run monitor + open-weight playground (FastAPI + React)
├── lab/                 checkpoint exploration + ensemble generation UI
├── analysis/            diversity, convergence, visualization
├── lab-notes/           chronological empirical log, per-run writeups, literature reviews
├── teaching/            01_tokens.md → 09_ensemble.md walkthrough + exercises
└── docs/                cluster layout, architecture, loss-blow-up post-mortem, data pipeline
```

## Running it

Requires Python 3.11+ and PyTorch 2.x. There is no lockfile yet — dependencies are declared
with lower bounds only in `pyproject.toml`.

The documented entry point is `scripts/run_sieve.sh`. To see what a run would do without
touching the cluster:

```bash
./scripts/run_sieve.sh --config config/profiles/smoke.yaml --dry-run
```

The orchestrator itself is a module with its own options
(`python -m controller.orchestrator --help`); `scripts/run_sieve.sh` is the wrapper that
resolves the config, discovers worker nodes and can deploy to them.

The from-scratch pipeline is `data/extract.py` → `data/clean.py` → `data/build_corpus.py`; the
staged successor is `data/pipeline/` (see `docs/data_pipeline.md`). Distribution across
worker nodes is coordinated through a Redis queue (`controller/dispatch.py`,
`worker/daemon.py`); `docs/cluster.md` describes the fleet layout.

**This is a sanitised snapshot** — host names, addresses and paths appear as placeholders,
and the dispatch scripts are readable but not runnable against the original fleet. See
`PUBLISHING.md` for exactly what was changed and why.

## Data and corpora

The training corpora are **not** redistributed here, and the per-corpus metadata that records
where each book came from is deliberately withheld from the public tree. Only source code,
configurations, tokenizer configuration (not artifacts), and the write-ups in `lab-notes/`
are published. If you want to reproduce a run, you will need to assemble an equivalently
dense corpus yourself; the curation heuristics are documented in `CLAUDE.md`.

## License

MIT — see `LICENSE`. Note that the licence covers the code in this repository only, not any
corpus material, model weights, or third-party texts referenced in the notes.
