# Lab Notes

Comprehensive running log of SIEVE experiments, data, artifacts, and findings.
Append-only in spirit — update existing pages when state changes, don't
rewrite history. If something is wrong, cross it out with a note about *when*
and *why* it was corrected.

## Navigation

| File | What's in it |
|------|-------------|
| [corpora.md](corpora.md) | Every corpus version built (v001…v006+), token counts, source lists, dedup status |
| [tokenizers.md](tokenizers.md) | Every BPE version, vocab size, domain-integrity metric |
| [artifacts.md](artifacts.md) | Full inventory of what sits on disk under `data/` and `runs/` |
| [experiments/](experiments/) | One page per run, dated, with full parameters and outcomes |
| [findings.md](findings.md) | Running list of empirical surprises and the questions they opened |
| [strategy-cluster-run.md](strategy-cluster-run.md) | Plan for moving from 1× GB10 smoke to 12× GB10 compute-cluster |
| [research-questions/](research-questions/) | Open questions with design docs (tokenizer scaling, etc.) |
| [research-questions/literature-small-corpus-ensembles.md](research-questions/literature-small-corpus-ensembles.md) | Literature review on training transformer ensembles at 10⁵–10⁸ tokens (diversity sources, aggregation methods, PBT/evolutionary, theoretical ceiling, regularization, small-data failure modes) |
| [experiments-planned.md](experiments-planned.md) | Queue of "probably-good" experiments (B1–B9) from the ensemble lit review, with design + expected-signal for each |

## How to use

- **When you build a new corpus**: append a row to `corpora.md` and, if it's
  a new BPE, a row to `tokenizers.md`.
- **When you kick off a training run**: create `experiments/NNN-<slug>.md`
  *before* starting so the parameters are captured even if the run crashes.
  Update the Results section when it completes (or fails).
- **When you discover something that surprises you**: add a bullet to
  `findings.md` with the date and a link to the experiment.
- **When an artifact moves or changes**: update `artifacts.md`.

## Date format

Use ISO-8601 (`2026-04-17`). Relative dates ("yesterday", "last week") decay;
absolute ones don't.

## What counts as an "experiment"

Any training run or systematic evaluation that produces a number we care
about. Includes: smoke loops, single-model ablations, test-set benchmarks,
tokenizer comparisons. Does **not** include: one-off script runs, syntax
checks, import-errors-debugging.

## Quick status (2026-04-17, late afternoon)

- **2 completed smoke runs**:
  exp-001 (v003 corpus, v002 tok) → ens val 8.7315;
  exp-002 (v005 corpus, v002 tok) → ens val 8.8247
- **1 smoke run in flight** (gpu-01, started 16:48):
  exp-004 (v006 corpus, v003 tok) — tests tokenizer-mismatch hypothesis
- **Tokenizers**: v002 (16K, 21 sources, 65.7% domain integrity),
  **v003** (16K, 37 sources, **75.7%** domain integrity, trained today)
- **Checkpoints on disk**: 36 transformers from completed runs,
  more incoming from exp-004
- **Strategy doc written**: [strategy-cluster-run.md](strategy-cluster-run.md)
  covers the path to 12× GB10 compute-cluster
- **Research doc in progress**: [research-questions/tokenizer-scaling.md](research-questions/tokenizer-scaling.md)
  analyzes whether bigger-BPE / multi-BPE / bigger-data is the right
  next lever; literature review attached when agent completes
