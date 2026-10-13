# SIEVE Data Pipeline

Functional, inspectable, resumable pipeline from a folder of extracted markdown
files to a train/val/test-split binary corpus ready for training.

At 100K–10M tokens every token is load-bearing, so every step is an auditable
transformation with per-document lineage preserved on disk.

## Layout

```
data/pipeline/
├── core.py        # Document, Paragraph, Provenance, @stage, pipe()
├── stages.py      # 11 pure text-cleaning stages + STANDARD_CLEAN_STAGES
├── paragraphs.py  # split, density score, MinHash dedup, stratified subsample
├── extract.py     # PDF/HTML → markdown (with caching + OCR fallback)
├── run.py         # orchestrator + CLI: build a corpus from data/extracted/
├── inspect.py     # CLI: walk any run, any doc, any stage
└── validate.py    # 15 structural + integrity checks for each run
```

## Design

A **Document** is an immutable `(Provenance, text, attrs)` triple. A **Stage**
is a pure `Document → Document`; the `@stage` decorator wraps a
`str → str | (str, notes)` function and appends a `StageRecord` (in/out chars,
wall_ms, notes, params) to the doc's provenance. `pipe(*stages)` composes them.

Because provenance is append-only, at any point you can reconstruct the full
history of any byte of text in the corpus: which file it came from, which
stages ran on it, how many chars each stage removed, how many matches each
regex fired, and whether its paragraph survived minhash + the budget sampler.

## Running

```bash
# Full corpus (writes data/corpus_vfull/ and data/pipeline/runs/full/):
python -m data.pipeline.run --version vfull --target-tokens 10000000 --run-id full

# Useful flags:
#   --source-dir data/extracted     (default)
#   --tokenizer v005                 tokenizer version under data/tokenizers/
#   --dedup-threshold 0.7            minhash Jaccard threshold
#   --stratification sqrt            sqrt | uniform | proportional
#   --keep-intermediates             write per-stage text snapshots to disk
#   --min-para-words 20
#   --min-doc-chars 500
#   --seed 42
```

The `--run-id` is the directory name under `data/pipeline/runs/`. A run dir
contains:

```
runs/<run_id>/
├── config.json            # exact flags + commit
├── summary.txt            # human-readable one-screen summary
├── index.jsonl            # one row per input doc: doc_id, title, sha256,
│                          # status, final_chars, every StageRecord
├── paragraphs.jsonl       # one row per paragraph: para_id, decision
│                          # (kept | dropped_minhash | dropped_subsample),
│                          # density, words, dup_of, text
└── stages/<stage_name>/<doc_id>.txt   # optional per-stage snapshots
                                       # when --keep-intermediates
```

`status` is one of `ok`, `skipped_too_small`, or `duplicate_sha256`. Byte-
identical files (same sha256, different filenames — e.g. textbook editions
that got extracted twice) are recorded as `duplicate_sha256` and skipped with
a `duplicate_of` pointer to the first occurrence.

## Inspecting

```bash
python -m data.pipeline.inspect list                  # all docs in latest run
python -m data.pipeline.inspect find <substr>          # doc by title/path
python -m data.pipeline.inspect trace <doc_id>         # stage-by-stage chars + notes
python -m data.pipeline.inspect show <doc_id> <stage>  # text at a stage (needs --keep-intermediates)
python -m data.pipeline.inspect diff <doc_id> <stage_a> <stage_b>
python -m data.pipeline.inspect paragraphs <doc_id>    # per-paragraph kept/dropped + density
python -m data.pipeline.inspect para <para_id>         # single paragraph record
python -m data.pipeline.inspect corpus                 # per-source: split→mh→pick, tokens
python -m data.pipeline.inspect summary                # run summary.txt
# every command takes --run <run_id> to pin a non-latest run.
```

`corpus` answers the original question — "what actually made it in?":

```
 tokens     split→mh→pick  density≥   title
 111792  15730→14833→1625  1.0        U.S. Navy Electricity and Electronics Course…
  87315    7482→7469→1112  1.0        THE GREAT CONVERSATION — Norman Melchert
  …
```

`split→mh→pick` shows per-document paragraph flow: how many paragraphs the
doc produced after splitting, how many survived minhash dedup, and how many
were picked by the stratified budget sampler. Everything else was dropped as
overflow / below-density for that doc's token budget.

## Validating

```bash
python -m data.pipeline.validate --run full
```

Runs 15 checks:

- structural: required files, well-formed jsonl, unique doc_ids
- stage chain: every doc has a non-empty stage chain; stage names match
  `STANDARD_CLEAN_STAGES` spec exactly
- snapshots: if `--keep-intermediates`, per-stage snapshots exist
- paragraphs: well-formed, reference only `ok`-status docs, counts match
- per-source flow: `split = minhash_dropped + after_minhash`, and
  `after_minhash = picked + overflow`
- metadata consistency: `corpus_v*/metadata.json` agrees with the run
- bin file sizes: `train/val/test.bin` byte counts match metadata tokens × 2
- sources files: `sources.txt`, `sources.jsonl`, `sources.annotated.md` agree
- per-source tokens sum to corpus total (±2% fudge for block-padding)
- status counts: `ok + skipped_too_small + duplicate_sha256` = total index rows

Every check prints `[PASS]` / `[FAIL]` with the count it checked against.
Exit code is non-zero if any check fails.

## Full-corpus run of record (2026-04-18)

```
runs/full
  source_dir:   data/extracted  (1537 files)
  corpus:       data/corpus_vfull/
  tokenizer:    custom v005  (vocab=8000)
  total_tokens: 8,770,126   (train 7,455,232 / val 876,544 / test 438,272)
  paragraphs:   split=346,058  kept=83,350  mh_drop=15,681  of_drop=247,027
  status:       1406 ok  89 too_small  42 duplicate_sha256
  corpus_sha:   b64db966bc641a3…
```

All 15 validation checks pass.

## Training on this corpus

`config/profiles/vfull.yaml` is the run-of-record profile pointed at this
corpus. For parallel dispatch across the GB10 fleet (which is where any
real SIEVE run should run — single-node training defeats the population
premise), see [`cluster.md`](cluster.md).
