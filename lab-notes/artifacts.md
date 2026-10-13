# Artifacts

What sits on disk, where, and how large. All paths are relative to the
repo root. Updated 2026-04-17.

## Data (`data/`)

```
data/
├── raw/                    3.0 GB    PDFs + (after 2026-04-17) epubs
│   ├── *.pdf               source PDFs originally rsync'd from corpus-host
│   └── corpus-expansion/      152 MB    second fetch (2026-04-17)
│       ├── epub/           10 epubs  Deleuze/Schuster/Mitchell/etc.
│       └── pdf/            8 PDFs    Lacan seminars, Kristeva, Latour, Fink
├── extracted/              34 MB     per-source markdown, one .md per book
│   ├── extraction_manifest.json      from data/extract.py
│   ├── epub_extraction_manifest.json from scripts/extract_epubs.py
│   └── reextraction_manifest.json    from scripts/reextract_broken_pdfs.py
├── cleaned/                34 MB     post-clean.py (NFKC, dedup, whitespace)
├── corpus_v001/            19 MB     tiktoken GPT-2, ~3M tokens
├── corpus_v002/            19 MB     custom 16K BPE, ~3M tokens
├── corpus_v003/            676 KB    99.7K tokens, 16 sources       <-- used by exp-001
├── corpus_v004/            46 MB     7.52M tokens, 37 sources       <-- expanded pool
├── corpus_v005/            628 KB    97.1K tokens, 35 sources       <-- used by exp-002
└── tokenizers/             1.8 MB
    ├── v001/               8K vocab (historical)
    └── v002/               16K vocab (current; used by v003 + v005)
```

## Runs (`runs/`)

```
runs/
├── smoke_loop/             1.5 GB    experiment 001 — v003 corpus
│   ├── checkpoints/
│   │   ├── gen-000-{6 ids}/*.pt       6 gen-0 members
│   │   ├── gen-001-{6 ids}/*.pt       6 gen-1 members
│   │   └── gen-002-{6 ids}/*.pt       6 gen-2 members
│   │                                  = 18 checkpoints, 7.4M params, ~88 MB each
│   └── results/
│       ├── lineage.db               SQLite; polluted with pre-fix debug runs
│       │                            (134 members rows; 18 canonical on disk)
│       ├── best_ensemble.json       3-member spec, val_loss 8.7315
│       └── gen2_final_rankings.json C(6,3)=20 combo rankings
│
└── smoke_loop_v005/        1.5 GB    experiment 002 — v005 corpus
    ├── checkpoints/                  same layout, 18 checkpoints
    ├── logs/run.log                  buffered stdout from the kicked-off run
    └── results/
        ├── lineage.db               Clean (18 members, 60 ensembles, 3 gens)
        └── best_ensemble.json       3-member spec, val_loss 8.8247
```

## Lineage DB schema

All three tables live in `results/lineage.db` per run.

```sql
CREATE TABLE members (
    id TEXT, generation INT, config_json TEXT,
    val_loss REAL, parent_ids TEXT, checkpoint_path TEXT,
    created_at TIMESTAMP
);
CREATE TABLE ensembles (
    id TEXT, generation INT, member_ids_json TEXT,
    ensemble_val_loss REAL, diversity_score REAL,
    created_at TIMESTAMP
);
CREATE TABLE generations (
    num INT, best_ensemble_loss REAL, best_individual_loss REAL,
    population_diversity REAL, wallclock_seconds REAL,
    created_at TIMESTAMP
);
```

Useful one-liners:

```bash
# per-gen progression
sqlite3 runs/smoke_loop_v005/results/lineage.db \
  "SELECT num, best_ensemble_loss, wallclock_seconds FROM generations ORDER BY num;"

# all ensembles at gen 2, worst to best
sqlite3 runs/smoke_loop_v005/results/lineage.db \
  "SELECT ensemble_val_loss, diversity_score FROM ensembles
   WHERE generation=2 ORDER BY ensemble_val_loss DESC;"

# member val losses (individual, after training — near-zero = memorized)
sqlite3 runs/smoke_loop_v005/results/lineage.db \
  "SELECT id, generation, val_loss FROM members ORDER BY generation, val_loss;"
```

## Known state oddities

- **v003 lineage.db contains 134 members + 260 ensembles across 10 "generations"**.
  Only the first 3 generations correspond to the 18 canonical checkpoints on
  disk. Gens 3-9 in the DB are debris from pre-fix debug attempts — wallclock
  times of 35-114s (vs ~5500s for a real gen) are the giveaway. Ignore those
  rows; trust the checkpoints directory as ground truth.
- **v005 lineage.db is clean** (18 / 60 / 3) — built after all orchestrator
  fixes were in place.
- **`scripts/` dir** has 5 session-created diagnostics: `eval_existing_gen0.py`
  (one-off gen-0 eval), `extract_epubs.py` (epub→md),
  `finalize_smoke_loop.py` (post-hoc `best_ensemble.json` builder),
  `reextract_broken_pdfs.py` (pymupdf raw fallback),
  `test_eval_after_profile_fix.py` (regression test for profile overrides)

## Uncommitted as of 2026-04-17

```
M  controller/orchestrator.py             # 3 profile-override bugfixes
?? config/profiles/smoke.yaml              # smoke profile
?? data/corpus_v004/
?? data/corpus_v005/
?? manual.md
?? lab-notes/                              # this directory
?? scripts/{5 diagnostic scripts}
```

Suggested atomic commits when ready:
1. `fix: orchestrator profile overrides (block_size, data_dir, results_dir, lineage_db)`
2. `infra: smoke profile + diagnostic scripts`
3. `data: pymupdf raw-text fallback for 7 under-extracted PDFs`
4. `data: corpus_v004 + v005 from expanded 37-source pool`
5. `docs: manual.md + lab-notes/`
