# Bench data assembly

Scripts that produce `benchmarks/lever_custom/data/{bench}/items.jsonl`
from source corpora. Three paths depending on authorship model:

| Bench | Path | Authorship |
|---|---|---|
| `polling_reasoning` | fully programmatic | template-generated; gold computed via numpy/scipy |
| `constraint_extraction` | seeded | pulls statute sections from lex; gold annotated by human |
| `policy_qa` | seeded | pulls sections from lex; question+answer authored by human |
| `claim_evidence_linking` | seeded | pulls claims from lever-engine, candidates via retrieval; labels by human |

Every assembler writes a `manifest.json` next to `items.jsonl` recording:
- source sha256s for provenance
- assembly seed + datetime
- authored/seeded item counts
- for seeded benches: how many items remain un-authored

Re-running an assembler is idempotent: existing items (by `id`) are kept
as-is; only new items and missing metadata are updated. Authors never
lose work to a rerun.

## Running

```bash
# polling_reasoning — pure code, runs anywhere
python -m benchmarks.lever_custom.assemble.assemble_polling_reasoning \
  --polling-snapshots data/polling/snapshots.jsonl \
  --out benchmarks/lever_custom/data/polling_reasoning/

# constraint_extraction — needs lex section pool
python -m benchmarks.lever_custom.assemble.assemble_constraint_extraction \
  --lex-sections data/lex/sections_federal.jsonl \
  --target-items 400 \
  --out benchmarks/lever_custom/data/constraint_extraction/

# policy_qa — same lex pool, different item shape
python -m benchmarks.lever_custom.assemble.assemble_policy_qa \
  --lex-sections data/lex/sections_federal.jsonl \
  --jurisdictional-mix federal=0.6,ontario=0.2,toronto=0.1,ottawa=0.1 \
  --target-items 800 \
  --out benchmarks/lever_custom/data/policy_qa/

# claim_evidence_linking — needs pig claim+candidate dump
python -m benchmarks.lever_custom.assemble.assemble_claim_evidence_linking \
  --claims data/lever/claims.jsonl \
  --candidates-index data/pig/snippets_index.jsonl \
  --target-items 300 \
  --out benchmarks/lever_custom/data/claim_evidence_linking/
```

None of these requires live Redis — they consume pre-dumped jsonl so
the assembly stage is decoupled from the Basin stack.
