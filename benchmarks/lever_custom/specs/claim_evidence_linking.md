# Benchmark: `lever/claim_evidence_linking`

Status: `spec-only`
Default weight in `FitnessConfig`: `0.6`
Oracle owner: lever-engine citation graph + manual review

## Purpose

Given a policy claim in natural language, identify which documents in
a candidate pool support, contradict, or are neutral toward the claim,
with a citation-level pointer. This is the retrieval-reasoning bridge
that lever-engine needs for its scrollytelling pipeline and for the
content-ID transclusion feature.

Distinct from `policy_qa` in that the member is **not** generating an
answer — it is making structured evidence-relation judgments over a
fixed candidate set.

## Schema

```json
// input
{
  "id": "clink-000052",
  "claim": "Toronto's 2026 housing starts are down >15% year-over-year.",
  "candidates": [
    {"doc_id": "pig:data:meta:a1b2...", "snippet": "<400-char excerpt>"},
    {"doc_id": "pig:data:meta:c3d4...", "snippet": "<400-char excerpt>"},
    ...  // k = 8..12 candidates per item
  ],
  "gold_labels": {
    "pig:data:meta:a1b2...": "supports",
    "pig:data:meta:c3d4...": "neutral",
    "pig:data:meta:e5f6...": "contradicts",
    ...
  }
}

// output
{
  "labels": {"<doc_id>": "supports|contradicts|neutral"}
}
```

## Size + splits

- Target: 300 items × ~10 candidates ≈ 3000 gold labels.
- 250 eval / 50 sequestered.
- Domain balance: 40% housing/urban, 30% immigration/demographics,
  30% fiscal/budget — where Lever's first deployments are expected.

## Oracle

Semi-automatic:

1. lever-engine's claim-extraction pass produces candidate claims
   from news + policy briefs.
2. Retrieval over the `pig` data lake pulls k=15 candidates per claim
   via hybrid BM25 + embedding search.
3. Manual annotator reviews each candidate and assigns
   `{supports, contradicts, neutral}`. Discards items where fewer
   than 2 `supports` or fewer than 2 `contradicts` are available after
   annotation — the label distribution must be non-degenerate.
4. Candidates are truncated to top-k=8..12 post-annotation to keep
   per-item serving cost bounded.

## Scoring

- **Macro-F1 over the 3 classes** (1.0 weight): weights `supports`
  and `contradicts` heavily (those are rare); `neutral` is the majority
  class and macro-averaging prevents it from dominating.

Aggregate: mean macro-F1 over items. Range `[0, 1]`.

## Hazards

- **Label imbalance.** Neutral is ~65% of labels. Micro-F1 would
  reward "always predict neutral". Macro-F1 chosen explicitly.
- **Citation-context smuggling.** Because the snippet includes the
  evidence in-line, the benchmark measures close-reading more than
  retrieval. That is intentional — retrieval is a separate pipeline
  stage in lever-engine, owned elsewhere.
- **Annotator disagreement on "supports" vs. "neutral".** Inter-
  annotator agreement floor is ~0.78 Cohen's κ on this task; a member
  reaching that is effectively at ceiling for this oracle.
