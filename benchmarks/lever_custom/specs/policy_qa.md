# Benchmark: `lever/policy_qa`

Status: `spec-only`
Default weight in `FitnessConfig`: `1.0`
Oracle owner: lex + manual review

## Purpose

Free-form QA over Canadian policy, regulation, and statute text. This
is the closest benchmark to the actual deployed use case (answering
policy questions grounded in the lex corpus) and therefore the primary
fitness signal for domain quality. Other benchmarks defend against
over-specialization; this one measures the thing we actually want.

## Schema

```json
// input item
{
  "id": "polqa-000123",
  "jurisdiction": "ca-federal|ca-on|ca-toronto|...",
  "question": "Under the Canada Elections Act, what is the maximum...",
  "context_refs": ["lex:section:cea/s.350"],
  "gold_answer": "The maximum is $X, per s.350(1).",
  "gold_citations": ["lex:section:cea/s.350"]
}

// output (what the member returns)
{
  "answer": "<free-form text>",
  "citations": ["lex:section:..."]   // optional; scored as bonus if provided
}
```

## Size + splits

- Target: 800 items at initial data-assembly.
- Split: 600 eval / 200 sequestered holdout. Sequestered set is used
  only for the final reporting pass at the end of a cluster run; never
  in fitness.
- Jurisdictional mix: ~60% federal, ~20% Ontario provincial, ~20%
  municipal (Toronto + Ottawa).

## Oracle

Gold answers + citations are produced by:

1. lex crawler provides the canonical section text.
2. A human author (with legal-adjacent background; the `pig` team has
   resources here) writes a question that the section cleanly answers.
3. Gold answer is written from the section; gold citation is the
   section's `lex:section:...` id.
4. Each item is reviewed by a second author before admission.

No LLM-generated golds. Authoring budget: ~1 item / 8 min → ~107 person-hours.

## Scoring

Composite, range `[0, 1]`:

- **Answer score** (0.7 weight): LLM-as-judge (GPT-4-class) comparing
  member's `answer` against `gold_answer`. Prompt constrains judge to
  score 0/0.5/1 on {factually correct, cites correct authority,
  complete}. Three-sample majority vote.
- **Citation score** (0.3 weight): recall@any of `gold_citations` in
  `output.citations`. If output has no citations, this is 0.

Aggregate: mean over all items.

## Hazards to guard against

- **LLM-as-judge bias toward verbose answers.** Mitigated by the
  rubric forcing 0/0.5/1 on three axes, and by periodically spot-checking
  judge accuracy against a 50-item human-scored subset.
- **Section-level gold leakage via the context window.** Do not include
  `context_refs` in the prompt for this benchmark — the member must
  retrieve. (A separate RAG-style benchmark can be added later.)
- **Training-on-test.** All items must post-date the fine-tuning
  corpus shards. Data-assembly tracks per-item `authored_at` and
  filters corpus crawl dates ≥ the earliest benchmark authored_at.
