# Benchmark: `lever/constraint_extraction`

Status: `spec-only`
Default weight in `FitnessConfig`: `0.8`
Oracle owner: lex-crawled statute text + manual annotation

## Purpose

Given a section of Canadian statute or regulation, extract the formal
constraints it imposes as a structured list. This is a direct
component of lever-engine's "statute → formal constraint" pipeline, so
a member that does well here accelerates downstream work regardless of
what else it can do.

This benchmark is deliberately narrow: it measures structured extraction,
not reasoning, not generation quality.

## Schema

```json
// input
{
  "id": "cext-000211",
  "jurisdiction": "ca-federal",
  "section_id": "lex:section:ita/s.118.3",
  "section_text": "<the full text of the section, ~200-800 words>",
  "gold_constraints": [
    {
      "subject": "individual",
      "predicate": "may claim",
      "object": "disability tax credit",
      "condition": "has been certified by a medical practitioner under s.118.3(1)(a.2)",
      "amount": null,
      "cites": ["ita/s.118.3(1)(a.2)"]
    },
    ...
  ]
}

// output
{
  "constraints": [ {...same shape...} ]
}
```

## Size + splits

- Target: 400 sections × avg. 3 constraints/section ≈ 1200 gold constraints.
- 300 eval / 100 sequestered (section-level, not constraint-level, to
  avoid within-section leakage).
- Jurisdictional mix: 70% federal (broader corpus), 30% provincial
  (ON + BC) to exercise cross-jurisdiction generalization.

## Oracle

Full manual annotation by a human familiar with statute structure.
Items are admitted in batches of 50 after double-annotation agreement
≥90% on the constraint-set Jaccard.

No LLM-extracted golds. This benchmark is specifically designed to
measure what LLMs can't do reliably yet, so LLM-extracted golds would
collapse the signal.

## Scoring

Composite over constraints in each section, then mean over sections:

- **Constraint-set F1** (0.7 weight): treating each gold constraint as
  correct if the output contains a constraint whose (subject, predicate,
  object) triple normalizes to the gold's (case-insensitive, lemmatized)
  and whose `condition` has token-F1 ≥ 0.6 with gold. Precision / recall /
  F1 computed per-section on the matched-constraint graph.
- **Citation score** (0.3 weight): for constraints matched by triple,
  citation recall@any against gold `cites`.

Aggregate per benchmark: mean F1 across sections. Range `[0, 1]`.

## Hazards

- **Structural ambiguity in gold.** "The minister may, where
  appropriate..." can be annotated as 1 constraint or 2. Annotation
  guideline: *one constraint per independent normative clause*; double-
  annotator agreement catches drift. Still, expect ~5% floor from
  annotation disagreement that no model can close.
- **Reward hacking via over-production.** A member that emits 50
  constraints per section will inflate recall at a precision cost.
  Mitigated by using F1, not recall.
- **Schema drift.** If the output schema's keys are slightly wrong
  (e.g., `"who"` instead of `"subject"`), the entire section scores 0.
  Adapter applies a tolerant JSON repair pass before scoring.
