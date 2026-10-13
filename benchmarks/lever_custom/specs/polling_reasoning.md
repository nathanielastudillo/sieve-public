# Benchmark: `lever/polling_reasoning`

Status: `spec-only`
Default weight in `FitnessConfig`: `0.7`
Oracle owner: Lever polling pipeline (programmatic) + spot review

## Purpose

Quantitative + qualitative reasoning over Canadian polling and survey
data. Tests whether the member can (a) read a structured polling
summary, (b) answer questions requiring multi-step arithmetic + sample
reasoning, (c) correctly flag when a claim is unsupportable given the
cited survey's sample size or margin of error.

## Schema

```json
// input item
{
  "id": "pollq-000045",
  "survey_snapshot": {
    "pollster": "Abacus Data",
    "field_dates": "2026-02-01 .. 2026-02-07",
    "n": 1200,
    "moe_pct": 2.8,
    "sample_frame": "eligible_voters_ca",
    "results": [
      {"question": "Vote choice", "option": "Liberal", "pct": 31.4},
      ...
    ]
  },
  "question": "If the MoE is ±2.8% at n=1200, is the 3-point Liberal...",
  "gold_answer": "No, the 3-point lead is within MoE at 95% confidence...",
  "gold_reasoning_tags": ["moe", "overlap", "cl_95"]
}
```

## Size + splits

- Target: 500 items.
- 400 eval / 100 sequestered.
- ~60% numerical-reasoning (MoE, confidence intervals, weighted
  aggregation), ~40% interpretive-reasoning (sample-frame caveats,
  pollster house effect, trend-vs-noise calls).

## Oracle

- Survey snapshots: extracted automatically from Lever's polling
  pipeline's JSON output. Real fields from real polls.
- Questions: generated from a template library (`lever-engine/polling/
  benchmark_templates.py`) that emits one question per tagged
  reasoning skill, with gold answer derived from a reference
  computation (numpy + scipy for MoE / CI items).
- Hand-written review on 50-item sample before admission of each
  template; template-level error rate must be <2% before admission.

Template-generated items are fine here because the *reasoning skill
inventory* (what skills should we test?) is hand-designed; only the
*surface content* is auto-assembled from real polls.

## Scoring

- **Exact-answer score** (0.5 weight): for numerical items, string match
  on the canonical numeric form (rounded to spec in template).
- **Reasoning-tag score** (0.5 weight): for the model's reasoning
  trace, award 1/N for each `gold_reasoning_tag` present in the output
  (regex-matched against a small canonical-phrase list per tag). N is
  the number of gold tags for the item.

Aggregate: mean over all items. Range `[0, 1]`.

## Hazards

- **Prompt-sensitivity of numerical reasoning.** The same model may
  solve the MoE calculation given a chain-of-thought prompt and fail
  given a terse prompt. Benchmark uses a fixed terse prompt template
  — that is the choice; it biases against verbose-by-default models
  but measures a real product constraint (API callers don't always
  send CoT prompts).
- **Template-gaming.** If a member fine-tunes on polling corpus with
  similar template structures, the tag-matching score inflates.
  Holdout tags (present in gold but filtered out of public corpus)
  would be the mitigation; out of scope for v1.
