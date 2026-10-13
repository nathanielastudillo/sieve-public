# SIEVE benchmarks

Benchmarks addressed by string name in `FitnessConfig.benchmarks`. A
run's fitness aggregator calls each benchmark by name; every benchmark
returns a float score in `[0, 1]` plus a JSON raw result, and the
weighted sum (per `FitnessConfig.benchmark_weights`) becomes the
`bench_agg` component of `FitnessVector`.

## Layout

```
benchmarks/
├── README.md                    ← this file
├── ensemble_metrics/            ← diversity / calibration / ensemble-only harness code
│   ├── diversity_prompts.jsonl  ← 200-prompt probe set for §1.7 composite diversity
│   ├── disagreement.py          ← pairwise JSD on MCQ / free-form outputs
│   └── calibration.py           ← ECE on MCQ benchmarks
└── lever_custom/                ← Lever-domain benchmarks
    ├── specs/                   ← spec.md per benchmark (schema, size, oracle, scoring)
    │   ├── policy_qa.md
    │   ├── polling_reasoning.md
    │   ├── constraint_extraction.md
    │   └── claim_evidence_linking.md
    └── data/                    ← gitignored; pulled from corpus-host on demand
```

## External benchmarks (no code in this repo)

These are pulled in via `lm-eval-harness` and HELM. Adapters live in
`benchmarks/external/`:

| Name | Harness | Role |
|---|---|---|
| `ifeval` | lm-eval | Instruction-following structural correctness |
| `mmlu-pro` | lm-eval | General-reasoning floor |
| `gpqa-diamond` | lm-eval | Hard science MCQ, signals overfit to easy trivia |
| `mt-bench` | FastChat | Multi-turn open-ended quality (LLM-as-judge) |
| `ruler` | NVIDIA | Long-context stress (128K) |
| `hellaswag` | lm-eval | Cheap sanity check (used in smoke only) |

## Custom benchmarks (Lever domain)

Four Lever-specific benchmarks in `lever_custom/specs/`. Each has a
spec file describing:

- **schema**: input/output JSON shape the benchmark expects
- **size**: number of items, train/val/test split (if any)
- **oracle**: how ground truth is produced (manual, lex derivation,
  polling dataset, etc.)
- **scoring**: metric + aggregation (exact match, F1, custom)
- **status**: `spec-only` | `data-assembled` | `implemented`

All four are `spec-only` as of 2026-04-18. Data assembly is a
prerequisite for cluster run 1.

## Invariants

1. Every benchmark returns `(score: float in [0,1], raw: dict)`.
2. Scores are **higher = better**. If the underlying metric is
   error-like (ECE, cross-entropy), the adapter flips the sign here.
3. Benchmarks must be deterministic given seed. Nondeterminism in the
   serving stack (temperature>0) is allowed only for MT-Bench.
4. Fitness aggregator weights live in run configs, not here. A new
   benchmark should ship with a recommended default weight in its spec
   but not hard-code one.

## Adding a new benchmark

1. Write a spec file under `lever_custom/specs/` using the schema
   in any existing spec as a template.
2. Commit the spec with `status: spec-only`.
3. Produce the data under `lever_custom/data/{name}/` (not committed).
4. Implement `lever_custom/{name}.py` exporting
   `run(member_endpoint, cfg) -> (score, raw)`.
5. Register the name in `openweight.fitness.KNOWN_BENCHMARKS`.
6. Update the spec's `status` field.
