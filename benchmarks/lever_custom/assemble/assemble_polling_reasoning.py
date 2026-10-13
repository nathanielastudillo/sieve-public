"""Assemble the ``lever/polling_reasoning`` benchmark.

Fully programmatic: real polling snapshots × hand-designed question
templates × numpy/scipy-computed gold answers. Output schema matches
``benchmarks/lever_custom/specs/polling_reasoning.md``.

Input: a jsonl of polling snapshots. Each record is:

    {
      "pollster": "Abacus Data",
      "field_dates": "2026-02-01 .. 2026-02-07",
      "n": 1200,
      "moe_pct": 2.8,
      "sample_frame": "eligible_voters_ca",
      "results": [{"question": "...", "option": "...", "pct": ...}]
    }

The assembler pairs each snapshot with one or more templates from
``TEMPLATES`` below and emits gold-graded items.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from benchmarks.lever_custom.assemble._common import (
    Manifest,
    deterministic_id,
    iter_jsonl,
    load_existing_items,
    now_iso,
    sha256_file,
    write_items,
    write_manifest,
)

BENCH = "pollq"


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


@dataclass
class Template:
    """A question template with a gold-answer generator.

    ``generate`` returns ``(question: str, gold_answer: str, reasoning_tags: list[str])``
    from a snapshot. Returning ``None`` means the template is not applicable
    to this snapshot (e.g. not enough options for a gap calculation).
    """

    name: str
    generate: Callable[[dict[str, Any]], tuple[str, str, list[str]] | None]


def _tmpl_moe_overlap(snap: dict[str, Any]) -> tuple[str, str, list[str]] | None:
    results = snap.get("results", [])
    if len(results) < 2:
        return None
    results = sorted(results, key=lambda r: -r["pct"])
    a, b = results[0], results[1]
    gap = a["pct"] - b["pct"]
    moe = snap["moe_pct"]
    within_moe = gap < 2 * moe
    q = (
        f"The {snap['pollster']} poll ({snap['field_dates']}, n={snap['n']}, "
        f"MoE ±{moe}%) shows {a['option']} at {a['pct']:.1f}% and "
        f"{b['option']} at {b['pct']:.1f}%. Is the {gap:.1f}-point lead "
        f"statistically significant at 95% confidence?"
    )
    if within_moe:
        ans = (
            f"No, the {gap:.1f}-point gap is within the ±{moe}% margin of error "
            f"(effective two-sided {2*moe}% band) — the lead is not "
            f"statistically significant at 95% confidence."
        )
    else:
        ans = (
            f"Yes, the {gap:.1f}-point gap exceeds twice the ±{moe}% MoE "
            f"({2*moe}%), so the lead is statistically significant at 95% confidence."
        )
    return q, ans, ["moe", "overlap", "cl_95"]


def _tmpl_sample_frame_caveat(snap: dict[str, Any]) -> tuple[str, str, list[str]] | None:
    frame = snap.get("sample_frame", "")
    if frame != "eligible_voters_ca":
        return None
    results = snap.get("results", [])
    if not results:
        return None
    top = max(results, key=lambda r: r["pct"])
    q = (
        f"The {snap['pollster']} poll surveyed eligible_voters_ca (not likely "
        f"voters). {top['option']} leads at {top['pct']:.1f}%. Can this "
        f"number be used directly to project election outcomes?"
    )
    ans = (
        "No — an eligible-voters frame does not account for turnout. Eligible "
        "voters who do not vote are included; actual electoral outcomes depend "
        "on a likely-voter screen or turnout weighting."
    )
    return q, ans, ["sample_frame", "turnout", "frame_caveat"]


def _tmpl_margin_precision(snap: dict[str, Any]) -> tuple[str, str, list[str]] | None:
    n = snap.get("n")
    moe = snap.get("moe_pct")
    if not n or not moe:
        return None
    # z=1.96 at 95%, worst-case p=0.5 → MoE ≈ 1.96 * sqrt(0.25/n) * 100
    implied_moe = 1.96 * math.sqrt(0.25 / n) * 100
    q = (
        f"A poll of n={n} reports MoE ±{moe}% at 95% confidence. Is this "
        f"consistent with the theoretical worst-case margin of error for a "
        f"simple random sample of this size?"
    )
    if abs(implied_moe - moe) < 0.3:
        ans = (
            f"Yes, the theoretical worst-case MoE for n={n} at 95% confidence "
            f"(p=0.5) is approximately ±{implied_moe:.1f}%, consistent with "
            f"the reported ±{moe}%."
        )
    else:
        ans = (
            f"No, the theoretical worst-case MoE for n={n} at 95% confidence "
            f"(p=0.5) is approximately ±{implied_moe:.1f}%, which differs "
            f"from the reported ±{moe}% — the pollster likely used a design "
            f"effect or weighted MoE."
        )
    return q, ans, ["moe", "design_effect", "cl_95"]


TEMPLATES: list[Template] = [
    Template("moe_overlap", _tmpl_moe_overlap),
    Template("sample_frame_caveat", _tmpl_sample_frame_caveat),
    Template("margin_precision", _tmpl_margin_precision),
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def assemble(
    snapshots_path: Path, out_dir: Path, seed: int = 0, max_items: int | None = None,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    items_path = out_dir / "items.jsonl"
    existing = load_existing_items(items_path)
    rng = random.Random(seed)

    items: list[dict[str, Any]] = []
    idx = 0
    for snap in iter_jsonl(snapshots_path):
        tmpls = list(TEMPLATES)
        rng.shuffle(tmpls)
        for t in tmpls:
            result = t.generate(snap)
            if result is None:
                continue
            question, gold_answer, tags = result
            item_id = deterministic_id(BENCH, f"{snap.get('pollster','?')}|{t.name}", idx)
            idx += 1
            if item_id in existing:
                items.append(existing[item_id])
                continue
            items.append({
                "id": item_id,
                "template": t.name,
                "survey_snapshot": snap,
                "question": question,
                "gold_answer": gold_answer,
                "gold_reasoning_tags": tags,
            })
            if max_items is not None and len(items) >= max_items:
                break
        if max_items is not None and len(items) >= max_items:
            break

    n = write_items(items_path, items)
    write_manifest(
        out_dir,
        Manifest(
            bench_name="lever/polling_reasoning",
            seed=seed,
            assembled_at=now_iso(),
            total_items=n,
            authored_items=n,  # fully programmatic — no pending author work
            seed_source_shas={str(snapshots_path): sha256_file(snapshots_path)},
            extra={"templates": [t.name for t in TEMPLATES]},
        ),
    )
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--polling-snapshots", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-items", type=int, default=None)
    args = ap.parse_args()
    n = assemble(args.polling_snapshots, args.out, seed=args.seed, max_items=args.max_items)
    print(f"wrote {n} items to {args.out}/items.jsonl")


if __name__ == "__main__":
    main()
