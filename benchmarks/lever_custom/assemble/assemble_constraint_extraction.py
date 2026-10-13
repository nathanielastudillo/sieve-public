"""Assemble ``lever/constraint_extraction`` seed items from a lex section pool.

Each output item is a section selected from the pool with
``gold_constraints: []`` left for a human annotator to fill. Re-runs
preserve any human-authored ``gold_constraints`` on existing items.

This assembler handles the MECHANICS of seeding (which sections, what
jurisdictional mix, id stability) and nothing else — the spec explicitly
forbids LLM-generated golds for this benchmark.

Input: jsonl of lex sections, each record:
    {
      "section_id": "lex:section:ita/s.118.3",
      "jurisdiction": "ca-federal",
      "text": "<section body>"
    }
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

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

BENCH = "cext"


def _parse_mix(s: str | None) -> dict[str, float]:
    if not s:
        return {"ca-federal": 0.7, "ca-on": 0.2, "ca-bc": 0.1}
    out: dict[str, float] = {}
    for kv in s.split(","):
        k, v = kv.split("=")
        out[k.strip()] = float(v)
    total = sum(out.values())
    if abs(total - 1.0) > 1e-3:
        raise ValueError(f"jurisdictional mix must sum to 1.0, got {total}")
    return out


def assemble(
    sections_path: Path,
    out_dir: Path,
    target_items: int = 400,
    jurisdictional_mix: str | None = None,
    min_section_len: int = 200,
    max_section_len: int = 2000,
    seed: int = 0,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    items_path = out_dir / "items.jsonl"
    existing = load_existing_items(items_path)
    mix = _parse_mix(jurisdictional_mix)
    rng = random.Random(seed)

    pool_by_juris: dict[str, list[dict[str, Any]]] = {}
    for sec in iter_jsonl(sections_path):
        text = sec.get("text", "")
        if not (min_section_len <= len(text.split()) <= max_section_len):
            continue
        pool_by_juris.setdefault(sec["jurisdiction"], []).append(sec)

    for j in pool_by_juris:
        rng.shuffle(pool_by_juris[j])

    items: list[dict[str, Any]] = []
    for juris, frac in mix.items():
        need = round(target_items * frac)
        candidates = pool_by_juris.get(juris, [])
        if not candidates:
            continue
        for i, sec in enumerate(candidates[:need]):
            item_id = deterministic_id(BENCH, sec["section_id"], i)
            if item_id in existing:
                items.append(existing[item_id])
                continue
            items.append({
                "id": item_id,
                "jurisdiction": juris,
                "section_id": sec["section_id"],
                "section_text": sec["text"],
                "gold_constraints": [],  # ← human annotator fills this
                "_annotation_status": "pending",
            })

    n = write_items(items_path, items)
    authored = sum(1 for it in items if it.get("gold_constraints"))
    write_manifest(
        out_dir,
        Manifest(
            bench_name="lever/constraint_extraction",
            seed=seed,
            assembled_at=now_iso(),
            total_items=n,
            authored_items=authored,
            seed_source_shas={str(sections_path): sha256_file(sections_path)},
            extra={"jurisdictional_mix": mix, "target_items": target_items},
        ),
    )
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lex-sections", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--target-items", type=int, default=400)
    ap.add_argument("--jurisdictional-mix", type=str, default=None,
                    help="e.g. 'ca-federal=0.7,ca-on=0.2,ca-bc=0.1'")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    n = assemble(
        args.lex_sections, args.out,
        target_items=args.target_items,
        jurisdictional_mix=args.jurisdictional_mix,
        seed=args.seed,
    )
    print(f"wrote {n} items to {args.out}/items.jsonl")


if __name__ == "__main__":
    main()
