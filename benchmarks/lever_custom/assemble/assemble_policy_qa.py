"""Assemble ``lever/policy_qa`` seed items from a lex section pool.

Each output item is a selected lex section with ``question`` and
``gold_answer`` left blank — the spec requires human-authored golds, so
this assembler only seeds the pool. A rerun preserves authored fields
on existing items by id.
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

BENCH = "polqa"


def _parse_mix(s: str | None) -> dict[str, float]:
    if not s:
        return {
            "ca-federal": 0.6,
            "ca-on": 0.2,
            "ca-toronto": 0.1,
            "ca-ottawa": 0.1,
        }
    out: dict[str, float] = {}
    for kv in s.split(","):
        k, v = kv.split("=")
        out[k.strip()] = float(v)
    total = sum(out.values())
    if abs(total - 1.0) > 1e-3:
        raise ValueError(f"mix must sum to 1.0, got {total}")
    return out


def assemble(
    sections_path: Path,
    out_dir: Path,
    target_items: int = 800,
    jurisdictional_mix: str | None = None,
    seed: int = 0,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    items_path = out_dir / "items.jsonl"
    existing = load_existing_items(items_path)
    mix = _parse_mix(jurisdictional_mix)
    rng = random.Random(seed)

    pool: dict[str, list[dict[str, Any]]] = {}
    for sec in iter_jsonl(sections_path):
        pool.setdefault(sec["jurisdiction"], []).append(sec)
    for j in pool:
        rng.shuffle(pool[j])

    items: list[dict[str, Any]] = []
    for juris, frac in mix.items():
        need = round(target_items * frac)
        for i, sec in enumerate(pool.get(juris, [])[:need]):
            item_id = deterministic_id(BENCH, sec["section_id"], i)
            if item_id in existing:
                items.append(existing[item_id])
                continue
            items.append({
                "id": item_id,
                "jurisdiction": juris,
                "question": "",  # ← human authors
                "context_refs": [sec["section_id"]],
                "gold_answer": "",  # ← human authors
                "gold_citations": [sec["section_id"]],
                "_annotation_status": "pending",
            })

    n = write_items(items_path, items)
    authored = sum(1 for it in items if it.get("gold_answer"))
    write_manifest(
        out_dir,
        Manifest(
            bench_name="lever/policy_qa",
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
    ap.add_argument("--target-items", type=int, default=800)
    ap.add_argument("--jurisdictional-mix", type=str, default=None)
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
