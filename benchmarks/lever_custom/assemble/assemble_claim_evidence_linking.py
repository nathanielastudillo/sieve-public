"""Assemble ``lever/claim_evidence_linking`` seed items.

Input sources:
- ``--claims``: jsonl of claims extracted upstream by lever-engine.
  Each record:
    {"claim_id": "...", "claim": "...", "topic": "housing|immigration|fiscal|..."}
- ``--candidates-index``: jsonl of candidate snippets pre-retrieved for
  each claim.
    {"claim_id": "...", "doc_id": "pig:data:meta:...", "snippet": "..."}

The assembler groups candidates by claim, caps k per item (default
k_max=12), and emits items with empty ``gold_labels`` for human
labeling. Reruns preserve existing labels by ``id``.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
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

BENCH = "clink"


def _parse_mix(s: str | None) -> dict[str, float]:
    if not s:
        return {"housing": 0.4, "immigration": 0.3, "fiscal": 0.3}
    out: dict[str, float] = {}
    for kv in s.split(","):
        k, v = kv.split("=")
        out[k.strip()] = float(v)
    total = sum(out.values())
    if abs(total - 1.0) > 1e-3:
        raise ValueError(f"topic mix must sum to 1.0, got {total}")
    return out


def assemble(
    claims_path: Path,
    candidates_path: Path,
    out_dir: Path,
    target_items: int = 300,
    topic_mix: str | None = None,
    k_max: int = 12,
    seed: int = 0,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    items_path = out_dir / "items.jsonl"
    existing = load_existing_items(items_path)
    mix = _parse_mix(topic_mix)
    rng = random.Random(seed)

    claims_by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for claim in iter_jsonl(claims_path):
        topic = claim.get("topic", "other")
        if topic in mix:
            claims_by_topic[topic].append(claim)

    candidates_by_claim: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cand in iter_jsonl(candidates_path):
        candidates_by_claim[cand["claim_id"]].append(cand)

    for topic in claims_by_topic:
        rng.shuffle(claims_by_topic[topic])

    items: list[dict[str, Any]] = []
    for topic, frac in mix.items():
        need = round(target_items * frac)
        pool = claims_by_topic.get(topic, [])
        for i, claim in enumerate(pool):
            if len(items) >= sum(round(target_items * f) for t2, f in mix.items() if t2 <= topic):
                break
            cands = candidates_by_claim.get(claim["claim_id"], [])
            if len(cands) < 3:
                continue
            cands = cands[:k_max]
            item_id = deterministic_id(BENCH, claim["claim_id"], i)
            if item_id in existing:
                items.append(existing[item_id])
                continue
            items.append({
                "id": item_id,
                "topic": topic,
                "claim": claim["claim"],
                "claim_id": claim["claim_id"],
                "candidates": [
                    {"doc_id": c["doc_id"], "snippet": c["snippet"]} for c in cands
                ],
                "gold_labels": {},  # ← human labels each doc_id
                "_annotation_status": "pending",
            })
            if sum(1 for it in items if it["topic"] == topic) >= need:
                break

    n = write_items(items_path, items)
    authored = sum(1 for it in items if it.get("gold_labels"))
    write_manifest(
        out_dir,
        Manifest(
            bench_name="lever/claim_evidence_linking",
            seed=seed,
            assembled_at=now_iso(),
            total_items=n,
            authored_items=authored,
            seed_source_shas={
                str(claims_path): sha256_file(claims_path),
                str(candidates_path): sha256_file(candidates_path),
            },
            extra={"topic_mix": mix, "target_items": target_items, "k_max": k_max},
        ),
    )
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", required=True, type=Path)
    ap.add_argument("--candidates-index", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--target-items", type=int, default=300)
    ap.add_argument("--topic-mix", type=str, default=None)
    ap.add_argument("--k-max", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    n = assemble(
        args.claims, args.candidates_index, args.out,
        target_items=args.target_items,
        topic_mix=args.topic_mix,
        k_max=args.k_max,
        seed=args.seed,
    )
    print(f"wrote {n} items to {args.out}/items.jsonl")


if __name__ == "__main__":
    main()
