"""Shared helpers for the per-bench assemblers."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


@dataclass
class Manifest:
    bench_name: str
    seed: int
    assembled_at: str
    total_items: int
    authored_items: int  # items with no gold-labeling work remaining
    seed_source_shas: dict[str, str]
    extra: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(
            {
                "bench_name": self.bench_name,
                "seed": self.seed,
                "assembled_at": self.assembled_at,
                "total_items": self.total_items,
                "authored_items": self.authored_items,
                "seed_source_shas": self.seed_source_shas,
                "extra": self.extra,
            },
            indent=2,
        )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_existing_items(items_path: Path) -> dict[str, dict[str, Any]]:
    """Load existing items by id so reruns preserve human-authored fields."""
    if not items_path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with items_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            out[item["id"]] = item
    return out


def write_items(items_path: Path, items: Iterable[dict[str, Any]]) -> int:
    items_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = items_path.with_suffix(items_path.suffix + ".tmp")
    n = 0
    with tmp.open("w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
            n += 1
    os.replace(tmp, items_path)
    return n


def write_manifest(out_dir: Path, manifest: Manifest) -> None:
    (out_dir / "manifest.json").write_text(manifest.to_json())


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def deterministic_id(bench: str, seed_material: str, idx: int) -> str:
    """Stable id so re-assembly does not invalidate human labels."""
    h = hashlib.sha1(f"{bench}|{seed_material}|{idx}".encode()).hexdigest()[:8]
    return f"{bench}-{h}-{idx:06d}"
