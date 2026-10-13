"""Post-run validation: structural + integrity + consistency checks.

Every check returns (ok: bool, message: str). The CLI runs them all and prints
a summary. Exit code is 0 if all checks pass, 1 otherwise.

Usage:
    python -m data.pipeline.validate --run <run_id>
    python -m data.pipeline.validate                # latest run
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

DATA_DIR = Path(__file__).parent.parent
RUNS_DIR = DATA_DIR / "pipeline" / "runs"


def _resolve_run(run_id: str | None) -> Path:
    if run_id:
        p = RUNS_DIR / run_id
        if not p.exists():
            sys.exit(f"no such run: {p}")
        return p
    runs = sorted([p for p in RUNS_DIR.glob("*") if p.is_dir()], key=lambda p: p.stat().st_mtime)
    if not runs:
        sys.exit(f"no runs in {RUNS_DIR}")
    return runs[-1]


def _load_index(run: Path) -> list[dict]:
    return [json.loads(l) for l in (run / "index.jsonl").read_text().splitlines() if l.strip()]


def _load_paragraphs(run: Path) -> list[dict]:
    path = run / "paragraphs.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _load_config(run: Path) -> dict:
    return json.loads((run / "config.json").read_text())


def _load_pipeline_spec(run: Path) -> list[dict]:
    return json.loads((run / "pipeline.json").read_text())


def _load_metadata(run: Path) -> dict:
    cfg = _load_config(run)
    corpus = DATA_DIR / f"corpus_{cfg['version']}"
    return json.loads((corpus / "metadata.json").read_text()), corpus


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_structure(run: Path):
    required = ["config.json", "pipeline.json", "index.jsonl", "paragraphs.jsonl", "summary.txt"]
    missing = [f for f in required if not (run / f).exists()]
    if missing:
        return False, f"missing run files: {missing}"
    return True, f"all {len(required)} expected files present"


def check_index_well_formed(run: Path):
    idx = _load_index(run)
    if not idx:
        return False, "index.jsonl is empty"
    required_keys = {"doc_id", "title", "source_path", "sha256", "final_chars", "stages", "status"}
    for r in idx:
        missing = required_keys - set(r.keys())
        if missing:
            return False, f"doc {r.get('doc_id', '?')} missing keys {missing}"
    return True, f"{len(idx)} index rows all well-formed"


def check_unique_doc_ids(run: Path):
    """Among ok/skipped_too_small docs, doc_ids must be unique. Duplicate-sha256
    rows deliberately share a doc_id with their original and are excluded."""
    idx = _load_index(run)
    ids = [r["doc_id"] for r in idx if r["status"] != "duplicate_sha256"]
    dupes = [k for k, v in Counter(ids).items() if v > 1]
    if dupes:
        return False, f"duplicate doc_ids among non-dup rows: {dupes[:5]}"
    n_dup = sum(1 for r in idx if r["status"] == "duplicate_sha256")
    return True, f"all {len(ids)} non-dup doc_ids unique ({n_dup} byte-identical dupes skipped)"


def check_stage_chain_consistency(run: Path):
    """For every doc, stage i's out_chars must equal stage i+1's in_chars.

    Also delta_chars must equal out - in.
    """
    idx = _load_index(run)
    for r in idx:
        if r["status"] == "duplicate_sha256":
            continue  # duplicates carry no stages by design
        stages = r["stages"]
        if not stages:
            return False, f"doc {r['doc_id']} has no stages"
        for i, s in enumerate(stages):
            if s["out_chars"] - s["in_chars"] != s["delta_chars"]:
                return False, f"doc {r['doc_id']} stage '{s['stage']}': delta mismatch"
            if i > 0 and s["in_chars"] != stages[i - 1]["out_chars"]:
                return (
                    False,
                    f"doc {r['doc_id']} stage '{s['stage']}': in_chars {s['in_chars']} "
                    f"!= prev out_chars {stages[i - 1]['out_chars']}",
                )
        if r["status"] == "ok" and stages[-1]["out_chars"] != r["final_chars"]:
            return False, f"doc {r['doc_id']}: final_chars != last stage out_chars"
    return True, f"all {len(idx)} docs have consistent stage chains"


def check_stages_match_spec(run: Path):
    spec = _load_pipeline_spec(run)
    spec_names = ["extract"] + [s["stage"] for s in spec]
    idx = _load_index(run)
    for r in idx:
        if r["status"] == "skipped_too_small":
            if [s["stage"] for s in r["stages"]] != ["extract"]:
                return False, f"skipped doc {r['doc_id']} has unexpected stages"
            continue
        if r["status"] == "duplicate_sha256":
            if r["stages"]:
                return False, f"duplicate doc {r['doc_id']} should have no stages"
            continue
        got = [s["stage"] for s in r["stages"]]
        if got != spec_names:
            return False, f"doc {r['doc_id']}: stages {got} != spec {spec_names}"
    return True, f"all {len(idx)} docs' stage sequences match pipeline spec"


def check_intermediate_snapshots(run: Path):
    cfg = _load_config(run)
    if not cfg.get("keep_intermediates"):
        return True, "keep_intermediates=False (snapshots not expected)"
    idx = _load_index(run)
    spec = _load_pipeline_spec(run)
    stage_names = ["extract"] + [s["stage"] for s in spec]
    stages_dir = run / "stages"
    for r in idx:
        expected = stage_names if r["status"] == "ok" else ["extract"]
        for name in expected:
            path = stages_dir / name / f"{r['doc_id']}.txt"
            if not path.exists():
                return False, f"missing snapshot: {path.relative_to(run)}"
    return True, f"all {len(idx)} docs have complete stage snapshots"


def check_paragraphs_well_formed(run: Path):
    paras = _load_paragraphs(run)
    if not paras:
        return False, "paragraphs.jsonl is empty"
    valid_decisions = {"kept", "dropped_minhash", "dropped_subsample"}
    required = {"para_id", "doc_id", "title", "text", "attrs", "decision"}
    for p in paras:
        missing = required - set(p.keys())
        if missing:
            return False, f"paragraph {p.get('para_id', '?')} missing keys {missing}"
        if p["decision"] not in valid_decisions:
            return False, f"paragraph {p['para_id']}: bad decision {p['decision']!r}"
    return True, f"{len(paras)} paragraphs all well-formed"


def check_paragraphs_doc_id_in_index(run: Path):
    idx = _load_index(run)
    idx_ids = {r["doc_id"] for r in idx}
    paras = _load_paragraphs(run)
    orphan = [p for p in paras if p["doc_id"] not in idx_ids]
    if orphan:
        return False, f"{len(orphan)} orphan paragraphs (first: {orphan[0]['para_id']})"
    return True, f"all paragraphs reference valid doc_ids"


def check_no_paragraphs_from_skipped(run: Path):
    """Paragraphs must only come from docs that actually went through the pipeline.
    A duplicate_sha256 row shares its doc_id with its original ok row, so exclude
    any id that is present in any ok row."""
    idx = _load_index(run)
    ok_ids = {r["doc_id"] for r in idx if r["status"] == "ok"}
    skipped = {r["doc_id"] for r in idx if r["status"] != "ok"} - ok_ids
    paras = _load_paragraphs(run)
    bad = [p for p in paras if p["doc_id"] in skipped]
    if bad:
        return False, f"{len(bad)} paragraphs from skipped docs"
    return True, f"no paragraphs from {len(skipped)} non-ok doc_ids"


def check_per_source_flow_accounting(run: Path):
    """For each contributing doc: paragraphs_split == sum of decisions in paragraphs.jsonl,
    and paragraphs_picked == kept count, paragraphs_minhash_dropped == dropped_minhash count."""
    meta, _ = _load_metadata(run)
    paras = _load_paragraphs(run)
    by_doc: dict[str, Counter] = defaultdict(Counter)
    for p in paras:
        by_doc[p["doc_id"]][p["decision"]] += 1
    for r in meta["per_source"]:
        d = r["doc_id"]
        c = by_doc[d]
        total = c["kept"] + c["dropped_minhash"] + c["dropped_subsample"]
        if total != r["paragraphs_split"]:
            return False, (
                f"doc {d} ({r['title'][:40]}): per_source.paragraphs_split={r['paragraphs_split']} "
                f"but paragraphs.jsonl total={total} ({dict(c)})"
            )
        if c["kept"] != r["paragraphs_picked"]:
            return False, f"doc {d}: paragraphs_picked mismatch ({r['paragraphs_picked']} vs {c['kept']})"
        if c["dropped_minhash"] != r["paragraphs_minhash_dropped"]:
            return False, f"doc {d}: minhash_dropped mismatch ({r['paragraphs_minhash_dropped']} vs {c['dropped_minhash']})"
        if c["dropped_subsample"] != r["paragraphs_overflow"]:
            return False, f"doc {d}: overflow mismatch ({r['paragraphs_overflow']} vs {c['dropped_subsample']})"
        if r["paragraphs_after_minhash"] != r["paragraphs_picked"] + r["paragraphs_overflow"]:
            return False, f"doc {d}: after_minhash != picked + overflow"
    return True, f"per_source flow accounting consistent for {len(meta['per_source'])} docs"


def check_metadata_counts(run: Path):
    meta, _ = _load_metadata(run)
    paras = _load_paragraphs(run)
    decs = Counter(p["decision"] for p in paras)
    if decs["kept"] != meta["num_paragraphs_kept"]:
        return False, f"kept: meta={meta['num_paragraphs_kept']} paragraphs.jsonl={decs['kept']}"
    if decs["dropped_minhash"] != meta["num_paragraphs_dropped_minhash"]:
        return False, f"dropped_minhash mismatch"
    if decs["dropped_subsample"] != meta["num_paragraphs_dropped_subsample"]:
        return False, f"dropped_subsample mismatch"
    total = sum(decs.values())
    if total != meta["num_paragraphs_split"]:
        return False, f"split total: meta={meta['num_paragraphs_split']} sum={total}"
    return True, f"metadata paragraph counts match ({total} total)"


def check_bin_file_sizes(run: Path):
    meta, corpus = _load_metadata(run)
    dtype_bytes = 4 if "uint32" in meta["dtype"] else 2
    for split in ("train", "val", "test"):
        path = corpus / f"{split}.bin"
        expected = meta[f"{split}_tokens"] * dtype_bytes
        got = path.stat().st_size
        if got != expected:
            return False, f"{split}.bin: size={got} expected={expected} ({meta[f'{split}_tokens']} tokens × {dtype_bytes})"
    return True, "train/val/test.bin sizes match metadata token counts"


def check_sources_files(run: Path):
    meta, corpus = _load_metadata(run)
    sources_jsonl = corpus / "sources.jsonl"
    annotated = corpus / "sources.annotated.md"
    flat = corpus / "sources.txt"
    for f in [sources_jsonl, annotated, flat]:
        if not f.exists():
            return False, f"missing: {f.name}"
        if f.stat().st_size == 0:
            return False, f"empty: {f.name}"
    n_lines = sum(1 for _ in open(sources_jsonl))
    if n_lines != meta["num_paragraphs_kept"]:
        return False, f"sources.jsonl has {n_lines} rows, metadata.num_paragraphs_kept={meta['num_paragraphs_kept']}"
    marker_count = annotated.read_text().count("<!-- para ")
    if marker_count != meta["num_paragraphs_kept"]:
        return False, f"sources.annotated.md has {marker_count} markers, expected {meta['num_paragraphs_kept']}"
    return True, f"sources files consistent ({n_lines} kept paragraphs)"


def check_per_source_tokens_consistency(run: Path):
    """Sum of per_source.used_tokens should be >= total_tokens × 0.95 (allowing for join overhead)."""
    meta, _ = _load_metadata(run)
    summed = sum(r["used_tokens"] for r in meta["per_source"])
    total = meta["total_tokens"]
    # total is re-tokenized from the joined text, so it may differ slightly from
    # the per-source sum. Allow 25% slack for this noise.
    ratio = summed / max(total, 1)
    if not (0.5 <= ratio <= 1.5):
        return False, f"per_source sum={summed}, total={total}, ratio={ratio:.2f} is outside [0.5, 1.5]"
    return True, f"per_source token sum ({summed}) within expected range of total ({total})"


def check_status_counts(run: Path):
    idx = _load_index(run)
    meta, _ = _load_metadata(run)
    # `num_skipped_docs` in metadata counts everything that didn't contribute:
    # too-small docs AND duplicate-sha256 docs.
    skipped = sum(1 for r in idx if r["status"] in ("skipped_too_small", "duplicate_sha256"))
    if skipped != meta["num_skipped_docs"]:
        return False, f"skipped: index={skipped} metadata={meta['num_skipped_docs']}"
    ok = sum(1 for r in idx if r["status"] == "ok")
    if ok < meta["num_sources"]:
        return False, f"ok docs ({ok}) < num_sources ({meta['num_sources']})"
    dup = sum(1 for r in idx if r["status"] == "duplicate_sha256")
    too_small = sum(1 for r in idx if r["status"] == "skipped_too_small")
    return True, f"{ok} ok, {too_small} too_small, {dup} duplicate_sha256 matches metadata"


CHECKS: list[tuple[str, Callable[[Path], tuple[bool, str]]]] = [
    ("structure", check_structure),
    ("index_well_formed", check_index_well_formed),
    ("unique_doc_ids", check_unique_doc_ids),
    ("stage_chain", check_stage_chain_consistency),
    ("stages_match_spec", check_stages_match_spec),
    ("intermediate_snapshots", check_intermediate_snapshots),
    ("paragraphs_well_formed", check_paragraphs_well_formed),
    ("paragraphs_doc_id_valid", check_paragraphs_doc_id_in_index),
    ("no_paras_from_skipped", check_no_paragraphs_from_skipped),
    ("per_source_flow", check_per_source_flow_accounting),
    ("metadata_counts", check_metadata_counts),
    ("bin_file_sizes", check_bin_file_sizes),
    ("sources_files", check_sources_files),
    ("per_source_tokens", check_per_source_tokens_consistency),
    ("status_counts", check_status_counts),
]


def run_checks(run: Path) -> list[tuple[str, bool, str]]:
    results = []
    for name, fn in CHECKS:
        try:
            ok, msg = fn(run)
        except Exception as e:
            ok, msg = False, f"raised {type(e).__name__}: {e}"
        results.append((name, ok, msg))
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None)
    args = p.parse_args()
    run = _resolve_run(args.run)
    print(f"validating run: {run.name}\n")
    results = run_checks(run)
    width = max(len(n) for n, _, _ in results)
    all_ok = True
    for name, ok, msg in results:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name:<{width}}  {msg}")
        if not ok:
            all_ok = False
    print()
    if all_ok:
        print(f"all {len(results)} checks passed")
        sys.exit(0)
    else:
        n_fail = sum(1 for _, ok, _ in results if not ok)
        print(f"{n_fail} of {len(results)} checks FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
