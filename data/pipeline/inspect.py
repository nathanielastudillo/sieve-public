"""Inspect a pipeline run.

Every pipeline run writes to ``data/pipeline/runs/<run_id>/``. These commands
let you walk through any document's journey through the pipeline:

    python -m data.pipeline.inspect list                        # show all docs in latest run
    python -m data.pipeline.inspect find <substr>                # find doc by title
    python -m data.pipeline.inspect trace <doc_id>               # stage-by-stage chars & notes
    python -m data.pipeline.inspect show <doc_id> <stage>        # print text at that stage
    python -m data.pipeline.inspect diff <doc_id> <stage_a> <stage_b>
    python -m data.pipeline.inspect paragraphs <doc_id>          # its paragraphs + decisions
    python -m data.pipeline.inspect para <para_id>               # one paragraph + journey
    python -m data.pipeline.inspect corpus [<corpus_dir>]        # per-source breakdown
    python -m data.pipeline.inspect summary                      # full run summary

Use --run <run_id> to pick a non-latest run.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path
from typing import Iterator, Optional

DATA_DIR = Path(__file__).parent.parent
RUNS_DIR = DATA_DIR / "pipeline" / "runs"


def resolve_run(run_id: Optional[str]) -> Path:
    if run_id:
        p = RUNS_DIR / run_id
        if not p.exists():
            sys.exit(f"no such run: {p}")
        return p
    runs = sorted([p for p in RUNS_DIR.glob("*") if p.is_dir()], key=lambda p: p.stat().st_mtime)
    if not runs:
        sys.exit(f"no runs in {RUNS_DIR}")
    return runs[-1]


def iter_index(run: Path) -> Iterator[dict]:
    with open(run / "index.jsonl") as fp:
        for line in fp:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_paragraphs(run: Path) -> Iterator[dict]:
    path = run / "paragraphs.jsonl"
    if not path.exists():
        return
    with open(path) as fp:
        for line in fp:
            line = line.strip()
            if line:
                yield json.loads(line)


def find_doc(run: Path, doc_id_or_substr: str) -> Optional[dict]:
    for row in iter_index(run):
        if row["doc_id"] == doc_id_or_substr:
            return row
    lowered = doc_id_or_substr.lower()
    for row in iter_index(run):
        if lowered in row["title"].lower() or lowered in row["source_path"].lower():
            return row
    return None


def cmd_list(run: Path, args):
    limit = args.limit
    rows = list(iter_index(run))
    rows.sort(key=lambda r: -r["final_chars"])
    if limit:
        rows = rows[:limit]
    print(f"{'doc_id':<14}{'final_chars':>12}  {'stages':>7}  title")
    for r in rows:
        print(f"{r['doc_id']:<14}{r['final_chars']:>12d}  {len(r['stages']):>7d}  {r['title'][:80]}")
    print(f"\n{len(list(iter_index(run)))} documents total")


def cmd_find(run: Path, args):
    q = args.query.lower()
    matches = [r for r in iter_index(run) if q in r["title"].lower() or q in r["source_path"].lower()]
    if not matches:
        sys.exit(f"no docs matching {args.query!r}")
    for r in matches:
        print(f"{r['doc_id']}  {r['final_chars']:>10d}  {r['title']}")


def cmd_trace(run: Path, args):
    row = find_doc(run, args.doc_id)
    if not row:
        sys.exit(f"doc not found: {args.doc_id}")
    print(f"doc_id:      {row['doc_id']}")
    print(f"title:       {row['title']}")
    print(f"source_path: {row['source_path']}")
    print(f"sha256:      {row['sha256']}")
    print(f"final_chars: {row['final_chars']}")
    print()
    print(f"{'#':>2} {'stage':<28} {'in':>8} {'out':>8} {'Δ':>8} {'ms':>5}  notes")
    print("-" * 100)
    for i, s in enumerate(row["stages"]):
        notes = ", ".join(f"{k}={v}" for k, v in (s.get("notes") or {}).items())
        print(
            f"{i:>2} {s['stage']:<28} {s['in_chars']:>8d} {s['out_chars']:>8d} "
            f"{s['delta_chars']:>+8d} {s['wall_ms']:>5d}  {notes}"
        )


def cmd_show(run: Path, args):
    row = find_doc(run, args.doc_id)
    if not row:
        sys.exit(f"doc not found: {args.doc_id}")
    stage_dir = run / "stages" / args.stage
    path = stage_dir / f"{row['doc_id']}.txt"
    if not path.exists():
        sys.exit(
            f"no snapshot for stage '{args.stage}' (did the run use --keep-intermediates?)"
        )
    text = path.read_text(encoding="utf-8")
    if args.head:
        print("\n".join(text.splitlines()[: args.head]))
    else:
        sys.stdout.write(text)


def cmd_diff(run: Path, args):
    row = find_doc(run, args.doc_id)
    if not row:
        sys.exit(f"doc not found: {args.doc_id}")
    a = run / "stages" / args.stage_a / f"{row['doc_id']}.txt"
    b = run / "stages" / args.stage_b / f"{row['doc_id']}.txt"
    if not a.exists() or not b.exists():
        sys.exit("stage snapshots missing (rerun with --keep-intermediates)")
    al = a.read_text(encoding="utf-8").splitlines()
    bl = b.read_text(encoding="utf-8").splitlines()
    for line in difflib.unified_diff(
        al, bl, fromfile=args.stage_a, tofile=args.stage_b, n=args.context, lineterm=""
    ):
        print(line)


def cmd_paragraphs(run: Path, args):
    doc_id = args.doc_id
    row = find_doc(run, doc_id)
    if row:
        doc_id = row["doc_id"]
    counts = {"kept": 0, "dropped_minhash": 0, "dropped_subsample": 0}
    rows = []
    for p in iter_paragraphs(run):
        if p["doc_id"] != doc_id:
            continue
        rows.append(p)
        counts[p.get("decision", "kept")] = counts.get(p.get("decision", "kept"), 0) + 1

    if not rows:
        sys.exit(f"no paragraphs recorded for {doc_id}")
    print(
        f"doc {doc_id}: {counts['kept']} kept  "
        f"{counts['dropped_minhash']} minhash  "
        f"{counts['dropped_subsample']} subsample"
    )
    if args.only in ("kept", "dropped", "all"):
        for p in rows:
            dec = p.get("decision", "?")
            if args.only == "kept" and dec != "kept":
                continue
            if args.only == "dropped" and dec == "kept":
                continue
            d = p.get("attrs", {}).get("density", 0.0)
            w = p.get("attrs", {}).get("words", 0)
            note = p.get("note", "")
            print(f"  [{dec:<18}] density={d:.3f} words={w:>4d} {p['para_id']} {note}")
            if args.show_text:
                print("    " + p["text"][:200].replace("\n", " ") + ("…" if len(p["text"]) > 200 else ""))


def cmd_para(run: Path, args):
    pid = args.para_id
    for p in iter_paragraphs(run):
        if p["para_id"] == pid:
            print(json.dumps(p, indent=2))
            return
    sys.exit(f"paragraph not found: {pid}")


def cmd_corpus(run: Path, args):
    corpus_dir = Path(args.corpus_dir) if args.corpus_dir else None
    if corpus_dir is None:
        # infer from config
        cfg = json.loads((run / "config.json").read_text())
        corpus_dir = DATA_DIR / f"corpus_{cfg['version']}"
    meta_path = corpus_dir / "metadata.json"
    if not meta_path.exists():
        sys.exit(f"no metadata.json at {meta_path}")
    meta = json.loads(meta_path.read_text())
    print(f"corpus: {corpus_dir.name}")
    print(f"  run_id:       {meta.get('run_id')}")
    print(f"  tokenizer:    {meta.get('tokenizer_label')}  (vocab={meta.get('vocab_size')})")
    print(
        f"  tokens:       total={meta['total_tokens']}  "
        f"train={meta['train_tokens']}  val={meta['val_tokens']}  test={meta['test_tokens']}"
    )
    print(
        f"  paragraphs:   split={meta.get('num_paragraphs_split')}  "
        f"kept={meta.get('num_paragraphs_kept')}  "
        f"drop_minhash={meta.get('num_paragraphs_dropped_minhash')}  "
        f"drop_overflow={meta.get('num_paragraphs_dropped_subsample')}"
    )
    print(f"  sources:      {meta['num_sources']} contributed  ({meta.get('num_skipped_docs', 0)} skipped)")
    print()
    print(f"{'tokens':>8}  {'split→mh→pick':>16}  density≥   title")
    for r in meta["per_source"][: args.limit or 30]:
        flow = f"{r['paragraphs_split']}→{r['paragraphs_after_minhash']}→{r['paragraphs_picked']}"
        print(
            f"{r['used_tokens']:>8d}  "
            f"{flow:>16}  "
            f"{str(r['min_density_kept']):<8}  {r['title'][:80]}"
        )


def cmd_summary(run: Path, args):
    summary = run / "summary.txt"
    if summary.exists():
        print(summary.read_text())
    else:
        print(f"no summary.txt in {run}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None, help="run_id (default: latest)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("list", help="list documents")
    sp.add_argument("--limit", type=int, default=None)

    sp = sub.add_parser("find", help="find doc by substring")
    sp.add_argument("query")

    sp = sub.add_parser("trace", help="show stage-by-stage trace for a doc")
    sp.add_argument("doc_id")

    sp = sub.add_parser("show", help="print text at a given stage")
    sp.add_argument("doc_id")
    sp.add_argument("stage")
    sp.add_argument("--head", type=int, default=None)

    sp = sub.add_parser("diff", help="diff two stages' output for a doc")
    sp.add_argument("doc_id")
    sp.add_argument("stage_a")
    sp.add_argument("stage_b")
    sp.add_argument("--context", type=int, default=3)

    sp = sub.add_parser("paragraphs", help="show paragraphs from a doc")
    sp.add_argument("doc_id")
    sp.add_argument("--only", choices=["kept", "dropped", "all"], default="all")
    sp.add_argument("--show-text", action="store_true")

    sp = sub.add_parser("para", help="show one paragraph")
    sp.add_argument("para_id")

    sp = sub.add_parser("corpus", help="per-source corpus breakdown")
    sp.add_argument("corpus_dir", nargs="?", default=None)
    sp.add_argument("--limit", type=int, default=None)

    sub.add_parser("summary", help="show run summary.txt")

    args = p.parse_args()
    run = resolve_run(args.run)

    funcs = {
        "list": cmd_list,
        "find": cmd_find,
        "trace": cmd_trace,
        "show": cmd_show,
        "diff": cmd_diff,
        "paragraphs": cmd_paragraphs,
        "para": cmd_para,
        "corpus": cmd_corpus,
        "summary": cmd_summary,
    }
    funcs[args.cmd](run, args)


if __name__ == "__main__":
    main()
