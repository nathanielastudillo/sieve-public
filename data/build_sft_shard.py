"""Build an SFT corpus shard as JSONL from a cleaned/ markdown directory.

Input: a directory of ``*.md`` files (e.g. ``data/cleaned_v009/``).
Output: ``{"text": "..."}`` JSONL with paragraphs packed up to
``max_chars`` per row. No tokenization; the SFT trainer's own
tokenizer handles that (``openweight.finetune`` uses the base model's
tokenizer via ``AutoTokenizer``).

Why paragraph-packed, not whole-document rows? 7B-13B models train
happily on ~2K-8K char rows (~512-2048 tokens after BPE). Whole-document
rows blow past ``max_seq_len`` and get truncated. Paragraph packing
preserves internal coherence within each row.

Usage
-----
    python -m data.build_sft_shard \
        --cleaned data/cleaned_v009 \
        --out data/sft_shards/v009.jsonl \
        --max-chars 6000

The output lands under ``data/sft_shards/`` by convention. The
resulting path is what gets passed as ``corpus_shard_path`` in
``SftJob``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


def iter_md_files(cleaned_dir: Path) -> Iterable[Path]:
    yield from sorted(p for p in cleaned_dir.glob("*.md") if p.is_file())


def pack_paragraphs(text: str, max_chars: int, min_chars: int = 200) -> Iterable[str]:
    """Group consecutive paragraphs into rows up to ``max_chars``.

    Paragraphs are split on double-newline. Rows shorter than
    ``min_chars`` are dropped (prevents flooding the shard with one-line
    fragments like bare section headers).
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    cur: list[str] = []
    cur_len = 0
    for para in paragraphs:
        plen = len(para)
        if cur_len + plen + 2 > max_chars and cur:
            row = "\n\n".join(cur)
            if len(row) >= min_chars:
                yield row
            cur = [para]
            cur_len = plen
        else:
            cur.append(para)
            cur_len += plen + 2
    if cur:
        row = "\n\n".join(cur)
        if len(row) >= min_chars:
            yield row


def build_shard(
    cleaned_dir: Path,
    out_path: Path,
    max_chars: int,
    min_chars: int,
) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    chars = 0
    files = 0
    sha = hashlib.sha256()
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for md in iter_md_files(cleaned_dir):
            text = md.read_text(encoding="utf-8", errors="ignore")
            sha.update(md.name.encode("utf-8"))
            sha.update(b"\x00")
            sha.update(text.encode("utf-8"))
            files += 1
            for row in pack_paragraphs(text, max_chars=max_chars, min_chars=min_chars):
                f.write(json.dumps({"text": row}, ensure_ascii=False) + "\n")
                rows += 1
                chars += len(row)
    tmp.replace(out_path)
    manifest = {
        "cleaned_dir": str(cleaned_dir),
        "out_path": str(out_path),
        "files": files,
        "rows": rows,
        "chars": chars,
        "max_chars": max_chars,
        "min_chars": min_chars,
        "sha256": sha.hexdigest(),
    }
    (out_path.with_suffix(out_path.suffix + ".manifest.json")).write_text(
        json.dumps(manifest, indent=2)
    )
    logger.info(
        "build_sft_shard: %d files → %d rows (%d chars) at %s",
        files, rows, chars, out_path,
    )
    return manifest


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cleaned", required=True, type=Path,
                    help="Directory of cleaned markdown files")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output JSONL path (e.g. data/sft_shards/v009.jsonl)")
    ap.add_argument("--max-chars", type=int, default=6000,
                    help="Max characters per JSONL row (roughly 1.5–2K tokens)")
    ap.add_argument("--min-chars", type=int, default=200,
                    help="Drop rows shorter than this")
    args = ap.parse_args()

    if not args.cleaned.is_dir():
        raise SystemExit(f"--cleaned {args.cleaned} is not a directory")

    manifest = build_shard(
        cleaned_dir=args.cleaned,
        out_path=args.out,
        max_chars=args.max_chars,
        min_chars=args.min_chars,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
