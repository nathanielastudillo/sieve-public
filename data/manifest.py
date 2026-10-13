"""Corpus manifest + diff tool.

Build a manifest of the current raw/ + extracted/ state, compare it to a
manifest built elsewhere (e.g. on `corpus-host`), and report what would be *added*
by a merge. Surfaces three kinds of collision:

  - exact file duplicate       (same sha256 of raw bytes)
  - extracted-text duplicate   (same normalized-text hash — different
                                 PDF rendering of the same document)
  - possible duplicate by name (normalized filename match — catches
                                 "Book (1994).pdf" vs "Book_scanned_.pdf")

Usage
-----
    # build manifest of local state
    python -m data.manifest build --out data/manifest_local.json

    # later, on corpus-host / another drive:
    python -m data.manifest build --raw /mnt/drive/books --extracted '' \
        --out /tmp/manifest_tub.json

    # diff: what would `other` add to `self` if merged?
    python -m data.manifest diff \
        --self data/manifest_local.json \
        --other /tmp/manifest_tub.json \
        --out data/diff_tub.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent
RAW_DIR = DATA_DIR / "raw"
EXTRACTED_DIR = DATA_DIR / "extracted"

# Filenames we never want in the corpus regardless of what a drive contains.
# Keep this conservative — diff just *flags* these, doesn't drop them.
PERSONAL_HINT_RE = re.compile(
    r"(emergency kit|1password|tax|resume|insurance|bank statement|passport|"
    r"invoice|receipt|payslip|driver.?licen[cs]e)",
    re.IGNORECASE,
)

# Extensions worth manifesting. PDFs are the primary corpus source; include
# epub/txt/md so future drives with non-PDF texts get surfaced.
TEXT_EXTS = {".pdf", ".epub", ".txt", ".md", ".markdown", ".djvu", ".mobi"}

# How many chars of extracted text to hash. A full-file hash is brittle
# (one page of OCR jitter flips it), a too-short prefix collides on cover
# pages. 50K chars is ~ the first chapter of most books.
TEXT_SAMPLE_CHARS = 50_000


def sha256_file(path: Path, *, bufsize: int = 1 << 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_filename(name: str) -> str:
    """Aggressive filename normalization for fuzzy dedup.

    Strips library suffixes, years, edition markers, punctuation — anything
    that tends to vary between copies of the same book.
    """
    s = name.lower()
    s = Path(s).stem  # drop extension
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    # library / source tags
    s = re.sub(r"\b(z[- ]?library|download-site\.?(?:li|lc|rs|is)?|b[- ]?ok|anna'?s? archive)\b", " ", s)
    # years in parens or bare
    s = re.sub(r"\(?\b(19|20)\d{2}\b\)?", " ", s)
    # edition markers
    s = re.sub(r"\b\d+(?:st|nd|rd|th)\s*(?:ed|edition)\b", " ", s)
    s = re.sub(r"\bed(?:ition)?\.?\s*\d+\b", " ", s)
    # trailing (1), (2) copies
    s = re.sub(r"\(\s*\d+\s*\)", " ", s)
    # publisher tails like "- Wiley", "- Springer"
    s = re.sub(r"[-_]\s*(wiley|springer|mit press|penguin|routledge|norton|mcgraw[- ]hill|oup|cambridge|harvard|princeton)\b.*$", " ", s)
    # collapse non-alphanum to spaces
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _normalize_text(text: str) -> str:
    """NFKC + lowercase + collapse whitespace. Matches clean.py's first step."""
    t = unicodedata.normalize("NFKC", text)
    t = t.lower()
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _text_hash(text: str) -> str:
    norm = _normalize_text(text)[:TEXT_SAMPLE_CHARS]
    return hashlib.sha256(norm.encode("utf-8", "ignore")).hexdigest()


@dataclass
class RawEntry:
    path: str                  # path relative to raw_root
    size: int
    sha256: str
    fname_key: str             # normalized filename
    personal_hint: bool        # matches PERSONAL_HINT_RE
    text_hash: Optional[str] = None  # set if a matching extracted file exists


@dataclass
class ExtractedEntry:
    path: str                  # path relative to extracted_root
    size: int
    text_hash: str             # hash of normalized first 50K chars
    fname_key: str             # normalized filename (minus extension)
    source_sha256: Optional[str] = None  # sha256 of originating raw file, if known


@dataclass
class Manifest:
    root: str
    raw_root: str
    extracted_root: str
    raw: list[RawEntry] = field(default_factory=list)
    extracted: list[ExtractedEntry] = field(default_factory=list)
    version: int = 1


def _iter_files(root: Path, exts: Optional[set[str]] = None):
    if not root or not root.exists():
        return
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if exts is not None and p.suffix.lower() not in exts:
            continue
        yield p


def _extracted_key_from_raw(raw_name: str) -> str:
    """extract.py names outputs with sha256 prefix; we can't easily reverse
    that. Fall back to filename-stem matching for the fname_key."""
    return _normalize_filename(raw_name)


def build_manifest(
    raw_root: Path,
    extracted_root: Optional[Path],
    *,
    progress_every: int = 50,
) -> Manifest:
    man = Manifest(
        root=str(raw_root.parent) if raw_root else "",
        raw_root=str(raw_root) if raw_root else "",
        extracted_root=str(extracted_root) if extracted_root else "",
    )

    # Extracted first — small markdown files, fast to hash. extract.py
    # names outputs {safe_stem}.md, so we link raw↔extracted via the
    # normalized filename key rather than a sha prefix.
    ext_by_fname: dict[str, ExtractedEntry] = {}
    if extracted_root and extracted_root.exists():
        files = list(_iter_files(extracted_root, {".md"}))
        logger.info("Hashing %d extracted files under %s", len(files), extracted_root)
        for i, p in enumerate(files):
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:  # noqa: BLE001
                logger.warning("skip %s: %s", p, e)
                continue
            th = _text_hash(text)
            stem = p.stem
            entry = ExtractedEntry(
                path=str(p.relative_to(extracted_root)),
                size=p.stat().st_size,
                text_hash=th,
                fname_key=_normalize_filename(stem),
                source_sha256=None,
            )
            man.extracted.append(entry)
            # First-writer-wins on fname_key collisions in extracted dir
            ext_by_fname.setdefault(entry.fname_key, entry)
            if progress_every and (i + 1) % progress_every == 0:
                logger.info("  extracted %d/%d", i + 1, len(files))

    # Raw files — expensive hash, but necessary.
    if raw_root and raw_root.exists():
        files = list(_iter_files(raw_root, TEXT_EXTS))
        logger.info("Hashing %d raw files under %s", len(files), raw_root)
        for i, p in enumerate(files):
            try:
                sha = sha256_file(p)
            except Exception as e:  # noqa: BLE001
                logger.warning("skip %s: %s", p, e)
                continue
            rel = str(p.relative_to(raw_root))
            fname_key = _normalize_filename(p.name)
            ext_match = ext_by_fname.get(fname_key)
            entry = RawEntry(
                path=rel,
                size=p.stat().st_size,
                sha256=sha,
                fname_key=fname_key,
                personal_hint=bool(PERSONAL_HINT_RE.search(p.name)),
                text_hash=ext_match.text_hash if ext_match else None,
            )
            man.raw.append(entry)
            if progress_every and (i + 1) % progress_every == 0:
                logger.info("  raw %d/%d", i + 1, len(files))

    return man


def save_manifest(man: Manifest, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": man.version,
        "root": man.root,
        "raw_root": man.raw_root,
        "extracted_root": man.extracted_root,
        "counts": {"raw": len(man.raw), "extracted": len(man.extracted)},
        "raw": [asdict(e) for e in man.raw],
        "extracted": [asdict(e) for e in man.extracted],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("Wrote manifest: %s (raw=%d, extracted=%d)",
                out_path, len(man.raw), len(man.extracted))


def load_manifest(path: Path) -> Manifest:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    man = Manifest(
        root=data.get("root", ""),
        raw_root=data.get("raw_root", ""),
        extracted_root=data.get("extracted_root", ""),
        version=data.get("version", 1),
    )
    for e in data.get("raw", []):
        man.raw.append(RawEntry(**e))
    for e in data.get("extracted", []):
        man.extracted.append(ExtractedEntry(**e))
    return man


def diff_manifests(self_man: Manifest, other_man: Manifest) -> dict:
    """What would merging other into self add?

    Classifies every raw file in `other` into one of:
      - new: no collision with self, worth adding
      - dup_file: same sha256 present in self (skip)
      - dup_text: same extracted-text hash in self (skip — different PDF,
        same content)
      - fuzzy_name: same normalized filename in self (human review)
      - personal: matches PERSONAL_HINT_RE (human review)
    """
    self_shas = {e.sha256 for e in self_man.raw}
    self_text_hashes: set[str] = set()
    for e in self_man.raw:
        if e.text_hash:
            self_text_hashes.add(e.text_hash)
    for e in self_man.extracted:
        self_text_hashes.add(e.text_hash)
    self_fname_keys: dict[str, str] = {}  # key -> first sample path
    for e in self_man.raw:
        self_fname_keys.setdefault(e.fname_key, e.path)

    buckets = {"new": [], "dup_file": [], "dup_text": [],
               "fuzzy_name": [], "personal": []}

    for e in other_man.raw:
        row = {
            "path": e.path, "size": e.size, "sha256": e.sha256,
            "fname_key": e.fname_key, "text_hash": e.text_hash,
        }
        if e.sha256 in self_shas:
            buckets["dup_file"].append(row)
            continue
        if e.text_hash and e.text_hash in self_text_hashes:
            buckets["dup_text"].append(row)
            continue
        if e.personal_hint:
            buckets["personal"].append(row)
            continue
        if e.fname_key and e.fname_key in self_fname_keys:
            row["self_match_path"] = self_fname_keys[e.fname_key]
            buckets["fuzzy_name"].append(row)
            continue
        buckets["new"].append(row)

    summary = {k: len(v) for k, v in buckets.items()}
    total = sum(summary.values())
    return {
        "summary": summary,
        "other_total": total,
        "self_raw_total": len(self_man.raw),
        "buckets": buckets,
    }


def save_diff(diff: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(diff, f, ensure_ascii=False, indent=2)


def print_diff_summary(diff: dict) -> None:
    s = diff["summary"]
    print(f"self raw total:    {diff['self_raw_total']}")
    print(f"other raw total:   {diff['other_total']}")
    print(f"  new:             {s['new']:>5}   (would be added)")
    print(f"  dup_file:        {s['dup_file']:>5}   (exact sha256 match — skip)")
    print(f"  dup_text:        {s['dup_text']:>5}   (same extracted content — skip)")
    print(f"  fuzzy_name:      {s['fuzzy_name']:>5}   (filename collision — review)")
    print(f"  personal:        {s['personal']:>5}   (looks personal — review)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_build = sub.add_parser("build", help="Build manifest of a corpus root")
    ap_build.add_argument("--raw", default=str(RAW_DIR),
                          help="Raw file root (recursive scan)")
    ap_build.add_argument("--extracted", default=str(EXTRACTED_DIR),
                          help="Extracted markdown root ('' to skip)")
    ap_build.add_argument("--out", required=True, help="Output manifest JSON")

    ap_diff = sub.add_parser("diff", help="Diff two manifests")
    ap_diff.add_argument("--self", dest="self_path", required=True,
                         help="This side (what I already have)")
    ap_diff.add_argument("--other", required=True,
                         help="Other side (what I might add)")
    ap_diff.add_argument("--out", help="Write full diff JSON here (optional)")

    ap_list = sub.add_parser("list-new",
                             help="Print paths of 'new' files from a diff")
    ap_list.add_argument("--diff", required=True)
    ap_list.add_argument("--bucket", default="new",
                         choices=["new", "dup_file", "dup_text",
                                  "fuzzy_name", "personal"])

    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    if args.cmd == "build":
        raw = Path(args.raw).resolve() if args.raw else None
        ext = Path(args.extracted).resolve() if args.extracted else None
        man = build_manifest(raw, ext)
        save_manifest(man, Path(args.out))
    elif args.cmd == "diff":
        self_man = load_manifest(Path(args.self_path))
        other_man = load_manifest(Path(args.other))
        d = diff_manifests(self_man, other_man)
        print_diff_summary(d)
        if args.out:
            save_diff(d, Path(args.out))
            print(f"\nWrote full diff: {args.out}")
    elif args.cmd == "list-new":
        with open(args.diff, "r", encoding="utf-8") as f:
            d = json.load(f)
        for row in d["buckets"][args.bucket]:
            print(row["path"])
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
