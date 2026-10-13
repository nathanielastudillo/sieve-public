"""EPUB extraction pipeline for SIEVE (corpus_v004 expansion).

Uses pymupdf (MuPDF backend) to open epubs as documents and extract
per-page markdown. Mirrors the shape of data/extract.py but targets
.epub files from data/raw/corpus-expansion/epub/.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path

import pymupdf

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SRC_DIR = DATA_DIR / "raw" / "corpus-expansion" / "epub"
OUT_DIR = DATA_DIR / "extracted"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_epub(epub_path: Path, output_dir: Path) -> dict:
    stem = epub_path.stem
    safe_stem = "".join(c if c.isalnum() or c in "-_ " else "_" for c in stem).strip()[:120]
    out_path = output_dir / f"{safe_stem}.md"

    if out_path.exists():
        logger.info("  [skip] %s (already extracted)", safe_stem)
        return {
            "source": epub_path.name,
            "output": out_path.name,
            "chars": out_path.stat().st_size,
            "status": "skipped",
        }

    try:
        doc = pymupdf.open(str(epub_path))
        chunks = []
        for page in doc:
            text = page.get_text("text")
            if text and text.strip():
                chunks.append(text)
        doc.close()
        md = "\n\n".join(chunks)
        out_path.write_text(md, encoding="utf-8")
        logger.info("  [ok]   %s → %d chars (%d pages)", safe_stem, len(md), len(chunks))
        return {
            "source": epub_path.name,
            "output": out_path.name,
            "chars": len(md),
            "pages": len(chunks),
            "sha256": sha256_file(epub_path),
            "status": "ok",
        }
    except Exception as e:
        logger.error("  [fail] %s: %s", safe_stem, e)
        return {
            "source": epub_path.name,
            "output": None,
            "chars": 0,
            "status": f"error: {e}",
        }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    epubs = sorted(SRC_DIR.glob("*.epub"))
    logger.info("Found %d epubs in %s", len(epubs), SRC_DIR)
    results = []
    for i, ep in enumerate(epubs):
        logger.info("[%d/%d] %s", i + 1, len(epubs), ep.name[:80])
        results.append(extract_epub(ep, OUT_DIR))

    manifest = {
        "source_dir": str(SRC_DIR),
        "output_dir": str(OUT_DIR),
        "total": len(epubs),
        "extracted": sum(1 for r in results if r["status"] in ("ok", "skipped")),
        "failed": sum(1 for r in results if r["status"].startswith("error")),
        "total_chars": sum(r["chars"] for r in results),
        "files": results,
    }
    manifest_path = OUT_DIR / "epub_extraction_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info(
        "Done: %d/%d ok, %d failed, %d total chars",
        manifest["extracted"], manifest["total"],
        manifest["failed"], manifest["total_chars"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
