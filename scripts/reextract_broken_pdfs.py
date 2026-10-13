"""Re-extract PDFs that pymupdf4llm mangled, using raw pymupdf.get_text().

pymupdf4llm's markdown conversion dropped 95%+ of text on multi-column or
unusual-layout PDFs (Fink, Lacan seminars, Mann, Laclau). Raw page text
extraction recovers the full content. Two PDFs (Collins IRC, Seminar XX)
return 0 chars even from raw — those are genuinely image-based and need
OCR, handled separately.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path

import pymupdf

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EXTRACTED = DATA / "extracted"

# PDFs under-extracted by pymupdf4llm (raw get_text recovers full content)
TARGETS = [
    "data/raw/corpus-expansion/pdf/Bruce Fink - The Lacanian Subject (1996).pdf",
    "data/raw/corpus-expansion/pdf/[The Seminar of Jacques Lacan , Book 11] Jacques Lacan, Jacques-Alain Miller, Alan Sheridan - The Four Fundamental Concepts of Psychoanalysis (1998, W. W. Norton & Company) - download-site.pdf",
    "data/raw/ Bruce Fink - A Clinical Introduction to Lacanian Psychoanalysis_ Theory and Technique (1999) - download-site.pdf",
    "data/raw/ Bruce Fink - Lacan to the Letter_ Reading Ecrits Closely (2004) - download-site.pdf",
    "data/raw/ Ernesto Laclau, Chantal Mouffe - Hegemony and Socialist Strategy_ Towards a Radical Democratic Politics (2001, Verso) - download-site.pdf",
    "data/raw/Mann_Sources_of_Social_Power_Vol1.pdf",
    "data/raw/Mann_Sources_of_Social_Power_Vol2.pdf",
]


def safe_stem(stem: str) -> str:
    return "".join(c if c.isalnum() or c in "-_ " else "_" for c in stem).strip()[:120]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def reextract(pdf_path: Path, out_dir: Path) -> dict:
    stem = safe_stem(pdf_path.stem)
    out_path = out_dir / f"{stem}.md"
    try:
        doc = pymupdf.open(str(pdf_path))
        chunks = []
        for page in doc:
            t = page.get_text("text")
            if t.strip():
                chunks.append(t)
        text = "\n\n".join(chunks)
        doc.close()
    except Exception as e:
        logger.error("  [fail] %s: %s", stem, e)
        return {"source": pdf_path.name, "status": f"error: {e}", "chars": 0}

    prior = out_path.stat().st_size if out_path.exists() else 0
    out_path.write_text(text, encoding="utf-8")
    logger.info("  [ok]   %s: %d → %d chars", stem, prior, len(text))
    return {
        "source": pdf_path.name,
        "output": out_path.name,
        "chars_before": prior,
        "chars_after": len(text),
        "sha256": sha256_file(pdf_path),
        "status": "ok",
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    results = []
    for rel in TARGETS:
        pdf = ROOT / rel
        if not pdf.exists():
            logger.warning("missing: %s", pdf)
            continue
        logger.info("[*] %s", pdf.name[:90])
        results.append(reextract(pdf, EXTRACTED))

    manifest = EXTRACTED / "reextraction_manifest.json"
    manifest.write_text(json.dumps(results, indent=2))
    logger.info(
        "Done. total chars_after: %d (from %d before)",
        sum(r.get("chars_after", 0) for r in results),
        sum(r.get("chars_before", 0) for r in results),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
