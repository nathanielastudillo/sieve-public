"""PDF extraction pipeline for SIEVE.

Extracts text from PDFs using pymupdf4llm (markdown output).
Handles the full raw/ → extracted/ pipeline with per-file metadata.
"""

import hashlib
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pymupdf
import pymupdf4llm

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent
RAW_DIR = DATA_DIR / "raw"
EXTRACTED_DIR = DATA_DIR / "extracted"

# Threshold below which we consider extraction "failed" and try fallbacks.
# Image-only scans typically yield <1K chars of (mostly garbage) text.
MIN_CHARS_OK = 1000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _try_pymupdf4llm(pdf_path: Path) -> str:
    """Primary extractor. May silently yield ~0 chars on image-only PDFs."""
    try:
        return pymupdf4llm.to_markdown(str(pdf_path))
    except Exception as e:
        logger.warning("    pymupdf4llm failed: %s", e)
        return ""


def _try_pymupdf_raw(pdf_path: Path) -> str:
    """Fallback 1: raw pymupdf.get_text() — sometimes works when pymupdf4llm's
    markdown layout inference gives up on multi-column / scanned pages."""
    try:
        doc = pymupdf.open(str(pdf_path))
        parts = []
        for page in doc:
            parts.append(page.get_text())
        doc.close()
        return "\n".join(parts)
    except Exception as e:
        logger.warning("    pymupdf raw failed: %s", e)
        return ""


def _try_ocrmypdf(pdf_path: Path) -> str:
    """Fallback 2: OCR with ocrmypdf + tesseract, then re-extract.

    Used for image-only scanned books (e.g. some Bourdieu / Padgett volumes).
    Writes OCR'd PDF to a temp file then runs pymupdf on it.
    """
    if shutil.which("ocrmypdf") is None or shutil.which("tesseract") is None:
        logger.warning("    ocrmypdf/tesseract not installed — skipping OCR fallback")
        return ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
            ocr_out = Path(tf.name)
        # --skip-text: keep pages that already have a text layer; OCR the rest.
        # --optimize 0: don't spend time compressing.
        # --jobs: tesseract parallelism.
        # -q: quiet.
        cmd = [
            "ocrmypdf",
            "--skip-text",
            "--optimize", "0",
            "--jobs", "4",
            "-q",
            str(pdf_path),
            str(ocr_out),
        ]
        logger.info("    running OCR (this may take several minutes)...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            logger.warning("    ocrmypdf exit %d: %s", result.returncode,
                           result.stderr[-400:] if result.stderr else "")
            return ""
        text = _try_pymupdf_raw(ocr_out)
        return text
    except subprocess.TimeoutExpired:
        logger.warning("    ocrmypdf timed out after 1h")
        return ""
    except Exception as e:
        logger.warning("    ocrmypdf exception: %s", e)
        return ""
    finally:
        try:
            ocr_out.unlink(missing_ok=True)
        except Exception:
            pass


def extract_pdf(pdf_path: Path, output_dir: Path, *, allow_ocr: bool = True) -> dict:
    """Extract a single PDF to markdown. Returns metadata dict.

    Tries in order: pymupdf4llm → raw pymupdf → ocrmypdf+tesseract.
    Each fallback only runs if the previous yielded < MIN_CHARS_OK chars.
    """
    stem = pdf_path.stem
    # Sanitize filename for filesystem
    safe_stem = "".join(c if c.isalnum() or c in "-_ " else "_" for c in stem).strip()[:120]
    out_path = output_dir / f"{safe_stem}.md"

    if out_path.exists() and out_path.stat().st_size >= MIN_CHARS_OK:
        logger.info("  [skip] %s (already extracted)", safe_stem)
        size = out_path.stat().st_size
        return {
            "source": str(pdf_path.name),
            "output": str(out_path.name),
            "chars": size,
            "status": "skipped",
        }

    method = "pymupdf4llm"
    text = _try_pymupdf4llm(pdf_path)

    if len(text) < MIN_CHARS_OK:
        logger.info("    pymupdf4llm yielded %d chars (<%d) — trying raw pymupdf",
                    len(text), MIN_CHARS_OK)
        raw = _try_pymupdf_raw(pdf_path)
        if len(raw) > len(text):
            text = raw
            method = "pymupdf_raw"

    if len(text) < MIN_CHARS_OK and allow_ocr:
        logger.info("    raw pymupdf yielded %d chars (<%d) — falling back to OCR",
                    len(text), MIN_CHARS_OK)
        ocr = _try_ocrmypdf(pdf_path)
        if len(ocr) > len(text):
            text = ocr
            method = "ocr"

    if not text:
        logger.error("  [fail] %s: all extraction methods returned empty", safe_stem)
        return {
            "source": str(pdf_path.name),
            "output": None,
            "chars": 0,
            "method": method,
            "status": "error: empty after all fallbacks",
        }

    out_path.write_text(text, encoding="utf-8")
    status = "ok" if len(text) >= MIN_CHARS_OK else "ok_small"
    logger.info("  [%s] %s → %d chars (%s)", status, safe_stem, len(text), method)
    return {
        "source": str(pdf_path.name),
        "output": str(out_path.name),
        "chars": len(text),
        "sha256": sha256_file(pdf_path),
        "method": method,
        "status": status,
    }


def extract_all(raw_dir: Path = RAW_DIR, output_dir: Path = EXTRACTED_DIR) -> list[dict]:
    """Extract all PDFs from raw_dir to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(raw_dir.glob("*.pdf"))
    # Also check subdirectories (corpus-host has some nested dirs)
    pdfs += sorted(raw_dir.rglob("*.pdf"))
    # Dedupe
    seen = set()
    unique_pdfs = []
    for p in pdfs:
        if p not in seen:
            seen.add(p)
            unique_pdfs.append(p)
    pdfs = unique_pdfs

    logger.info("Found %d PDFs in %s", len(pdfs), raw_dir)

    results = []
    for i, pdf in enumerate(pdfs):
        logger.info("[%d/%d] %s", i + 1, len(pdfs), pdf.name[:80])
        meta = extract_pdf(pdf, output_dir)
        results.append(meta)

    # Write manifest
    manifest = {
        "source_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "total_pdfs": len(pdfs),
        "extracted": sum(1 for r in results if r["status"] in ("ok", "skipped")),
        "failed": sum(1 for r in results if r["status"].startswith("error")),
        "total_chars": sum(r["chars"] for r in results),
        "files": results,
    }
    manifest_path = output_dir / "extraction_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info(
        "Extraction complete: %d/%d ok, %d failed, %d total chars",
        manifest["extracted"], manifest["total_pdfs"],
        manifest["failed"], manifest["total_chars"],
    )
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raw = Path(sys.argv[1]) if len(sys.argv) > 1 else RAW_DIR
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else EXTRACTED_DIR
    extract_all(raw, out)
