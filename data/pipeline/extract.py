"""Source stages: produce a Document from an on-disk PDF or pre-extracted .md.

These are the pipeline's entry points. They attach initial Provenance
(doc_id, title, sha256, source_path) and a first StageRecord named "extract".
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

from .core import Document, StageRecord, make_provenance, sha256_bytes

logger = logging.getLogger(__name__)

MIN_CHARS_OK = 1000


def _try_pymupdf4llm(path: Path) -> str:
    try:
        import pymupdf4llm

        return pymupdf4llm.to_markdown(str(path))
    except Exception as e:
        logger.warning("    pymupdf4llm failed on %s: %s", path.name, e)
        return ""


def _try_pymupdf_raw(path: Path) -> str:
    try:
        import pymupdf

        doc = pymupdf.open(str(path))
        parts = [page.get_text() for page in doc]
        doc.close()
        return "\n".join(parts)
    except Exception as e:
        logger.warning("    pymupdf raw failed on %s: %s", path.name, e)
        return ""


def _try_ocr(path: Path) -> str:
    if shutil.which("ocrmypdf") is None or shutil.which("tesseract") is None:
        return ""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
        out = Path(tf.name)
    try:
        r = subprocess.run(
            [
                "ocrmypdf",
                "--skip-text",
                "--optimize",
                "0",
                "--jobs",
                "4",
                "-q",
                str(path),
                str(out),
            ],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if r.returncode != 0:
            return ""
        return _try_pymupdf_raw(out)
    finally:
        try:
            out.unlink(missing_ok=True)
        except Exception:
            pass


def extract_pdf_document(path: Path, *, allow_ocr: bool = True) -> Document:
    """Extract a PDF from scratch. Records the extraction method used."""
    raw_bytes = path.read_bytes()
    prov = make_provenance(path, sha256_bytes(raw_bytes))

    t0 = time.time()
    text = _try_pymupdf4llm(path)
    method = "pymupdf4llm"

    if len(text) < MIN_CHARS_OK:
        raw = _try_pymupdf_raw(path)
        if len(raw) > len(text):
            text, method = raw, "pymupdf_raw"

    if len(text) < MIN_CHARS_OK and allow_ocr:
        ocr = _try_ocr(path)
        if len(ocr) > len(text):
            text, method = ocr, "ocr"

    wall_ms = int((time.time() - t0) * 1000)
    rec = StageRecord(
        stage="extract",
        in_chars=0,
        out_chars=len(text),
        wall_ms=wall_ms,
        notes={"method": method, "source_bytes": len(raw_bytes)},
        params={"allow_ocr": allow_ocr},
    )
    return Document(prov=prov.with_stage(rec), text=text)


def load_extracted_document(source_path: Path, md_path: Path) -> Document:
    """Load a previously-extracted .md file as a Document.

    ``source_path`` is the original PDF (used for provenance sha256 if
    available); ``md_path`` is the cached extracted markdown.
    """
    if source_path.exists() and source_path.is_file():
        sha = sha256_bytes(source_path.read_bytes())
        prov_path = source_path
    else:
        sha = sha256_bytes(md_path.read_bytes())
        prov_path = md_path

    prov = make_provenance(prov_path, sha)
    text = md_path.read_text(encoding="utf-8", errors="replace")
    rec = StageRecord(
        stage="extract",
        in_chars=0,
        out_chars=len(text),
        wall_ms=0,
        notes={"method": "pre-extracted", "md_path": str(md_path)},
    )
    return Document(prov=prov.with_stage(rec), text=text)
