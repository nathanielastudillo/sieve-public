"""Inspectable, functional-style data pipeline for SIEVE.

Usage:
    from sieve.data.pipeline import core, stages as S, paragraphs as P
    from sieve.data.pipeline.extract import load_extracted_document

    clean = core.pipe(S.normalize_unicode, S.strip_page_numbers, ...)
    doc = load_extracted_document(pdf, md)
    doc = clean(doc)                    # doc.prov.stages now records every step

Or via CLI:
    python -m data.pipeline.run    --version v010 --source-dir data/extracted --limit 20
    python -m data.pipeline.inspect trace <doc_id>
"""

from . import core, stages, paragraphs, extract, run, inspect  # noqa: F401
