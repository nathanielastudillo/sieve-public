"""Library of composable document-text cleaning stages.

Each stage is a pure Document -> Document function produced by ``@stage``.
Compose them with ``core.pipe(...)``. Every stage records the char delta it
produced and, where useful, the number of pattern matches it removed.

The ``STANDARD_CLEAN`` pipeline is semantically equivalent to the old
monolithic ``clean.py`` ``clean_text()`` but decomposed into inspectable steps.
"""

from __future__ import annotations

import re
import unicodedata

from .core import pipe, stage


_SMART_CHARS = ("\u2018", "\u2019", "\u201c", "\u201d", "\u2013", "\u2014", "\u2026", "\xa0")


@stage("normalize_unicode", form="NFKC")
def normalize_unicode(text: str):
    return unicodedata.normalize("NFKC", text)


@stage("smart_quotes_to_ascii")
def smart_quotes_to_ascii(text: str):
    subs = sum(text.count(c) for c in _SMART_CHARS)
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "--").replace("\u2014", "---")
    text = text.replace("\u2026", "...")
    text = text.replace("\xa0", " ")
    return text, {"substitutions": subs}


_PAGE_NUMBER_RE = re.compile(r"(?m)^\s*\d{1,4}\s*$")


@stage("strip_page_numbers", pattern=_PAGE_NUMBER_RE.pattern)
def strip_page_numbers(text: str):
    matches = len(_PAGE_NUMBER_RE.findall(text))
    return _PAGE_NUMBER_RE.sub("", text), {"matches": matches}


_CHAPTER_HEADER_RE = re.compile(r"(?m)^#{0,3}\s*(?:CHAPTER|Chapter)\s+\d+\s*$")


@stage("strip_chapter_headers", pattern=_CHAPTER_HEADER_RE.pattern)
def strip_chapter_headers(text: str):
    matches = len(_CHAPTER_HEADER_RE.findall(text))
    return _CHAPTER_HEADER_RE.sub("", text), {"matches": matches}


_PAGE_REF_RE = re.compile(r"\(p\.\s*\d+\)|\(pp\.\s*\d+[-\u2013]\d+\)")


@stage("strip_page_references", pattern=_PAGE_REF_RE.pattern)
def strip_page_references(text: str):
    matches = len(_PAGE_REF_RE.findall(text))
    return _PAGE_REF_RE.sub("", text), {"matches": matches}


_MD_IMAGE_RE = re.compile(r"!\[.*?\]\(.*?\)")


@stage("strip_markdown_images", pattern=_MD_IMAGE_RE.pattern)
def strip_markdown_images(text: str):
    matches = len(_MD_IMAGE_RE.findall(text))
    return _MD_IMAGE_RE.sub("", text), {"matches": matches}


_HRULE_RE = re.compile(r"(?m)^(?:[-_*]{3,}|={3,})\s*$")


@stage("strip_horizontal_rules", pattern=_HRULE_RE.pattern)
def strip_horizontal_rules(text: str):
    matches = len(_HRULE_RE.findall(text))
    return _HRULE_RE.sub("", text), {"matches": matches}


@stage("collapse_blank_lines")
def collapse_blank_lines(text: str):
    return re.sub(r"\n{3,}", "\n\n", text)


@stage("rstrip_lines")
def rstrip_lines(text: str):
    return "\n".join(line.rstrip() for line in text.split("\n"))


@stage("trim")
def trim(text: str):
    return text.strip()


@stage("dedup_paragraphs_exact", min_length=50)
def dedup_paragraphs_exact(text: str):
    paragraphs = text.split("\n\n")
    seen: set[str] = set()
    unique: list[str] = []
    dropped = 0
    for para in paragraphs:
        stripped = para.strip()
        if len(stripped) < 50:
            unique.append(para)
            continue
        key = re.sub(r"\s+", " ", stripped.lower())
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        unique.append(para)
    return "\n\n".join(unique), {"dropped_paragraphs": dropped, "total_paragraphs": len(paragraphs)}


STANDARD_CLEAN_STAGES = [
    normalize_unicode,
    smart_quotes_to_ascii,
    strip_page_numbers,
    strip_chapter_headers,
    strip_page_references,
    strip_markdown_images,
    strip_horizontal_rules,
    collapse_blank_lines,
    rstrip_lines,
    trim,
    dedup_paragraphs_exact,
]

STANDARD_CLEAN = pipe(*STANDARD_CLEAN_STAGES)
