"""Paragraph-level operations: split, density score, dedup, stratified subsample.

These work on ``list[Paragraph]`` and always return (kept, dropped) pairs (or
append decision metadata) so every paragraph's fate is inspectable downstream.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import replace
from typing import Callable

import numpy as np

from .core import Document, Paragraph

logger = logging.getLogger(__name__)


def density_score(text: str, n: int = 3) -> float:
    words = text.lower().split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def split_paragraphs(doc: Document, *, min_words: int = 20) -> list[Paragraph]:
    raw = [p.strip() for p in doc.text.split("\n\n")]
    kept: list[Paragraph] = []
    for i, p in enumerate(raw):
        if len(p.split()) < min_words:
            continue
        kept.append(
            Paragraph(
                prov=doc.prov,
                para_id=f"{doc.prov.doc_id}#p{i:04d}",
                text=p,
                attrs={
                    "density": round(density_score(p), 4),
                    "words": len(p.split()),
                    "chars": len(p),
                },
            )
        )
    return kept


def minhash_dedup_paragraphs(
    paragraphs: list[Paragraph],
    *,
    threshold: float = 0.7,
    num_perm: int = 128,
) -> tuple[list[Paragraph], list[Paragraph]]:
    """Return (kept, dropped). Dropped paragraphs carry ``attrs['dup_of']``."""
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        logger.warning("datasketch not installed, skipping MinHash dedup")
        return list(paragraphs), []

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept: list[Paragraph] = []
    dropped: list[Paragraph] = []

    for p in paragraphs:
        if len(p.text) < 80:
            kept.append(p)
            continue
        m = MinHash(num_perm=num_perm)
        text = p.text
        for j in range(len(text) - 4):
            m.update(text[j : j + 5].encode("utf-8"))
        try:
            hits = lsh.query(m)
        except Exception:
            kept.append(p)
            continue
        if hits:
            new_attrs = dict(p.attrs)
            new_attrs["dup_of"] = hits[0]
            dropped.append(replace(p, attrs=new_attrs))
        else:
            lsh.insert(p.para_id, m)
            kept.append(p)

    return kept, dropped


def stratified_budget_subsample(
    paragraphs: list[Paragraph],
    *,
    target_tokens: int,
    token_count_fn: Callable[[str], int],
    stratification: str = "sqrt",
    min_budget: int = 500,
) -> tuple[list[Paragraph], list[Paragraph], dict]:
    """Pick paragraphs per doc up to a per-doc token budget, density-first.

    Returns (kept, dropped, per_doc_stats) where per_doc_stats maps each doc_id
    to a summary of its budget, how many paragraphs were picked, etc.
    """
    by_doc: dict[str, list[Paragraph]] = defaultdict(list)
    for p in paragraphs:
        by_doc[p.prov.doc_id].append(p)
    doc_ids = sorted(by_doc.keys())

    sizes = np.array(
        [sum(p.attrs.get("words", len(p.text.split())) for p in by_doc[d]) for d in doc_ids],
        dtype=float,
    )
    if sizes.sum() == 0:
        return [], list(paragraphs), {}

    if stratification == "uniform":
        weights = np.ones_like(sizes)
    elif stratification == "proportional":
        weights = sizes
    elif stratification == "sqrt":
        weights = np.sqrt(sizes)
    else:
        raise ValueError(f"unknown stratification: {stratification}")
    weights = weights / weights.sum()
    budgets = np.maximum((weights * target_tokens).astype(int), min_budget)

    kept: list[Paragraph] = []
    dropped: list[Paragraph] = []
    per_doc: dict = {}

    for d, budget in zip(doc_ids, budgets):
        ordered = sorted(by_doc[d], key=lambda p: -p.attrs.get("density", 0.0))
        picked: list[Paragraph] = []
        picked_tokens = 0
        threshold_d = None
        for p in ordered:
            n = token_count_fn(p.text)
            if picked_tokens + n > budget and picked:
                break
            picked.append(p)
            picked_tokens += n
            threshold_d = p.attrs.get("density")
        picked_ids = {p.para_id for p in picked}
        for p in by_doc[d]:
            if p.para_id in picked_ids:
                kept.append(p)
            else:
                new_attrs = dict(p.attrs)
                new_attrs["not_picked_reason"] = "budget_exhausted_or_low_density"
                dropped.append(replace(p, attrs=new_attrs))

        per_doc[d] = {
            "doc_id": d,
            "title": by_doc[d][0].prov.title,
            "source_path": by_doc[d][0].prov.source_path,
            # paragraphs entering the subsample (= survived minhash)
            "paragraphs_after_minhash": len(by_doc[d]),
            "paragraphs_picked": len(picked),
            "paragraphs_overflow": len(by_doc[d]) - len(picked),
            "source_words": int(sum(p.attrs.get("words", 0) for p in by_doc[d])),
            "budget_tokens": int(budget),
            "used_tokens": picked_tokens,
            "min_density_kept": round(threshold_d, 4) if threshold_d is not None else None,
            "mean_density_kept": round(
                float(np.mean([p.attrs.get("density", 0.0) for p in picked])) if picked else 0.0,
                4,
            ),
        }

    return kept, dropped, per_doc
