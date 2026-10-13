"""Corpus quality metrics and deduplication for SIEVE.

At 100K tokens every token is load-bearing.  These utilities measure
information density, detect near-duplicates, and balance domain coverage
so the curated corpus is maximally useful for training.
"""

from __future__ import annotations

import hashlib
import re
import struct
from collections import Counter, defaultdict
from typing import Optional, Sequence

# ---------------------------------------------------------------------------
# Information density
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\b\w+\b")


def _tokenize_simple(text: str) -> list[str]:
    """Lowercase whitespace tokenization (for density scoring, not model input)."""
    return _WORD_RE.findall(text.lower())


def compute_density_score(text: str, n: int = 3) -> float:
    """Information density heuristic: unique n-grams / total n-grams.

    Higher values indicate more information-dense text with less repetition.
    A passage that repeats the same phrases scores low; one packed with
    distinct concepts scores high.

    Parameters
    ----------
    text : raw text passage.
    n : n-gram size (default 3 for trigrams).

    Returns
    -------
    Float in [0, 1].  Returns 0.0 for text shorter than *n* words.
    """
    words = _tokenize_simple(text)
    if len(words) < n:
        return 0.0

    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    total = len(ngrams)
    unique = len(set(ngrams))
    return unique / total


# ---------------------------------------------------------------------------
# MinHash near-duplicate detection
# ---------------------------------------------------------------------------

_MAX_HASH = 2**32 - 1
_MERSENNE_PRIME = (1 << 61) - 1


def _shingle(text: str, k: int = 5) -> set[str]:
    """Create character-level k-shingles from text."""
    text = re.sub(r"\s+", " ", text.lower().strip())
    if len(text) < k:
        return {text}
    return {text[i : i + k] for i in range(len(text) - k + 1)}


def _minhash_signature(shingles: set[str], num_perm: int = 128) -> list[int]:
    """Compute a MinHash signature for a set of shingles.

    Uses *num_perm* random hash functions simulated via ``(a * h + b) mod p mod MAX``
    with a fixed seed for reproducibility.
    """
    import random as _random

    rng = _random.Random(42)
    # Pre-generate hash function coefficients
    a_coeffs = [rng.randint(1, _MERSENNE_PRIME - 1) for _ in range(num_perm)]
    b_coeffs = [rng.randint(0, _MERSENNE_PRIME - 1) for _ in range(num_perm)]

    sig = [_MAX_HASH] * num_perm
    for s in shingles:
        h = int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:4], "little")
        for i in range(num_perm):
            val = (a_coeffs[i] * h + b_coeffs[i]) % _MERSENNE_PRIME % _MAX_HASH
            if val < sig[i]:
                sig[i] = val
    return sig


def _jaccard_from_sigs(sig_a: list[int], sig_b: list[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    assert len(sig_a) == len(sig_b)
    return sum(a == b for a, b in zip(sig_a, sig_b)) / len(sig_a)


def deduplicate(texts: list[str], threshold: float = 0.8, num_perm: int = 128) -> list[str]:
    """Remove near-duplicate texts using MinHash.

    Parameters
    ----------
    texts : list of text passages.
    threshold : Jaccard similarity threshold above which a passage is
        considered a duplicate and dropped.
    num_perm : number of permutations for MinHash signatures (more = more
        accurate but slower).

    Returns
    -------
    Deduplicated list of texts (preserves order of first occurrence).
    """
    if not texts:
        return []

    # Compute signatures
    sigs: list[list[int]] = []
    shingle_sets: list[set[str]] = []
    for text in texts:
        sh = _shingle(text)
        shingle_sets.append(sh)
        sigs.append(_minhash_signature(sh, num_perm=num_perm))

    # Greedily keep passages, skipping those too similar to an already-kept one
    kept_indices: list[int] = []
    for i in range(len(texts)):
        is_dup = False
        for j in kept_indices:
            sim = _jaccard_from_sigs(sigs[i], sigs[j])
            if sim >= threshold:
                is_dup = True
                break
        if not is_dup:
            kept_indices.append(i)

    return [texts[i] for i in kept_indices]


# ---------------------------------------------------------------------------
# Passage scoring
# ---------------------------------------------------------------------------

def score_passages(texts: list[str], n: int = 3) -> list[tuple[float, str]]:
    """Rank passages by information density (descending).

    Parameters
    ----------
    texts : list of text passages.
    n : n-gram size for density scoring.

    Returns
    -------
    List of ``(score, text)`` tuples sorted highest-density first.
    """
    scored = [(compute_density_score(t, n=n), t) for t in texts]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored


# ---------------------------------------------------------------------------
# Domain balancing
# ---------------------------------------------------------------------------

def balance_domains(
    texts: list[str],
    domain_labels: list[str],
    target_ratio: Optional[dict[str, float]] = None,
    seed: int = 42,
) -> list[str]:
    """Ensure domain balance in the corpus by down-sampling over-represented domains.

    Parameters
    ----------
    texts : list of text passages.
    domain_labels : parallel list of domain labels (e.g., ``"physics"``,
        ``"clinical"``).
    target_ratio : mapping from domain label to desired fraction of the
        corpus (must sum to 1.0).  If ``None``, all domains are targeted
        at equal representation.
    seed : random seed for deterministic down-sampling.

    Returns
    -------
    Balanced list of texts.  Passages are selected by information density
    (highest first) within each domain.
    """
    assert len(texts) == len(domain_labels), "texts and domain_labels must have equal length"

    import random as _random

    rng = _random.Random(seed)

    # Group by domain
    domain_passages: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for idx, (text, label) in enumerate(zip(texts, domain_labels)):
        score = compute_density_score(text)
        domain_passages[label].append((score, idx))

    # Sort each domain by density (highest first)
    for label in domain_passages:
        domain_passages[label].sort(key=lambda x: x[0], reverse=True)

    domains = sorted(domain_passages.keys())

    # Compute target counts
    if target_ratio is None:
        target_ratio = {d: 1.0 / len(domains) for d in domains}

    # Normalize target_ratio in case it doesn't sum to 1
    ratio_sum = sum(target_ratio.get(d, 0.0) for d in domains)
    if ratio_sum == 0:
        return list(texts)
    normalized = {d: target_ratio.get(d, 0.0) / ratio_sum for d in domains}

    total = len(texts)
    target_counts: dict[str, int] = {}
    for d in domains:
        available = len(domain_passages[d])
        desired = int(total * normalized[d])
        target_counts[d] = min(available, desired)

    # Select top passages per domain
    selected_indices: list[int] = []
    for d in domains:
        count = target_counts[d]
        indices = [idx for _score, idx in domain_passages[d][:count]]
        selected_indices.extend(indices)

    # Return in original order
    selected_indices.sort()
    return [texts[i] for i in selected_indices]
