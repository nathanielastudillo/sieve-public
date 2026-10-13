"""Density-weighted stratified subsample builder for SIEVE.

Given the full cleaned/ corpus (~3M tokens), produce a dense ~100K-token
subsample that preserves per-source diversity.

Strategy:
  1. For each source, split into paragraphs and score each by density
     (unique 3-grams / total 3-grams — same heuristic as clean.py).
  2. Rank paragraphs within each source (descending density).
  3. Compute per-source token budget using sqrt-proportional stratification:
     B_i ∝ sqrt(S_i) where S_i is source i's size. This compromises between
     uniform (each source equal) and size-proportional (big sources dominate):
     Mann gets more than Stavrakakis, but not 10x more.
  4. Walk density-sorted paragraphs per source, accumulating until per-source
     token budget is met.
  5. MinHash dedup across the union.
  6. Tokenize with the specified BPE tokenizer; write train/val/test bins
     and metadata.json (incl. per-source counts, selection thresholds).

Usage:
  python data/build_corpus_subsample.py \\
      --version v003 \\
      --target-tokens 100000 \\
      --tokenizer custom \\
      --tokenizer-path data/tokenizers/v002/tokenizer.json
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent
CLEANED_DIR = DATA_DIR / "cleaned"


def density_score(text: str, n: int = 3) -> float:
    words = text.lower().split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def split_paragraphs(text: str, min_words: int = 20) -> list[str]:
    paras = [p.strip() for p in text.split("\n\n")]
    return [p for p in paras if len(p.split()) >= min_words]


def load_tokenizer(tokenizer_type: str, tokenizer_path: Optional[str], encoding: str):
    if tokenizer_type == "custom" and tokenizer_path:
        from tokenizers import Tokenizer as HFTokenizer
        tok = HFTokenizer.from_file(tokenizer_path)
        return ("custom", tok, tok.get_vocab_size())
    import tiktoken
    enc = tiktoken.get_encoding(encoding)
    return ("tiktoken", enc, enc.n_vocab)


def encode(tok_kind, tok, text: str) -> list[int]:
    if tok_kind == "custom":
        return tok.encode(text).ids
    return tok.encode(text)


def minhash_dedup(paragraphs: list[str], threshold: float = 0.7, num_perm: int = 128) -> list[str]:
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        logger.warning("datasketch not installed; skipping MinHash dedup")
        return paragraphs
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept = []
    for i, para in enumerate(paragraphs):
        if len(para) < 80:
            kept.append(para)
            continue
        m = MinHash(num_perm=num_perm)
        for j in range(len(para) - 4):
            m.update(para[j : j + 5].encode("utf-8"))
        try:
            if not lsh.query(m):
                lsh.insert(f"p-{i}", m)
                kept.append(para)
        except Exception:
            kept.append(para)
    removed = len(paragraphs) - len(kept)
    if removed:
        logger.info("MinHash dedup: removed %d/%d paragraphs", removed, len(paragraphs))
    return kept


def build_subsample(
    cleaned_dir: Path = CLEANED_DIR,
    version: str = "v003",
    target_tokens: int = 100_000,
    tokenizer_type: str = "custom",
    tokenizer_path: Optional[str] = "data/tokenizers/v002/tokenizer.json",
    tiktoken_encoding: str = "gpt2",
    val_fraction: float = 0.10,
    test_fraction: float = 0.05,
    dedup_threshold: float = 0.7,
    min_para_words: int = 20,
    stratification: str = "sqrt",  # uniform | sqrt | proportional
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> dict:
    if output_dir is None:
        output_dir = DATA_DIR / f"corpus_{version}"
    output_dir.mkdir(parents=True, exist_ok=True)

    tok_kind, tok, vocab_size = load_tokenizer(tokenizer_type, tokenizer_path, tiktoken_encoding)
    logger.info("Tokenizer: %s (vocab=%d)", tok_kind, vocab_size)

    files = sorted(cleaned_dir.glob("*.md"))
    logger.info("Found %d cleaned source files", len(files))

    # Per-source: split into paragraphs, count words as size proxy
    sources = []
    for f in files:
        text = f.read_text(encoding="utf-8").strip()
        if len(text) < 500:
            continue
        paras = split_paragraphs(text, min_words=min_para_words)
        size = sum(len(p.split()) for p in paras)
        if size == 0:
            continue
        sources.append({"name": f.name, "path": f, "paragraphs": paras, "words": size})

    total_words = sum(s["words"] for s in sources)
    logger.info("Total words across %d sources: %d", len(sources), total_words)

    # Per-source budget (in tokens) via stratification
    if stratification == "uniform":
        weights = np.ones(len(sources))
    elif stratification == "proportional":
        weights = np.array([s["words"] for s in sources], dtype=float)
    elif stratification == "sqrt":
        weights = np.sqrt([s["words"] for s in sources])
    else:
        raise ValueError(f"unknown stratification: {stratification}")
    weights = weights / weights.sum()
    budgets = (weights * target_tokens).astype(int)
    # Ensure everyone gets at least min_budget tokens (1% of target)
    min_budget = max(500, target_tokens // 200)
    budgets = np.maximum(budgets, min_budget)
    logger.info("Stratification=%s; min_budget=%d", stratification, min_budget)

    # For each source: rank paragraphs by density (desc) and pick until budget
    selected_per_source: list[dict] = []
    all_kept: list[str] = []
    for s, budget in zip(sources, budgets):
        paras = s["paragraphs"]
        scored = [(p, density_score(p)) for p in paras]
        scored.sort(key=lambda x: -x[1])

        # Pick top-density paragraphs until token budget is met
        picked: list[str] = []
        used_tokens = 0
        threshold_density = None
        for p, d in scored:
            n = len(encode(tok_kind, tok, p))
            if used_tokens + n > budget and picked:
                break
            picked.append(p)
            used_tokens += n
            threshold_density = d
        selected_per_source.append({
            "source": s["name"],
            "total_paragraphs": len(paras),
            "picked_paragraphs": len(picked),
            "source_words": s["words"],
            "budget_tokens": int(budget),
            "used_tokens": used_tokens,
            "min_density_kept": round(threshold_density, 4) if threshold_density is not None else None,
            "mean_density_kept": round(
                float(np.mean([density_score(p) for p in picked])) if picked else 0.0, 4
            ),
        })
        all_kept.extend(picked)
        logger.info(
            "  %s: %d/%d paras kept, %d tokens (budget=%d), min_density=%.3f",
            s["name"][:55], len(picked), len(paras), used_tokens, budget,
            threshold_density if threshold_density is not None else 0.0,
        )

    # Cross-source dedup
    all_kept = minhash_dedup(all_kept, threshold=dedup_threshold)

    # Shuffle paragraph order (document-level shuffle preserves local coherence)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(all_kept))
    rng.shuffle(idx)
    all_kept = [all_kept[i] for i in idx]

    full_text = "\n\n".join(all_kept)
    (output_dir / "sources.txt").write_text(full_text, encoding="utf-8")

    # Tokenize
    token_ids = encode(tok_kind, tok, full_text)
    total_tokens = len(token_ids)
    logger.info("Final corpus: %d tokens (target=%d)", total_tokens, target_tokens)

    dtype = np.uint16 if vocab_size < 65536 else np.uint32
    arr = np.array(token_ids, dtype=dtype)

    # Chunk-level split to preserve local coherence
    chunk_size = 512
    n_chunks = total_tokens // chunk_size
    chunk_indices = np.arange(n_chunks)
    rng2 = np.random.default_rng(seed + 1)
    rng2.shuffle(chunk_indices)
    n_test = max(1, int(n_chunks * test_fraction))
    n_val = max(1, int(n_chunks * val_fraction))
    test_c = sorted(chunk_indices[:n_test])
    val_c = sorted(chunk_indices[n_test : n_test + n_val])
    train_c = sorted(chunk_indices[n_test + n_val :])

    def gather(ix):
        parts = [arr[i * chunk_size : (i + 1) * chunk_size] for i in ix]
        return np.concatenate(parts) if parts else np.array([], dtype=dtype)

    train = gather(train_c)
    val = gather(val_c)
    test = gather(test_c)
    train.tofile(output_dir / "train.bin")
    val.tofile(output_dir / "val.bin")
    test.tofile(output_dir / "test.bin")

    metadata = {
        "version": version,
        "derived_from": "cleaned/ (18 sources) via density-weighted stratified subsample",
        "target_tokens": target_tokens,
        "stratification": stratification,
        "tokenizer_type": tokenizer_type,
        "tokenizer_path": tokenizer_path,
        "vocab_size": vocab_size,
        "dtype": str(dtype),
        "total_tokens": total_tokens,
        "train_tokens": len(train),
        "val_tokens": len(val),
        "test_tokens": len(test),
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "dedup_threshold": dedup_threshold,
        "min_para_words": min_para_words,
        "seed": seed,
        "num_sources": len(sources),
        "per_source": selected_per_source,
        "corpus_sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info(
        "Corpus %s: train=%d val=%d test=%d (vocab=%d)",
        version, len(train), len(val), len(test), vocab_size,
    )
    return metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="Build density-weighted stratified subsample")
    p.add_argument("--version", default="v003")
    p.add_argument("--cleaned-dir", default=str(CLEANED_DIR))
    p.add_argument("--target-tokens", type=int, default=100_000)
    p.add_argument("--tokenizer", default="custom", choices=["tiktoken", "custom"])
    p.add_argument("--tokenizer-path", default="data/tokenizers/v002/tokenizer.json")
    p.add_argument("--encoding", default="gpt2")
    p.add_argument("--val-fraction", type=float, default=0.10)
    p.add_argument("--test-fraction", type=float, default=0.05)
    p.add_argument("--dedup-threshold", type=float, default=0.7)
    p.add_argument("--min-para-words", type=int, default=20)
    p.add_argument("--stratification", default="sqrt", choices=["uniform", "sqrt", "proportional"])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    build_subsample(
        cleaned_dir=Path(args.cleaned_dir),
        version=args.version,
        target_tokens=args.target_tokens,
        tokenizer_type=args.tokenizer,
        tokenizer_path=args.tokenizer_path,
        tiktoken_encoding=args.encoding,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        dedup_threshold=args.dedup_threshold,
        min_para_words=args.min_para_words,
        stratification=args.stratification,
        seed=args.seed,
    )
