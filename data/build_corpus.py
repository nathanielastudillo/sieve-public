"""Corpus builder for SIEVE.

Selects files from cleaned/, applies dedup, builds train/val/test splits,
tokenizes with a specified tokenizer, and writes versioned corpus artifacts.

Each corpus version is a complete, reproducible snapshot:
  corpus_v{NNN}/
    train.bin        — tokenized training data (memmap)
    val.bin          — tokenized validation data (memmap)
    test.bin         — tokenized tertiary test set (NEVER used for selection)
    metadata.json    — full provenance: files, splits, tokenizer, hashes
    sources.txt      — concatenated cleaned text (for BPE training)
"""

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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def minhash_dedup_paragraphs(
    text: str, threshold: float = 0.7, num_perm: int = 128
) -> str:
    """Remove near-duplicate paragraphs using MinHash LSH."""
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        logger.warning("datasketch not installed, skipping MinHash dedup")
        return text

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) < 2:
        return text

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept = []

    for i, para in enumerate(paragraphs):
        if len(para) < 80:
            kept.append(para)
            continue

        m = MinHash(num_perm=num_perm)
        # Character-level 5-shingles
        for j in range(len(para) - 4):
            m.update(para[j : j + 5].encode("utf-8"))

        try:
            result = lsh.query(m)
            if not result:
                lsh.insert(f"p-{i}", m)
                kept.append(para)
            # else: near-duplicate, skip
        except Exception:
            kept.append(para)  # keep on error

    removed = len(paragraphs) - len(kept)
    if removed > 0:
        logger.info("  MinHash dedup: removed %d/%d paragraphs", removed, len(paragraphs))
    return "\n\n".join(kept)


def build_corpus(
    cleaned_dir: Path = CLEANED_DIR,
    output_dir: Optional[Path] = None,
    version: str = "v001",
    tokenizer_type: str = "tiktoken",  # tiktoken | custom
    tokenizer_path: Optional[str] = None,  # path to custom tokenizer
    tiktoken_encoding: str = "gpt2",
    val_fraction: float = 0.10,
    test_fraction: float = 0.05,
    dedup_threshold: float = 0.7,
    include_files: Optional[list[str]] = None,  # whitelist, or None for all
    exclude_files: Optional[list[str]] = None,
    seed: int = 42,
) -> dict:
    """Build a versioned corpus from cleaned text files.

    Returns metadata dict with full provenance.
    """
    if output_dir is None:
        output_dir = DATA_DIR / f"corpus_{version}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect files
    files = sorted(cleaned_dir.glob("*.md"))
    if include_files:
        files = [f for f in files if f.name in include_files]
    if exclude_files:
        files = [f for f in files if f.name not in exclude_files]

    logger.info("Building corpus %s from %d files", version, len(files))

    # Read and concatenate
    sources = []
    all_text_parts = []
    for f in files:
        text = f.read_text(encoding="utf-8").strip()
        if len(text) < 100:
            continue
        sources.append({
            "file": f.name,
            "chars": len(text),
            "sha256": sha256_text(text),
        })
        all_text_parts.append(text)

    full_text = "\n\n".join(all_text_parts)
    logger.info("Raw concatenated text: %d chars, %d words", len(full_text), len(full_text.split()))

    # MinHash dedup across the full corpus
    full_text = minhash_dedup_paragraphs(full_text, threshold=dedup_threshold)
    logger.info("After dedup: %d chars", len(full_text))

    # Save sources.txt (for BPE training)
    (output_dir / "sources.txt").write_text(full_text, encoding="utf-8")

    # Tokenize
    if tokenizer_type == "custom" and tokenizer_path:
        from tokenizers import Tokenizer as HFTokenizer
        tok = HFTokenizer.from_file(tokenizer_path)
        encoded = tok.encode(full_text)
        token_ids = encoded.ids
        vocab_size = tok.get_vocab_size()
    else:
        import tiktoken
        enc = tiktoken.get_encoding(tiktoken_encoding)
        token_ids = enc.encode(full_text)
        vocab_size = enc.n_vocab

    total_tokens = len(token_ids)
    logger.info("Tokenized: %d tokens (vocab_size=%d)", total_tokens, vocab_size)

    # Determine dtype
    dtype = np.uint16 if vocab_size < 65536 else np.uint32
    tokens_array = np.array(token_ids, dtype=dtype)

    # Stratified split by position (preserving document order)
    rng = np.random.default_rng(seed)
    n_test = int(total_tokens * test_fraction)
    n_val = int(total_tokens * val_fraction)
    n_train = total_tokens - n_val - n_test

    # Shuffle at chunk level (not token level) to preserve local coherence
    chunk_size = 512
    n_chunks = total_tokens // chunk_size
    chunk_indices = np.arange(n_chunks)
    rng.shuffle(chunk_indices)

    # Assign chunks to splits
    n_test_chunks = max(1, int(n_chunks * test_fraction))
    n_val_chunks = max(1, int(n_chunks * val_fraction))

    test_chunks = sorted(chunk_indices[:n_test_chunks])
    val_chunks = sorted(chunk_indices[n_test_chunks : n_test_chunks + n_val_chunks])
    train_chunks = sorted(chunk_indices[n_test_chunks + n_val_chunks :])

    def gather_chunks(indices):
        parts = []
        for idx in indices:
            start = idx * chunk_size
            end = start + chunk_size
            parts.append(tokens_array[start:end])
        return np.concatenate(parts) if parts else np.array([], dtype=dtype)

    train_tokens = gather_chunks(train_chunks)
    val_tokens = gather_chunks(val_chunks)
    test_tokens = gather_chunks(test_chunks)

    # Write binary files
    train_tokens.tofile(str(output_dir / "train.bin"))
    val_tokens.tofile(str(output_dir / "val.bin"))
    test_tokens.tofile(str(output_dir / "test.bin"))

    metadata = {
        "version": version,
        "tokenizer_type": tokenizer_type,
        "tokenizer_path": tokenizer_path,
        "tiktoken_encoding": tiktoken_encoding if tokenizer_type == "tiktoken" else None,
        "vocab_size": vocab_size,
        "dtype": str(dtype),
        "total_tokens": total_tokens,
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "test_tokens": len(test_tokens),
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "dedup_threshold": dedup_threshold,
        "seed": seed,
        "num_sources": len(sources),
        "total_source_chars": sum(s["chars"] for s in sources),
        "sources": sources,
        "corpus_sha256": hashlib.sha256(tokens_array.tobytes()).hexdigest(),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    logger.info(
        "Corpus %s built: train=%d val=%d test=%d tokens (vocab=%d)",
        version, len(train_tokens), len(val_tokens), len(test_tokens), vocab_size,
    )
    return metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import argparse
    parser = argparse.ArgumentParser(description="Build versioned SIEVE corpus")
    parser.add_argument("--version", default="v001")
    parser.add_argument("--cleaned-dir", default=str(CLEANED_DIR))
    parser.add_argument("--tokenizer", default="tiktoken", choices=["tiktoken", "custom"])
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--encoding", default="gpt2")
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--dedup-threshold", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_corpus(
        cleaned_dir=Path(args.cleaned_dir),
        version=args.version,
        tokenizer_type=args.tokenizer,
        tokenizer_path=args.tokenizer_path,
        tiktoken_encoding=args.encoding,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        dedup_threshold=args.dedup_threshold,
        seed=args.seed,
    )
