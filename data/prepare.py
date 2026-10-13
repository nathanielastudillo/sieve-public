"""Corpus extraction and tokenization for SIEVE.

Walks a source directory of PDFs, text, and markdown files, cleans the text,
tokenizes it, splits into train/val, and writes memory-mapped numpy arrays
ready for the dataloader.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Optional

import fitz  # pymupdf
import numpy as np
import tiktoken

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_pdf(path: Path) -> str:
    """Extract text from a PDF using pymupdf/fitz."""
    doc = fitz.open(str(path))
    pages: list[str] = []
    for page in doc:
        text = page.get_text("text")
        if text:
            pages.append(text)
    doc.close()
    return "\n\n".join(pages)


def extract_text(path: Path) -> str:
    """Read plain text or markdown file."""
    return path.read_text(encoding="utf-8", errors="replace")


def extract(path: Path) -> str:
    """Dispatch extraction based on file extension."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return extract_pdf(path)
    if ext in (".txt", ".md"):
        return extract_text(path)
    raise ValueError(f"Unsupported file type: {ext}")


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

_HEADER_FOOTER_RE = re.compile(
    r"(?m)^(?:\s*(?:Page\s+\d+|[-–—]\s*\d+\s*[-–—]|\d+\s*\|)\s*)$"
)
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r"[^\S\n]{2,}")


def clean_text(text: str) -> str:
    """Strip headers/footers, normalize whitespace, remove duplicate paragraphs."""
    # Strip common header/footer patterns (page numbers, dividers)
    text = _HEADER_FOOTER_RE.sub("", text)

    # Normalize whitespace
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    text = text.strip()

    # Remove duplicate paragraphs (preserves first occurrence)
    paragraphs = text.split("\n\n")
    seen: set[str] = set()
    deduped: list[str] = []
    for para in paragraphs:
        normalized = para.strip()
        if not normalized:
            continue
        key = re.sub(r"\s+", " ", normalized.lower())
        if key not in seen:
            seen.add(key)
            deduped.append(normalized)

    return "\n\n".join(deduped)


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def _get_tiktoken_encoder(tokenizer_name: str) -> tiktoken.Encoding:
    """Get a tiktoken encoding by name."""
    return tiktoken.get_encoding(tokenizer_name)


def _load_custom_tokenizer(path: str):
    """Load a sentencepiece model and return an encode function + vocab size."""
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.Load(path)
    return sp


def tokenize_text(
    text: str,
    tokenizer_name: str = "gpt2",
    custom_tokenizer_path: Optional[str] = None,
) -> np.ndarray:
    """Tokenize text into a uint16/uint32 numpy array of token IDs.

    Uses tiktoken by default.  If *custom_tokenizer_path* points to a
    sentencepiece ``.model`` file, that tokenizer is used instead.
    """
    if custom_tokenizer_path is not None:
        sp = _load_custom_tokenizer(custom_tokenizer_path)
        ids = sp.Encode(text)
        vocab_size = sp.GetPieceSize()
    else:
        enc = _get_tiktoken_encoder(tokenizer_name)
        ids = enc.encode(text, allowed_special=set())
        vocab_size = enc.n_vocab

    dtype = np.uint16 if vocab_size < 65536 else np.uint32
    return np.array(ids, dtype=dtype)


# ---------------------------------------------------------------------------
# Train / val split
# ---------------------------------------------------------------------------

def _stratified_split(
    doc_tokens: list[tuple[str, np.ndarray]],
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split token arrays 90/10 stratified by source document.

    Each document contributes proportionally to both train and val so that
    the val set covers all sources.
    """
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []

    for _name, tokens in doc_tokens:
        n = len(tokens)
        if n == 0:
            continue
        n_val = max(1, int(n * val_fraction))
        # Choose a random contiguous val slice so that val has coherent text
        start = rng.integers(0, max(1, n - n_val))
        val_parts.append(tokens[start : start + n_val])
        train_parts.append(np.concatenate([tokens[:start], tokens[start + n_val :]]))

    train = np.concatenate(train_parts) if train_parts else np.array([], dtype=np.uint16)
    val = np.concatenate(val_parts) if val_parts else np.array([], dtype=np.uint16)
    return train, val


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prepare_corpus(
    source_dir: str | Path,
    output_dir: str | Path,
    tokenizer_name: str = "gpt2",
    custom_tokenizer_path: Optional[str] = None,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> dict:
    """End-to-end corpus preparation.

    1. Walk *source_dir* and extract text from .pdf, .txt, .md files.
    2. Clean each document.
    3. Tokenize.
    4. Split train/val (90/10, stratified by source document).
    5. Write ``train.bin`` and ``val.bin`` as memory-mapped numpy arrays.
    6. Write ``metadata.json`` with token counts, source files, and split info.

    Returns the metadata dict.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover and extract
    doc_tokens: list[tuple[str, np.ndarray]] = []
    source_files: list[dict] = []

    for path in sorted(source_dir.rglob("*")):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if not path.is_file():
            continue

        rel = str(path.relative_to(source_dir))
        logger.info("Processing %s", rel)

        try:
            raw = extract(path)
        except Exception:
            logger.exception("Failed to extract %s — skipping", rel)
            continue

        cleaned = clean_text(raw)
        if not cleaned:
            logger.warning("Empty after cleaning: %s — skipping", rel)
            continue

        tokens = tokenize_text(cleaned, tokenizer_name, custom_tokenizer_path)
        doc_tokens.append((rel, tokens))

        source_files.append({
            "path": rel,
            "raw_chars": len(raw),
            "clean_chars": len(cleaned),
            "tokens": len(tokens),
            "sha256": hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest(),
        })

    if not doc_tokens:
        raise RuntimeError(f"No valid documents found in {source_dir}")

    # Split
    train, val = _stratified_split(doc_tokens, val_fraction=val_fraction, seed=seed)

    # Write memory-mapped arrays
    train_path = output_dir / "train.bin"
    val_path = output_dir / "val.bin"

    train_mm = np.memmap(str(train_path), dtype=train.dtype, mode="w+", shape=train.shape)
    train_mm[:] = train
    train_mm.flush()
    del train_mm

    val_mm = np.memmap(str(val_path), dtype=val.dtype, mode="w+", shape=val.shape)
    val_mm[:] = val
    val_mm.flush()
    del val_mm

    # Metadata
    metadata = {
        "tokenizer": custom_tokenizer_path or tokenizer_name,
        "dtype": str(train.dtype),
        "train_tokens": int(len(train)),
        "val_tokens": int(len(val)),
        "total_tokens": int(len(train) + len(val)),
        "val_fraction": val_fraction,
        "seed": seed,
        "num_sources": len(source_files),
        "sources": source_files,
    }

    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    logger.info(
        "Prepared corpus: %d train tokens, %d val tokens from %d files → %s",
        len(train),
        len(val),
        len(source_files),
        output_dir,
    )
    return metadata


def train_custom_tokenizer(
    text_path: str | Path,
    vocab_size: int = 8192,
    output_path: Optional[str | Path] = None,
) -> Path:
    """Train a BPE tokenizer on a text file using sentencepiece.

    Parameters
    ----------
    text_path : path to a UTF-8 text file (the full cleaned corpus).
    vocab_size : target vocabulary size.
    output_path : where to write the ``.model`` and ``.vocab`` files.
        Defaults to ``<text_path_parent>/tokenizer``.

    Returns
    -------
    Path to the saved ``.model`` file.
    """
    import sentencepiece as spm

    text_path = Path(text_path)
    if output_path is None:
        output_path = text_path.parent / "tokenizer"
    else:
        output_path = Path(output_path)

    # sentencepiece wants a prefix, not a full path with extension
    prefix = str(output_path)

    spm.SentencePieceTrainer.Train(
        input=str(text_path),
        model_prefix=prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=3,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        num_threads=4,
    )

    model_path = Path(f"{prefix}.model")
    logger.info("Trained custom tokenizer (%d vocab) → %s", vocab_size, model_path)
    return model_path
