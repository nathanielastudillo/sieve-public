"""BPE tokenizer training for SIEVE.

BPE is NOT a one-shot step — it's a parameter we evolve.
Different vocab sizes, normalization, pre-tokenization rules produce
different tokenizers → different token streams → different models.

Each tokenizer version gets its own directory with:
  tokenizers/v{NNN}/
    tokenizer.json   — HuggingFace tokenizers format
    config.json      — training config (reproducible)
    metrics.json     — quality metrics (compression, fertility, domain integrity)

Usage:
  python data/train_bpe.py --corpus data/cleaned/ --version v001 --vocab-size 8000
  python data/train_bpe.py --corpus data/cleaned/ --version v002 --vocab-size 16000
  python data/train_bpe.py --corpus data/cleaned/ --version v003 --vocab-size 8000 --no-lowercase
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent
TOKENIZERS_DIR = DATA_DIR / "tokenizers"

# Domain terms that should ideally be single tokens
DOMAIN_TERMS_CLINICAL = [
    "psychoanalysis", "transference", "countertransference", "narcissism",
    "jouissance", "signifier", "unconscious", "repression", "sublimation",
    "identification", "introjection", "projection", "splitting",
    "borderline", "neurosis", "psychosis", "hysteria", "obsessional",
    "castration", "phallus", "oedipus", "superego", "libido",
    "reconsolidation", "coherence", "symptom", "pathological",
    "attachment", "mentalization", "countertransference",
]

DOMAIN_TERMS_THEORY = [
    "hegemony", "assemblage", "deterritorialization", "reterritorialization",
    "rhizome", "multiplicity", "immanence", "transcendence",
    "infrastructure", "ideology", "dialectical", "epistemology",
    "ontological", "phenomenology", "hermeneutics", "praxis",
    "stratification", "legitimacy", "coercion", "bureaucratic",
    "interaction", "solidarity", "microsociology", "macrosociology",
]

DOMAIN_TERMS_PHYSICS = [
    "hamiltonian", "lagrangian", "eigenvalue", "eigenvector",
    "thermodynamic", "entropy", "electromagnetic", "photon",
    "quantum", "superposition", "entanglement", "wavefunction",
    "schrodinger", "perturbation", "renormalization", "divergence",
]

ALL_DOMAIN_TERMS = DOMAIN_TERMS_CLINICAL + DOMAIN_TERMS_THEORY + DOMAIN_TERMS_PHYSICS


def train_bpe(
    corpus_dir: Path,
    version: str = "v001",
    vocab_size: int = 8000,
    min_frequency: int = 2,
    lowercase: bool = True,
    strip_accents: bool = False,
    special_tokens: Optional[list[str]] = None,
    output_dir: Optional[Path] = None,
) -> dict:
    """Train a BPE tokenizer on the corpus.

    Returns metrics dict.
    """
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers

    if output_dir is None:
        output_dir = TOKENIZERS_DIR / version
    output_dir.mkdir(parents=True, exist_ok=True)

    if special_tokens is None:
        special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]

    # Collect training files
    files = sorted(corpus_dir.glob("*.md")) + sorted(corpus_dir.glob("*.txt"))
    if not files:
        raise ValueError(f"No .md or .txt files found in {corpus_dir}")
    file_paths = [str(f) for f in files]
    logger.info("Training BPE v%s: vocab=%d, %d files", version, vocab_size, len(files))

    # Build normalizer
    norm_steps = [normalizers.NFKC()]
    if lowercase:
        norm_steps.append(normalizers.Lowercase())
    if strip_accents:
        norm_steps.append(normalizers.StripAccents())
    normalizer = normalizers.Sequence(norm_steps)

    # Build tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizer
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

    # Decoder (needed for detokenization)
    from tokenizers import decoders
    tokenizer.decoder = decoders.ByteLevel()

    # Train
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )

    t0 = time.time()
    tokenizer.train(files=file_paths, trainer=trainer)
    train_time = time.time() - t0

    # Save tokenizer
    tok_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tok_path))

    # Save config
    config = {
        "version": version,
        "vocab_size": vocab_size,
        "actual_vocab_size": tokenizer.get_vocab_size(),
        "min_frequency": min_frequency,
        "lowercase": lowercase,
        "strip_accents": strip_accents,
        "special_tokens": special_tokens,
        "normalizer": "NFKC" + ("+lowercase" if lowercase else "") + ("+strip_accents" if strip_accents else ""),
        "pre_tokenizer": "byte_level",
        "num_train_files": len(files),
        "train_time_seconds": round(train_time, 2),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    # Evaluate
    metrics = evaluate_tokenizer(tokenizer, files, output_dir)

    logger.info(
        "BPE %s trained: vocab=%d, compression=%.2f, fertility=%.2f, domain_integrity=%.1f%%",
        version, tokenizer.get_vocab_size(),
        metrics["compression_ratio"], metrics["fertility"],
        metrics["domain_integrity_pct"],
    )

    return metrics


def evaluate_tokenizer(
    tokenizer,
    corpus_files: list[Path],
    output_dir: Path,
) -> dict:
    """Evaluate tokenizer quality metrics."""
    # Read corpus
    full_text = ""
    for f in corpus_files[:20]:  # sample first 20 files for speed
        full_text += Path(f).read_text(encoding="utf-8") + "\n"

    encoded = tokenizer.encode(full_text)
    n_tokens = len(encoded.ids)
    n_bytes = len(full_text.encode("utf-8"))
    n_words = len(full_text.split())

    # 1. Compression ratio: bytes / tokens
    compression = n_bytes / max(1, n_tokens)

    # 2. Fertility: tokens per word
    fertility = n_tokens / max(1, n_words)

    # 3. Domain term integrity: % of domain terms that are single tokens
    single_count = 0
    term_results = {}
    for term in ALL_DOMAIN_TERMS:
        enc = tokenizer.encode(term)
        n_subtokens = len(enc.ids)
        term_results[term] = n_subtokens
        if n_subtokens == 1:
            single_count += 1

    domain_integrity = single_count / max(1, len(ALL_DOMAIN_TERMS))

    # 4. Vocab utilization: unique tokens used / vocab size
    unique_used = len(set(encoded.ids))
    utilization = unique_used / max(1, tokenizer.get_vocab_size())

    # 5. Sequence length: how many tokens for the whole corpus
    # (Lower = better compression = more content per context window)

    metrics = {
        "compression_ratio": round(compression, 3),
        "fertility": round(fertility, 3),
        "domain_integrity_pct": round(100 * domain_integrity, 1),
        "single_token_terms": single_count,
        "total_domain_terms": len(ALL_DOMAIN_TERMS),
        "vocab_utilization_pct": round(100 * utilization, 1),
        "corpus_tokens": n_tokens,
        "corpus_bytes": n_bytes,
        "corpus_words": n_words,
        "term_subtokens": {k: v for k, v in sorted(term_results.items()) if v > 1},
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import argparse
    parser = argparse.ArgumentParser(description="Train BPE tokenizer for SIEVE")
    parser.add_argument("--corpus", default=str(DATA_DIR / "cleaned"),
                        help="Directory with cleaned .md/.txt files")
    parser.add_argument("--version", default="v001")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--no-lowercase", action="store_true")
    parser.add_argument("--strip-accents", action="store_true")
    args = parser.parse_args()

    train_bpe(
        corpus_dir=Path(args.corpus),
        version=args.version,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        lowercase=not args.no_lowercase,
        strip_accents=args.strip_accents,
    )
