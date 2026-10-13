"""Text cleaning pipeline for SIEVE.

Cleans extracted markdown text for high-density corpus creation.
Every token is load-bearing at 100K scale — cleaning IS architecture.
"""

import hashlib
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent
EXTRACTED_DIR = DATA_DIR / "extracted"
CLEANED_DIR = DATA_DIR / "cleaned"


def clean_text(text: str) -> str:
    """Clean extracted text for training corpus.

    Pipeline:
    1. Unicode normalization (NFKC)
    2. Strip boilerplate (page numbers, headers, footers)
    3. Normalize whitespace
    4. Remove low-signal content
    5. Preserve domain-critical content (math, clinical terms)
    """
    # 1. Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # 2. Smart quotes → ASCII
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "--").replace("\u2014", "---")
    text = text.replace("\u2026", "...")
    text = text.replace("\xa0", " ")  # non-breaking space

    # 3. Strip page numbers (standalone numbers on their own line)
    text = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", text)

    # 4. Strip common headers/footers patterns
    # "Chapter N" style headers (but keep section titles)
    text = re.sub(r"(?m)^#{0,3}\s*(?:CHAPTER|Chapter)\s+\d+\s*$", "", text)
    # Page references like "see page 42" or "(p. 123)"
    text = re.sub(r"\(p\.\s*\d+\)", "", text)
    text = re.sub(r"\(pp\.\s*\d+[-–]\d+\)", "", text)

    # 5. Strip markdown image references (we don't have the images)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)

    # 6. Strip horizontal rules and decorative elements
    text = re.sub(r"(?m)^[-_*]{3,}\s*$", "", text)
    text = re.sub(r"(?m)^={3,}\s*$", "", text)

    # 7. Collapse multiple blank lines → double newline
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 8. Strip leading/trailing whitespace per line
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    # 9. Strip leading/trailing whitespace overall
    text = text.strip()

    return text


def compute_density_score(text: str, n: int = 3) -> float:
    """Information density heuristic: unique n-grams / total n-grams."""
    words = text.lower().split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def remove_duplicate_paragraphs(text: str, min_length: int = 50) -> str:
    """Remove exact duplicate paragraphs (keeps first occurrence)."""
    paragraphs = text.split("\n\n")
    seen = set()
    unique = []
    for para in paragraphs:
        stripped = para.strip()
        if len(stripped) < min_length:
            unique.append(para)  # keep short ones (headers, etc.)
            continue
        key = re.sub(r"\s+", " ", stripped.lower())
        if key not in seen:
            seen.add(key)
            unique.append(para)
    return "\n\n".join(unique)


def clean_file(input_path: Path, output_dir: Path) -> dict:
    """Clean a single extracted file. Returns metadata dict."""
    text = input_path.read_text(encoding="utf-8")
    raw_chars = len(text)

    cleaned = clean_text(text)
    cleaned = remove_duplicate_paragraphs(cleaned)
    clean_chars = len(cleaned)

    density = compute_density_score(cleaned)

    out_path = output_dir / input_path.name
    out_path.write_text(cleaned, encoding="utf-8")

    return {
        "source": input_path.name,
        "raw_chars": raw_chars,
        "clean_chars": clean_chars,
        "reduction_pct": round(100 * (1 - clean_chars / max(1, raw_chars)), 1),
        "density_score": round(density, 4),
        "word_count": len(cleaned.split()),
    }


def clean_all(
    input_dir: Path = EXTRACTED_DIR,
    output_dir: Path = CLEANED_DIR,
    min_chars: int = 500,
) -> list[dict]:
    """Clean all extracted files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.md"))
    logger.info("Cleaning %d files from %s", len(files), input_dir)

    results = []
    for i, f in enumerate(files):
        meta = clean_file(f, output_dir)
        if meta["clean_chars"] < min_chars:
            logger.info(
                "  [skip] %s — only %d chars after cleaning",
                f.name[:60], meta["clean_chars"],
            )
            # Remove the too-small output
            (output_dir / f.name).unlink(missing_ok=True)
            meta["status"] = "too_small"
        else:
            logger.info(
                "  [ok]   %s — %d→%d chars (-%s%%), density=%.3f",
                f.name[:60], meta["raw_chars"], meta["clean_chars"],
                meta["reduction_pct"], meta["density_score"],
            )
            meta["status"] = "ok"
        results.append(meta)

    kept = [r for r in results if r.get("status") == "ok"]
    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "total_files": len(files),
        "kept": len(kept),
        "dropped": len(files) - len(kept),
        "total_words": sum(r["word_count"] for r in kept),
        "mean_density": round(
            sum(r["density_score"] for r in kept) / max(1, len(kept)), 4
        ),
        "files": sorted(kept, key=lambda r: -r["density_score"]),
    }
    manifest_path = output_dir / "cleaning_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    logger.info(
        "Cleaning complete: %d kept, %d dropped, %d total words, mean density=%.3f",
        len(kept), len(files) - len(kept),
        manifest["total_words"], manifest["mean_density"],
    )
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    inp = Path(sys.argv[1]) if len(sys.argv) > 1 else EXTRACTED_DIR
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else CLEANED_DIR
    clean_all(inp, out)
