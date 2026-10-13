"""Pipeline runner: extract → clean → split → dedup → subsample → tokenize.

Every stage's output is captured to disk so any source can be traced end-to-end:

    data/pipeline/runs/<run_id>/
      config.json                    overall config
      pipeline.json                  ordered stage list + params
      index.jsonl                    one line per document (full stage trace)
      stages/<stage>/<doc_id>.txt    text snapshot after each stage (optional)
      paragraphs.jsonl               every paragraph + kept/dropped decision
      summary.txt                    human-readable run summary

    data/corpus_<version>/
      train.bin val.bin test.bin
      sources.txt                    flat training corpus (same as before)
      sources.annotated.md           each paragraph prefixed with its provenance
      sources.jsonl                  machine-readable: which para came from where
      metadata.json                  per-source contribution + tokenizer info

Usage:
    python -m data.pipeline.run --version v010 --limit 20
    python -m data.pipeline.run --version v010 --source-dir data/extracted
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import paragraphs as P
from . import stages as S
from .core import Document, Paragraph, run_with_history
from .extract import load_extracted_document

logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).parent.parent


class PipelineRun:
    """Captures a pipeline run's artifacts to disk."""

    def __init__(self, run_dir: Path, keep_intermediates: bool = False):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "stages").mkdir(exist_ok=True)
        self.keep_intermediates = keep_intermediates
        self._idx = open(self.run_dir / "index.jsonl", "w")
        self._par = open(self.run_dir / "paragraphs.jsonl", "w")

    def record_document(self, doc: Document, history: list[Document], status: str = "ok") -> None:
        row = {
            "doc_id": doc.prov.doc_id,
            "title": doc.prov.title,
            "source_path": doc.prov.source_path,
            "sha256": doc.prov.sha256,
            "status": status,
            "final_chars": len(doc.text),
            "stages": [s.to_dict() for s in doc.prov.stages],
        }
        self._idx.write(json.dumps(row) + "\n")
        self._idx.flush()
        if self.keep_intermediates:
            # history[0] is the doc after source/extract stage; history[i] is after stage i of the clean pipeline.
            for d in history:
                if not d.prov.stages:
                    continue
                name = d.prov.stages[-1].stage
                out_dir = self.run_dir / "stages" / name
                out_dir.mkdir(exist_ok=True)
                (out_dir / f"{doc.prov.doc_id}.txt").write_text(d.text, encoding="utf-8")

    def record_paragraph(self, p: Paragraph, decision: str, note: Optional[str] = None) -> None:
        row = p.to_dict()
        row["decision"] = decision
        if note:
            row["note"] = note
        self._par.write(json.dumps(row) + "\n")

    def write_pipeline_spec(self, stages: list) -> None:
        spec = [
            {
                "stage": getattr(s, "__stage_name__", "?"),
                "params": getattr(s, "__stage_params__", {}),
            }
            for s in stages
        ]
        (self.run_dir / "pipeline.json").write_text(json.dumps(spec, indent=2))

    def write_config(self, config: dict) -> None:
        (self.run_dir / "config.json").write_text(json.dumps(config, indent=2))

    def close(self):
        self._idx.close()
        self._par.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _make_tokenizer(tokenizer_type: str, tokenizer_path: Optional[str], encoding: str):
    if tokenizer_type == "custom" and tokenizer_path:
        from tokenizers import Tokenizer as HFTokenizer

        tok = HFTokenizer.from_file(tokenizer_path)

        def encode(s: str) -> list[int]:
            return tok.encode(s).ids

        return encode, tok.get_vocab_size(), "custom"

    import tiktoken

    enc = tiktoken.get_encoding(encoding)

    def encode(s: str) -> list[int]:
        return enc.encode(s)

    return encode, enc.n_vocab, f"tiktoken:{encoding}"


def build(
    *,
    version: str,
    source_dir: Path,
    raw_dir: Optional[Path] = None,
    clean_stages: Optional[list] = None,
    run_id: Optional[str] = None,
    keep_intermediates: bool = False,
    target_tokens: int = 5_000_000,
    tokenizer_type: str = "custom",
    tokenizer_path: Optional[str] = "data/tokenizers/v005/tokenizer.json",
    encoding: str = "gpt2",
    val_fraction: float = 0.10,
    test_fraction: float = 0.05,
    dedup_threshold: float = 0.7,
    min_para_words: int = 20,
    min_doc_chars: int = 500,
    stratification: str = "sqrt",
    seed: int = 42,
    limit: Optional[int] = None,
) -> dict:
    if clean_stages is None:
        clean_stages = list(S.STANDARD_CLEAN_STAGES)

    if run_id is None:
        run_id = f"{version}-{int(time.time())}"
    run_dir = DATA_DIR / "pipeline" / "runs" / run_id
    corpus_dir = DATA_DIR / f"corpus_{version}"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    encode, vocab_size, tok_label = _make_tokenizer(tokenizer_type, tokenizer_path, encoding)

    with PipelineRun(run_dir, keep_intermediates=keep_intermediates) as pr:
        pr.write_pipeline_spec(clean_stages)
        pr.write_config(
            {
                "version": version,
                "run_id": run_id,
                "source_dir": str(source_dir),
                "raw_dir": str(raw_dir) if raw_dir else None,
                "tokenizer_label": tok_label,
                "tokenizer_type": tokenizer_type,
                "tokenizer_path": tokenizer_path,
                "target_tokens": target_tokens,
                "val_fraction": val_fraction,
                "test_fraction": test_fraction,
                "dedup_threshold": dedup_threshold,
                "min_para_words": min_para_words,
                "min_doc_chars": min_doc_chars,
                "stratification": stratification,
                "seed": seed,
                "limit": limit,
                "keep_intermediates": keep_intermediates,
            }
        )

        files = sorted(Path(source_dir).glob("*.md"))
        if limit:
            files = files[:limit]
        logger.info("pipeline[%s]: %d source files in %s", run_id, len(files), source_dir)

        all_paragraphs: list[Paragraph] = []
        skipped = 0
        seen_sha: dict[str, str] = {}  # sha256 -> doc_id of first occurrence
        for i, f in enumerate(files):
            src = f
            if raw_dir:
                candidate = Path(raw_dir) / f"{f.stem}.pdf"
                if candidate.exists():
                    src = candidate
            try:
                doc = load_extracted_document(src, f)
            except Exception as e:
                logger.warning("  [skip-load] %s: %s", f.name, e)
                skipped += 1
                continue

            # Byte-identical duplicate files (same sha256) would produce the
            # same doc_id. Record and skip them to keep doc_id unique.
            if doc.prov.sha256 in seen_sha:
                orig = seen_sha[doc.prov.sha256]
                row = {
                    "doc_id": doc.prov.doc_id,
                    "title": doc.prov.title,
                    "source_path": doc.prov.source_path,
                    "sha256": doc.prov.sha256,
                    "status": "duplicate_sha256",
                    "duplicate_of": orig,
                    "final_chars": 0,
                    "stages": [],
                }
                pr._idx.write(json.dumps(row) + "\n")
                pr._idx.flush()
                skipped += 1
                continue
            seen_sha[doc.prov.sha256] = doc.prov.doc_id

            if len(doc.text) < min_doc_chars:
                # record a minimal trace so the inspect CLI can still see why
                pr.record_document(doc, [doc], status="skipped_too_small")
                skipped += 1
                continue

            final, history = run_with_history(doc, clean_stages)
            pr.record_document(final, history, status="ok")

            paras = P.split_paragraphs(final, min_words=min_para_words)
            all_paragraphs.extend(paras)

            logger.info(
                "  [%d/%d] %-60s  %6d→%-6d chars  %4d paragraphs",
                i + 1,
                len(files),
                doc.prov.title[:60],
                len(doc.text),
                len(final.text),
                len(paras),
            )

        # Per-doc counts BEFORE minhash — needed so metadata.per_source can
        # show the full flow (split → minhash → subsample → picked).
        split_by_doc: dict[str, int] = {}
        for p in all_paragraphs:
            split_by_doc[p.prov.doc_id] = split_by_doc.get(p.prov.doc_id, 0) + 1

        # Cross-document near-duplicate removal
        kept, deduped = P.minhash_dedup_paragraphs(
            all_paragraphs, threshold=dedup_threshold
        )
        minhash_dropped_by_doc: dict[str, int] = {}
        for p in deduped:
            minhash_dropped_by_doc[p.prov.doc_id] = (
                minhash_dropped_by_doc.get(p.prov.doc_id, 0) + 1
            )
        logger.info(
            "minhash_dedup: %d → %d (dropped %d)",
            len(all_paragraphs),
            len(kept),
            len(deduped),
        )
        for p in deduped:
            pr.record_paragraph(
                p, decision="dropped_minhash", note=f"dup_of={p.attrs.get('dup_of')}"
            )

        # Stratified density-weighted subsample to hit target_tokens
        selected, overflow, per_doc = P.stratified_budget_subsample(
            kept,
            target_tokens=target_tokens,
            token_count_fn=lambda s: len(encode(s)),
            stratification=stratification,
        )
        for p in overflow:
            pr.record_paragraph(p, decision="dropped_subsample")
        for p in selected:
            pr.record_paragraph(p, decision="kept")
        logger.info(
            "stratified_subsample: kept %d, overflow %d, across %d docs",
            len(selected),
            len(overflow),
            len(per_doc),
        )

        # Paragraph-level shuffle (preserves intra-paragraph coherence)
        rng = np.random.default_rng(seed)
        idx = np.arange(len(selected))
        rng.shuffle(idx)
        selected = [selected[i] for i in idx]

        full_text = "\n\n".join(p.text for p in selected)
        (corpus_dir / "sources.txt").write_text(full_text, encoding="utf-8")

        # Annotated sources.md: HTML-comment marker per paragraph so the human
        # can see exactly which book + which para each chunk came from.
        annotated_parts = [
            (
                f"<!-- para {p.para_id} | from: {p.prov.title} | "
                f"density: {p.attrs.get('density', 0):.3f} | "
                f"words: {p.attrs.get('words', 0)} -->\n{p.text}"
            )
            for p in selected
        ]
        (corpus_dir / "sources.annotated.md").write_text(
            "\n\n".join(annotated_parts), encoding="utf-8"
        )

        with open(corpus_dir / "sources.jsonl", "w") as fp:
            for p in selected:
                fp.write(json.dumps(p.to_dict()) + "\n")

        # Tokenize + split into train/val/test (chunk-level to preserve coherence)
        token_ids = encode(full_text)
        total_tokens = len(token_ids)
        dtype = np.uint16 if vocab_size < 65536 else np.uint32
        arr = np.array(token_ids, dtype=dtype)
        logger.info("tokenized: %d tokens (vocab=%d)", total_tokens, vocab_size)

        chunk_size = 512
        n_chunks = total_tokens // chunk_size
        chunk_idx = np.arange(n_chunks)
        rng2 = np.random.default_rng(seed + 1)
        rng2.shuffle(chunk_idx)
        n_test = max(1, int(n_chunks * test_fraction))
        n_val = max(1, int(n_chunks * val_fraction))
        test_c = sorted(chunk_idx[:n_test])
        val_c = sorted(chunk_idx[n_test : n_test + n_val])
        train_c = sorted(chunk_idx[n_test + n_val :])

        def gather(ix):
            parts = [arr[i * chunk_size : (i + 1) * chunk_size] for i in ix]
            return np.concatenate(parts) if parts else np.array([], dtype=dtype)

        train_arr = gather(train_c)
        val_arr = gather(val_c)
        test_arr = gather(test_c)
        train_arr.tofile(corpus_dir / "train.bin")
        val_arr.tofile(corpus_dir / "val.bin")
        test_arr.tofile(corpus_dir / "test.bin")

        # Merge pre-minhash / minhash-drop counts into per_doc so metadata can
        # show the full per-source flow.
        for d, row in per_doc.items():
            row["paragraphs_split"] = split_by_doc.get(d, row["paragraphs_after_minhash"])
            row["paragraphs_minhash_dropped"] = minhash_dropped_by_doc.get(d, 0)
        # Also include docs whose paragraphs were 100% minhash-eliminated
        # (they wouldn't appear in per_doc otherwise).
        for d, n_split in split_by_doc.items():
            if d in per_doc:
                continue
            n_drop = minhash_dropped_by_doc.get(d, 0)
            sample = next(p for p in all_paragraphs if p.prov.doc_id == d)
            per_doc[d] = {
                "doc_id": d,
                "title": sample.prov.title,
                "source_path": sample.prov.source_path,
                "paragraphs_split": n_split,
                "paragraphs_minhash_dropped": n_drop,
                "paragraphs_after_minhash": 0,
                "paragraphs_picked": 0,
                "paragraphs_overflow": 0,
                "source_words": int(sum(p.attrs.get("words", 0) for p in all_paragraphs if p.prov.doc_id == d)),
                "budget_tokens": 0,
                "used_tokens": 0,
                "min_density_kept": None,
                "mean_density_kept": 0.0,
            }

        per_source_list = sorted(per_doc.values(), key=lambda d: -d["used_tokens"])
        metadata = {
            "version": version,
            "run_id": run_id,
            "pipeline_run_dir": str(run_dir),
            "source_dir": str(source_dir),
            "target_tokens": target_tokens,
            "stratification": stratification,
            "tokenizer_type": tokenizer_type,
            "tokenizer_path": tokenizer_path,
            "tokenizer_label": tok_label,
            "vocab_size": vocab_size,
            "dtype": str(dtype),
            "total_tokens": total_tokens,
            "train_tokens": int(len(train_arr)),
            "val_tokens": int(len(val_arr)),
            "test_tokens": int(len(test_arr)),
            "val_fraction": val_fraction,
            "test_fraction": test_fraction,
            "dedup_threshold": dedup_threshold,
            "min_para_words": min_para_words,
            "min_doc_chars": min_doc_chars,
            "seed": seed,
            "num_sources": len(per_doc),
            "num_skipped_docs": skipped,
            "num_paragraphs_split": len(all_paragraphs),
            "num_paragraphs_kept": len(selected),
            "num_paragraphs_dropped_minhash": len(deduped),
            "num_paragraphs_dropped_subsample": len(overflow),
            "per_source": per_source_list,
            "corpus_sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
        }
        (corpus_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        # Also write a brief human summary into the run dir
        summary_lines = [
            f"pipeline run {run_id}",
            f"  source_dir: {source_dir}",
            f"  corpus:     {corpus_dir}",
            f"  sources:    {len(per_doc)} docs ({skipped} skipped)",
            f"  paragraphs: split={len(all_paragraphs)}  kept={len(selected)}  "
            f"minhash_drop={len(deduped)}  overflow_drop={len(overflow)}",
            f"  tokens:     total={total_tokens}  train={len(train_arr)}  "
            f"val={len(val_arr)}  test={len(test_arr)}",
            f"  tokenizer:  {tok_label}  (vocab={vocab_size})",
            "",
            "top-10 sources by contribution:",
        ]
        for r in per_source_list[:10]:
            summary_lines.append(
                f"  {r['used_tokens']:>7d} tok  "
                f"{r['paragraphs_picked']:>4d}/{r['paragraphs_split']:<4d} paras  "
                f"(−{r['paragraphs_minhash_dropped']} mh −{r['paragraphs_overflow']} of)  "
                f"density≥{r['min_density_kept']}  {r['title'][:60]}"
            )
        (run_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n")

        logger.info(
            "corpus %s: train=%d val=%d test=%d (vocab=%d)",
            version,
            len(train_arr),
            len(val_arr),
            len(test_arr),
            vocab_size,
        )
        logger.info("run artifacts: %s", run_dir)
        return metadata


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--version", required=True)
    p.add_argument("--source-dir", default=str(DATA_DIR / "extracted"))
    p.add_argument("--raw-dir", default=str(DATA_DIR / "raw"))
    p.add_argument("--target-tokens", type=int, default=5_000_000)
    p.add_argument("--tokenizer", default="custom", choices=["tiktoken", "custom"])
    p.add_argument("--tokenizer-path", default="data/tokenizers/v005/tokenizer.json")
    p.add_argument("--encoding", default="gpt2")
    p.add_argument("--val-fraction", type=float, default=0.10)
    p.add_argument("--test-fraction", type=float, default=0.05)
    p.add_argument("--dedup-threshold", type=float, default=0.7)
    p.add_argument("--min-para-words", type=int, default=20)
    p.add_argument("--min-doc-chars", type=int, default=500)
    p.add_argument("--stratification", default="sqrt", choices=["uniform", "sqrt", "proportional"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep-intermediates", action="store_true", help="Save text after each stage")
    p.add_argument("--run-id", default=None)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    build(
        version=args.version,
        source_dir=Path(args.source_dir),
        raw_dir=Path(args.raw_dir) if args.raw_dir else None,
        run_id=args.run_id,
        keep_intermediates=args.keep_intermediates,
        target_tokens=args.target_tokens,
        tokenizer_type=args.tokenizer,
        tokenizer_path=args.tokenizer_path,
        encoding=args.encoding,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        dedup_threshold=args.dedup_threshold,
        min_para_words=args.min_para_words,
        min_doc_chars=args.min_doc_chars,
        stratification=args.stratification,
        seed=args.seed,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
