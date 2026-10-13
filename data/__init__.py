"""SIEVE data module — corpus preparation, loading, and curation.

Imports are lazy so worker nodes without pymupdf/other extraction deps can
import `data.loader` at training time without pulling in PDF tooling.
"""

__all__ = [
    "prepare_corpus",
    "train_custom_tokenizer",
    "SieveDataset",
    "create_dataloader",
    "EpochShuffler",
    "compute_density_score",
    "deduplicate",
    "score_passages",
    "balance_domains",
]


def __getattr__(name):
    if name in ("prepare_corpus", "train_custom_tokenizer"):
        from data import prepare
        return getattr(prepare, name)
    if name in ("SieveDataset", "create_dataloader", "EpochShuffler"):
        from data import loader
        return getattr(loader, name)
    if name in ("compute_density_score", "deduplicate", "score_passages", "balance_domains"):
        from data import curate
        return getattr(curate, name)
    raise AttributeError(f"module 'data' has no attribute {name!r}")
