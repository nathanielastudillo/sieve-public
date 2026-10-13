"""DataLoader with epoch-aware shuffling for SIEVE.

Implements the Slowrun finding that per-epoch shuffle order has outsized
impact on multi-epoch training in the data-constrained regime.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


class SieveDataset(Dataset):
    """Memory-mapped token dataset.

    Loads a ``.bin`` file produced by ``prepare.py`` as a numpy memmap and
    returns ``(x, y)`` chunks of ``block_size`` tokens, where ``y`` is ``x``
    shifted right by one position (standard causal LM target).
    """

    def __init__(self, data_path: str | Path, block_size: int, dtype: str = "uint16"):
        self.data_path = Path(data_path)
        self.block_size = block_size
        self.data = np.memmap(str(self.data_path), dtype=dtype, mode="r")
        # Number of complete (x, y) chunks we can form.
        # Each chunk needs block_size + 1 contiguous tokens.
        self.num_chunks = max(0, len(self.data) - block_size)

    def __len__(self) -> int:
        return self.num_chunks

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.block_size + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


class EpochShuffler(Sampler):
    """Custom sampler that re-seeds its shuffle each epoch.

    Addresses the Slowrun finding that data ordering across epochs has
    outsized impact.  Each epoch uses ``base_seed + epoch`` as its RNG
    seed so that the order is reproducible yet distinct per epoch.

    Call :meth:`set_epoch` at the start of each training epoch.
    """

    def __init__(self, data_source: Dataset, seed: int = 42, shuffle: bool = True):
        self.data_source = data_source
        self.base_seed = seed
        self.shuffle = shuffle
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch (changes the shuffle seed)."""
        self._epoch = epoch

    @property
    def epoch(self) -> int:
        return self._epoch

    def __len__(self) -> int:
        return len(self.data_source)

    def __iter__(self):
        n = len(self.data_source)
        if self.shuffle:
            gen = torch.Generator()
            gen.manual_seed(self.base_seed + self._epoch)
            indices = torch.randperm(n, generator=gen).tolist()
        else:
            indices = list(range(n))
        return iter(indices)


def _worker_init_fn(worker_id: int) -> None:
    """Seed numpy and stdlib RNG per worker for reproducibility.

    Each DataLoader worker gets a different but deterministic seed derived
    from the base worker seed that PyTorch already sets, combined with
    the worker id.
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)


def create_dataloader(
    data_path: str | Path,
    block_size: int,
    batch_size: int,
    seed: int = 42,
    shuffle: bool = True,
    dtype: str = "uint16",
    num_workers: int = 2,
    pin_memory: bool = True,
) -> tuple[DataLoader, EpochShuffler]:
    """Create a DataLoader with epoch-aware shuffling.

    Parameters
    ----------
    data_path : path to a ``.bin`` memmap file.
    block_size : context window length (number of input tokens per sample).
    batch_size : mini-batch size.
    seed : base random seed.  The effective seed for epoch *e* is ``seed + e``.
    shuffle : whether to shuffle (should be True for training, False for val).
    dtype : numpy dtype string for the memmap (must match what ``prepare.py`` wrote).
    num_workers : DataLoader worker processes.
    pin_memory : pin tensors into CUDA pinned memory.

    Returns
    -------
    (dataloader, sampler) — the sampler is returned so the training loop can
    call ``sampler.set_epoch(e)`` each epoch.
    """
    dataset = SieveDataset(data_path, block_size, dtype=dtype)
    sampler = EpochShuffler(dataset, seed=seed, shuffle=shuffle)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        worker_init_fn=_worker_init_fn,
    )

    return loader, sampler
