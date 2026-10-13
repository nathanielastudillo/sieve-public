"""Worker-side handler for ``sieve:jobs:sft`` jobs.

Pulls an SFT job off the queue, runs LoRA/DoRA training via
``openweight.finetune``, and writes the adapter path back to the
result key. Mirrors ``worker/trainer.py`` for the small-corpus track.

Invoked from ``worker/daemon.py`` when a job's kind is ``"sft"``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from openweight.finetune import fine_tune
from openweight.genome import SftConfig

logger = logging.getLogger(__name__)


def handle_sft_job(job: dict[str, Any], sieve_root: Path) -> dict[str, Any]:
    """Run one SFT job end-to-end; return a result dict for the dispatcher."""
    cfg = SftConfig(**job["sft_config"])
    corpus_path = sieve_root / job["corpus_shard_path"]
    out_dir = sieve_root / job["adapter_out_dir"]

    t0 = time.time()
    adapter_path = fine_tune(cfg, corpus_path, out_dir)
    elapsed = time.time() - t0

    return {
        "member_id": job["member_id"],
        "generation": job["generation"],
        "adapter_path": str(adapter_path),
        "wallclock_s": elapsed,
        "status": "ok",
    }
