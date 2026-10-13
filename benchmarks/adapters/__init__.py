"""Benchmark adapters — harness-agnostic ``(name, endpoint) → (score, raw)``.

Every benchmark is a callable ``run(base_url, model_name, proxy=False)
-> (score: float, raw: dict)``. Score is higher-is-better in ``[0, 1]``
(per ``benchmarks/README.md`` invariants); error-like metrics are flipped
inside the adapter.

Adapters go here rather than in ``openweight/`` because they are the
interface between *any* member serving surface (sglang, vLLM, hosted
API) and the fitness aggregator. They don't know about LoRA, merges, or
genomes.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


# Registry: name → dotted module path. Lazy-imported so a missing optional
# dep (e.g. lm-eval) doesn't break imports of unrelated adapters.
_REGISTRY: dict[str, str] = {
    "smoke_mcq":      "benchmarks.adapters.smoke_mcq",
    "ifeval":         "benchmarks.adapters.lm_eval_task",
    "hellaswag":      "benchmarks.adapters.lm_eval_task",
    "mmlu_pro":       "benchmarks.adapters.lm_eval_task",
    "gpqa_diamond":   "benchmarks.adapters.lm_eval_task",
}


AdapterFn = Callable[..., tuple[float, dict[str, Any]]]


def get_adapter(name: str) -> AdapterFn:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown benchmark {name!r}; known: {sorted(_REGISTRY)}"
        )
    mod = importlib.import_module(_REGISTRY[name])
    return getattr(mod, "run")


def run_benchmark(
    name: str,
    base_url: str,
    model_name: str,
    proxy: bool = False,
) -> tuple[float, dict[str, Any]]:
    """Entry point called by ``worker.openweight_runner.run_eval``.

    ``base_url`` is the OpenAI-compatible endpoint root (ends with ``/v1``).
    ``model_name`` is the ``served-model-name`` passed to sglang (the
    adapter id for LoRA, the model path for merged).
    """
    fn = get_adapter(name)
    return fn(base_url=base_url, model_name=model_name, proxy=proxy, task=name)
