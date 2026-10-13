"""Genome representation for the openweight track.

See ``docs/architecture-openweight.md`` §"Genome representation" for the
design rationale. In short: a member is a tagged union over three kinds
(``sft``, ``merge``, ``lora_merge``). Keeping the kinds explicit lets
consumers pattern-match on ``kind`` instead of guessing which fields are
populated.

All dataclasses are plain ``@dataclass`` (no pydantic) so the lineage DB
can serialize them as JSON without a framework dependency, matching the
small-corpus track's style.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

Kind = Literal["sft", "merge", "lora_merge"]
TokenizerFamily = Literal["llama3", "qwen2", "gemma2", "mistral"]


# ---------------------------------------------------------------------------
# Kind-specific configs
# ---------------------------------------------------------------------------


@dataclass
class SftConfig:
    """Hyperparameters for a LoRA/DoRA fine-tune over a data shard."""

    base_model_id: str
    adapter_kind: Literal["lora", "dora", "qlora"]
    rank: int
    alpha: int
    dropout: float
    target_modules: list[str]
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    epochs: int
    batch_size: int
    max_seq_len: int
    corpus_shard_id: str
    seed: int
    optimizer: Literal["adamw_torch", "adamw_8bit", "paged_adamw_32bit"] = "adamw_torch"
    precision: Literal["bf16", "fp16"] = "bf16"


@dataclass
class MergeRecipe:
    """A mergekit recipe for combining 2+ parents into one merged model.

    ``yaml_dict`` is the raw mergekit recipe (parsed from YAML into a
    dict). ``coefficients`` is the flat vector that CMA-ES optimizes
    against (the merge coefficients extracted from the recipe for
    search-space purposes).
    """

    method: Literal["ties", "dare_ties", "slerp", "task_arithmetic", "linear"]
    parents: list[str]  # member_ids (OR HF ids for "root" parents)
    yaml_dict: dict[str, Any]
    coefficients: list[float]
    cma_es_step: Optional[int] = None  # which CMA-ES iteration produced this


@dataclass
class LoraMergeRecipe:
    """Adapter-level merge (LoraHub style) over a shared base.

    Cheaper than full-weight merge because only the rank-k deltas are
    combined. Base model must be identical across parents.
    """

    base_model_id: str
    parent_adapter_paths: list[str]
    weights: list[float]
    normalize: bool = True


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------


@dataclass
class FitnessVector:
    """Multi-objective fitness for NSGA-II.

    Sign convention: higher is better for ``bench_agg`` and
    ``diversity``; lower is better for ``calibration_ece``,
    ``latency_ms``, ``memory_gb``. The NSGA-II dominator normalizes
    these before comparison.

    Per-benchmark breakdown lives in the ``fitness_evals`` table so
    aggregate weights can be changed post-hoc without re-evaluating.
    """

    bench_agg: float
    diversity: float
    calibration_ece: float
    latency_ms: float
    memory_gb: float


# ---------------------------------------------------------------------------
# Member
# ---------------------------------------------------------------------------


@dataclass
class OpenweightMember:
    id: str
    generation: int
    kind: Kind
    parent_ids: list[str]
    lineage: list[int]

    tokenizer_family: TokenizerFamily
    base_model_id: str

    sft_config: Optional[SftConfig] = None
    merge_recipe: Optional[MergeRecipe] = None
    lora_merge_recipe: Optional[LoraMergeRecipe] = None

    adapter_path: Optional[str] = None
    merged_checkpoint_path: Optional[str] = None
    fitness: Optional[FitnessVector] = None
    # Worker that trained this adapter. Used by run_eval as a pull-fallback
    # when the adapter isn't yet present on the eval worker (P2P replication
    # lag or partial failure). Populated from SftResultMsg.worker_id.
    sft_worker_id: Optional[str] = None

    def __post_init__(self) -> None:
        configs = [
            (self.kind == "sft", self.sft_config),
            (self.kind == "merge", self.merge_recipe),
            (self.kind == "lora_merge", self.lora_merge_recipe),
        ]
        active = [(expected, cfg) for expected, cfg in configs if expected]
        if len(active) != 1 or active[0][1] is None:
            raise ValueError(
                f"member {self.id} kind={self.kind} has inconsistent config fields"
            )

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, payload: str) -> "OpenweightMember":
        data = json.loads(payload)
        data["sft_config"] = (
            SftConfig(**data["sft_config"]) if data.get("sft_config") else None
        )
        data["merge_recipe"] = (
            MergeRecipe(**data["merge_recipe"]) if data.get("merge_recipe") else None
        )
        data["lora_merge_recipe"] = (
            LoraMergeRecipe(**data["lora_merge_recipe"])
            if data.get("lora_merge_recipe")
            else None
        )
        data["fitness"] = (
            FitnessVector(**data["fitness"]) if data.get("fitness") else None
        )
        return cls(**data)


def new_member_id() -> str:
    return uuid.uuid4().hex[:12]
