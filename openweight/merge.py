"""Thin wrapper around mergekit for evolutionary merge jobs.

See ``docs/architecture-openweight.md`` §"Generation loop" step 3 for how
merge jobs flow through the dispatcher. This module produces mergekit
recipes from parent members and runs them. The CMA-ES search over merge
coefficients lives in ``openweight/search.py`` (not yet written); this
module just converts a ``MergeRecipe`` into a merged safetensors dir.

Expected runtime environment: mergekit >=0.0.5 installed; workers have
local HF cache at ``/data/hf_cache``.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

import yaml

from openweight.genome import MergeRecipe, OpenweightMember

logger = logging.getLogger(__name__)


def recipe_to_yaml(recipe: MergeRecipe, out_path: Path) -> None:
    """Serialize a ``MergeRecipe`` as a mergekit YAML file."""
    out_path.write_text(yaml.safe_dump(recipe.yaml_dict, sort_keys=False))


def run_mergekit(recipe_path: Path, out_dir: Path, allow_crimes: bool = False) -> Path:
    """Invoke ``mergekit-yaml`` and return the output directory."""
    cmd = ["mergekit-yaml", str(recipe_path), str(out_dir)]
    if allow_crimes:
        cmd.append("--allow-crimes")
    logger.info("mergekit: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_dir


def materialize(
    member: OpenweightMember,
    parents: list[OpenweightMember],
    work_dir: Path,
) -> Path:
    """Produce a merged checkpoint for ``member`` and return its path.

    Consumes ``member.merge_recipe``, writes the recipe YAML beside the
    output, runs mergekit, and returns the dir containing
    ``model.safetensors`` (or the shard index).
    """
    if member.kind != "merge" or member.merge_recipe is None:
        raise ValueError(f"member {member.id} is not a merge kind")
    out_dir = work_dir / f"merge-{member.id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    recipe_path = out_dir / "recipe.yaml"
    recipe_to_yaml(member.merge_recipe, recipe_path)
    return run_mergekit(recipe_path, out_dir)


def propose_ties_recipe(
    base: str, parent_adapters_or_models: list[str], weights: list[float]
) -> dict[str, Any]:
    """Build a minimal TIES recipe dict. Consumers can perturb before yaml-ing."""
    if len(parent_adapters_or_models) != len(weights):
        raise ValueError("parents and weights must align")
    models = [
        {"model": p, "parameters": {"density": 0.5, "weight": w}}
        for p, w in zip(parent_adapters_or_models, weights)
    ]
    return {
        "merge_method": "ties",
        "base_model": base,
        "models": models,
        "parameters": {"normalize": True, "int8_mask": True},
        "dtype": "bfloat16",
    }


def propose_dare_ties_recipe(
    base: str, parent_adapters_or_models: list[str], weights: list[float]
) -> dict[str, Any]:
    raise NotImplementedError


def propose_slerp_recipe(
    base: str, other: str, t: float
) -> dict[str, Any]:
    raise NotImplementedError
