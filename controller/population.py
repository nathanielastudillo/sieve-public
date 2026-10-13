"""Population management, selection, and reproduction for SIEVE.

Implements PBT-style evolutionary operators where the unit of selection is
ensemble contribution, not individual model fitness.  Members are hyperparameter
configurations that get trained into checkpoints; selection pressure comes from
how often a member appears in top-performing ensembles.
"""

from __future__ import annotations

import math
import random
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Member:
    """A single population member (one training run / checkpoint).

    Attributes:
        id: Short unique identifier (uuid4 hex[:8]).
        generation: Which evolutionary generation this member was created in.
        config: Full hyperparameter configuration dict (model + training params).
        checkpoint_path: Path to saved checkpoint after training, or None if untrained.
        val_loss: Individual validation loss, or None if not yet evaluated.
        parent_ids: IDs of parent member(s) this was derived from (empty for random init).
        lineage: List of generation numbers tracing back through ancestors.
    """

    id: str
    generation: int
    config: dict[str, Any]
    checkpoint_path: str | None = None
    val_loss: float | None = None
    parent_ids: list[str] = field(default_factory=list)
    lineage: list[int] = field(default_factory=list)


class Population:
    """Manages a population of Members through evolutionary cycles.

    The search space dict maps parameter names to either:
      - A list of discrete values (e.g. [4, 6, 8]) -> random choice / step mutation
      - A two-element [min, max] list of floats -> log-uniform sampling / multiplicative mutation

    Continuous vs discrete is determined by checking whether the list has exactly
    two numeric elements that look like a range (min < max, both float-like) vs
    a list of discrete options.
    """

    def __init__(self, search_space: dict[str, Any], size: int) -> None:
        self.search_space = search_space
        self.size = size
        self._generation = 0

    @staticmethod
    def _is_continuous(values: list) -> bool:
        """Check if a search space entry represents a continuous range [min, max]."""
        if not isinstance(values, list) or len(values) != 2:
            return False
        try:
            lo, hi = float(values[0]), float(values[1])
            return lo < hi
        except (TypeError, ValueError):
            return False

    def sample_random_config(self, search_space: dict[str, Any] | None = None) -> dict[str, Any]:
        """Sample a single random configuration from the search space.

        For discrete parameters: uniform random choice from the list.
        For continuous parameters [min, max]: log-uniform sample.
        """
        space = search_space or self.search_space
        config: dict[str, Any] = {}

        for param, values in space.items():
            if self._is_continuous(values):
                lo, hi = float(values[0]), float(values[1])
                # Log-uniform: sample uniformly in log-space then exponentiate.
                # Guard against zero/negative bounds by falling back to uniform.
                if lo > 0 and hi > 0:
                    log_lo, log_hi = math.log(lo), math.log(hi)
                    config[param] = math.exp(random.uniform(log_lo, log_hi))
                else:
                    config[param] = random.uniform(lo, hi)
            else:
                config[param] = random.choice(values)

        # Ensure n_embd % n_head == 0 (architectural constraint)
        if "n_embd" in config and "n_head" in config:
            n_embd = config["n_embd"]
            n_head = config["n_head"]
            if n_embd % n_head != 0:
                # Pick a valid n_head from the search space
                valid_heads = [h for h in space.get("n_head", [n_head])
                               if n_embd % h == 0]
                if valid_heads:
                    config["n_head"] = random.choice(valid_heads)
                else:
                    # Fall back to n_head=4 which divides all options
                    config["n_head"] = 4

        return config

    def initialize(self) -> list[Member]:
        """Create the initial population (generation 0).

        Samples ``self.size`` random configurations from the search space,
        each with a unique short ID.
        """
        members: list[Member] = []
        for _ in range(self.size):
            member = Member(
                id=uuid.uuid4().hex[:8],
                generation=0,
                config=self.sample_random_config(),
                lineage=[0],
            )
            members.append(member)

        self._generation = 0
        return members

    def mutate_config(
        self,
        config: dict[str, Any],
        search_space: dict[str, Any] | None = None,
        strength: float = 0.2,
    ) -> dict[str, Any]:
        """PBT-style mutation of a hyperparameter config.

        Continuous params: multiply by (1 + strength) or (1 - strength), i.e. x0.8 / x1.2
        at default strength=0.2.  Result is clamped to [min, max].

        Discrete params: step +/-1 in the allowed list (wrapping at boundaries).
        """
        space = search_space or self.search_space
        mutated: dict[str, Any] = {}

        for param, value in config.items():
            if param not in space:
                # Pass through params not in search space unchanged
                mutated[param] = value
                continue

            bounds = space[param]

            if self._is_continuous(bounds):
                lo, hi = float(bounds[0]), float(bounds[1])
                factor = random.choice([1.0 - strength, 1.0 + strength])
                new_val = float(value) * factor
                mutated[param] = max(lo, min(hi, new_val))
            else:
                # Discrete: step +/-1 in the ordered list
                options = list(bounds)
                if value in options:
                    idx = options.index(value)
                    step = random.choice([-1, 1])
                    new_idx = max(0, min(len(options) - 1, idx + step))
                    mutated[param] = options[new_idx]
                else:
                    # Value not in options (shouldn't happen), pick randomly
                    mutated[param] = random.choice(options)

        return mutated

    def crossover_configs(
        self,
        config_a: dict[str, Any],
        config_b: dict[str, Any],
        search_space: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a child config by crossing over two parent configs.

        Continuous params: interpolate (midpoint) between the two parents.
        Discrete params: randomly pick from either parent.
        """
        space = search_space or self.search_space
        child: dict[str, Any] = {}

        all_params = set(config_a.keys()) | set(config_b.keys())

        for param in all_params:
            val_a = config_a.get(param)
            val_b = config_b.get(param)

            # If only one parent has this param, take that value
            if val_a is None:
                child[param] = val_b
                continue
            if val_b is None:
                child[param] = val_a
                continue

            bounds = space.get(param)
            if bounds is not None and self._is_continuous(bounds):
                # Interpolate continuous params
                lo, hi = float(bounds[0]), float(bounds[1])
                mid = (float(val_a) + float(val_b)) / 2.0
                child[param] = max(lo, min(hi, mid))
            else:
                # Discrete: pick from either parent
                child[param] = random.choice([val_a, val_b])

        return child

    def select(
        self,
        members: list[Member],
        ensemble_rankings: list[dict],
        elite_fraction: float = 0.25,
        immigration_fraction: float = 0.1,
    ) -> list[Member]:
        """Evolve the population based on ensemble-level fitness.

        The core SIEVE insight: fitness is not individual val_loss but how much
        a member contributes to top-performing ensembles.

        Args:
            members: Current generation's trained members.
            ensemble_rankings: Sorted list of ensemble evaluation dicts, each with
                ``member_ids`` (list[str]) and ``ensemble_val_loss`` (float).
                Must be sorted ascending by ensemble_val_loss (best first).
            elite_fraction: Fraction of population to keep as elites.
            immigration_fraction: Fraction of population filled with fresh random configs.

        Returns:
            New generation of Members ready for training.
        """
        self._generation += 1
        gen = self._generation
        target_size = self.size

        # --- Score members by ensemble contribution ---
        top_k = max(1, len(ensemble_rankings) // 10)  # top 10% of ensembles
        top_ensembles = ensemble_rankings[:top_k]

        # Count appearances in top-K ensembles
        appearance_counts: Counter[str] = Counter()
        for ens in top_ensembles:
            for mid in ens["member_ids"]:
                appearance_counts[mid] += 1

        # Build lookup for current members
        member_map = {m.id: m for m in members}

        # --- Identify elites (most frequent in top ensembles) ---
        n_elite = max(1, int(target_size * elite_fraction))
        sorted_by_contribution = sorted(
            members,
            key=lambda m: appearance_counts.get(m.id, 0),
            reverse=True,
        )
        elites = sorted_by_contribution[:n_elite]

        # --- Identify dead weight (never in any top-K ensemble) ---
        all_top_members = set(appearance_counts.keys())
        dead_weight = [m for m in members if m.id not in all_top_members]

        # --- Identify complementary pairs ---
        # Pairs that appear together in top ensembles AND have maximally different configs
        complementary_pairs = self._find_complementary_pairs(
            elites, top_ensembles, max_pairs=n_elite // 2
        )

        # --- Build next generation ---
        new_members: list[Member] = []

        # 1. Reproduction: clone elites with perturbed hyperparams
        for elite in elites:
            child_config = self.mutate_config(elite.config)
            child = Member(
                id=uuid.uuid4().hex[:8],
                generation=gen,
                config=child_config,
                parent_ids=[elite.id],
                lineage=elite.lineage + [gen],
            )
            new_members.append(child)
            if len(new_members) >= target_size:
                break

        # 2. Crossover: children from complementary pairs
        for parent_a, parent_b in complementary_pairs:
            if len(new_members) >= target_size:
                break
            child_config = self.crossover_configs(parent_a.config, parent_b.config)
            child = Member(
                id=uuid.uuid4().hex[:8],
                generation=gen,
                config=child_config,
                parent_ids=[parent_a.id, parent_b.id],
                lineage=list(set(parent_a.lineage + parent_b.lineage)) + [gen],
            )
            new_members.append(child)

        # 3. Immigration: fill remaining slots with fresh random samples
        n_immigrants = max(1, int(target_size * immigration_fraction))
        remaining = target_size - len(new_members)
        n_immigrants = min(n_immigrants, remaining)

        for _ in range(n_immigrants):
            if len(new_members) >= target_size:
                break
            immigrant = Member(
                id=uuid.uuid4().hex[:8],
                generation=gen,
                config=self.sample_random_config(),
                lineage=[gen],
            )
            new_members.append(immigrant)

        # 4. If still under target, produce more elite mutations
        while len(new_members) < target_size:
            parent = random.choice(elites)
            child_config = self.mutate_config(parent.config)
            child = Member(
                id=uuid.uuid4().hex[:8],
                generation=gen,
                config=child_config,
                parent_ids=[parent.id],
                lineage=parent.lineage + [gen],
            )
            new_members.append(child)

        return new_members[:target_size]

    def _find_complementary_pairs(
        self,
        elites: list[Member],
        top_ensembles: list[dict],
        max_pairs: int = 5,
    ) -> list[tuple[Member, Member]]:
        """Find pairs of elites that co-occur in top ensembles and are maximally different.

        "Complementary" means they appear together in good ensembles (they're
        synergistic) AND their configs differ substantially (they cover different
        regions of solution space).
        """
        if len(elites) < 2:
            return []

        elite_ids = {m.id for m in elites}
        elite_map = {m.id: m for m in elites}

        # Count co-occurrences in top ensembles
        cooccur: Counter[tuple[str, str]] = Counter()
        for ens in top_ensembles:
            ens_elites = [mid for mid in ens["member_ids"] if mid in elite_ids]
            for i, a in enumerate(ens_elites):
                for b in ens_elites[i + 1 :]:
                    pair = tuple(sorted([a, b]))
                    cooccur[pair] += 1

        if not cooccur:
            return []

        # Score pairs: co-occurrence count * config distance
        scored_pairs: list[tuple[float, str, str]] = []
        for (a_id, b_id), count in cooccur.items():
            dist = self._config_distance(
                elite_map[a_id].config, elite_map[b_id].config
            )
            score = count * dist
            scored_pairs.append((score, a_id, b_id))

        # Take highest-scoring pairs
        scored_pairs.sort(reverse=True)
        pairs: list[tuple[Member, Member]] = []
        used: set[str] = set()
        for _score, a_id, b_id in scored_pairs:
            if a_id in used or b_id in used:
                continue
            pairs.append((elite_map[a_id], elite_map[b_id]))
            used.add(a_id)
            used.add(b_id)
            if len(pairs) >= max_pairs:
                break

        return pairs

    def _config_distance(self, config_a: dict, config_b: dict) -> float:
        """Compute a normalized distance between two configs.

        Continuous params: |log(a) - log(b)| / |log(hi) - log(lo)|
        Discrete params: 0 if same, 1 if different
        Average across all params.
        """
        distances: list[float] = []

        for param, bounds in self.search_space.items():
            val_a = config_a.get(param)
            val_b = config_b.get(param)
            if val_a is None or val_b is None:
                continue

            if self._is_continuous(bounds):
                lo, hi = float(bounds[0]), float(bounds[1])
                if lo > 0 and hi > 0:
                    log_range = math.log(hi) - math.log(lo)
                    if log_range > 0:
                        d = abs(math.log(float(val_a)) - math.log(float(val_b))) / log_range
                        distances.append(min(d, 1.0))
                    else:
                        distances.append(0.0)
                else:
                    rng = hi - lo
                    if rng > 0:
                        distances.append(abs(float(val_a) - float(val_b)) / rng)
                    else:
                        distances.append(0.0)
            else:
                distances.append(0.0 if val_a == val_b else 1.0)

        return sum(distances) / len(distances) if distances else 0.0
