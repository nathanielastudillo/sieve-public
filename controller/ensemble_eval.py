"""Combinatorial ensemble evaluation for SIEVE.

Evaluates subsets of trained model checkpoints as ensembles (logit averaging)
on a held-out validation set.  Supports random sampling of combinations and
greedy construction strategies to make the combinatorial search tractable.
"""

from __future__ import annotations

import itertools
import logging
import random
import uuid
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from controller.population import Member

logger = logging.getLogger(__name__)


class EnsembleEvaluator:
    """Evaluate model checkpoint combinations as logit-averaging ensembles.

    Loads checkpoints, runs forward passes on the validation set, and
    averages logits across ensemble members to compute ensemble val loss.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        data_dir: str,
        device: str = "cuda",
        batch_size: int = 128,
        block_size: int | None = None,
        aggregator: str = "logit_mean_smoothed",
        smoothing_eps: float = 0.05,
        meta_val_fraction: float = 0.1,
        meta_val_seed: int = 1729,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.data_dir = data_dir
        self.device = device
        self.batch_size = batch_size
        self._block_size_override = block_size
        # Sequestered meta-val slice: a deterministic fraction of val.bin
        # that is NEVER used for ensemble selection — only for end-of-gen
        # reporting. Purpose: detect val-set meta-overfitting at scale.
        # See lab-notes/research-questions/literature-small-corpus-ensembles.md
        # §8.2. At 30-50 generations × hundreds of ensembles per gen, the
        # winning ensemble's val loss is a biased estimator of its true
        # held-out loss; the meta-val slice gives us an unbiased canary.
        # The `meta_val_seed` is fixed so the split is identical across
        # runs, making per-generation meta-val trajectories comparable.
        self.meta_val_fraction = float(meta_val_fraction)
        self.meta_val_seed = int(meta_val_seed)
        # Aggregator choices:
        #   "logit_mean"          : mean logits + softmax. Equivalent to
        #                           geometric-mean-of-probabilities (Wood 2023).
        #                           Catastrophic when any member assigns ~0
        #                           probability to the truth.
        #   "logit_mean_smoothed" : ε-smooth each member's softmax with
        #                           uniform(1/V) at α = smoothing_eps, then
        #                           geometric-mean-of-smoothed-probs. Floors
        #                           probabilities, kills geometric-mean blowup.
        #                           Default.
        #   "prob_mean"           : arithmetic mean of softmax (BMA-style).
        #                           Bounded by log(V), robust to one zero-prob
        #                           member but overweights overconfident ones.
        if aggregator not in {"logit_mean", "logit_mean_smoothed", "prob_mean"}:
            raise ValueError(f"Unknown aggregator: {aggregator}")
        self.aggregator = aggregator
        self.smoothing_eps = float(smoothing_eps)

    def _load_val_data(
        self, split: str = "selection",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Load validation data, returning the selection-val OR meta-val slice.

        We split val.bin deterministically (seeded by ``self.meta_val_seed``)
        into two disjoint chunk-level slices:

        - ``selection`` — the slice the evolutionary loop optimizes
          against. Default caller; preserves the prior behavior if
          meta_val_fraction=0.
        - ``meta_val``   — a reserved slice that the selector never
          sees. Used by ``evaluate_meta_val`` for end-of-gen reporting
          to detect selection bias (see §8.2 of the ensemble lit
          review).

        The split is at the chunk level (after chunking) rather than the
        token level so that each slice is composed of complete
        block_size-sized windows — this keeps per-chunk CE comparable
        between the two slices.
        """
        from pathlib import Path

        val_path = Path(self.data_dir) / "val.bin"
        # Determine dtype and block_size from metadata if available
        meta_path = Path(self.data_dir) / "metadata.json"
        dtype = np.uint16
        block_size = self._block_size_override or 512
        if meta_path.exists():
            import json
            meta = json.loads(meta_path.read_text())
            dtype_str = meta.get("dtype", "uint16")
            if "32" in dtype_str:
                dtype = np.uint32

        val_data = np.memmap(str(val_path), dtype=dtype, mode="r")
        n_tokens = len(val_data)

        # Create chunks of block_size+1 (input + target offset by 1)
        n_chunks = max(1, (n_tokens - 1) // block_size)
        inputs_list: list[np.ndarray] = []
        targets_list: list[np.ndarray] = []

        for i in range(n_chunks):
            start = i * block_size
            end = start + block_size + 1
            if end > n_tokens:
                break
            chunk = val_data[start:end]
            inputs_list.append(chunk[:-1])
            targets_list.append(chunk[1:])

        inputs = torch.tensor(np.array(inputs_list), dtype=torch.long)
        targets = torch.tensor(np.array(targets_list), dtype=torch.long)

        # Split chunks into selection-val and meta-val via a fixed-seed
        # permutation. Using a *fixed* seed (not time/run-dependent)
        # means the same val chunks land in meta-val across every run
        # on this corpus, so meta-val trajectories are directly
        # comparable between experiments. Fall through if the fraction
        # is 0 — preserves the prior single-slice behavior.
        total = inputs.shape[0]
        if self.meta_val_fraction <= 0.0 or total < 2:
            if split == "meta_val":
                # Empty meta-val; caller should skip.
                return inputs[:0], targets[:0]
            return inputs, targets

        rng = np.random.default_rng(self.meta_val_seed)
        perm = rng.permutation(total)
        n_meta = max(1, int(round(self.meta_val_fraction * total)))
        meta_idx = np.sort(perm[:n_meta])
        sel_idx = np.sort(perm[n_meta:])

        if split == "meta_val":
            return inputs[meta_idx], targets[meta_idx]
        return inputs[sel_idx], targets[sel_idx]

    def evaluate_meta_val(
        self,
        member_ids: list[str],
        logits_cache: dict[str, torch.Tensor],
        targets_meta: torch.Tensor | None = None,
    ) -> float:
        """Compute ensemble loss on the sequestered meta-val slice.

        This is a read-only diagnostic. It must NEVER feed back into
        selection — that would defeat the purpose. See §8.2 of the
        ensemble lit review.

        Expects ``logits_cache`` to already contain meta-val logits for
        each member (caller must have called ``_get_logits_for_checkpoint``
        against meta-val inputs). If ``targets_meta`` is not supplied,
        it will be loaded fresh.
        """
        if not member_ids:
            return float("inf")
        if targets_meta is None:
            _, targets_meta = self._load_val_data(split="meta_val")
        if targets_meta.numel() == 0:
            return float("inf")
        combo_logits = [logits_cache[mid] for mid in member_ids]
        return self._compute_ensemble_loss(combo_logits, targets_meta)

    def compute_meta_val_loss(
        self,
        member_ids: list[str],
        members: list[Member],
    ) -> float:
        """End-to-end meta-val diagnostic for a picked ensemble.

        Loads the reserved meta-val slice, forward-passes each of the
        ``member_ids``' checkpoints against it, and returns the
        aggregated ensemble loss. A fresh, selection-never-touched
        canary — exactly what §8.2 of the lit review prescribes to
        detect selection bias at scale.

        Returns ``float('inf')`` if meta-val is empty (e.g.
        ``meta_val_fraction=0`` or val set too small to carve a slice)
        or no members are resolvable.
        """
        if not member_ids:
            return float("inf")

        meta_inputs, meta_targets = self._load_val_data(split="meta_val")
        if meta_targets.numel() == 0:
            # No meta-val slice configured; silently skip.
            return float("inf")

        member_map = {m.id: m for m in members if m.checkpoint_path is not None}
        meta_logits: list[torch.Tensor] = []
        for mid in member_ids:
            m = member_map.get(mid)
            if m is None or m.checkpoint_path is None:
                # Skip unresolvable members; ensemble may end up smaller,
                # but that is preferable to failing the whole diagnostic.
                continue
            meta_logits.append(
                self._get_logits_for_checkpoint(m.checkpoint_path, meta_inputs)
            )

        if not meta_logits:
            return float("inf")

        return self._compute_ensemble_loss(meta_logits, meta_targets)

    def _get_logits_for_checkpoint(
        self,
        checkpoint_path: str,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Load a checkpoint and compute logits on all input batches.

        Returns a tensor of shape (N, T, V) where N is number of chunks,
        T is block_size, V is vocab_size.
        """
        # Import here to avoid circular deps at module level
        try:
            from worker.checkpoint import load_model_from_checkpoint
        except ImportError:
            logger.warning(
                "worker.checkpoint not available; returning dummy logits "
                "for %s", checkpoint_path
            )
            # Stub: return random logits so the pipeline can be tested
            N, T = inputs.shape
            return torch.randn(N, T, 50257)

        model = load_model_from_checkpoint(checkpoint_path, device=self.device)
        model.eval()

        all_logits: list[torch.Tensor] = []
        with torch.no_grad():
            for batch_start in range(0, len(inputs), self.batch_size):
                batch = inputs[batch_start : batch_start + self.batch_size].to(self.device)
                # Pass dummy targets to get full-sequence logits (not just last position)
                targets_dummy = batch  # same shape, values don't matter for logit extraction
                logits, _ = model(batch, targets=targets_dummy)
                all_logits.append(logits.cpu())

        return torch.cat(all_logits, dim=0)

    def _compute_ensemble_loss(
        self,
        member_logits: list[torch.Tensor],
        targets: torch.Tensor,
    ) -> float:
        """Aggregate member outputs and compute cross-entropy.

        Dispatches on ``self.aggregator``. See __init__ for semantics.

        Args:
            member_logits: List of (N, T, V) logit tensors, one per member.
            targets: (N, T) target token indices.
        """
        stacked = torch.stack(member_logits, dim=0)  # (M, N, T, V)
        flat_targets = targets.view(-1)
        V = stacked.size(-1)

        if self.aggregator == "logit_mean":
            # Unsmoothed geometric-mean-of-probs. Kept for ablation.
            avg_logits = stacked.mean(dim=0)
            loss = F.cross_entropy(
                avg_logits.view(-1, V), flat_targets, ignore_index=-1,
            )
            return loss.item()

        if self.aggregator == "logit_mean_smoothed":
            # Per-member smoothed log-probs:
            #   p_i    = (1-α) softmax(z_i) + α/V
            #   log p_i in log-space, then mean over members = log of
            #   unnormalized geometric mean of smoothed probs.
            # Renormalize via logsumexp over V to get a proper distribution.
            α = self.smoothing_eps
            if α <= 0.0:
                # Fall back to plain logit mean (no floor).
                avg = stacked.mean(dim=0)
                return F.cross_entropy(
                    avg.view(-1, V), flat_targets, ignore_index=-1,
                ).item()
            log_probs = F.log_softmax(stacked, dim=-1)  # (M, N, T, V)
            # logaddexp(log((1-α)p), log(α/V)) — stable smoothing in log space
            log_one_minus_α = torch.log(torch.tensor(1.0 - α, dtype=log_probs.dtype))
            log_α_over_V = torch.log(torch.tensor(α / V, dtype=log_probs.dtype))
            smoothed_lp = torch.logaddexp(
                log_probs + log_one_minus_α,
                log_α_over_V.expand_as(log_probs),
            )  # (M, N, T, V)
            agg = smoothed_lp.mean(dim=0)  # (N, T, V) — log of unnormalized GM
            # Renormalize so it's a valid log-distribution
            agg = agg - torch.logsumexp(agg, dim=-1, keepdim=True)
            loss = F.nll_loss(
                agg.view(-1, V), flat_targets, ignore_index=-1,
            )
            return loss.item()

        # prob_mean
        probs = F.softmax(stacked, dim=-1).mean(dim=0)  # (N, T, V)
        loss = F.nll_loss(
            probs.clamp_min(1e-12).log().view(-1, V),
            flat_targets,
            ignore_index=-1,
        )
        return loss.item()

    def _compute_diversity_score(
        self,
        member_logits: list[torch.Tensor],
    ) -> float:
        """Measure prediction diversity across ensemble members.

        Uses average pairwise Jensen-Shannon divergence on a subsample of
        positions for efficiency.  Higher = more diverse = more Solomonoff-like.
        """
        if len(member_logits) < 2:
            return 0.0

        # Subsample positions for efficiency (max 1000)
        N, T, V = member_logits[0].shape
        total_positions = N * T
        n_sample = min(1000, total_positions)
        indices = random.sample(range(total_positions), n_sample)

        # Flatten to (N*T, V) and subsample
        flat_logits = []
        for logits in member_logits:
            flat = logits.view(-1, V)
            flat_logits.append(F.softmax(flat[indices], dim=-1))

        # Average pairwise JSD
        jsd_sum = 0.0
        n_pairs = 0
        for i in range(len(flat_logits)):
            for j in range(i + 1, len(flat_logits)):
                p, q = flat_logits[i], flat_logits[j]
                m = (p + q) / 2
                kl_pm = (p * (p / m).log()).sum(dim=-1).mean()
                kl_qm = (q * (q / m).log()).sum(dim=-1).mean()
                jsd = ((kl_pm + kl_qm) / 2).item()
                jsd_sum += jsd
                n_pairs += 1

        return jsd_sum / n_pairs if n_pairs > 0 else 0.0

    def evaluate_all_combinations(
        self,
        members: list[Member],
        ensemble_size: int,
        max_combinations: int = 5000,
        strategy: str = "caruana_greedy",
    ) -> list[dict[str, Any]]:
        """Evaluate ensemble combinations and return ranked results.

        Args:
            members: Trained members with checkpoint_paths set.
            ensemble_size: Number of models per ensemble.
            max_combinations: Maximum number of combinations to evaluate.
            strategy: One of
                - "caruana_greedy" (default): forward-stepwise greedy
                  selection *with replacement* from Caruana, Niculescu-
                  Mizil, Crew & Ksikes 2004
                  (https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf).
                  Re-selection is allowed, which (a) soft-weights strong
                  members by counting how often they're picked, and
                  (b) empirically resists overfitting on the selection
                  set, per Caruana 2004 §5.1. Still SOTA in AutoML
                  post-hoc ensembling (Purucker et al. 2023,
                  arxiv.org/abs/2307.00286).
                - "greedy": legacy no-repeat greedy (kept for ablation).
                - "random": random subset sampling.

        Returns:
            List of dicts sorted by ensemble_val_loss (ascending / best first):
            {combination_id, member_ids, ensemble_val_loss, diversity_score}
        """
        # Filter to members that have checkpoints
        trainable = [m for m in members if m.checkpoint_path is not None]
        if len(trainable) < ensemble_size:
            logger.warning(
                "Only %d members have checkpoints, need %d for ensemble. "
                "Evaluating with all available.",
                len(trainable), ensemble_size,
            )
            ensemble_size = len(trainable)

        if ensemble_size < 1:
            return []

        logger.info(
            "Evaluating ensembles: %d members, ensemble_size=%d, strategy=%s",
            len(trainable), ensemble_size, strategy,
        )

        # Load validation data
        inputs, targets = self._load_val_data()

        # Pre-compute logits for all members (cache to avoid redundant forward passes)
        logger.info("Computing logits for %d members...", len(trainable))
        logits_cache: dict[str, torch.Tensor] = {}
        for member in trainable:
            logger.info("  Loading checkpoint: %s (%s)", member.id, member.checkpoint_path)
            logits_cache[member.id] = self._get_logits_for_checkpoint(
                member.checkpoint_path, inputs
            )

        # Generate combinations to evaluate
        if strategy == "caruana_greedy":
            combinations = self._caruana_greedy_combinations(
                trainable, ensemble_size, max_combinations, logits_cache, targets
            )
        elif strategy == "greedy":
            combinations = self._greedy_combinations(
                trainable, ensemble_size, max_combinations, logits_cache, targets
            )
        else:
            combinations = self._random_combinations(
                trainable, ensemble_size, max_combinations
            )

        # Evaluate each combination
        results: list[dict[str, Any]] = []
        for i, combo_member_ids in enumerate(combinations):
            if (i + 1) % 100 == 0:
                logger.info("  Evaluated %d / %d combinations", i + 1, len(combinations))

            combo_logits = [logits_cache[mid] for mid in combo_member_ids]
            ens_loss = self._compute_ensemble_loss(combo_logits, targets)
            diversity = self._compute_diversity_score(combo_logits)

            results.append({
                "combination_id": uuid.uuid4().hex[:12],
                "member_ids": list(combo_member_ids),
                "ensemble_val_loss": ens_loss,
                "diversity_score": diversity,
            })

        # Sort by ensemble_val_loss ascending (best first)
        results.sort(key=lambda r: r["ensemble_val_loss"])

        if results:
            logger.info(
                "Best ensemble loss: %.4f (diversity: %.4f), worst: %.4f",
                results[0]["ensemble_val_loss"],
                results[0]["diversity_score"],
                results[-1]["ensemble_val_loss"],
            )

        return results

    def _random_combinations(
        self,
        members: list[Member],
        ensemble_size: int,
        max_combinations: int,
    ) -> list[list[str]]:
        """Generate random subsets of member IDs."""
        member_ids = [m.id for m in members]
        total_possible = 1
        for i in range(ensemble_size):
            total_possible = total_possible * (len(member_ids) - i) // (i + 1)

        if total_possible <= max_combinations:
            # Enumerate all combinations
            return [list(c) for c in itertools.combinations(member_ids, ensemble_size)]

        # Random sampling (no replacement of full combinations)
        seen: set[tuple[str, ...]] = set()
        combos: list[list[str]] = []
        while len(combos) < max_combinations:
            sample = tuple(sorted(random.sample(member_ids, ensemble_size)))
            if sample not in seen:
                seen.add(sample)
                combos.append(list(sample))

        return combos

    def _caruana_greedy_combinations(
        self,
        members: list[Member],
        ensemble_size: int,
        max_combinations: int,
        logits_cache: dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> list[list[str]]:
        """Caruana 2004 greedy ensemble selection with replacement.

        Paper: https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf
        Post-hoc AutoML validation: Purucker et al. 2023
        (https://arxiv.org/abs/2307.00286).

        Procedure
        ---------
        - Seed the ensemble with the best single member (lowest val loss).
        - At each step, consider *every* library member — including members
          already in the ensemble — and add the one whose inclusion most
          reduces ensemble loss. Re-selection is the key difference from
          subset-selection greedy: by picking the same member twice we
          implicitly give it weight 2/k in the averaged output. This lets
          the selector discover a weighted ensemble without an explicit
          weight-search step, and it empirically resists overfitting on
          the selection set (Caruana 2004 §5.1; demonstrated repeatedly
          in AutoML benchmarks).

        For SIEVE we run this from multiple seeds (best-N single members)
        to get diverse trajectories and return up to ``max_combinations``
        distinct resulting ensembles. Each ensemble is returned as a
        list of member IDs with multiplicity (sorted only for dedup; the
        downstream `_compute_ensemble_loss` averages over entries, so
        repeats increase the member's weight).

        Complexity: O(S · k · N) ensemble-loss evaluations where S =
        number of seed starts, k = ensemble_size, N = |library|. At our
        pop=32, ensemble=8, S=8 → ~2k evals vs C(32,8)=10.5M for brute
        force.
        """
        if not members:
            return []

        # Seed starts: the best (up to 8) individual members by val loss.
        # Caruana recommends ≥1 seed; multi-seed starts protect against
        # the initial greedy step committing to a local optimum.
        sorted_members = sorted(
            members,
            key=lambda m: m.val_loss if m.val_loss is not None else float("inf"),
        )
        n_seeds = min(len(sorted_members), max(1, min(8, max_combinations)))
        seed_ids = [m.id for m in sorted_members[:n_seeds]]
        library_ids = [m.id for m in members]

        combos: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()

        for seed_id in seed_ids:
            if len(combos) >= max_combinations:
                break

            current_ids: list[str] = [seed_id]

            while len(current_ids) < ensemble_size:
                best_candidate: str | None = None
                best_loss = float("inf")

                # Consider all members *including* ones already picked —
                # that is the Caruana-with-replacement twist.
                for candidate_id in library_ids:
                    trial_logits = [
                        logits_cache[mid] for mid in current_ids
                    ] + [logits_cache[candidate_id]]
                    trial_loss = self._compute_ensemble_loss(
                        trial_logits, targets
                    )
                    if trial_loss < best_loss:
                        best_loss = trial_loss
                        best_candidate = candidate_id

                if best_candidate is None:
                    break
                current_ids.append(best_candidate)

            # Dedup by sorted member-id multiset. A given ensemble with
            # repeats (e.g. [A, A, B, C]) is distinct from its sibling
            # [A, B, B, C], so we key on the sorted sequence.
            key = tuple(sorted(current_ids))
            if key not in seen and len(current_ids) == ensemble_size:
                seen.add(key)
                combos.append(current_ids)

        return combos

    def _greedy_combinations(
        self,
        members: list[Member],
        ensemble_size: int,
        max_combinations: int,
        logits_cache: dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> list[list[str]]:
        """Greedy ensemble construction.

        Start from each member as seed, greedily add the member that most
        reduces ensemble loss at each step.  Returns up to max_combinations
        distinct ensembles.
        """
        # Sort members by individual val_loss to prioritize better seeds
        sorted_members = sorted(
            members,
            key=lambda m: m.val_loss if m.val_loss is not None else float("inf"),
        )

        combos: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()

        for seed in sorted_members:
            if len(combos) >= max_combinations:
                break

            current_ids = [seed.id]
            remaining_ids = [m.id for m in members if m.id != seed.id]

            while len(current_ids) < ensemble_size and remaining_ids:
                best_candidate = None
                best_loss = float("inf")

                for candidate_id in remaining_ids:
                    trial_logits = [logits_cache[mid] for mid in current_ids] + [
                        logits_cache[candidate_id]
                    ]
                    trial_loss = self._compute_ensemble_loss(trial_logits, targets)
                    if trial_loss < best_loss:
                        best_loss = trial_loss
                        best_candidate = candidate_id

                if best_candidate is not None:
                    current_ids.append(best_candidate)
                    remaining_ids.remove(best_candidate)
                else:
                    break

            key = tuple(sorted(current_ids))
            if key not in seen and len(current_ids) == ensemble_size:
                seen.add(key)
                combos.append(current_ids)

        return combos

    def rank_members_by_ensemble_contribution(
        self,
        combinations: list[dict[str, Any]],
        top_k: int,
    ) -> dict[str, float]:
        """Score each member by how often it appears in the top-K ensembles.

        Args:
            combinations: Ensemble evaluation results (sorted by ensemble_val_loss).
            top_k: Number of top ensembles to consider.

        Returns:
            Dict mapping member_id -> contribution_score (count of appearances
            in top-K ensembles, normalized by top_k).
        """
        top = combinations[:top_k]
        counts: dict[str, int] = {}

        for ens in top:
            for mid in ens["member_ids"]:
                counts[mid] = counts.get(mid, 0) + 1

        # Normalize by top_k so score is in [0, 1]
        return {mid: count / top_k for mid, count in counts.items()}
