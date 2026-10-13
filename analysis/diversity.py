"""Ensemble diversity metrics for SIEVE.

Measures how different ensemble members are from each other in prediction
space and hyperparameter space. High diversity is essential -- the
Solomonoff-inspired thesis requires that each member covers different
solution-space territory rather than collapsing to the same mode.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along *axis*."""
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def prediction_disagreement(logits_list: list[np.ndarray]) -> float:
    """Average pairwise disagreement rate across ensemble members.

    For each pair of members, computes the fraction of positions where
    the argmax prediction differs. Returns the mean over all pairs.

    Parameters
    ----------
    logits_list : list of arrays, each of shape ``(T, V)`` where
        *T* is the number of positions and *V* is the vocabulary size.
        All arrays must share the same shape.

    Returns
    -------
    Float in [0, 1]. 0 means all members agree everywhere;
    1 means every pair disagrees on every position.
    """
    n = len(logits_list)
    if n < 2:
        return 0.0

    predictions = [np.argmax(l, axis=-1) for l in logits_list]

    total_disagreement = 0.0
    n_pairs = 0
    for i, j in combinations(range(n), 2):
        disagreement = np.mean(predictions[i] != predictions[j])
        total_disagreement += float(disagreement)
        n_pairs += 1

    return total_disagreement / n_pairs


def kl_diversity(logits_list: list[np.ndarray], eps: float = 1e-10) -> float:
    """Average pairwise symmetric KL divergence between softmax distributions.

    Computes ``(KL(p||q) + KL(q||p)) / 2`` for each pair and returns the
    mean. This is equivalent to the Jensen-Shannon divergence's numerator
    (without the log-2 scaling), giving a non-negative symmetric measure.

    Parameters
    ----------
    logits_list : list of arrays, each of shape ``(T, V)``.
    eps : small constant added to probabilities to avoid log(0).

    Returns
    -------
    Non-negative float. Higher values indicate more diverse predictions.
    """
    n = len(logits_list)
    if n < 2:
        return 0.0

    probs = [_softmax(l) + eps for l in logits_list]

    total_kl = 0.0
    n_pairs = 0
    for i, j in combinations(range(n), 2):
        p, q = probs[i], probs[j]
        # KL(p || q) averaged over positions
        kl_pq = np.mean(np.sum(p * np.log(p / q), axis=-1))
        kl_qp = np.mean(np.sum(q * np.log(q / p), axis=-1))
        total_kl += float(kl_pq + kl_qp) / 2.0
        n_pairs += 1

    return total_kl / n_pairs


def config_diversity(configs: list[dict], search_space: dict) -> float:
    """Normalized distance between hyperparameter configurations.

    For continuous parameters (specified as ``(lo, hi)`` tuples in
    *search_space*), the distance is ``|a - b| / (hi - lo)``.
    For discrete/categorical parameters (specified as lists), the
    distance is ``0`` if equal, ``1`` if different.

    The per-parameter distances are averaged within each pair, and the
    result is the mean across all pairs.

    Parameters
    ----------
    configs : list of dicts mapping param names to values.
    search_space : dict mapping param names to either a list of
        allowed values (discrete/categorical) or a ``[lo, hi]`` pair
        (continuous range).

    Returns
    -------
    Float in [0, 1]. 0 means all configs are identical;
    1 means maximal separation on every parameter.
    """
    n = len(configs)
    if n < 2:
        return 0.0

    # Identify which params to compare (intersection of config keys and search space)
    param_names = [k for k in search_space if any(k in c for c in configs)]
    if not param_names:
        return 0.0

    def _param_distance(name: str, val_a, val_b) -> float:
        spec = search_space[name]
        if isinstance(spec, (list, tuple)) and len(spec) == 2 and all(
            isinstance(s, (int, float)) for s in spec
        ):
            # Could be continuous [lo, hi] or a 2-element discrete list.
            # Heuristic: if both values are numeric and the spec looks like a
            # range (lo < hi, and values are within that range), treat as
            # continuous. Otherwise treat as categorical.
            lo, hi = spec[0], spec[1]
            if (
                isinstance(val_a, (int, float))
                and isinstance(val_b, (int, float))
                and lo < hi
            ):
                span = hi - lo
                if span == 0:
                    return 0.0
                return abs(float(val_a) - float(val_b)) / span
        # Categorical / discrete
        return 0.0 if val_a == val_b else 1.0

    total_dist = 0.0
    n_pairs = 0
    for i, j in combinations(range(n), 2):
        pair_dist = 0.0
        counted = 0
        for name in param_names:
            val_a = configs[i].get(name)
            val_b = configs[j].get(name)
            if val_a is None or val_b is None:
                continue
            pair_dist += _param_distance(name, val_a, val_b)
            counted += 1
        if counted > 0:
            total_dist += pair_dist / counted
        n_pairs += 1

    return total_dist / n_pairs if n_pairs > 0 else 0.0


def ensemble_redundancy(logits_list: list[np.ndarray]) -> float:
    """Measure how redundant each member is within the ensemble.

    For each member *i*, computes the ensemble prediction with and without
    it. Redundancy is defined as ``1 - mean_impact``, where *impact* for
    member *i* is the fraction of positions where removing it changes the
    ensemble's argmax prediction.

    Low redundancy (close to 0) means every member is critical -- removing
    any single member changes the ensemble's predictions significantly.
    High redundancy (close to 1) means members are interchangeable.

    Parameters
    ----------
    logits_list : list of arrays, each of shape ``(T, V)``.

    Returns
    -------
    Float in [0, 1].
    """
    n = len(logits_list)
    if n < 2:
        return 0.0

    # Full ensemble prediction (average logits, then argmax)
    full_mean = np.mean(logits_list, axis=0)
    full_pred = np.argmax(full_mean, axis=-1)

    total_impact = 0.0
    for i in range(n):
        # Leave-one-out ensemble
        remaining = [logits_list[j] for j in range(n) if j != i]
        loo_mean = np.mean(remaining, axis=0)
        loo_pred = np.argmax(loo_mean, axis=-1)
        # Impact: fraction of positions where prediction changed
        impact = float(np.mean(full_pred != loo_pred))
        total_impact += impact

    mean_impact = total_impact / n
    return 1.0 - mean_impact
