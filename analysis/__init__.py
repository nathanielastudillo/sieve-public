"""SIEVE analysis module -- metrics, convergence tracking, and visualization."""

from analysis.diversity import (
    prediction_disagreement,
    kl_diversity,
    config_diversity,
    ensemble_redundancy,
)
from analysis.convergence import ConvergenceTracker
from analysis.visualize import (
    plot_loss_curve,
    plot_diversity_vs_loss,
    plot_lineage_tree,
    plot_hyperparameter_evolution,
    plot_ensemble_composition,
)

__all__ = [
    "prediction_disagreement",
    "kl_diversity",
    "config_diversity",
    "ensemble_redundancy",
    "ConvergenceTracker",
    "plot_loss_curve",
    "plot_diversity_vs_loss",
    "plot_lineage_tree",
    "plot_hyperparameter_evolution",
    "plot_ensemble_composition",
]
