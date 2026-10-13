"""Visualization for SIEVE evolutionary runs.

Generates static PNG plots from the LineageDB. All functions accept a
*db_path* (path to the LineageDB SQLite file) and an *output_path*
(where to write the PNG). Pass ``show=True`` to also call
``plt.show()`` for interactive inspection.

Requires matplotlib. No interactive backends or GUI dependencies.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from typing import Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")  # non-interactive backend -- no display required
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize


def _connect(db_path: str) -> sqlite3.Connection:
    """Open a read-only connection to the LineageDB."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


# ------------------------------------------------------------------
# 1. Loss curve
# ------------------------------------------------------------------

def plot_loss_curve(
    db_path: str,
    output_path: str,
    show: bool = False,
) -> None:
    """Plot best ensemble val_loss over generations.

    Draws both the best-ensemble curve and the population mean/min/max
    member losses as a shaded band for context.

    Parameters
    ----------
    db_path : path to the LineageDB.
    output_path : path to write the PNG.
    show : if True, also display the plot interactively.
    """
    conn = _connect(db_path)

    gen_rows = conn.execute(
        "SELECT num, best_ensemble_loss, best_individual_loss FROM generations ORDER BY num"
    ).fetchall()

    if not gen_rows:
        conn.close()
        raise ValueError("No generations found in LineageDB")

    gens = [r["num"] for r in gen_rows]
    best_losses = [r["best_ensemble_loss"] for r in gen_rows]
    # Use best_individual_loss as a proxy for mean (generations table doesn't store mean)
    mean_losses = [r["best_individual_loss"] for r in gen_rows]

    # Per-generation member loss range
    min_member = []
    max_member = []
    for g in gens:
        rows = conn.execute(
            "SELECT MIN(val_loss) as lo, MAX(val_loss) as hi "
            "FROM members WHERE generation = ?",
            (g,),
        ).fetchone()
        min_member.append(rows["lo"] if rows["lo"] is not None else float("nan"))
        max_member.append(rows["hi"] if rows["hi"] is not None else float("nan"))

    conn.close()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Member loss band
    ax.fill_between(
        gens, min_member, max_member,
        alpha=0.15, color="steelblue", label="Member loss range",
    )
    # Mean member loss
    ax.plot(gens, mean_losses, color="steelblue", alpha=0.5, linewidth=1,
            linestyle="--", label="Mean member loss")
    # Best ensemble loss (the primary metric)
    ax.plot(gens, best_losses, color="crimson", linewidth=2,
            label="Best ensemble loss")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Validation Loss")
    ax.set_title("SIEVE -- Ensemble Loss Over Generations")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


# ------------------------------------------------------------------
# 2. Diversity vs loss scatter
# ------------------------------------------------------------------

def plot_diversity_vs_loss(
    db_path: str,
    output_path: str,
    show: bool = False,
) -> None:
    """Scatter plot: config diversity vs best ensemble loss, per generation.

    Each point is one generation. Color encodes generation number so
    temporal progression is visible.

    Parameters
    ----------
    db_path : path to the LineageDB.
    output_path : path to write the PNG.
    show : if True, also display interactively.
    """
    from analysis.convergence import ConvergenceTracker

    tracker = ConvergenceTracker(db_path)
    try:
        gens_loss, losses = tracker.loss_curve()
        gens_div, divs = tracker.diversity_curve()
    finally:
        tracker.close()

    if not gens_loss:
        raise ValueError("No generations found in LineageDB")

    # Align by generation number (both should match, but be safe)
    gen_set = set(gens_loss) & set(gens_div)
    loss_map = dict(zip(gens_loss, losses))
    div_map = dict(zip(gens_div, divs))
    gens_aligned = sorted(gen_set)

    x = [div_map[g] for g in gens_aligned]
    y = [loss_map[g] for g in gens_aligned]
    c = list(gens_aligned)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(x, y, c=c, cmap="viridis", s=40, alpha=0.8, edgecolors="k", linewidths=0.3)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Generation")

    ax.set_xlabel("Config Diversity")
    ax.set_ylabel("Best Ensemble Loss")
    ax.set_title("SIEVE -- Diversity vs Loss")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


# ------------------------------------------------------------------
# 3. Lineage tree
# ------------------------------------------------------------------

def plot_lineage_tree(
    db_path: str,
    output_path: str,
    max_generations: int = 50,
    show: bool = False,
) -> None:
    """Plot member parentage as a tree, color-coded by val_loss.

    Each node is a member. Edges connect children to parents. Nodes
    are positioned by generation (x-axis) and spread within each
    generation (y-axis). Color intensity encodes val_loss (darker =
    lower = better).

    Parameters
    ----------
    db_path : path to the LineageDB.
    output_path : path to write the PNG.
    max_generations : limit to the most recent N generations.
    show : if True, also display interactively.
    """
    conn = _connect(db_path)

    # Determine generation range
    max_gen_row = conn.execute("SELECT MAX(num) as mg FROM generations").fetchone()
    if max_gen_row["mg"] is None:
        conn.close()
        raise ValueError("No generations found in LineageDB")

    max_gen = max_gen_row["mg"]
    min_gen = max(0, max_gen - max_generations + 1)

    members = conn.execute(
        "SELECT id, generation, val_loss, parent_ids FROM members "
        "WHERE generation >= ? AND generation <= ? "
        "ORDER BY generation, val_loss",
        (min_gen, max_gen),
    ).fetchall()
    conn.close()

    if not members:
        raise ValueError("No members found in the requested generation range")

    # Assign positions: x = generation, y = index within generation
    gen_members: dict[int, list] = defaultdict(list)
    for m in members:
        gen_members[m["generation"]].append(m)

    # Map member_id -> (x, y) position
    positions: dict[str, tuple[float, float]] = {}
    val_losses: dict[str, float] = {}

    for gen in sorted(gen_members):
        mems = gen_members[gen]
        n = len(mems)
        for idx, m in enumerate(mems):
            # Spread members vertically, centered at 0
            y = (idx - (n - 1) / 2.0) if n > 1 else 0.0
            positions[m["id"]] = (gen, y)
            val_losses[m["id"]] = m["val_loss"] if m["val_loss"] is not None else float("nan")

    # Collect edges (child -> parent)
    edges: list[tuple[str, str]] = []
    for m in members:
        if m["parent_ids"]:
            try:
                pids = json.loads(m["parent_ids"])
            except (json.JSONDecodeError, TypeError):
                pids = []
            for pid in pids:
                if pid in positions:
                    edges.append((pid, m["id"]))

    # Color mapping
    valid_losses = [v for v in val_losses.values() if not np.isnan(v)]
    if valid_losses:
        vmin, vmax = min(valid_losses), max(valid_losses)
    else:
        vmin, vmax = 0.0, 1.0

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap("RdYlGn_r")  # red = high loss, green = low loss

    # Plot
    n_gens_shown = max_gen - min_gen + 1
    fig_width = max(10, min(40, n_gens_shown * 0.5))
    max_pop = max(len(v) for v in gen_members.values()) if gen_members else 1
    fig_height = max(6, min(20, max_pop * 0.3))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Draw edges
    for parent_id, child_id in edges:
        px, py = positions[parent_id]
        cx, cy = positions[child_id]
        ax.plot([px, cx], [py, cy], color="gray", alpha=0.2, linewidth=0.5, zorder=1)

    # Draw nodes
    xs = [positions[mid][0] for mid in positions]
    ys = [positions[mid][1] for mid in positions]
    colors = [cmap(norm(val_losses[mid])) if not np.isnan(val_losses[mid]) else (0.5, 0.5, 0.5, 1.0)
              for mid in positions]

    ax.scatter(xs, ys, c=colors, s=15, zorder=2, edgecolors="k", linewidths=0.2)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7)
    cbar.set_label("Validation Loss")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Population Member")
    ax.set_title(f"SIEVE -- Lineage Tree (generations {min_gen}-{max_gen})")
    ax.grid(True, alpha=0.15, axis="x")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


# ------------------------------------------------------------------
# 4. Hyperparameter evolution
# ------------------------------------------------------------------

def plot_hyperparameter_evolution(
    db_path: str,
    param_name: str,
    output_path: str,
    show: bool = False,
) -> None:
    """Show how a specific hyperparameter's distribution evolves.

    For numeric parameters, plots a box/violin per generation showing the
    distribution. For categorical parameters, plots stacked bars showing
    the proportion of each category.

    Parameters
    ----------
    db_path : path to the LineageDB.
    param_name : the hyperparameter key to track (e.g. ``"dropout"``,
        ``"n_layer"``, ``"optimizer"``).
    output_path : path to write the PNG.
    show : if True, also display interactively.
    """
    conn = _connect(db_path)

    rows = conn.execute(
        "SELECT generation, config_json FROM members ORDER BY generation"
    ).fetchall()
    conn.close()

    if not rows:
        raise ValueError("No members found in LineageDB")

    # Extract per-generation values
    gen_values: dict[int, list] = defaultdict(list)
    for r in rows:
        if not r["config_json"]:
            continue
        try:
            cfg = json.loads(r["config_json"])
        except (json.JSONDecodeError, TypeError):
            continue
        if param_name in cfg:
            gen_values[r["generation"]].append(cfg[param_name])

    if not gen_values:
        raise ValueError(f"Parameter '{param_name}' not found in any member config")

    gens = sorted(gen_values.keys())

    # Determine if numeric or categorical
    all_values = [v for vals in gen_values.values() for v in vals]
    is_numeric = all(isinstance(v, (int, float)) for v in all_values)

    if is_numeric:
        _plot_numeric_param(gens, gen_values, param_name, output_path, show)
    else:
        _plot_categorical_param(gens, gen_values, param_name, output_path, show)


def _plot_numeric_param(
    gens: list[int],
    gen_values: dict[int, list],
    param_name: str,
    output_path: str,
    show: bool,
) -> None:
    """Box plot for a numeric hyperparameter across generations."""
    # Subsample generations if there are too many for readable boxes
    max_boxes = 60
    if len(gens) > max_boxes:
        step = len(gens) // max_boxes
        gens = gens[::step]

    data = [gen_values[g] for g in gens]

    fig, ax = plt.subplots(figsize=(max(10, len(gens) * 0.25), 6))
    bp = ax.boxplot(
        data, positions=gens, widths=0.6,
        patch_artist=True, showfliers=False,
    )
    for box in bp["boxes"]:
        box.set_facecolor("steelblue")
        box.set_alpha(0.6)

    # Overlay individual points with jitter
    for g, vals in zip(gens, data):
        jitter = np.random.default_rng(42).uniform(-0.2, 0.2, size=len(vals))
        ax.scatter(
            [g + j for j in jitter], vals,
            color="navy", alpha=0.3, s=8, zorder=3,
        )

    ax.set_xlabel("Generation")
    ax.set_ylabel(param_name)
    ax.set_title(f"SIEVE -- Evolution of '{param_name}'")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def _plot_categorical_param(
    gens: list[int],
    gen_values: dict[int, list],
    param_name: str,
    output_path: str,
    show: bool,
) -> None:
    """Stacked bar chart for a categorical hyperparameter."""
    # Find all unique categories
    categories = sorted(set(str(v) for vals in gen_values.values() for v in vals))
    cat_index = {c: i for i, c in enumerate(categories)}

    # Build proportion matrix: (n_gens, n_categories)
    proportions = np.zeros((len(gens), len(categories)))
    for gi, g in enumerate(gens):
        vals = gen_values[g]
        n = len(vals)
        if n == 0:
            continue
        for v in vals:
            proportions[gi, cat_index[str(v)]] += 1.0 / n

    fig, ax = plt.subplots(figsize=(max(10, len(gens) * 0.15), 6))

    cmap = cm.get_cmap("tab10")
    bottom = np.zeros(len(gens))
    for ci, cat in enumerate(categories):
        color = cmap(ci % 10)
        ax.bar(
            gens, proportions[:, ci], bottom=bottom,
            width=0.8, color=color, label=cat, alpha=0.8,
        )
        bottom += proportions[:, ci]

    ax.set_xlabel("Generation")
    ax.set_ylabel("Proportion")
    ax.set_title(f"SIEVE -- Evolution of '{param_name}'")
    ax.legend(title=param_name, loc="upper right", fontsize="small")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


# ------------------------------------------------------------------
# 5. Ensemble composition heatmap
# ------------------------------------------------------------------

def plot_ensemble_composition(
    db_path: str,
    output_path: str,
    top_k: int = 20,
    show: bool = False,
) -> None:
    """Heatmap showing which members appear in the top-K ensembles.

    Rows are ensembles (sorted by val_loss, best at top). Columns are
    members (sorted by frequency of inclusion). A filled cell means
    the member is part of that ensemble.

    Parameters
    ----------
    db_path : path to the LineageDB.
    output_path : path to write the PNG.
    top_k : number of top ensembles to include (by val_loss).
    show : if True, also display interactively.
    """
    conn = _connect(db_path)

    ens_rows = conn.execute(
        "SELECT id, member_ids_json, ensemble_val_loss FROM ensembles "
        "ORDER BY ensemble_val_loss ASC LIMIT ?",
        (top_k,),
    ).fetchall()

    if not ens_rows:
        conn.close()
        raise ValueError("No ensembles found in LineageDB")

    # Parse member IDs and collect all unique members
    ensembles: list[tuple[str, float, list[str]]] = []
    member_freq: dict[str, int] = defaultdict(int)

    for row in ens_rows:
        ens_id = row["id"]
        val_loss = row["ensemble_val_loss"]
        try:
            mids = json.loads(row["member_ids_json"])
        except (json.JSONDecodeError, TypeError):
            continue
        ensembles.append((ens_id, val_loss, mids))
        for mid in mids:
            member_freq[mid] += 1

    if not ensembles:
        conn.close()
        raise ValueError("Could not parse any ensemble member lists")

    # Get member val_losses for labeling
    all_member_ids = list(member_freq.keys())
    placeholders = ",".join("?" * len(all_member_ids))
    member_rows = conn.execute(
        f"SELECT id, val_loss, generation FROM members WHERE id IN ({placeholders})",
        all_member_ids,
    ).fetchall()
    conn.close()

    member_info = {r["id"]: r for r in member_rows}

    # Sort members by frequency (most common first)
    sorted_members = sorted(member_freq.keys(), key=lambda m: -member_freq[m])

    # Build inclusion matrix
    n_ens = len(ensembles)
    n_mem = len(sorted_members)
    member_index = {mid: i for i, mid in enumerate(sorted_members)}

    matrix = np.zeros((n_ens, n_mem))
    for ei, (ens_id, val_loss, mids) in enumerate(ensembles):
        for mid in mids:
            if mid in member_index:
                matrix[ei, member_index[mid]] = 1.0

    # Labels
    ens_labels = [f"{val_loss:.4f}" for _, val_loss, _ in ensembles]
    mem_labels = []
    for mid in sorted_members:
        info = member_info.get(mid)
        if info:
            gen = info["generation"]
            vloss = info["val_loss"]
            short_id = mid[:8] if len(mid) > 8 else mid
            mem_labels.append(f"g{gen}/{short_id}\n({vloss:.3f})" if vloss else f"g{gen}/{short_id}")
        else:
            mem_labels.append(mid[:8])

    # Plot
    fig_w = max(8, min(30, n_mem * 0.6 + 2))
    fig_h = max(4, min(20, n_ens * 0.4 + 2))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(
        matrix, cmap="YlOrRd", aspect="auto",
        interpolation="nearest", vmin=0, vmax=1,
    )

    ax.set_xticks(range(n_mem))
    ax.set_xticklabels(mem_labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n_ens))
    ax.set_yticklabels(ens_labels, fontsize=8)

    ax.set_xlabel("Member (sorted by inclusion frequency)")
    ax.set_ylabel("Ensemble val_loss")
    ax.set_title(f"SIEVE -- Top-{top_k} Ensemble Composition")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
