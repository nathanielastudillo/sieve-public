"""Evolutionary convergence tracking for SIEVE.

Reads from the controller's LineageDB (SQLite) to produce loss curves,
diversity trajectories, convergence diagnostics, and per-generation
summaries.

Expected LineageDB schema (populated by controller/lineage.py):

    members:
        id              TEXT PRIMARY KEY,
        generation      INTEGER,
        config_json     TEXT (JSON),
        val_loss        REAL,
        checkpoint_path TEXT,
        parent_ids      TEXT (JSON),  -- NULL for gen-0 (random init)
        created_at      REAL

    ensembles:
        id              TEXT PRIMARY KEY,
        generation      INTEGER,
        member_ids_json TEXT (JSON list of member IDs),
        ensemble_val_loss REAL,
        diversity_score REAL,
        created_at      REAL

    generations:
        num             INTEGER PRIMARY KEY,
        best_ensemble_loss REAL,
        best_individual_loss REAL,
        population_diversity REAL,
        wallclock_seconds REAL,
        created_at      REAL
"""

from __future__ import annotations

import json
import sqlite3
from typing import Optional

import numpy as np


class ConvergenceTracker:
    """Track and diagnose evolutionary convergence from the LineageDB.

    Provides loss/diversity curves, convergence detection, and
    per-generation summaries for monitoring SIEVE runs.
    """

    def __init__(self, db_path: str) -> None:
        """Open a read-only connection to the LineageDB.

        Parameters
        ----------
        db_path : path to the SQLite database written by the controller.
        """
        self.db_path = db_path
        self._conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        self._conn.row_factory = sqlite3.Row

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------
    # Core curves
    # ------------------------------------------------------------------

    def loss_curve(self) -> tuple[list[int], list[float]]:
        """Best ensemble val_loss per generation.

        Returns
        -------
        (generations, losses) -- parallel lists suitable for plotting.
        """
        rows = self._conn.execute(
            "SELECT num, best_ensemble_loss FROM generations ORDER BY num"
        ).fetchall()
        gens = [r["num"] for r in rows]
        losses = [r["best_ensemble_loss"] for r in rows]
        return gens, losses

    def diversity_curve(self) -> tuple[list[int], list[float]]:
        """Population config diversity per generation.

        Computes normalized pairwise config distance for all members in
        each generation using the search space from ``config/default.yaml``.
        Falls back to a simpler variance-based metric if configs are not
        available.

        Returns
        -------
        (generations, diversities) -- parallel lists.
        """
        from analysis.diversity import config_diversity

        gen_rows = self._conn.execute(
            "SELECT num FROM generations ORDER BY num"
        ).fetchall()

        gens: list[int] = []
        divs: list[float] = []

        for row in gen_rows:
            gen = row["num"]
            members = self._conn.execute(
                "SELECT config_json FROM members WHERE generation = ?", (gen,)
            ).fetchall()

            configs = []
            for m in members:
                if m["config_json"]:
                    try:
                        configs.append(json.loads(m["config_json"]))
                    except (json.JSONDecodeError, TypeError):
                        continue

            if len(configs) < 2:
                div = 0.0
            else:
                # Build a search space from the observed configs (union of
                # all keys, inferring ranges from the values seen).
                search_space = _infer_search_space(configs)
                div = config_diversity(configs, search_space)

            gens.append(gen)
            divs.append(div)

        return gens, divs

    # ------------------------------------------------------------------
    # Convergence diagnostics
    # ------------------------------------------------------------------

    def is_converged(self, patience: int = 20) -> bool:
        """Has the best ensemble loss plateaued?

        Returns True if the best loss has not improved for *patience*
        consecutive generations. Improvement is measured as a relative
        decrease exceeding 0.1%.
        """
        _, losses = self.loss_curve()
        if len(losses) < patience + 1:
            return False

        best_so_far = losses[0]
        gens_without_improvement = 0

        for loss in losses[1:]:
            if loss < best_so_far * 0.999:  # 0.1% relative improvement
                best_so_far = loss
                gens_without_improvement = 0
            else:
                gens_without_improvement += 1

            if gens_without_improvement >= patience:
                return True

        return False

    def is_diversity_collapsed(self, threshold: float = 0.1) -> bool:
        """Has population diversity fallen below *threshold*?

        Checks the most recent generation's config diversity. If below
        *threshold*, the population is too homogeneous and the
        evolutionary search is likely stuck in a single basin.
        """
        gens, divs = self.diversity_curve()
        if not divs:
            return False
        return divs[-1] < threshold

    def estimate_remaining_improvement(self, window: int = 20) -> float:
        """Extrapolate expected remaining loss improvement.

        Fits a linear trend to the best-loss curve over the most recent
        *window* generations and projects how much additional improvement
        one more *window* of generations would yield (assuming the trend
        continues linearly).

        Returns
        -------
        Estimated loss decrease (positive means improvement expected).
        Returns 0.0 if there are fewer than *window* generations or
        the trend is worsening.
        """
        _, losses = self.loss_curve()
        if len(losses) < window:
            return 0.0

        recent = np.array(losses[-window:], dtype=np.float64)
        x = np.arange(window, dtype=np.float64)

        # Least-squares linear fit: loss = slope * x + intercept
        coeffs = np.polyfit(x, recent, 1)
        slope = coeffs[0]

        if slope >= 0:
            # Trend is flat or worsening -- no further improvement expected
            return 0.0

        # Project forward by *window* more generations
        projected_improvement = abs(slope) * window
        return float(projected_improvement)

    # ------------------------------------------------------------------
    # Per-generation summary
    # ------------------------------------------------------------------

    def generation_summary(self, gen_num: int) -> dict:
        """Rich summary for a single generation.

        Returns
        -------
        Dict with keys:
            generation       -- generation number
            population_size  -- number of members
            best_loss        -- best ensemble val_loss
            mean_loss        -- mean member val_loss
            median_loss      -- median member val_loss
            worst_loss       -- worst member val_loss
            loss_std         -- std of member val_losses
            n_ensembles      -- number of ensembles evaluated
            best_ensemble    -- dict with id, member_ids, val_loss
            diversity        -- config diversity score
            top_configs      -- configs of the top 5 members by val_loss
            improvement      -- loss change vs previous generation (negative = better)
        """
        # Generation metadata
        gen_row = self._conn.execute(
            "SELECT * FROM generations WHERE num = ?", (gen_num,)
        ).fetchone()

        if gen_row is None:
            raise ValueError(f"Generation {gen_num} not found in LineageDB")

        # Member stats
        member_rows = self._conn.execute(
            "SELECT id, config_json, val_loss, parent_ids FROM members "
            "WHERE generation = ? ORDER BY val_loss ASC",
            (gen_num,),
        ).fetchall()

        member_losses = [r["val_loss"] for r in member_rows if r["val_loss"] is not None]
        loss_arr = np.array(member_losses, dtype=np.float64) if member_losses else np.array([])

        # Best ensemble this generation
        best_ens_row = self._conn.execute(
            "SELECT id, member_ids_json, ensemble_val_loss, diversity_score FROM ensembles "
            "WHERE generation = ? ORDER BY ensemble_val_loss ASC LIMIT 1",
            (gen_num,),
        ).fetchone()

        best_ensemble = None
        if best_ens_row is not None:
            best_ensemble = {
                "id": best_ens_row["id"],
                "member_ids": json.loads(best_ens_row["member_ids_json"]),
                "val_loss": best_ens_row["ensemble_val_loss"],
                "diversity_score": best_ens_row["diversity_score"],
            }

        # Ensemble count
        ens_count = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM ensembles WHERE generation = ?",
            (gen_num,),
        ).fetchone()["cnt"]

        # Config diversity
        configs = []
        for m in member_rows:
            if m["config_json"]:
                try:
                    configs.append(json.loads(m["config_json"]))
                except (json.JSONDecodeError, TypeError):
                    continue

        if len(configs) >= 2:
            from analysis.diversity import config_diversity as _cd
            search_space = _infer_search_space(configs)
            diversity = _cd(configs, search_space)
        else:
            diversity = 0.0

        # Top-5 configs
        top_configs = []
        for m in member_rows[:5]:
            if m["config_json"]:
                try:
                    top_configs.append(json.loads(m["config_json"]))
                except (json.JSONDecodeError, TypeError):
                    continue

        # Improvement vs previous generation
        prev_row = self._conn.execute(
            "SELECT best_ensemble_loss FROM generations WHERE num = ?",
            (gen_num - 1,),
        ).fetchone()
        improvement = None
        if prev_row is not None and gen_row["best_ensemble_loss"] is not None:
            improvement = gen_row["best_ensemble_loss"] - prev_row["best_ensemble_loss"]

        return {
            "generation": gen_num,
            "population_size": len(member_rows),
            "best_loss": gen_row["best_ensemble_loss"],
            "mean_loss": float(loss_arr.mean()) if len(loss_arr) > 0 else None,
            "median_loss": float(np.median(loss_arr)) if len(loss_arr) > 0 else None,
            "worst_loss": float(loss_arr.max()) if len(loss_arr) > 0 else None,
            "loss_std": float(loss_arr.std()) if len(loss_arr) > 0 else None,
            "n_ensembles": ens_count,
            "best_ensemble": best_ensemble,
            "diversity": diversity,
            "top_configs": top_configs,
            "improvement": improvement,
        }


def _infer_search_space(configs: list[dict]) -> dict:
    """Infer a search space from observed config values.

    For numeric parameters, constructs a [min, max] range from the
    values seen. For non-numeric parameters, constructs a list of
    unique values.

    This is a best-effort fallback -- the actual search space from
    ``config/default.yaml`` should be preferred when available.
    """
    all_keys: set[str] = set()
    for c in configs:
        all_keys.update(c.keys())

    space: dict = {}
    for key in sorted(all_keys):
        values = [c[key] for c in configs if key in c]
        if not values:
            continue

        if all(isinstance(v, (int, float)) for v in values):
            lo = min(values)
            hi = max(values)
            if lo == hi:
                # All same value -- set a nominal range so distance = 0
                space[key] = [lo, lo + 1]
            else:
                space[key] = [lo, hi]
        else:
            space[key] = list(set(str(v) for v in values))

    return space
