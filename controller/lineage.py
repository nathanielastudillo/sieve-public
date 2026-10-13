"""Lineage tracking for SIEVE evolutionary runs.

Stores every member, ensemble evaluation, and generation summary in a SQLite
database.  Provides queries for tracing ancestry, plotting convergence curves,
and exporting the full evolutionary graph for visualization.
"""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any

from controller.population import Member


class LineageDB:
    """SQLite-backed evolutionary history tracker.

    Tables:
        members     - every member across all generations
        ensembles   - every evaluated ensemble combination
        generations - per-generation summary statistics
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        """Create tables if they don't exist."""
        cur = self.conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS members (
                id TEXT PRIMARY KEY,
                generation INTEGER NOT NULL,
                config_json TEXT NOT NULL,
                val_loss REAL,
                parent_ids TEXT NOT NULL DEFAULT '[]',
                checkpoint_path TEXT,
                created_at REAL NOT NULL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS ensembles (
                id TEXT PRIMARY KEY,
                generation INTEGER NOT NULL,
                member_ids_json TEXT NOT NULL,
                ensemble_val_loss REAL NOT NULL,
                diversity_score REAL,
                created_at REAL NOT NULL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS generations (
                num INTEGER PRIMARY KEY,
                best_ensemble_loss REAL,
                best_individual_loss REAL,
                population_diversity REAL,
                wallclock_seconds REAL,
                created_at REAL NOT NULL
            )
        """)

        # meta_val_loss was added in 2026-04-18 for the sequestered
        # meta-val diagnostic (lit review §8.2). Use ALTER TABLE so
        # databases from prior runs don't need a migration script.
        existing_cols = {
            r[1] for r in cur.execute("PRAGMA table_info(generations)").fetchall()
        }
        if "meta_val_loss" not in existing_cols:
            cur.execute(
                "ALTER TABLE generations ADD COLUMN meta_val_loss REAL"
            )

        # Openweight track additions (2026-04-18). Additive — the small-corpus
        # track leaves these NULL. See docs/architecture-openweight.md §"Population storage".
        member_cols = {
            r[1] for r in cur.execute("PRAGMA table_info(members)").fetchall()
        }
        for col_name, col_type in [
            ("kind", "TEXT"),
            ("tokenizer_family", "TEXT"),
            ("base_model_id", "TEXT"),
            ("adapter_path", "TEXT"),
            ("merged_checkpoint_path", "TEXT"),
        ]:
            if col_name not in member_cols:
                cur.execute(
                    f"ALTER TABLE members ADD COLUMN {col_name} {col_type}"
                )

        cur.execute("""
            CREATE TABLE IF NOT EXISTS fitness_evals (
                member_id TEXT NOT NULL,
                benchmark_name TEXT NOT NULL,
                score REAL NOT NULL,
                wallclock_s REAL NOT NULL,
                raw_json TEXT,
                evaluated_at REAL NOT NULL,
                PRIMARY KEY (member_id, benchmark_name)
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS merge_recipes (
                recipe_id TEXT PRIMARY KEY,
                member_id TEXT NOT NULL,
                method TEXT NOT NULL,
                yaml_text TEXT NOT NULL,
                coefficients_json TEXT NOT NULL,
                cma_es_step INTEGER,
                created_at REAL NOT NULL
            )
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_fitness_evals_member
            ON fitness_evals(member_id)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_merge_recipes_member
            ON merge_recipes(member_id)
        """)

        # Indexes for common queries
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_members_generation
            ON members(generation)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_ensembles_generation
            ON ensembles(generation)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_ensembles_loss
            ON ensembles(ensemble_val_loss)
        """)

        self.conn.commit()

    def record_member(self, member: Member) -> None:
        """Insert or update a member record."""
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO members
                (id, generation, config_json, val_loss, parent_ids, checkpoint_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                member.id,
                member.generation,
                json.dumps(member.config),
                member.val_loss,
                json.dumps(member.parent_ids),
                member.checkpoint_path,
                time.time(),
            ),
        )
        self.conn.commit()

    def record_ensemble(self, combination: dict[str, Any], generation: int) -> None:
        """Record an evaluated ensemble combination.

        Args:
            combination: Dict with keys: combination_id, member_ids,
                         ensemble_val_loss, diversity_score.
            generation: The generation number this evaluation belongs to.
        """
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO ensembles
                (id, generation, member_ids_json, ensemble_val_loss, diversity_score, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                combination["combination_id"],
                generation,
                json.dumps(combination["member_ids"]),
                combination["ensemble_val_loss"],
                combination.get("diversity_score"),
                time.time(),
            ),
        )
        self.conn.commit()

    def record_generation(self, gen_num: int, stats: dict[str, Any]) -> None:
        """Record per-generation summary statistics.

        Args:
            gen_num: Generation number.
            stats: Dict with optional keys: best_ensemble_loss,
                   best_individual_loss, population_diversity, wallclock_seconds.
        """
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO generations
                (num, best_ensemble_loss, best_individual_loss,
                 population_diversity, wallclock_seconds, meta_val_loss,
                 created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                gen_num,
                stats.get("best_ensemble_loss"),
                stats.get("best_individual_loss"),
                stats.get("population_diversity"),
                stats.get("wallclock_seconds"),
                stats.get("meta_val_loss"),
                time.time(),
            ),
        )
        self.conn.commit()

    def get_member_lineage(self, member_id: str) -> list[dict[str, Any]]:
        """Trace a member's ancestry back through parent_ids.

        Returns a list of member dicts from the queried member back to
        the earliest ancestor(s), ordered newest-first.
        """
        lineage: list[dict[str, Any]] = []
        visited: set[str] = set()
        queue = [member_id]

        while queue:
            mid = queue.pop(0)
            if mid in visited:
                continue
            visited.add(mid)

            row = self.conn.execute(
                "SELECT * FROM members WHERE id = ?", (mid,)
            ).fetchone()

            if row is None:
                continue

            member_dict = dict(row)
            member_dict["config"] = json.loads(member_dict.pop("config_json"))
            member_dict["parent_ids"] = json.loads(member_dict["parent_ids"])
            lineage.append(member_dict)

            # Queue parents for traversal
            for parent_id in member_dict["parent_ids"]:
                if parent_id not in visited:
                    queue.append(parent_id)

        return lineage

    def get_generation_history(self) -> list[dict[str, Any]]:
        """Return generation summary stats ordered by generation number.

        This is the convergence curve data -- best_ensemble_loss over generations.
        """
        rows = self.conn.execute(
            "SELECT * FROM generations ORDER BY num ASC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_best_ensemble(self, top_k: int = 1) -> list[dict[str, Any]]:
        """Return the top-K best ensembles across all generations.

        Sorted by ensemble_val_loss ascending (best first).
        """
        rows = self.conn.execute(
            "SELECT * FROM ensembles ORDER BY ensemble_val_loss ASC LIMIT ?",
            (top_k,),
        ).fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            d = dict(row)
            d["member_ids"] = json.loads(d.pop("member_ids_json"))
            results.append(d)

        return results

    def export_graph(self) -> dict[str, Any]:
        """Export the full lineage as a graph structure for visualization.

        Returns:
            {
                "nodes": [
                    {"id": str, "generation": int, "val_loss": float|None,
                     "config": dict, "type": "member"},
                    ...
                ],
                "edges": [
                    {"source": parent_id, "target": child_id, "type": "parent_of"},
                    ...
                ],
                "generations": [
                    {"num": int, "best_ensemble_loss": float, ...},
                    ...
                ]
            }
        """
        # Nodes: all members
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []

        rows = self.conn.execute("SELECT * FROM members ORDER BY generation ASC").fetchall()
        for row in rows:
            d = dict(row)
            config = json.loads(d.pop("config_json"))
            parent_ids = json.loads(d["parent_ids"])

            nodes.append({
                "id": d["id"],
                "generation": d["generation"],
                "val_loss": d["val_loss"],
                "config": config,
                "checkpoint_path": d["checkpoint_path"],
                "type": "member",
            })

            for pid in parent_ids:
                edges.append({
                    "source": pid,
                    "target": d["id"],
                    "type": "parent_of",
                })

        # Generations
        gen_rows = self.conn.execute(
            "SELECT * FROM generations ORDER BY num ASC"
        ).fetchall()
        generations = [dict(r) for r in gen_rows]

        return {
            "nodes": nodes,
            "edges": edges,
            "generations": generations,
        }

    # ------------------------------------------------------------------
    # Openweight track — additive helpers. Small-corpus code paths
    # never call these; they operate on the same ``members`` table via
    # the new columns.
    # ------------------------------------------------------------------

    def record_openweight_member(
        self,
        member_id: str,
        generation: int,
        kind: str,
        config_json: str,
        parent_ids: list[str],
        tokenizer_family: str,
        base_model_id: str,
        adapter_path: str | None = None,
        merged_checkpoint_path: str | None = None,
    ) -> None:
        """Insert/update an openweight member.

        ``config_json`` holds the kind-specific blob (SftConfig,
        MergeRecipe, or LoraMergeRecipe serialized as JSON). See
        ``openweight/genome.py`` for the schema.
        """
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO members
                (id, generation, config_json, val_loss, parent_ids,
                 checkpoint_path, created_at,
                 kind, tokenizer_family, base_model_id,
                 adapter_path, merged_checkpoint_path)
            VALUES (?, ?, ?, NULL, ?, NULL, ?, ?, ?, ?, ?, ?)
            """,
            (
                member_id,
                generation,
                config_json,
                json.dumps(parent_ids),
                time.time(),
                kind,
                tokenizer_family,
                base_model_id,
                adapter_path,
                merged_checkpoint_path,
            ),
        )
        self.conn.commit()

    def record_fitness_eval(
        self,
        member_id: str,
        benchmark_name: str,
        score: float,
        wallclock_s: float,
        raw: dict[str, Any] | None = None,
    ) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO fitness_evals
                (member_id, benchmark_name, score, wallclock_s, raw_json, evaluated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                member_id,
                benchmark_name,
                score,
                wallclock_s,
                json.dumps(raw) if raw is not None else None,
                time.time(),
            ),
        )
        self.conn.commit()

    def get_fitness_evals(self, member_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM fitness_evals WHERE member_id = ?", (member_id,)
        ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            if d.get("raw_json"):
                d["raw"] = json.loads(d.pop("raw_json"))
            else:
                d.pop("raw_json", None)
            out.append(d)
        return out

    def record_merge_recipe(
        self,
        recipe_id: str,
        member_id: str,
        method: str,
        yaml_text: str,
        coefficients: list[float],
        cma_es_step: int | None = None,
    ) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO merge_recipes
                (recipe_id, member_id, method, yaml_text,
                 coefficients_json, cma_es_step, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                recipe_id,
                member_id,
                method,
                yaml_text,
                json.dumps(coefficients),
                cma_es_step,
                time.time(),
            ),
        )
        self.conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

    def __enter__(self) -> LineageDB:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
