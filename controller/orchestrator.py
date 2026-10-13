"""Main orchestration loop for SIEVE.

Coordinates the full evolutionary pipeline: initialize population, dispatch
training, evaluate ensemble combinations, select/reproduce, iterate until
convergence or budget exhausted.  The final artifact is an ensemble spec --
a JSON document listing the checkpoint paths and configs of the best-performing
ensemble of model checkpoints.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from controller.population import Member, Population
from controller.ensemble_eval import EnsembleEvaluator
from controller.lineage import LineageDB

logger = logging.getLogger(__name__)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    merged = dict(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


class SieveOrchestrator:
    """Main evolutionary loop controller.

    Loads configuration, initializes all subsystems, and runs the
    evolutionary training/evaluation/selection cycle.
    """

    def __init__(
        self,
        config_path: str,
        workers: list[dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            config_path: Path to the YAML config file (default.yaml).
                         Profile overrides are merged on top via --profile.
            workers: Optional list of worker node descriptors
                     [{"host": str, "port": int, "device": str}, ...].
                     If None, runs locally on the current machine.
        """
        self.config = self._load_config(config_path)
        self.workers = workers or []

        # Resolve paths relative to config file location
        config_dir = Path(config_path).parent.parent  # sieve/ root
        paths = self.config.get("paths", {})
        self.checkpoint_dir = str(config_dir / paths.get("checkpoint_dir", "checkpoints"))
        self.results_dir = str(config_dir / paths.get("results_dir", "results"))
        self.data_dir = str(config_dir / paths.get("corpus_dir", "corpus"))
        lineage_db_path = str(config_dir / paths.get("lineage_db", "results/lineage.db"))

        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        # Extract evolution params
        evo = self.config.get("evolution", {})
        self.population_size = evo.get("population_size", 32)
        self.ensemble_size = evo.get("ensemble_size", 8)
        self.elite_fraction = evo.get("elite_fraction", 0.25)
        self.immigration_fraction = evo.get("immigration_fraction", 0.1)
        self.mutation_strength = evo.get("mutation_strength", 0.2)
        self.max_generations = evo.get("max_generations", 200)
        self.stagnation_patience = evo.get("stagnation_patience", 20)
        self.eval_combinations = evo.get("eval_combinations", 5000)

        # Device
        device = "cuda" if _cuda_available() else "cpu"
        ens_cfg = self.config.get("ensemble", {})
        eval_batch_size = ens_cfg.get("eval_batch_size", 128)
        # Default aggregator is ε-smoothed logit mean (lit-review rec #2) —
        # floors per-member probabilities to kill geometric-mean blowup on
        # overconfident memorizers. See docs/ensemble_lit_review.md.
        aggregator = ens_cfg.get("aggregator", "logit_mean_smoothed")
        smoothing_eps = float(ens_cfg.get("smoothing_eps", 0.05))
        self.aggregator = aggregator
        self.smoothing_eps = smoothing_eps

        # Initialize subsystems
        search_space = self.config.get("search_space", {})
        self.population = Population(search_space, self.population_size)
        model_block_size = self.config.get("model", {}).get("block_size", 512)
        self.evaluator = EnsembleEvaluator(
            checkpoint_dir=self.checkpoint_dir,
            data_dir=self.data_dir,
            device=device,
            batch_size=eval_batch_size,
            block_size=model_block_size,
            aggregator=aggregator,
            smoothing_eps=smoothing_eps,
        )
        self.lineage = LineageDB(lineage_db_path)

        # Track convergence
        self._best_losses: list[float] = []

    def _load_config(self, config_path: str) -> dict[str, Any]:
        """Load default config and merge with profile override if present."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(path) as f:
            config = yaml.safe_load(f) or {}

        return config

    def load_profile(self, profile_name: str) -> None:
        """Merge a profile override on top of the current config.

        Profiles live in config/profiles/<name>.yaml relative to the
        default config's directory.
        """
        config_root = Path(self.config.get("_config_path", "config"))
        profile_path = config_root / "profiles" / f"{profile_name}.yaml"

        if not profile_path.exists():
            # Try relative to cwd
            profile_path = Path("config") / "profiles" / f"{profile_name}.yaml"

        if profile_path.exists():
            with open(profile_path) as f:
                override = yaml.safe_load(f) or {}
            self.config = _deep_merge(self.config, override)

            # Re-extract evolution params after merge
            evo = self.config.get("evolution", {})
            self.population_size = evo.get("population_size", self.population_size)
            self.ensemble_size = evo.get("ensemble_size", self.ensemble_size)
            self.elite_fraction = evo.get("elite_fraction", self.elite_fraction)
            self.immigration_fraction = evo.get("immigration_fraction", self.immigration_fraction)
            self.max_generations = evo.get("max_generations", self.max_generations)
            self.stagnation_patience = evo.get("stagnation_patience", self.stagnation_patience)
            self.eval_combinations = evo.get("eval_combinations", self.eval_combinations)

            self.population.size = self.population_size

            # Re-extract search_space after merge so profile overrides
            # actually take effect on sampling/mutation
            self.population.search_space = self.config.get(
                "search_space", self.population.search_space
            )

            # Re-extract model.block_size so ensemble_eval uses the profile's
            # context window (evaluator was constructed before profile merge).
            self.evaluator._block_size_override = self.config.get("model", {}).get(
                "block_size", self.evaluator._block_size_override
            )

            logger.info("Loaded profile: %s", profile_name)
        else:
            logger.warning("Profile not found: %s", profile_name)

    def run(self) -> dict[str, Any]:
        """Execute the full evolutionary loop.

        Returns the best ensemble spec as a dict.
        """
        logger.info("=" * 60)
        logger.info("SIEVE — Solomonoff-Inspired Evolutionary Ensemble")
        logger.info("=" * 60)
        logger.info("Population size: %d", self.population_size)
        logger.info("Ensemble size: %d", self.ensemble_size)
        logger.info("Max generations: %d", self.max_generations)
        logger.info("Checkpoint dir: %s", self.checkpoint_dir)
        logger.info("Data dir: %s", self.data_dir)
        logger.info("Workers: %s", "local" if not self.workers else f"{len(self.workers)} nodes")
        logger.info("=" * 60)

        run_start = time.time()

        # --- Phase 1: Initialize population (Generation 0) ---
        logger.info("\n[Generation 0] Initializing population...")
        members = self.population.initialize()
        for m in members:
            self.lineage.record_member(m)

        generation = 0
        best_ensemble_spec: dict[str, Any] = {}

        while generation <= self.max_generations:
            gen_start = time.time()

            # --- Train all members ---
            logger.info("\n[Generation %d] Training %d members...", generation, len(members))
            if getattr(self, "_dispatcher", None) is not None:
                members = self.dispatch_training_redis(members)
            else:
                members = self.dispatch_training(members)

            # Record trained members
            for m in members:
                self.lineage.record_member(m)

            # --- Ensemble evaluation ---
            logger.info("[Generation %d] Evaluating ensemble combinations...", generation)
            trainable = [m for m in members if m.checkpoint_path is not None]

            if len(trainable) >= self.ensemble_size:
                eval_ensemble_size = self.ensemble_size
            else:
                eval_ensemble_size = max(1, len(trainable))

            # Caruana 2004 greedy with replacement — default as of 2026-04-18.
            # See controller/ensemble_eval.py._caruana_greedy_combinations for
            # rationale. Brute-force C(N,K) is intractable for any N>12, and
            # random sampling misses the high-weight regions of the combo
            # space. Caruana is O(S·k·N) and has 22 years of AutoML validation.
            ensemble_rankings = self.evaluator.evaluate_all_combinations(
                trainable,
                ensemble_size=eval_ensemble_size,
                max_combinations=self.eval_combinations,
                strategy="caruana_greedy",
            )

            # Record ensembles in lineage DB
            for combo in ensemble_rankings[:100]:  # store top 100 per generation
                self.lineage.record_ensemble(combo, generation)

            # --- Generation stats ---
            gen_elapsed = time.time() - gen_start
            best_ens_loss = ensemble_rankings[0]["ensemble_val_loss"] if ensemble_rankings else None
            best_ind_loss = min(
                (m.val_loss for m in members if m.val_loss is not None),
                default=None,
            )

            # Population diversity: average pairwise config distance
            pop_diversity = self._compute_population_diversity(members)

            # Sequestered meta-val diagnostic — evaluate the picked top
            # ensemble on a slice of val.bin the selector never touched.
            # The selection-val loss is a biased estimator of the
            # ensemble's true held-out loss because the Caruana greedy
            # search optimizes directly against it; meta-val is the
            # unbiased canary. See lab-notes/research-questions/
            # literature-small-corpus-ensembles.md §8.2 and §9 (P0).
            # Strictly read-only — NEVER feeds back into selection.
            meta_val_loss: float | None = None
            if ensemble_rankings:
                try:
                    top_member_ids = ensemble_rankings[0]["member_ids"]
                    meta_val_loss = self.evaluator.compute_meta_val_loss(
                        top_member_ids, trainable,
                    )
                    if meta_val_loss == float("inf"):
                        meta_val_loss = None  # meta-val empty / skipped
                except Exception:
                    # Never let diagnostic failure break the training loop.
                    logger.exception(
                        "[Generation %d] meta-val diagnostic failed",
                        generation,
                    )
                    meta_val_loss = None

            gen_stats = {
                "best_ensemble_loss": best_ens_loss,
                "best_individual_loss": best_ind_loss,
                "population_diversity": pop_diversity,
                "wallclock_seconds": gen_elapsed,
                "meta_val_loss": meta_val_loss,
            }
            self.lineage.record_generation(generation, gen_stats)

            meta_str = (
                f"{meta_val_loss:.4f}" if meta_val_loss is not None else "n/a"
            )
            logger.info(
                "[Generation %d] Best ensemble loss: %.4f | Meta-val: %s | "
                "Best individual: %.4f | Diversity: %.4f | Time: %.1fs",
                generation,
                best_ens_loss or float("inf"),
                meta_str,
                best_ind_loss or float("inf"),
                pop_diversity,
                gen_elapsed,
            )

            if best_ens_loss is not None:
                self._best_losses.append(best_ens_loss)

            # Update best ensemble spec.
            # We track GLOBAL-best across generations, not per-gen-top-1.
            # findings.md 2026-04-17 documents the bug this replaces:
            # exp-002 gen-0 best=9.39 → gen-1 best=8.68 → gen-2 best=8.82.
            # The old code overwrote best_ensemble_spec each generation, so
            # best_ensemble.json reported 8.82 (gen 2) when the true run-best
            # was 8.68 (gen 1). Evolutionary runs are non-monotonic at small
            # population sizes — track the best seen so far, not the last.
            # Also see literature-small-corpus-ensembles.md §9 P1-#8.
            if ensemble_rankings:
                top = ensemble_rankings[0]
                member_map = {m.id: m for m in members}
                candidate = {
                    "ensemble_id": f"sieve-gen-{generation}-top1",
                    "members": [
                        {
                            "checkpoint": member_map[mid].checkpoint_path,
                            "config": member_map[mid].config,
                            "val_loss": member_map[mid].val_loss,
                            "id": mid,
                        }
                        for mid in top["member_ids"]
                        if mid in member_map
                    ],
                    "ensemble_size": len(top["member_ids"]),
                    "val_loss": top["ensemble_val_loss"],
                    "diversity_score": top["diversity_score"],
                    "aggregation": self.aggregator,
                    "smoothing_eps": self.smoothing_eps,
                    "generation": generation,
                }
                prev_best = best_ensemble_spec.get("val_loss", float("inf"))
                if candidate["val_loss"] < prev_best:
                    best_ensemble_spec = candidate
                    logger.info(
                        "[Generation %d] New global-best ensemble: %.4f "
                        "(previous best: %.4f)",
                        generation, candidate["val_loss"], prev_best,
                    )
                else:
                    logger.info(
                        "[Generation %d] Current gen best %.4f did not "
                        "beat global best %.4f (gen %s) — keeping previous.",
                        generation, candidate["val_loss"], prev_best,
                        best_ensemble_spec.get("generation", "?"),
                    )

            # --- Convergence check ---
            if self.check_convergence(generation):
                logger.info(
                    "\n[Generation %d] Converged — stopping early.", generation
                )
                break

            # --- Selection & reproduction ---
            if generation < self.max_generations:
                logger.info("[Generation %d] Evolving population...", generation)
                members = self.population.select(
                    members,
                    ensemble_rankings,
                    elite_fraction=self.elite_fraction,
                    immigration_fraction=self.immigration_fraction,
                )
                for m in members:
                    self.lineage.record_member(m)

            generation += 1

        # --- Completion ---
        total_time = time.time() - run_start
        logger.info("\n" + "=" * 60)
        logger.info("SIEVE run complete in %.1f seconds (%d generations)", total_time, generation)

        if best_ensemble_spec:
            spec_path = Path(self.results_dir) / "best_ensemble.json"
            spec_path.write_text(json.dumps(best_ensemble_spec, indent=2))
            logger.info("Best ensemble spec written to: %s", spec_path)
            logger.info("Best ensemble val loss: %.4f", best_ensemble_spec["val_loss"])
            logger.info("Ensemble members: %d", best_ensemble_spec["ensemble_size"])
            logger.info("\n%s", json.dumps(best_ensemble_spec, indent=2))
        else:
            logger.warning("No ensembles were evaluated.")

        logger.info("=" * 60)
        return best_ensemble_spec

    def dispatch_training(self, members: list[Member]) -> list[Member]:
        """Train each member and populate checkpoint_path / val_loss.

        Currently runs sequentially on the local GPU.  The worker list is
        accepted in __init__ for future distributed dispatch via HTTP.

        Args:
            members: Members with configs but no checkpoints yet.

        Returns:
            Same members with checkpoint_path and val_loss filled in.
        """
        try:
            from worker.trainer import train_model
        except ImportError:
            logger.warning(
                "worker module not available; using stub training. "
                "Members will get dummy val_loss and no checkpoints."
            )
            return self._stub_training(members)

        for i, member in enumerate(members):
            logger.info(
                "  Training member %d/%d [%s] (gen %d)",
                i + 1, len(members), member.id, member.generation,
            )

            # Build training config from member's evolved hyperparams
            # Merge model defaults with member-specific overrides
            model_defaults = self.config.get("model", {})
            training_defaults = self.config.get("training", {})

            # Member config can override both model and training params
            model_params = dict(model_defaults)
            train_params = dict(training_defaults)

            model_keys = {"n_layer", "n_head", "n_embd", "block_size", "dropout", "bias", "vocab_size"}
            for key, val in member.config.items():
                if key in model_keys:
                    model_params[key] = val
                else:
                    train_params[key] = val

            try:
                # Build train config dict with correct field names
                train_cfg_dict = {
                    "epochs": int(train_params.get("epochs", 50)),
                    "batch_size": int(train_params.get("batch_size", 64)),
                    "learning_rate": float(train_params.get("learning_rate", 3e-4)),
                    "weight_decay": float(train_params.get("weight_decay", 0.3)),
                    "warmup_fraction": float(train_params.get("warmup_fraction", 0.05)),
                    "lr_schedule": str(train_params.get("lr_schedule", "cosine")),
                    "optimizer_name": str(train_params.get("optimizer", "muon")),
                    "gradient_clip": float(train_params.get("gradient_clip", 1.0)),
                    "shuffle_per_epoch": True,
                    "seed": hash(member.id) % (2**31),
                    "checkpoint_dir": os.path.join(
                        self.checkpoint_dir,
                        f"gen-{member.generation:03d}-{member.id}",
                    ),
                    "checkpoint_every": int(train_params.get("epochs", 50)),  # only final
                }

                result = train_model(model_params, train_cfg_dict, self.data_dir)

                # train_model already saves checkpoints; use the last one
                if result.checkpoint_paths:
                    member.checkpoint_path = result.checkpoint_paths[-1]
                member.val_loss = result.best_loss

            except Exception:
                logger.exception("Training failed for member %s", member.id)
                member.val_loss = float("inf")

        return members

    def dispatch_training_redis(self, members: list[Member]) -> list[Member]:
        """Parallel dispatch: enqueue one job per member, block on completion.

        Requires ``self._dispatcher`` (RedisDispatcher) and
        ``self._dispatch_cfg`` dict (run_id, remote_corpus_dir,
        remote_checkpoint_dir) to be set by ``main``.
        """
        from controller.dispatch import make_job

        cfg = self._dispatch_cfg
        dispatcher = self._dispatcher

        model_defaults = self.config.get("model", {})
        training_defaults = self.config.get("training", {})
        model_keys = {"n_layer", "n_head", "n_embd", "block_size",
                      "dropout", "bias", "vocab_size"}

        jobs = []
        member_ids = []
        for member in members:
            model_params = dict(model_defaults)
            train_params = dict(training_defaults)
            for key, val in member.config.items():
                if key in model_keys:
                    model_params[key] = val
                else:
                    train_params[key] = val

            train_cfg = {
                "epochs": int(train_params.get("epochs", 50)),
                "batch_size": int(train_params.get("batch_size", 64)),
                "learning_rate": float(train_params.get("learning_rate", 3e-4)),
                "weight_decay": float(train_params.get("weight_decay", 0.3)),
                "warmup_fraction": float(train_params.get("warmup_fraction", 0.05)),
                "lr_schedule": str(train_params.get("lr_schedule", "cosine")),
                "optimizer_name": str(train_params.get("optimizer", "muon")),
                "gradient_clip": float(train_params.get("gradient_clip", 1.0)),
                "shuffle_per_epoch": True,
                "seed": hash(member.id) % (2**31),
                "checkpoint_every": int(train_params.get("epochs", 50)),
            }

            job = make_job(
                run_id=cfg["run_id"],
                generation=member.generation,
                member_id=member.id,
                model_config=model_params,
                train_config=train_cfg,
                corpus_dir=cfg["remote_corpus_dir"],
                checkpoint_dir=cfg["remote_checkpoint_dir"],
                results_dir=cfg.get("remote_results_dir"),
            )
            jobs.append(job)
            member_ids.append(member.id)

        t0 = time.time()
        dispatcher.submit_generation(jobs)
        logger.info(
            "  Dispatched %d jobs (gen %d). Waiting on workers: %s",
            len(jobs), members[0].generation if members else -1,
            [w[0] for w in dispatcher.list_workers()],
        )

        results = dispatcher.wait_for_generation(
            run_id=cfg["run_id"],
            generation=members[0].generation,
            member_ids=member_ids,
            timeout_s=float(cfg.get("job_timeout_s", 7200.0)),
        )
        wall = time.time() - t0

        import re
        import subprocess
        ssh_pass = cfg.get("ssh_pass", os.environ.get("SSH_PASS"))
        ssh_user = cfg.get("ssh_user", os.environ.get("SSH_USER", "<user>"))
        local_ckpt_root = Path(self.checkpoint_dir)
        local_ckpt_root.mkdir(parents=True, exist_ok=True)

        for member in members:
            msg = results.get(member.id)
            if msg is None or msg.status != "ok":
                err = getattr(msg, "error", "unknown") if msg else "no result"
                logger.error("  Member %s failed: %s", member.id, err)
                member.val_loss = float("inf")
                continue
            member.val_loss = msg.val_loss if msg.val_loss is not None else float("inf")
            remote_path = msg.checkpoint_path
            if remote_path and msg.worker_id:
                # worker_id = "<node>-<pid>" — extract node label
                node = re.sub(r"-\d+$", "", msg.worker_id)
                # If the worker already encoded a user (user@host), don't re-prepend.
                target_host = node if "@" in node else f"{ssh_user}@{node}"
                local_dir = local_ckpt_root / member.id
                local_dir.mkdir(parents=True, exist_ok=True)
                local_path = local_dir / Path(remote_path).name
                try:
                    subprocess.run(
                        ["rsync", "-az",
                         "-e", "ssh -o StrictHostKeyChecking=no",
                         f"{target_host}:{remote_path}", str(local_path)],
                        check=True, timeout=120,
                    )
                    member.checkpoint_path = str(local_path)
                except Exception as e:
                    logger.error("  Could not pull checkpoint for %s from %s: %s",
                                 member.id, target_host, e)
                    # Keep the successful val_loss so best_individual still reports
                    # correctly; checkpoint_path stays None so this member is
                    # excluded from ensemble eval (can't load without file).
                    member.checkpoint_path = None
            else:
                member.checkpoint_path = None

        ok = sum(1 for m in members if m.val_loss != float("inf"))
        logger.info(
            "  Generation complete: %d/%d ok in %.1fs (avg %.1fs/member)",
            ok, len(members), wall, wall / max(1, len(members)),
        )
        return members

    def _stub_training(self, members: list[Member]) -> list[Member]:
        """Dummy training for testing when the worker module is unavailable.

        Assigns random val_loss values and dummy checkpoint paths so the
        rest of the pipeline can be exercised.
        """
        import random

        for member in members:
            member.val_loss = 2.0 + random.random() * 3.0
            member.checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"gen-{member.generation:03d}-member-{member.id}.pt",
            )

        return members

    def check_convergence(self, generation: int) -> bool:
        """Check if the run should stop early.

        Criteria:
        1. Best ensemble loss hasn't improved in stagnation_patience generations.
        2. Population diversity has collapsed below a threshold.
        """
        if generation < self.stagnation_patience:
            return False

        # Check loss stagnation
        if len(self._best_losses) >= self.stagnation_patience:
            recent = self._best_losses[-self.stagnation_patience:]
            best_recent = min(recent)
            older = self._best_losses[:-self.stagnation_patience]
            if older:
                best_older = min(older)
                # No improvement (or negligible improvement < 0.1%)
                if best_recent >= best_older * 0.999:
                    logger.info(
                        "Stagnation detected: best recent %.4f vs best older %.4f "
                        "(patience=%d)",
                        best_recent, best_older, self.stagnation_patience,
                    )
                    return True

        return False

    def _compute_population_diversity(self, members: list[Member]) -> float:
        """Average pairwise config distance across the population.

        Uses a subsample for large populations to keep it O(n) instead of O(n^2).
        """
        import random as rng

        if len(members) < 2:
            return 0.0

        # Subsample for large populations
        sample = members if len(members) <= 50 else rng.sample(members, 50)

        total_dist = 0.0
        n_pairs = 0
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                total_dist += self.population._config_distance(
                    sample[i].config, sample[j].config
                )
                n_pairs += 1

        return total_dist / n_pairs if n_pairs > 0 else 0.0

    def close(self) -> None:
        """Clean up resources."""
        self.lineage.close()


def _cuda_available() -> bool:
    """Check if CUDA is available without importing torch at module level."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def main() -> None:
    """CLI entry point for SIEVE orchestrator."""
    parser = argparse.ArgumentParser(
        description="SIEVE — Solomonoff-Inspired Evolutionary Ensemble trainer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m controller.orchestrator --config config/default.yaml\n"
            "  python -m controller.orchestrator --config config/default.yaml --profile small\n"
            "  python -m controller.orchestrator --config config/default.yaml --corpus-dir /data/corpus\n"
        ),
    )

    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to the YAML config file (default: config/default.yaml)",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Profile name to merge on top of default config (e.g. small, medium, large, full)",
    )
    parser.add_argument(
        "--corpus-dir",
        default=None,
        help="Override the corpus/data directory path",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Override the checkpoint directory path",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Override the results directory path",
    )
    parser.add_argument(
        "--dispatch",
        default=None,
        help=(
            "Redis URL for parallel dispatch (e.g. redis://<node-ip>:6379/0). "
            "When set, training jobs are enqueued to a worker pool instead of "
            "run sequentially on the local GPU."
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Identifier prefix for this run when using --dispatch (default: auto)",
    )
    parser.add_argument(
        "--remote-sieve-root",
        default=None,
        help=(
            "Sieve repo root path on worker nodes (e.g. /home/<user>/sieve). "
            "Required with --dispatch: paths in jobs are relative to this on workers."
        ),
    )
    parser.add_argument(
        "--remote-corpus-dir",
        default=None,
        help="Corpus path relative to --remote-sieve-root on workers (overrides local --corpus-dir for job payloads)",
    )
    parser.add_argument(
        "--remote-checkpoint-dir",
        default=None,
        help="Checkpoint path relative to --remote-sieve-root on workers",
    )
    parser.add_argument(
        "--ssh-user",
        default=None,
        help="SSH user for pulling checkpoints back from workers (default: $SSH_USER or '<user>')",
    )
    parser.add_argument(
        "--ssh-pass",
        default=None,
        help="SSH password for worker rsync (required: set $SSH_PASS or pass --ssh-pass)",
    )
    parser.add_argument(
        "--job-timeout-s",
        type=float,
        default=None,
        help="Per-generation wait deadline for worker results in seconds "
             "(default: $SIEVE_JOB_TIMEOUT_S or 7200). Must exceed single-member "
             "training wallclock; for corpus_v009 (4.3M tok) empirically 5-11h.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Create orchestrator
    orchestrator = SieveOrchestrator(config_path=args.config)

    # Apply profile override
    if args.profile:
        orchestrator.load_profile(args.profile)

    # Apply CLI overrides
    if args.corpus_dir:
        orchestrator.data_dir = args.corpus_dir
        # Propagate to evaluator (already constructed in __init__)
        orchestrator.evaluator.data_dir = args.corpus_dir
    if args.checkpoint_dir:
        orchestrator.checkpoint_dir = args.checkpoint_dir
        orchestrator.evaluator.checkpoint_dir = args.checkpoint_dir
        os.makedirs(orchestrator.checkpoint_dir, exist_ok=True)
    if args.results_dir:
        orchestrator.results_dir = args.results_dir
        os.makedirs(orchestrator.results_dir, exist_ok=True)
        # Rebuild LineageDB at the overridden path so lineage/records land
        # alongside the run's other artifacts, not in the default config dir.
        orchestrator.lineage = LineageDB(
            os.path.join(orchestrator.results_dir, "lineage.db")
        )

    # Parallel dispatch: attach Redis dispatcher before starting the run.
    if args.dispatch:
        if not args.remote_sieve_root:
            logger.error("--dispatch requires --remote-sieve-root")
            sys.exit(2)
        from controller.dispatch import RedisDispatcher, new_run_id
        dispatcher = RedisDispatcher(args.dispatch)
        run_id = args.run_id or new_run_id()
        orchestrator._dispatcher = dispatcher  # consumed by dispatch_training_redis
        orchestrator._dispatch_cfg = {
            "run_id": run_id,
            "remote_sieve_root": args.remote_sieve_root,
            "remote_corpus_dir": args.remote_corpus_dir or args.corpus_dir or "data/corpus",
            "remote_checkpoint_dir": args.remote_checkpoint_dir or args.checkpoint_dir or "runs/current/checkpoints",
            "remote_results_dir": args.results_dir,
            "job_timeout_s": float(
                args.job_timeout_s
                if args.job_timeout_s is not None
                else os.environ.get("SIEVE_JOB_TIMEOUT_S", 7200.0)
            ),
            "ssh_user": args.ssh_user or os.environ.get("SSH_USER", "<user>"),
            "ssh_pass": args.ssh_pass or os.environ.get("SSH_PASS"),
        }
        logger.info("Parallel dispatch enabled: redis=%s run_id=%s workers=%d",
                    args.dispatch, run_id, len(dispatcher.list_workers()))

    # Run
    try:
        best = orchestrator.run()
        if best:
            sys.exit(0)
        else:
            logger.error("No ensemble was produced.")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user.")
        sys.exit(130)
    except Exception:
        logger.exception("Fatal error in SIEVE orchestrator")
        sys.exit(1)
    finally:
        orchestrator.close()


if __name__ == "__main__":
    main()
