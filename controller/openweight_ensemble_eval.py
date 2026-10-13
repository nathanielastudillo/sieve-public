"""Posthoc ensemble evaluation for the openweight track.

Reads per-question items out of ``fitness_evals.raw_json`` in a lineage
DB, aggregates ensemble answers by majority vote (tokenizer-agnostic —
works for homogeneous and cross-family rosters alike), and writes the
ensemble score back as a pseudo-member row with ``kind="ensemble"``.

Why majority vote instead of logit averaging
--------------------------------------------
Logit averaging requires aligned vocabularies or an EVA/DeePEn layer
(§1.4 of the inference-arch research). For the immediate question
"does the SFT'd ensemble beat the SFT'd best individual?", majority
vote over already-decoded answers is a cheap, correct, unbiased
estimator that needs no extra inference passes — the workers already
emitted per-question answers during the per-member eval stage.

Aggregation strategies
----------------------
- ``majority``          plurality over A/B/C/D per question; ties broken
                        lexicographically ("A" wins). Default.
- ``weighted_majority`` weight each member's vote by their individual
                        benchmark score (stronger members carry more).
                        Degenerates to ``majority`` when scores are tied.

Only implemented for MCQ-shape benchmarks today (``smoke_mcq`` raw has
``items=[{i, got, answer, raw}, ...]``). ``ifeval`` and ``mmlu_pro`` have
different raw shapes — add adapters when those benchmarks are wired.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_here = Path(__file__).resolve().parent
if str(_here.parent) not in sys.path:
    sys.path.insert(0, str(_here.parent))

logger = logging.getLogger("sieve.openweight.ensemble")


@dataclass
class MemberEval:
    member_id: str
    generation: int
    base_model_id: str
    tokenizer_family: str
    score: float
    items: list[dict]  # per-question [{i, got, answer, raw?}]


def _load_mcq_evals(
    db_path: str, benchmark: str,
) -> list[MemberEval]:
    """Read all (member, benchmark) rows whose raw_json is MCQ-shaped."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT m.id AS member_id, m.generation, m.base_model_id,
               m.tokenizer_family, m.kind,
               f.score, f.raw_json
          FROM fitness_evals f
          JOIN members m ON m.id = f.member_id
         WHERE f.benchmark_name = ?
           AND COALESCE(m.kind, 'ensemble') != 'ensemble'
        """,
        (benchmark,),
    ).fetchall()
    out: list[MemberEval] = []
    for r in rows:
        if not r["raw_json"]:
            continue
        try:
            raw = json.loads(r["raw_json"])
        except json.JSONDecodeError:
            continue
        items = raw.get("items") or []
        if not items or not all("got" in it and "answer" in it for it in items):
            continue
        out.append(MemberEval(
            member_id=r["member_id"],
            generation=r["generation"],
            base_model_id=r["base_model_id"] or "",
            tokenizer_family=r["tokenizer_family"] or "",
            score=float(r["score"]),
            items=items,
        ))
    conn.close()
    return out


def _majority_vote(
    members: list[MemberEval],
    strategy: str = "majority",
) -> tuple[float, list[dict]]:
    """Aggregate per-question votes across ``members``.

    Returns (ensemble_score, per_question_records).
    """
    if not members:
        return 0.0, []
    # Validate: all members should have identical question indices and
    # identical reference answers. smoke_mcq uses the same 20 items per
    # member; if counts differ it's likely one member ran proxy=True and
    # another didn't. Truncate to the shortest for safety.
    min_len = min(len(m.items) for m in members)
    if min_len == 0:
        return 0.0, []

    per_q: list[dict] = []
    correct = 0
    for i in range(min_len):
        answer = members[0].items[i]["answer"]
        votes = Counter()
        # Tiebreakers: weighted vs plain
        if strategy == "weighted_majority":
            for m in members:
                got = (m.items[i].get("got") or "").upper()
                if got in ("A", "B", "C", "D"):
                    votes[got] += float(m.score)
        else:
            for m in members:
                got = (m.items[i].get("got") or "").upper()
                if got in ("A", "B", "C", "D"):
                    votes[got] += 1
        if not votes:
            winner = ""
        else:
            # Lexicographic tiebreak for determinism
            max_vote = max(votes.values())
            winner = min(letter for letter, v in votes.items() if v == max_vote)
        ok = winner == answer
        correct += int(ok)
        per_q.append({
            "i": i, "answer": answer, "got": winner,
            "votes": dict(votes), "ok": ok,
        })
    score = correct / min_len
    return score, per_q


def _record_ensemble_member(
    db_path: str,
    *,
    ensemble_id: str,
    generation: int,
    member_ids: list[str],
    roster_name: str,
    strategy: str,
    benchmark: str,
    score: float,
    per_q: list[dict],
) -> None:
    """Write the ensemble as a pseudo-member (kind='ensemble') + its fitness_evals row.

    The ensemble's ``parent_ids`` contains every member that voted, so
    downstream lineage queries can recover the composition.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    config = {
        "roster": roster_name,
        "strategy": strategy,
        "benchmark": benchmark,
        "n_voters": len(member_ids),
    }
    cur.execute(
        """
        INSERT OR REPLACE INTO members
            (id, generation, config_json, val_loss, parent_ids,
             checkpoint_path, created_at, kind, tokenizer_family,
             base_model_id, adapter_path, merged_checkpoint_path)
        VALUES (?, ?, ?, NULL, ?, NULL, ?, 'ensemble', 'mixed', NULL, NULL, NULL)
        """,
        (
            ensemble_id, generation,
            json.dumps(config),
            json.dumps(member_ids),
            time.time(),
        ),
    )
    raw = {
        "n": len(per_q),
        "correct": sum(1 for q in per_q if q["ok"]),
        "strategy": strategy,
        "items": per_q,
        "member_ids": member_ids,
    }
    cur.execute(
        """
        INSERT OR REPLACE INTO fitness_evals
            (member_id, benchmark_name, score, wallclock_s, raw_json, evaluated_at)
        VALUES (?, ?, ?, 0.0, ?, ?)
        """,
        (ensemble_id, benchmark, float(score), json.dumps(raw), time.time()),
    )
    conn.commit()
    conn.close()


def aggregate_from_lineage(
    db_path: str,
    *,
    benchmark: str,
    roster_name: str,
    strategy: str = "majority",
    generation: Optional[int] = None,
    ensemble_id: Optional[str] = None,
) -> dict:
    """Run posthoc aggregation over all members of ``generation``.

    Returns a summary dict with the ensemble score and composition.
    """
    evals = _load_mcq_evals(db_path, benchmark)
    if generation is not None:
        evals = [e for e in evals if e.generation == generation]
    if not evals:
        return {"status": "no_members", "benchmark": benchmark, "generation": generation}

    best_individual = max(evals, key=lambda e: e.score)
    score, per_q = _majority_vote(evals, strategy=strategy)
    eid = ensemble_id or f"ensemble-{roster_name}-g{evals[0].generation}-{uuid.uuid4().hex[:8]}"
    _record_ensemble_member(
        db_path,
        ensemble_id=eid,
        generation=evals[0].generation,
        member_ids=[e.member_id for e in evals],
        roster_name=roster_name,
        strategy=strategy,
        benchmark=benchmark,
        score=score,
        per_q=per_q,
    )
    return {
        "status": "ok",
        "ensemble_id": eid,
        "benchmark": benchmark,
        "strategy": strategy,
        "n_voters": len(evals),
        "ensemble_score": score,
        "best_individual_score": best_individual.score,
        "best_individual_id": best_individual.member_id,
        "best_individual_base": best_individual.base_model_id,
        "delta_vs_best_individual": score - best_individual.score,
        "member_scores": {e.member_id: e.score for e in evals},
        "tokenizer_families": sorted({e.tokenizer_family for e in evals}),
    }


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lineage-db", required=True)
    ap.add_argument("--benchmark", default="smoke_mcq",
                    help="Benchmark name whose raw_json has MCQ items.")
    ap.add_argument("--roster", default="unknown",
                    help="Roster name used for this run (stored in config_json).")
    ap.add_argument("--strategy", default="majority",
                    choices=["majority", "weighted_majority"])
    ap.add_argument("--generation", type=int, default=None,
                    help="Limit to a single generation; defaults to all.")
    args = ap.parse_args()

    summary = aggregate_from_lineage(
        args.lineage_db,
        benchmark=args.benchmark,
        roster_name=args.roster,
        strategy=args.strategy,
        generation=args.generation,
    )
    print(json.dumps(summary, indent=2))
    if summary.get("status") != "ok":
        return 1
    logger.info(
        "ensemble=%.3f  best_individual=%.3f  Δ=%+.3f  voters=%d  families=%s",
        summary["ensemble_score"], summary["best_individual_score"],
        summary["delta_vs_best_individual"], summary["n_voters"],
        ",".join(summary["tokenizer_families"]),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
