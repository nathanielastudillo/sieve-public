"""SIEVE dashboard backend -- FastAPI server with WebSocket for realtime updates.

Reads from the lineage.db SQLite database (written by the SIEVE orchestrator)
and serves:
  - REST endpoints for generation history, members, ensembles, lineage, hyperparams
  - WebSocket endpoint that pushes updates when new data appears
  - Static files for the React frontend SPA

Usage:
    python -m dashboard.server --db ./results/lineage.db --port 8096
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sqlite3
import statistics
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import ast
import os
import re
import shlex
import subprocess
import threading
import urllib.error
import urllib.request
import uuid

from fastapi import Body, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

logger = logging.getLogger("sieve.dashboard")

# ---------------------------------------------------------------------------
# Globals set at startup from CLI args
# ---------------------------------------------------------------------------

DB_PATH: str = "./results/lineage.db"
CHECKPOINT_DIR: Optional[str] = None
FRONTEND_DIR: Path = Path(__file__).parent / "frontend" / "dist"
SIEVE_ROOT: Path = Path(__file__).parent.parent  # for launch paths
LAUNCH_DISPATCH_CMD: Optional[str] = None  # when set, spawn launches via this wrapper


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _get_connection() -> Optional[sqlite3.Connection]:
    """Open the lineage DB in read-only mode.

    Returns None if the database file does not exist, allowing endpoints
    to return empty data instead of crashing.
    """
    db_file = Path(DB_PATH)
    if not db_file.exists():
        return None
    try:
        uri = f"file:{db_file.resolve()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.OperationalError:
        logger.warning("Could not open database at %s", DB_PATH)
        return None


def _query(sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    """Execute a read query and return results as a list of dicts."""
    conn = _get_connection()
    if conn is None:
        return []
    try:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError as exc:
        logger.warning("Query failed: %s -- %s", sql.strip()[:80], exc)
        return []
    finally:
        conn.close()


def _query_one(sql: str, params: tuple = ()) -> Optional[dict[str, Any]]:
    """Execute a read query and return the first row or None."""
    conn = _get_connection()
    if conn is None:
        return None
    try:
        row = conn.execute(sql, params).fetchone()
        return dict(row) if row else None
    except sqlite3.OperationalError as exc:
        logger.warning("Query failed: %s -- %s", sql.strip()[:80], exc)
        return None
    finally:
        conn.close()


def _scalar(sql: str, params: tuple = (), default: Any = None) -> Any:
    """Execute a query expecting a single scalar value."""
    conn = _get_connection()
    if conn is None:
        return default
    try:
        row = conn.execute(sql, params).fetchone()
        return row[0] if row else default
    except sqlite3.OperationalError:
        return default
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# WebSocket client manager
# ---------------------------------------------------------------------------

class ConnectionManager:
    """Track active WebSocket connections and broadcast updates."""

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.append(ws)
        logger.info("WebSocket client connected (%d total)", len(self._connections))

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._connections:
            self._connections.remove(ws)
        logger.info("WebSocket client disconnected (%d remaining)", len(self._connections))

    async def broadcast(self, message: dict[str, Any]) -> None:
        payload = json.dumps(message)
        stale: list[WebSocket] = []
        for ws in self._connections:
            try:
                await ws.send_text(payload)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self.disconnect(ws)

    @property
    def has_clients(self) -> bool:
        return len(self._connections) > 0


manager = ConnectionManager()


# ---------------------------------------------------------------------------
# Background poller -- detects new data and pushes via WebSocket
# ---------------------------------------------------------------------------

_poll_task: Optional[asyncio.Task] = None
_last_known_generation: Optional[int] = None
_last_known_member_count: int = 0
_last_known_ensemble_count: int = 0

POLL_INTERVAL_SECONDS = 2.0


async def _poll_loop() -> None:
    """Periodically check the DB for new data and broadcast changes."""
    global _last_known_generation, _last_known_member_count, _last_known_ensemble_count

    while True:
        await asyncio.sleep(POLL_INTERVAL_SECONDS)

        if not manager.has_clients:
            continue

        try:
            current_gen = _scalar(
                "SELECT MAX(num) FROM generations", default=None
            )
            member_count = _scalar(
                "SELECT COUNT(*) FROM members", default=0
            )
            ensemble_count = _scalar(
                "SELECT COUNT(*) FROM ensembles", default=0
            )

            gen_changed = current_gen is not None and current_gen != _last_known_generation
            members_changed = member_count != _last_known_member_count
            ensembles_changed = ensemble_count != _last_known_ensemble_count

            if gen_changed:
                _last_known_generation = current_gen
                _last_known_member_count = member_count
                _last_known_ensemble_count = ensemble_count

                # Send both generation data and status update
                gen_data = _query_one(
                    "SELECT * FROM generations WHERE num = ?",
                    (current_gen,),
                )
                if gen_data:
                    await manager.broadcast({
                        "type": "generation_update",
                        "data": gen_data,
                    })

                status = _build_status()
                await manager.broadcast({
                    "type": "status_update",
                    "data": status,
                })

            elif members_changed or ensembles_changed:
                _last_known_member_count = member_count
                _last_known_ensemble_count = ensemble_count

                status = _build_status()
                await manager.broadcast({
                    "type": "status_update",
                    "data": status,
                })

        except Exception:
            logger.exception("Error in poll loop")


# ---------------------------------------------------------------------------
# Data builders
# ---------------------------------------------------------------------------

def _build_status() -> dict[str, Any]:
    """Build the current run status payload."""
    current_generation = _scalar("SELECT MAX(num) FROM generations", default=None)
    total_members = _scalar("SELECT COUNT(*) FROM members", default=0)
    total_ensembles = _scalar("SELECT COUNT(*) FROM ensembles", default=0)

    best_ensemble_loss = _scalar(
        "SELECT MIN(best_ensemble_loss) FROM generations"
    )
    best_individual_loss = _scalar(
        "SELECT MIN(best_individual_loss) FROM generations"
    )

    # Elapsed time: first generation's created_at to latest
    first_created = _scalar(
        "SELECT MIN(created_at) FROM generations"
    )
    latest_created = _scalar(
        "SELECT MAX(created_at) FROM generations"
    )
    elapsed_seconds = None
    if first_created is not None and latest_created is not None:
        elapsed_seconds = latest_created - first_created

    # Running: latest generation was created within the last 120 seconds
    is_running = False
    if latest_created is not None:
        is_running = (time.time() - latest_created) < 120.0

    # Population diversity from latest generation
    population_diversity = None
    if current_generation is not None:
        population_diversity = _scalar(
            "SELECT population_diversity FROM generations WHERE num = ?",
            (current_generation,),
        )

    return {
        "current_generation": current_generation,
        "total_members": total_members,
        "total_ensembles": total_ensembles,
        "best_ensemble_loss": best_ensemble_loss,
        "best_individual_loss": best_individual_loss,
        "elapsed_seconds": elapsed_seconds,
        "is_running": is_running,
        "population_diversity": population_diversity,
    }


def _parse_member_row(row: dict[str, Any]) -> dict[str, Any]:
    """Parse a raw member row into the API response format."""
    config_raw = row.get("config_json")
    config = {}
    if config_raw:
        try:
            config = json.loads(config_raw)
        except (json.JSONDecodeError, TypeError):
            config = {}

    parent_ids_raw = row.get("parent_ids")
    parent_ids: list[str] = []
    if parent_ids_raw:
        try:
            parent_ids = json.loads(parent_ids_raw)
        except (json.JSONDecodeError, TypeError):
            parent_ids = []

    return {
        "id": row["id"],
        "generation": row["generation"],
        "config": config,
        "val_loss": row.get("val_loss"),
        "parent_ids": parent_ids,
        "checkpoint_path": row.get("checkpoint_path"),
    }


def _parse_ensemble_row(row: dict[str, Any]) -> dict[str, Any]:
    """Parse a raw ensemble row into the API response format."""
    member_ids_raw = row.get("member_ids_json")
    member_ids: list[str] = []
    if member_ids_raw:
        try:
            member_ids = json.loads(member_ids_raw)
        except (json.JSONDecodeError, TypeError):
            member_ids = []

    return {
        "id": row["id"],
        "generation": row["generation"],
        "member_ids": member_ids,
        "ensemble_val_loss": row.get("ensemble_val_loss"),
        "diversity_score": row.get("diversity_score"),
    }


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

_reap_task: Optional[asyncio.Task] = None
_metrics_ingest_task: Optional[asyncio.Task] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start and stop the background poll task."""
    global _poll_task, _reap_task, _metrics_ingest_task, _last_known_generation, _last_known_member_count, _last_known_ensemble_count

    # Initialize last-known state so the poller doesn't immediately fire
    _last_known_generation = _scalar("SELECT MAX(num) FROM generations", default=None)
    _last_known_member_count = _scalar("SELECT COUNT(*) FROM members", default=0)
    _last_known_ensemble_count = _scalar("SELECT COUNT(*) FROM ensembles", default=0)

    _poll_task = asyncio.create_task(_poll_loop())
    _reap_task = asyncio.create_task(_reap_launches_loop())
    _metrics_ingest_task = asyncio.create_task(_launch_metrics_ingest_loop())
    logger.info("Dashboard poller + reaper + metrics ingest started (interval=%.1fs)", POLL_INTERVAL_SECONDS)

    # Discover serving_shim orphans across the fleet — shims left running
    # from before this dashboard process started. Background-threaded so a
    # slow fleet doesn't delay startup; ensure() will use whatever's been
    # discovered by the time it runs.
    asyncio.create_task(asyncio.to_thread(_populate_orphan_cache))

    yield

    for t in (_poll_task, _reap_task, _metrics_ingest_task):
        if t is not None:
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
    logger.info("Dashboard background tasks stopped")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SIEVE Dashboard",
    description="Realtime dashboard for the SIEVE evolutionary ensemble trainer",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/api/status")
async def get_status() -> JSONResponse:
    """Current run status."""
    return JSONResponse(_build_status())


@app.get("/api/generations")
async def get_generations() -> JSONResponse:
    """All generation history, sorted by generation number ascending."""
    rows = _query("SELECT * FROM generations ORDER BY num ASC")
    return JSONResponse(rows)


@app.get("/api/members")
async def get_members(generation: Optional[int] = Query(None)) -> JSONResponse:
    """Members for a given generation, or all members if no generation specified."""
    if generation is not None:
        rows = _query(
            "SELECT * FROM members WHERE generation = ? ORDER BY val_loss ASC",
            (generation,),
        )
    else:
        rows = _query("SELECT * FROM members ORDER BY generation ASC, val_loss ASC")

    return JSONResponse([_parse_member_row(r) for r in rows])


@app.get("/api/ensembles")
async def get_ensembles(
    generation: Optional[int] = Query(None),
    top_k: int = Query(10, ge=1),
) -> JSONResponse:
    """Top ensembles, optionally filtered by generation.

    Sorted by ensemble_val_loss ascending (best first).
    """
    if generation is not None:
        rows = _query(
            "SELECT * FROM ensembles WHERE generation = ? "
            "ORDER BY ensemble_val_loss ASC LIMIT ?",
            (generation, top_k),
        )
    else:
        rows = _query(
            "SELECT * FROM ensembles ORDER BY ensemble_val_loss ASC LIMIT ?",
            (top_k,),
        )

    return JSONResponse([_parse_ensemble_row(r) for r in rows])


@app.get("/api/lineage/{member_id}")
async def get_lineage(member_id: str) -> JSONResponse:
    """Ancestry trace for a member via BFS through parent_ids.

    Returns array of member objects from newest to oldest ancestor.
    """
    lineage: list[dict[str, Any]] = []
    visited: set[str] = set()
    queue: list[str] = [member_id]

    conn = _get_connection()
    if conn is None:
        return JSONResponse([])

    try:
        while queue:
            mid = queue.pop(0)
            if mid in visited:
                continue
            visited.add(mid)

            row = conn.execute(
                "SELECT * FROM members WHERE id = ?", (mid,)
            ).fetchone()
            if row is None:
                continue

            parsed = _parse_member_row(dict(row))
            lineage.append(parsed)

            for parent_id in parsed["parent_ids"]:
                if parent_id not in visited:
                    queue.append(parent_id)
    finally:
        conn.close()

    return JSONResponse(lineage)


@app.get("/api/hyperparams")
async def get_hyperparams() -> JSONResponse:
    """Hyperparameter evolution across generations.

    For each generation, computes distribution stats for numeric hyperparams
    and category counts for categorical hyperparams.
    """
    rows = _query(
        "SELECT generation, config_json FROM members ORDER BY generation ASC"
    )

    if not rows:
        return JSONResponse({"numeric": {}, "categorical": {}})

    # Group configs by generation
    gen_configs: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        gen = row["generation"]
        try:
            config = json.loads(row["config_json"])
        except (json.JSONDecodeError, TypeError):
            continue
        gen_configs[gen].append(config)

    # Identify all hyperparameter keys and classify as numeric or categorical
    all_keys: set[str] = set()
    for configs in gen_configs.values():
        for config in configs:
            all_keys.update(config.keys())

    # Classify each key by inspecting actual values across the full population
    numeric_keys: set[str] = set()
    categorical_keys: set[str] = set()

    for key in all_keys:
        is_numeric = False
        for configs in gen_configs.values():
            for config in configs:
                val = config.get(key)
                if val is not None:
                    if isinstance(val, (int, float)) and not isinstance(val, bool):
                        is_numeric = True
                    else:
                        is_numeric = False
                    break
            if not is_numeric:
                break

        if is_numeric:
            numeric_keys.add(key)
        else:
            categorical_keys.add(key)

    # Build numeric stats per generation per parameter
    numeric_result: dict[str, list[dict]] = defaultdict(list)
    for gen in sorted(gen_configs.keys()):
        configs = gen_configs[gen]
        for key in sorted(numeric_keys):
            values = [
                c[key] for c in configs
                if key in c and isinstance(c[key], (int, float)) and not isinstance(c[key], bool)
            ]
            if not values:
                continue

            entry: dict[str, Any] = {
                "generation": gen,
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "std": statistics.stdev(values) if len(values) >= 2 else 0.0,
            }
            numeric_result[key].append(entry)

    # Build categorical counts per generation per parameter
    categorical_result: dict[str, list[dict]] = defaultdict(list)
    for gen in sorted(gen_configs.keys()):
        configs = gen_configs[gen]
        for key in sorted(categorical_keys):
            counts: dict[str, int] = defaultdict(int)
            for config in configs:
                if key in config:
                    val = config[key]
                    # Normalize value to string for counting
                    if isinstance(val, bool):
                        counts[str(val).lower()] += 1
                    elif isinstance(val, list):
                        counts[json.dumps(val)] += 1
                    else:
                        counts[str(val)] += 1

            if counts:
                categorical_result[key].append({
                    "generation": gen,
                    "counts": dict(counts),
                })

    return JSONResponse({
        "numeric": dict(numeric_result),
        "categorical": dict(categorical_result),
    })


@app.get("/api/population_scatter")
async def get_population_scatter(
    generation: Optional[int] = Query(None),
) -> JSONResponse:
    """Population scatter data for a given generation (default: latest).

    Returns members with val_loss, config summary, and ensemble contribution
    (how many of the top 50 ensembles each member appears in).
    """
    # Resolve generation
    if generation is None:
        generation = _scalar("SELECT MAX(num) FROM generations", default=None)
        if generation is None:
            # Fall back to max generation from members table
            generation = _scalar("SELECT MAX(generation) FROM members", default=None)
    if generation is None:
        return JSONResponse([])

    # Get members for this generation
    member_rows = _query(
        "SELECT * FROM members WHERE generation = ?",
        (generation,),
    )
    if not member_rows:
        return JSONResponse([])

    # Get top ensembles for this generation to compute ensemble contribution
    ensemble_rows = _query(
        "SELECT member_ids_json FROM ensembles WHERE generation = ? "
        "ORDER BY ensemble_val_loss ASC LIMIT 50",
        (generation,),
    )

    # Count how many top ensembles each member appears in
    contribution_counts: dict[str, int] = defaultdict(int)
    for erow in ensemble_rows:
        try:
            member_ids = json.loads(erow["member_ids_json"])
        except (json.JSONDecodeError, TypeError):
            continue
        for mid in member_ids:
            contribution_counts[mid] += 1

    # Build response
    result = []
    for row in member_rows:
        parsed = _parse_member_row(row)
        config = parsed["config"]

        # Build a compact config summary for tooltip display
        summary_keys = ["n_layer", "n_head", "n_embd", "dropout", "learning_rate",
                        "weight_decay", "optimizer", "lr_schedule"]
        config_summary = {
            k: config[k] for k in summary_keys if k in config
        }

        result.append({
            "id": parsed["id"],
            "val_loss": parsed["val_loss"],
            "config_summary": config_summary,
            "ensemble_contribution": contribution_counts.get(parsed["id"], 0),
        })

    return JSONResponse(result)


# ---------------------------------------------------------------------------
# Openweight — monitor endpoints
# ---------------------------------------------------------------------------

def _parse_openweight_member_row(row: dict[str, Any]) -> dict[str, Any]:
    """Parse a members row that carries openweight columns."""
    parsed = _parse_member_row(row)
    parsed.update({
        "kind": row.get("kind"),
        "tokenizer_family": row.get("tokenizer_family"),
        "base_model_id": row.get("base_model_id"),
        "adapter_path": row.get("adapter_path"),
        "merged_checkpoint_path": row.get("merged_checkpoint_path"),
        "created_at": row.get("created_at"),
    })
    return parsed


def _fitness_evals_by_member() -> dict[str, dict[str, dict[str, Any]]]:
    """Return {member_id: {bench_name: {score, wallclock_s, evaluated_at, raw}}}."""
    rows = _query("SELECT * FROM fitness_evals")
    out: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        raw = None
        if r.get("raw_json"):
            try:
                raw = json.loads(r["raw_json"])
            except (json.JSONDecodeError, TypeError):
                raw = None
        out[r["member_id"]][r["benchmark_name"]] = {
            "score": r["score"],
            "wallclock_s": r["wallclock_s"],
            "evaluated_at": r["evaluated_at"],
            "raw": raw,
        }
    return out


@app.get("/api/openweight/members")
async def get_openweight_members() -> JSONResponse:
    """All openweight members (kind IS NOT NULL), with fitness evals joined."""
    rows = _query(
        "SELECT * FROM members WHERE kind IS NOT NULL "
        "ORDER BY generation ASC, created_at ASC"
    )
    evals = _fitness_evals_by_member()
    out: list[dict[str, Any]] = []
    for r in rows:
        m = _parse_openweight_member_row(r)
        m["fitness_evals"] = evals.get(m["id"], {})
        out.append(m)
    return JSONResponse(out)


@app.get("/api/openweight/fitness_evals/{member_id}")
async def get_openweight_fitness_evals(member_id: str) -> JSONResponse:
    rows = _query(
        "SELECT * FROM fitness_evals WHERE member_id = ?",
        (member_id,),
    )
    out = []
    for r in rows:
        raw = None
        if r.get("raw_json"):
            try:
                raw = json.loads(r["raw_json"])
            except (json.JSONDecodeError, TypeError):
                raw = None
        out.append({
            "member_id": r["member_id"],
            "benchmark_name": r["benchmark_name"],
            "score": r["score"],
            "wallclock_s": r["wallclock_s"],
            "evaluated_at": r["evaluated_at"],
            "raw": raw,
        })
    return JSONResponse(out)


@app.get("/api/openweight/compare")
async def openweight_compare(
    a: str = Query(..., description="member_id A (baseline)"),
    b: str = Query(..., description="member_id B (post-SFT)"),
) -> JSONResponse:
    """Side-by-side bench comparison for two members.

    Returns per-benchmark {a, b, delta, rel_delta}. Useful for the
    before/after view of a fine-tune.
    """
    evals = _fitness_evals_by_member()
    ea = evals.get(a, {})
    eb = evals.get(b, {})
    benches = sorted(set(ea) | set(eb))
    rows = []
    for bench in benches:
        sa = ea.get(bench, {}).get("score")
        sb = eb.get(bench, {}).get("score")
        delta = None
        rel = None
        if sa is not None and sb is not None:
            delta = sb - sa
            if sa != 0:
                rel = delta / sa
        rows.append({
            "benchmark": bench,
            "a_score": sa, "b_score": sb,
            "delta": delta, "rel_delta": rel,
            "a_wallclock_s": ea.get(bench, {}).get("wallclock_s"),
            "b_wallclock_s": eb.get(bench, {}).get("wallclock_s"),
        })
    return JSONResponse({"a": a, "b": b, "benches": rows})


@app.get("/api/openweight/runs")
async def openweight_runs() -> JSONResponse:
    """Discover runs/openweight_* directories under SIEVE_ROOT."""
    runs_dir = SIEVE_ROOT / "runs"
    if not runs_dir.is_dir():
        return JSONResponse([])
    out = []
    for d in sorted(runs_dir.glob("openweight*")):
        if not d.is_dir():
            continue
        db = d / "lineage.db"
        entry = {
            "name": d.name,
            "path": str(d),
            "has_db": db.exists(),
            "db_mtime": db.stat().st_mtime if db.exists() else None,
        }
        out.append(entry)
    return JSONResponse(out)


# ---------------------------------------------------------------------------
# Openweight — launch (experimentation panel)
# ---------------------------------------------------------------------------

def _launches_db_path() -> Path:
    return Path(DB_PATH).parent / "launches.db"


def _launches_conn() -> sqlite3.Connection:
    """Open (and initialize) the launches DB. Separate file so the main
    lineage.db can stay read-only from the dashboard's perspective."""
    p = _launches_db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS launches (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            spec_json TEXT NOT NULL,
            status TEXT NOT NULL,
            started_at REAL NOT NULL,
            finished_at REAL,
            exit_code INTEGER,
            log_path TEXT NOT NULL,
            member_id TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS launch_metrics (
            launch_id TEXT NOT NULL,
            step INTEGER NOT NULL,
            wall_time REAL NOT NULL,
            metrics_json TEXT NOT NULL,
            PRIMARY KEY (launch_id, step)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS launch_ingest_state (
            launch_id TEXT PRIMARY KEY,
            log_byte_offset INTEGER NOT NULL DEFAULT 0,
            logging_steps INTEGER NOT NULL DEFAULT 10,
            records_ingested INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.commit()
    return conn


_LAUNCH_PROCS: dict[str, subprocess.Popen] = {}


def _launches_dir() -> Path:
    d = Path(DB_PATH).parent / "launches"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _record_launch_status(
    launch_id: str, status: str,
    finished_at: Optional[float] = None,
    exit_code: Optional[int] = None,
) -> None:
    c = _launches_conn()
    try:
        c.execute(
            "UPDATE launches SET status=?, finished_at=?, exit_code=? WHERE id=?",
            (status, finished_at, exit_code, launch_id),
        )
        c.commit()
    finally:
        c.close()


async def _reap_launches_loop() -> None:
    """Periodically check subprocess state and update launch rows."""
    while True:
        await asyncio.sleep(3.0)
        try:
            done_ids: list[str] = []
            for lid, proc in list(_LAUNCH_PROCS.items()):
                rc = proc.poll()
                if rc is not None:
                    status = "succeeded" if rc == 0 else "failed"
                    _record_launch_status(lid, status, time.time(), rc)
                    done_ids.append(lid)
            for lid in done_ids:
                _LAUNCH_PROCS.pop(lid, None)
        except Exception:
            logger.exception("reap_launches error")


@app.get("/api/openweight/launches")
async def list_launches(limit: int = Query(50, ge=1, le=500)) -> JSONResponse:
    c = _launches_conn()
    try:
        rows = c.execute(
            "SELECT * FROM launches ORDER BY started_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return JSONResponse([
            {**dict(r), "spec": json.loads(r["spec_json"])}
            for r in rows
        ])
    finally:
        c.close()


@app.get("/api/openweight/launch/{launch_id}")
async def get_launch(launch_id: str) -> JSONResponse:
    c = _launches_conn()
    try:
        r = c.execute(
            "SELECT * FROM launches WHERE id=?", (launch_id,)
        ).fetchone()
        if r is None:
            raise HTTPException(404, "launch not found")
        d = dict(r)
        d["spec"] = json.loads(d["spec_json"])
        return JSONResponse(d)
    finally:
        c.close()


@app.get("/api/openweight/launch/{launch_id}/log", response_class=PlainTextResponse)
async def get_launch_log(
    launch_id: str,
    tail: int = Query(500, ge=1, le=10000),
) -> PlainTextResponse:
    c = _launches_conn()
    try:
        r = c.execute(
            "SELECT log_path FROM launches WHERE id=?", (launch_id,)
        ).fetchone()
        if r is None:
            raise HTTPException(404, "launch not found")
        log_path = Path(r["log_path"])
    finally:
        c.close()
    if not log_path.exists():
        return PlainTextResponse("(no log yet)\n")
    try:
        data = log_path.read_text(errors="replace").splitlines()
        return PlainTextResponse("\n".join(data[-tail:]) + "\n")
    except Exception as e:  # noqa: BLE001
        return PlainTextResponse(f"(log read error: {e})\n")


@app.post("/api/openweight/launch")
async def launch_experiment(spec: dict = Body(...)) -> JSONResponse:
    """Kick off a bench or SFT run. Spec schema:

    {
      "kind": "baseline_bench" | "sft" | "sft_bench",
      "base_model_id": "...",            # required for baseline_bench/sft
      "member_id": "...",                 # required
      "benchmarks": ["smoke_mcq", ...],  # baseline_bench/sft_bench
      "proxy": true,                       # baseline_bench/sft_bench
      "generation": 0,
      "parent_ids": [],
      "corpus_shard": "data/sft_shards/v009.jsonl",  # sft
      "adapter_path": "...",               # sft_bench (pre-merged) or for SFT input
      "merged_checkpoint_path": "...",     # sft_bench (post-merge)
      "tokenizer_family": "qwen25",       # optional, default qwen25
      "epochs": 1, "rank": 16, "batch_size": 2, ...   # sft hypers
      "lineage_db": "runs/openweight_smoke/lineage.db",  # optional override
    }

    Launches via LAUNCH_DISPATCH_CMD if set (a wrapper that handles
    remote SSH to the compute node); otherwise runs locally.
    """
    kind = spec.get("kind")
    if kind not in ("baseline_bench", "sft", "sft_bench"):
        raise HTTPException(400, f"unknown kind {kind!r}")
    member_id = spec.get("member_id")
    if not member_id:
        raise HTTPException(400, "member_id is required")

    lineage_db = spec.get("lineage_db") or "runs/openweight_smoke/lineage.db"

    # Build remote-relative command fragments. LAUNCH_DISPATCH_CMD (a shell
    # wrapper) receives them as a single string and is responsible for
    # invoking the right python / transport.
    if kind in ("baseline_bench", "sft_bench"):
        base_model_id = spec.get("base_model_id")
        benchmarks = spec.get("benchmarks") or ["smoke_mcq"]
        proxy_flag = "--proxy" if spec.get("proxy", True) else ""
        kind_val = "base" if kind == "baseline_bench" else "sft"
        model_path = spec.get("merged_checkpoint_path") or base_model_id
        if not model_path:
            raise HTTPException(400, "model_path (base_model_id or merged_checkpoint_path) required")
        fragments = [
            "python -m scripts.bench_openweight_member",
            f"--model-path {shlex.quote(str(model_path))}",
            f"--member-id {shlex.quote(member_id)}",
            f"--kind {kind_val}",
            f"--tokenizer-family {shlex.quote(spec.get('tokenizer_family', 'qwen25'))}",
            f"--generation {int(spec.get('generation', 0))}",
            f"--benchmarks {shlex.quote(','.join(benchmarks))}",
            proxy_flag,
            f"--lineage-db {shlex.quote(lineage_db)}",
        ]
        if base_model_id:
            fragments.append(f"--base-model-id {shlex.quote(base_model_id)}")
        if spec.get("adapter_path"):
            fragments.append(f"--adapter-path {shlex.quote(spec['adapter_path'])}")
        if spec.get("merged_checkpoint_path"):
            fragments.append(f"--merged-checkpoint-path {shlex.quote(spec['merged_checkpoint_path'])}")
        for pid in spec.get("parent_ids", []) or []:
            fragments.append(f"--parent-id {shlex.quote(pid)}")
    else:  # sft
        base_model_id = spec.get("base_model_id")
        corpus_shard = spec.get("corpus_shard")
        out_dir = spec.get("out_dir") or f"runs/openweight_smoke/adapters/{member_id}"
        if not base_model_id or not corpus_shard:
            raise HTTPException(400, "sft requires base_model_id and corpus_shard")
        fragments = [
            "python -m scripts.sft_openweight_member",
            f"--base-model-id {shlex.quote(base_model_id)}",
            f"--corpus-shard {shlex.quote(corpus_shard)}",
            f"--member-id {shlex.quote(member_id)}",
            f"--out-dir {shlex.quote(out_dir)}",
            f"--lineage-db {shlex.quote(lineage_db)}",
            f"--generation {int(spec.get('generation', 1))}",
            f"--epochs {int(spec.get('epochs', 1))}",
            f"--rank {int(spec.get('rank', 16))}",
            f"--batch-size {int(spec.get('batch_size', 2))}",
            f"--max-seq-len {int(spec.get('max_seq_len', 1024))}",
            f"--learning-rate {float(spec.get('learning_rate', 1e-4))}",
        ]
        parent_ids = spec.get("parent_ids") or []
        if parent_ids:
            fragments.append(f"--parent-ids {shlex.quote(json.dumps(parent_ids))}")

    cmdline = " ".join(f for f in fragments if f)

    launch_id = f"lx-{int(time.time())}-{uuid.uuid4().hex[:6]}"
    log_path = _launches_dir() / f"{launch_id}.log"

    # Env the training process needs so DashboardMetricsCallback can POST
    # live loss points back to us. The dispatch wrapper must forward env
    # across the SSH hop for this to take effect remotely — if it can't,
    # the log-parse ingester is our fallback and still fills the chart.
    dashboard_url = os.environ.get(
        "SIEVE_DASHBOARD_PUBLIC_URL", f"http://127.0.0.1:{os.environ.get('SIEVE_DASHBOARD_PORT','8096')}"
    )
    metric_env_prefix = (
        f"SIEVE_DASHBOARD_LAUNCH_ID={shlex.quote(launch_id)} "
        f"SIEVE_DASHBOARD_URL={shlex.quote(dashboard_url)} "
    )

    if LAUNCH_DISPATCH_CMD:
        # Prepend env as shell-visible vars so dispatch scripts that
        # transport the command string over SSH still carry them.
        full_cmd = [*shlex.split(LAUNCH_DISPATCH_CMD), metric_env_prefix + cmdline]
    else:
        # Local: inject via subprocess env as well as the command string.
        full_cmd = ["bash", "-lc", metric_env_prefix + cmdline]

    log_f = open(log_path, "wb")
    log_f.write(f"# launch_id={launch_id}\n# kind={kind}\n# cmd={cmdline}\n".encode())
    log_f.flush()
    try:
        child_env = os.environ.copy()
        child_env["SIEVE_DASHBOARD_LAUNCH_ID"] = launch_id
        child_env["SIEVE_DASHBOARD_URL"] = dashboard_url
        proc = subprocess.Popen(
            full_cmd, stdout=log_f, stderr=subprocess.STDOUT,
            cwd=str(SIEVE_ROOT), env=child_env,
        )
    except Exception as e:  # noqa: BLE001
        log_f.close()
        raise HTTPException(500, f"spawn failed: {e}")

    _LAUNCH_PROCS[launch_id] = proc
    c = _launches_conn()
    try:
        c.execute(
            "INSERT INTO launches (id, kind, spec_json, status, started_at, log_path, member_id) "
            "VALUES (?,?,?,?,?,?,?)",
            (launch_id, kind, json.dumps(spec), "running",
             time.time(), str(log_path), member_id),
        )
        # Seed ingest state so the log-parse fallback kicks in if the
        # callback's HTTP POST can't reach back through the firewall.
        c.execute(
            "INSERT OR REPLACE INTO launch_ingest_state "
            "(launch_id, log_byte_offset, logging_steps, records_ingested) "
            "VALUES (?, 0, ?, 0)",
            (launch_id, int(spec.get("logging_steps", 10))),
        )
        c.commit()
    finally:
        c.close()

    return JSONResponse({
        "launch_id": launch_id,
        "status": "running",
        "log_path": str(log_path),
        "cmd": cmdline,
    })


# ---------------------------------------------------------------------------
# Openweight — per-launch training metrics (loss curves)
# ---------------------------------------------------------------------------
#
# Two sources feed ``launch_metrics``:
#   1. ``DashboardMetricsCallback`` inside the training process POSTs to
#      ``/api/openweight/launch/{id}/metrics`` on every TRL ``on_log``.
#   2. ``_launch_metrics_ingest_loop`` below tails each running launch's
#      log file for TRL's console dict lines (``{'loss': '2.4', ...}``)
#      and inserts them directly. This covers runs launched outside the
#      dashboard (CLI, ad-hoc) and acts as a belt-and-braces if the POST
#      path is unreachable.
#
# The table is ``(launch_id, step)`` PK with ``INSERT OR REPLACE`` so the
# two sources can't create duplicates.


# TRL's default console log format prints the metrics dict as Python-dict
# repr: ``{'loss': '2.403', 'grad_norm': '0.4615', ..., 'epoch': '...'}``.
# String-quoted values, single quotes, comma-separated. Parseable with
# ``ast.literal_eval`` once we isolate the dict from surrounding text.
_TRL_METRIC_LINE = re.compile(r"(\{'loss':\s*'[^']*'.*?\})")


def _parse_trl_metric_line(text: str) -> Optional[dict[str, Any]]:
    m = _TRL_METRIC_LINE.search(text)
    if not m:
        return None
    try:
        raw = ast.literal_eval(m.group(1))
    except (ValueError, SyntaxError):
        return None
    if not isinstance(raw, dict):
        return None
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if isinstance(v, str):
            try:
                out[k] = float(v)
                continue
            except ValueError:
                pass
        out[k] = v
    return out


def _insert_metric(
    conn: sqlite3.Connection,
    launch_id: str,
    step: int,
    wall_time: float,
    metrics: dict[str, Any],
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO launch_metrics "
        "(launch_id, step, wall_time, metrics_json) VALUES (?,?,?,?)",
        (launch_id, step, wall_time, json.dumps(metrics)),
    )


@app.post("/api/openweight/launch/{launch_id}/metrics")
async def post_launch_metric(launch_id: str, point: dict = Body(...)) -> JSONResponse:
    """Accept a metric point from the training callback.

    Body: ``{"step": <int>, "wall_time": <float>, "metrics": {...}}``
    or batch ``{"points": [{...}, {...}]}`` for backfill.
    """
    points = point.get("points")
    if points is None:
        points = [point]
    if not isinstance(points, list):
        raise HTTPException(400, "points must be a list or body must be a single point")

    c = _launches_conn()
    try:
        exists = c.execute(
            "SELECT 1 FROM launches WHERE id=?", (launch_id,)
        ).fetchone()
        if not exists:
            raise HTTPException(404, "unknown launch_id")
        accepted = 0
        for p in points:
            try:
                step = int(p.get("step", 0))
                wall_time = float(p.get("wall_time") or time.time())
                metrics = p.get("metrics") or {}
                if not isinstance(metrics, dict):
                    continue
                _insert_metric(c, launch_id, step, wall_time, metrics)
                accepted += 1
            except (TypeError, ValueError):
                continue
        c.commit()
        return JSONResponse({"launch_id": launch_id, "accepted": accepted})
    finally:
        c.close()


@app.get("/api/openweight/launch/{launch_id}/metrics")
async def get_launch_metrics(
    launch_id: str,
    since: Optional[int] = Query(None, description="only return points with step > since"),
    limit: int = Query(5000, ge=1, le=50000),
) -> JSONResponse:
    c = _launches_conn()
    try:
        if since is None:
            rows = c.execute(
                "SELECT step, wall_time, metrics_json FROM launch_metrics "
                "WHERE launch_id=? ORDER BY step ASC LIMIT ?",
                (launch_id, limit),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT step, wall_time, metrics_json FROM launch_metrics "
                "WHERE launch_id=? AND step>? ORDER BY step ASC LIMIT ?",
                (launch_id, since, limit),
            ).fetchall()
        out = [
            {"step": r["step"], "wall_time": r["wall_time"],
             "metrics": json.loads(r["metrics_json"])}
            for r in rows
        ]
        max_step = out[-1]["step"] if out else since
        return JSONResponse({"launch_id": launch_id, "points": out, "max_step": max_step})
    finally:
        c.close()


@app.post("/api/openweight/launch/register")
async def register_external_launch(spec: dict = Body(...)) -> JSONResponse:
    """Register a training run that was started outside the dashboard.

    Body must include ``log_path`` and ``member_id``. Optional: ``kind``
    (default ``sft``), ``launch_id`` (generated if omitted),
    ``started_at`` (default: log file mtime or now), ``logging_steps``
    (default 10 — how TRL reports). The log-parse ingester will start
    tailing the file on its next tick.
    """
    log_path = spec.get("log_path")
    member_id = spec.get("member_id")
    if not log_path or not member_id:
        raise HTTPException(400, "log_path and member_id are required")
    kind = spec.get("kind", "sft")
    launch_id = spec.get("launch_id") or f"ext-{int(time.time())}-{uuid.uuid4().hex[:6]}"
    logging_steps = int(spec.get("logging_steps", 10))
    lp = Path(log_path)
    started_at = float(spec.get("started_at") or (
        lp.stat().st_mtime if lp.exists() else time.time()
    ))

    c = _launches_conn()
    try:
        c.execute(
            "INSERT OR REPLACE INTO launches "
            "(id, kind, spec_json, status, started_at, log_path, member_id) "
            "VALUES (?,?,?,?,?,?,?)",
            (launch_id, kind, json.dumps(spec), "running",
             started_at, str(log_path), member_id),
        )
        c.execute(
            "INSERT OR REPLACE INTO launch_ingest_state "
            "(launch_id, log_byte_offset, logging_steps, records_ingested) "
            "VALUES (?, 0, ?, 0)",
            (launch_id, logging_steps),
        )
        c.commit()
    finally:
        c.close()
    return JSONResponse({"launch_id": launch_id, "status": "registered"})


def _ingest_one_launch(c: sqlite3.Connection, launch_id: str, log_path: str) -> int:
    """Tail new bytes of ``log_path`` and insert any TRL metric lines found.

    Returns the number of new points inserted.
    """
    p = Path(log_path)
    if not p.exists():
        return 0
    row = c.execute(
        "SELECT log_byte_offset, logging_steps, records_ingested "
        "FROM launch_ingest_state WHERE launch_id=?",
        (launch_id,),
    ).fetchone()
    if row is None:
        c.execute(
            "INSERT INTO launch_ingest_state (launch_id) VALUES (?)",
            (launch_id,),
        )
        c.commit()
        offset = 0
        logging_steps = 10
        n_done = 0
    else:
        offset = row["log_byte_offset"]
        logging_steps = row["logging_steps"] or 10
        n_done = row["records_ingested"] or 0

    try:
        size = p.stat().st_size
    except OSError:
        return 0
    if size < offset:
        # Log rotated / truncated; reset.
        offset = 0
        n_done = 0
    if size == offset:
        return 0

    try:
        with p.open("rb") as fh:
            fh.seek(offset)
            chunk = fh.read(size - offset)
    except OSError:
        return 0
    text = chunk.decode("utf-8", errors="replace")
    # Split on newlines but also handle the \r-separated progress-bar output.
    inserted = 0
    # Find every dict occurrence independently — a single "line" may contain
    # multiple \r-separated status updates plus one metric dict at the end.
    for match in _TRL_METRIC_LINE.finditer(text):
        metrics = _parse_trl_metric_line(match.group(1))
        if not metrics:
            continue
        n_done += 1
        step = n_done * logging_steps
        _insert_metric(c, launch_id, step, time.time(), metrics)
        inserted += 1

    c.execute(
        "UPDATE launch_ingest_state "
        "SET log_byte_offset=?, records_ingested=? WHERE launch_id=?",
        (size, n_done, launch_id),
    )
    c.commit()
    return inserted


async def _launch_metrics_ingest_loop() -> None:
    """Periodically tail each running launch's log file for TRL metric dicts."""
    while True:
        await asyncio.sleep(4.0)
        try:
            c = _launches_conn()
            try:
                rows = c.execute(
                    "SELECT id, log_path FROM launches WHERE status='running'"
                ).fetchall()
                for r in rows:
                    try:
                        _ingest_one_launch(c, r["id"], r["log_path"])
                    except Exception:
                        logger.exception("ingest failed for launch %s", r["id"])
            finally:
                c.close()
        except Exception:
            logger.exception("launch_metrics_ingest_loop error")


@app.post("/api/openweight/launch/{launch_id}/cancel")
async def cancel_launch(launch_id: str) -> JSONResponse:
    proc = _LAUNCH_PROCS.get(launch_id)
    if proc is None:
        raise HTTPException(404, "not running")
    try:
        proc.terminate()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"terminate failed: {e}")
    _record_launch_status(launch_id, "cancelled", time.time(), -15)
    _LAUNCH_PROCS.pop(launch_id, None)
    return JSONResponse({"launch_id": launch_id, "status": "cancelled"})


# ---------------------------------------------------------------------------
# Openweight — playground (side-by-side inference against live member shims)
#
# Wraps ``openweight.inference.launch_*_server`` with a per-member registry so
# the dashboard can boot-or-reuse a ``serving_shim`` subprocess for any member
# and proxy OpenAI-compat chat completions to it. This is the openweight
# equivalent of ``lab/server.py``'s fan-out UI for local ``.pt`` checkpoints.
#
# Lifecycle:
#   - ensure  → kicks off a background thread that calls launch_*_server;
#               registry entry transitions idle → starting → ready/error.
#   - chat    → POSTs to the member's shim at /v1/chat/completions.
#   - stop    → terminates the subprocess and removes the registry entry.
#   - status  → lists all active shims (for the UI poll).
# ---------------------------------------------------------------------------

_PLAYGROUND_SHIMS: dict[str, dict[str, Any]] = {}
_PLAYGROUND_LOCK = threading.RLock()
# Serialize node-pick + reserve so two simultaneous ensures can't pick the
# same node. Held only across pick_node + the immediate registry write.
_NODE_PICK_LOCK = threading.Lock()

# Orphaned serving_shims discovered on the fleet at dashboard startup —
# left over from a previous dashboard run that died/restarted with shims
# still up. ``playground_ensure`` consults this list before launching: if
# the requested (model_path, adapter) matches an unadopted orphan, it
# adopts it instead of spending the launch latency + RAM to stand up a
# fresh copy. Populated once during ``lifespan`` startup.
_DISCOVERED_ORPHANS: list[Any] = []  # list[DiscoveredShim]
_DISCOVERED_ORPHANS_LOCK = threading.Lock()


def _self_exclusion() -> tuple[set[str], set[str]]:
    """Names + IPs of the dashboard host itself — never co-locate a shim here.

    The dashboard process serves the orchestration HTTP API and also tends
    to do CPU-side work (LoRA merges, lineage queries) that competes with
    a shim for unified memory. Hard-exclude.
    """
    from openweight.node_picker import local_hostname_short, local_ips
    return ({local_hostname_short()}, local_ips())


def _resolve_playground_model_path(
    member_row: dict[str, Any],
) -> tuple[Optional[str], Optional[str]]:
    """Return (model_path, adapter_path) for serving this member.

    - kind=base            → (base_model_id, None)
    - kind=merge|lora_merge → (merged_checkpoint_path, None)
    - kind=sft             → (base_model_id, adapter_path) — triggers merge-on-demand
    - anything else        → (None, None)
    """
    kind = member_row.get("kind")
    base = member_row.get("base_model_id")
    adapter = member_row.get("adapter_path")
    merged = member_row.get("merged_checkpoint_path")
    if kind == "base":
        return (base, None)
    if kind in ("merge", "lora_merge"):
        return (merged, None)
    if kind == "sft":
        if not base or not adapter:
            return (None, None)
        return (base, adapter)
    return (None, None)


def _tail_log_file(path: Optional[str], max_lines: int = 5, max_bytes: int = 32_768) -> list[str]:
    """Return last ``max_lines`` non-empty lines from ``path`` (best-effort, non-raising)."""
    if not path:
        return []
    try:
        p = Path(path)
        if not p.exists():
            return []
        size = p.stat().st_size
        with p.open("rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
            chunk = f.read()
        text = chunk.decode(errors="replace")
        lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
        return lines[-max_lines:]
    except Exception:
        return []


def _tail_remote_log_quiet(node_name: str, node_ip: str, log_path: str, max_lines: int) -> list[str]:
    """ssh-tail a log on a remote node, swallowing all errors.

    Used in the realtime status snapshot — we don't want a transient ssh
    hiccup to surface as a frontend error. Keep ``max_lines`` small (3-5)
    to keep the ssh round-trip cheap.
    """
    try:
        from openweight.node_picker import Node
        from openweight.remote_launch import tail_remote_log
        return tail_remote_log(Node(name=node_name, ip=node_ip), log_path, tail=max_lines)
    except Exception:
        return []


def _playground_public(member_id: str, include_log_tail: bool = True) -> dict[str, Any]:
    """Snapshot registry entry minus the non-serializable handle.

    When ``include_log_tail`` and the shim is still starting, append the last
    few log lines so the client can show meaningful loading progress.
    Logs live on the remote node (one-model-per-node rule); we ssh-tail
    on demand.
    """
    with _PLAYGROUND_LOCK:
        e = _PLAYGROUND_SHIMS.get(member_id)
        if not e:
            return {"member_id": member_id, "status": "idle"}
        snap = {
            "member_id": member_id,
            "status": e.get("status", "idle"),
            "model_path": e.get("model_path"),
            "adapter_path": e.get("adapter_path"),
            "port": e.get("port"),
            "remote_port": e.get("remote_port"),
            "base_url": e.get("base_url"),
            "served_model": e.get("served_model"),
            "started_at": e.get("started_at"),
            "ready_at": e.get("ready_at"),
            "error": e.get("error"),
            "log_path": e.get("log_path"),
            "node": e.get("node"),
            "host": e.get("host"),
            "free_gb_at_pick": e.get("free_gb_at_pick"),
            "replicate_from": e.get("replicate_from"),
        }
    started_at = snap.get("started_at")
    ready_at = snap.get("ready_at")
    if started_at:
        end = ready_at if ready_at else time.time()
        snap["elapsed_s"] = max(0.0, end - started_at)
    if include_log_tail and snap.get("status") == "starting":
        node_name = snap.get("node")
        node_ip = snap.get("host")
        log_path = snap.get("log_path")
        if node_name and node_ip and log_path:
            snap["log_tail"] = _tail_remote_log_quiet(node_name, node_ip, log_path, 3)
        else:
            snap["log_tail"] = []
    return snap


def _start_shim_thread(
    member_id: str, model_path: str, adapter_path: Optional[str], node: "Node",
) -> None:
    """Blocking: ssh-launch ``serving_shim`` on ``node``, then publish the handle.

    Per the one-model-per-node rule: the shim ALWAYS lives on a remote
    GB10, never on the dashboard host. Caller has already reserved
    ``node`` in the registry (via the ``_NODE_PICK_LOCK`` path in
    ``playground_ensure``); on launch failure we flip the entry to
    ``error`` but leave the registry record so the operator can see it.
    """
    from openweight.remote_launch import launch_remote_shim

    # Phase 0: replicate weights to the target if the picker flagged it.
    # The reservation already exists with status="replicating"; we run the
    # rsync here (in the worker thread, not under the pick lock) so multiple
    # concurrent ensures fan out in parallel.
    with _PLAYGROUND_LOCK:
        e = _PLAYGROUND_SHIMS.get(member_id, {})
        repl_from = e.get("replicate_from")
        repl_from_ip = e.get("replicate_from_ip")

    if repl_from:
        try:
            from openweight.node_picker import Node as _Node, replicate_weights
            source = _Node(name=repl_from, ip=repl_from_ip or "127.0.0.1")
            t0 = time.time()
            logger.info(
                "playground replicate %s: %s → %s (model=%s)",
                member_id, source.name, node.name, model_path,
            )
            replicate_weights(
                model_path, source, node,
                log_cb=lambda line: logger.info("[%s] %s", member_id, line),
            )
            logger.info(
                "playground replicate %s done in %.1fs", member_id, time.time() - t0,
            )
            with _PLAYGROUND_LOCK:
                e = _PLAYGROUND_SHIMS.setdefault(member_id, {})
                e["status"] = "starting"
                e["replicate_from"] = None
        except Exception as exc:  # noqa: BLE001
            logger.exception("playground replication failed for %s", member_id)
            with _PLAYGROUND_LOCK:
                e = _PLAYGROUND_SHIMS.setdefault(member_id, {})
                e.update({"status": "error", "error": f"replication failed: {exc}"})
            return

    def _on_spawn(info: dict) -> None:
        """Invoked by the launcher right after Popen on the remote — stamps
        log_path+port into the registry so the frontend can tail output
        during model load."""
        with _PLAYGROUND_LOCK:
            e = _PLAYGROUND_SHIMS.setdefault(member_id, {})
            e["log_path"] = info.get("log_path")
            e["port"] = info.get("port")
            e["base_url"] = info.get("base_url")
            e["remote_pid"] = info.get("pid")
            e["tunnel_pid"] = info.get("tunnel_pid")
            e["remote_port"] = info.get("remote_port")

    try:
        handle = launch_remote_shim(
            node=node,
            model_path=model_path,
            adapter_path=adapter_path,
            ready_timeout_s=900.0,
            on_spawn=_on_spawn,
        )
        with _PLAYGROUND_LOCK:
            e = _PLAYGROUND_SHIMS.setdefault(member_id, {})
            e.update({
                "status": "ready", "handle": handle,
                "port": handle.port, "base_url": handle.base_url,
                "served_model": handle.served_model_name,
                "log_path": str(handle.log_path),
                "remote_pid": handle.remote_pid,
                "tunnel_pid": handle.tunnel_pid,
                "remote_port": handle.remote_port,
                "ready_at": time.time(), "error": None,
            })
        logger.info(
            "playground shim ready: %s → %s on %s",
            member_id, handle.base_url, node.name,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("playground shim launch failed for %s on %s", member_id, node.name)
        with _PLAYGROUND_LOCK:
            e = _PLAYGROUND_SHIMS.setdefault(member_id, {})
            e.update({"status": "error", "error": str(exc)})


def _populate_orphan_cache() -> None:
    """One-shot fleet scan for live ``serving_shim`` orphans (dashboard-startup).

    Runs on a background thread off ``lifespan`` so a slow fleet doesn't
    delay startup. Best-effort: any exception is logged and the cache
    stays empty, in which case ``playground_ensure`` falls through to the
    normal pick + launch path.
    """
    try:
        from openweight.remote_launch import discover_remote_shims
        shims = discover_remote_shims()
    except Exception:  # noqa: BLE001
        logger.exception("_populate_orphan_cache: discovery failed")
        return
    with _DISCOVERED_ORPHANS_LOCK:
        _DISCOVERED_ORPHANS[:] = shims
    logger.info(
        "_populate_orphan_cache: discovered %d orphan shim(s) on the fleet",
        len(shims),
    )


def _try_adopt_orphan(
    member_id: str, model_path: str, adapter: Optional[str],
) -> bool:
    """If an unadopted orphan matches (model_path, adapter), attach to it.

    Returns True if a match was found, the ssh -L tunnel was opened, and
    ``_PLAYGROUND_SHIMS[member_id]`` was registered as ``ready``. Returns
    False otherwise — caller should fall through to the normal launch path.

    Mutates ``_DISCOVERED_ORPHANS`` to remove the consumed entry so two
    concurrent ensures can't claim the same orphan.

    Matching follows the launcher's serve-path convention: SFT members
    pre-merge base+adapter on the remote into
    ``/tmp/sieve-merged/{adapter_basename}`` and the shim's ``--model-path``
    points at the merged dir. So an orphan from a previous run records
    that merged path, not the original (base, adapter) pair — we recompute
    the expected serve-path here and compare to the orphan.
    """
    from openweight.remote_launch import REMOTE_MERGED_CACHE, adopt_remote_shim

    if adapter:
        expected_serve_path = f"{REMOTE_MERGED_CACHE}/{Path(adapter).name}"
    else:
        expected_serve_path = model_path

    with _DISCOVERED_ORPHANS_LOCK:
        chosen = None
        for i, d in enumerate(_DISCOVERED_ORPHANS):
            if d.model_path == expected_serve_path:
                chosen = _DISCOVERED_ORPHANS.pop(i)
                break
    if chosen is None:
        return False

    try:
        handle = adopt_remote_shim(chosen)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "playground_ensure: adopt failed for %s on %s (%s) — falling through to launch",
            model_path, chosen.node.name, exc,
        )
        return False

    with _PLAYGROUND_LOCK:
        _PLAYGROUND_SHIMS[member_id] = {
            "status": "ready",
            "model_path": model_path,
            "adapter_path": adapter,
            "node": chosen.node.name,
            "host": chosen.node.ip,
            "free_gb_at_pick": None,
            "replicate_from": None,
            "replicate_from_ip": None,
            "started_at": time.time(),
            "port": handle.port,
            "base_url": handle.base_url,
            "served_model": handle.served_model_name,
            "ready_at": time.time(),
            "error": None,
            "log_path": str(handle.log_path),
            "handle": handle,
            "remote_pid": handle.remote_pid,
            "tunnel_pid": handle.tunnel_pid,
            "remote_port": handle.remote_port,
            "adopted": True,
        }
    logger.info(
        "playground_ensure: adopted orphan shim %s on %s for %s (remote pid=%d)",
        model_path, chosen.node.name, member_id, chosen.remote_pid,
    )
    return True


def _pick_and_reserve(
    member_id: str, model_path: str, adapter: Optional[str],
) -> "Node":
    """Pick a free node (excluding self + currently-reserved) and reserve it.

    Serialized via ``_NODE_PICK_LOCK`` so concurrent ensures don't race on
    the same node. Raises ``NoFreeNode`` if the fleet is exhausted —
    caller should translate that into a 503.
    """
    from openweight.node_picker import pick_node

    self_hosts, self_ips = _self_exclusion()
    with _NODE_PICK_LOCK:
        # Snapshot current reservations under the main lock.
        with _PLAYGROUND_LOCK:
            excl_hosts = set(self_hosts)
            excl_ips = set(self_ips)
            for mid, e in _PLAYGROUND_SHIMS.items():
                if mid == member_id:
                    continue
                if e.get("status") not in ("replicating", "starting", "ready"):
                    continue
                if n := e.get("node"):
                    excl_hosts.add(n)
                if h := e.get("host"):
                    excl_ips.add(h)
        node, probe = pick_node(
            model_path,
            exclude_hosts=excl_hosts,
            exclude_ips=excl_ips,
            auto_replicate=True,
        )
        needs_repl = bool(probe.get("needs_replication"))
        # Commit reservation atomically with the pick decision so a second
        # call into _pick_and_reserve sees this node as taken. If the picker
        # selected a target that needs weights replicated, hold the entry in
        # ``replicating`` until the worker thread finishes the rsync.
        with _PLAYGROUND_LOCK:
            _PLAYGROUND_SHIMS[member_id] = {
                "status": "replicating" if needs_repl else "starting",
                "model_path": model_path,
                "adapter_path": adapter,
                "node": node.name,
                "host": node.ip,
                "free_gb_at_pick": probe.get("free_gb"),
                "replicate_from": probe.get("source_name") if needs_repl else None,
                "replicate_from_ip": probe.get("source_ip") if needs_repl else None,
                "started_at": time.time(),
                "port": None, "base_url": None, "served_model": None,
                "ready_at": None, "error": None, "log_path": None,
                "handle": None, "remote_pid": None,
            }
    return node


@app.post("/api/openweight/playground/ensure/{member_id}")
async def playground_ensure(member_id: str) -> JSONResponse:
    """Start (or reuse) a serving_shim on a remote GB10 for ``member_id``.

    Hard rule: one shim per node. If every fleet node is either the
    dashboard host or already serving a shim (or doesn't have the
    weights), this returns **503** — we never co-locate.

    Returns the current registry snapshot once the shim is reserved on a
    node. Model load itself takes ~30-200s; clients poll ``/status`` for
    the ready transition.
    """
    from openweight.node_picker import NoFreeNode

    row = _query_one("SELECT * FROM members WHERE id = ?", (member_id,))
    if not row:
        raise HTTPException(status_code=404, detail=f"no member {member_id!r}")
    parsed = _parse_openweight_member_row(row)
    model_path, adapter = _resolve_playground_model_path(parsed)
    if not model_path:
        raise HTTPException(
            status_code=400,
            detail=f"cannot resolve serve path for kind={parsed.get('kind')!r}",
        )

    # Fast path: already starting or ready — return current snapshot.
    with _PLAYGROUND_LOCK:
        existing = _PLAYGROUND_SHIMS.get(member_id)
        if existing and existing.get("status") in ("replicating", "starting", "ready"):
            return JSONResponse(_playground_public(member_id))

    # Adoption path: a previous dashboard run may have left a shim for this
    # exact (model_path, adapter) running on the fleet. If startup discovery
    # found one, attach to it instead of launching a fresh copy.
    adopted = await asyncio.to_thread(_try_adopt_orphan, member_id, model_path, adapter)
    if adopted:
        return JSONResponse(_playground_public(member_id))

    # Slow path: pick + reserve a remote node. ssh probes block ~1-3s
    # per candidate, so do it in a thread to keep the event loop free.
    try:
        node = await asyncio.to_thread(_pick_and_reserve, member_id, model_path, adapter)
    except NoFreeNode as ex:
        # Panic, per the one-model-per-node rule. Surface the reasons so
        # the operator can see which candidates failed (shim already up?
        # weights missing? ssh denied?).
        logger.warning("playground_ensure: no free node for %s: %s", member_id, ex)
        raise HTTPException(status_code=503, detail=f"no free node: {ex}")

    t = threading.Thread(
        target=_start_shim_thread,
        args=(member_id, model_path, adapter, node),
        daemon=True, name=f"shim-{member_id}",
    )
    t.start()
    return JSONResponse(_playground_public(member_id))


@app.get("/api/openweight/playground/status")
async def playground_status() -> JSONResponse:
    with _PLAYGROUND_LOCK:
        ids = list(_PLAYGROUND_SHIMS.keys())
    return JSONResponse({"shims": [_playground_public(mid) for mid in ids]})


@app.get("/api/openweight/playground/log/{member_id}")
async def playground_log(member_id: str, tail: int = 40) -> JSONResponse:
    """Return the last ``tail`` lines of the shim's stdout/stderr log.

    The log lives on the remote shim node; we ssh-tail it on demand.
    """
    tail = max(1, min(int(tail), 400))
    with _PLAYGROUND_LOCK:
        e = _PLAYGROUND_SHIMS.get(member_id)
    if not e:
        raise HTTPException(status_code=404, detail="no shim")
    log_path = e.get("log_path")
    node_name = e.get("node")
    node_ip = e.get("host")
    lines: list[str] = []
    if log_path and node_name and node_ip:
        lines = await asyncio.to_thread(
            _tail_remote_log_quiet, node_name, node_ip, log_path, tail,
        )
    return JSONResponse({
        "member_id": member_id,
        "status": e.get("status"),
        "log_path": log_path,
        "node": node_name,
        "host": node_ip,
        "lines": lines,
    })


@app.post("/api/openweight/playground/stop/{member_id}")
async def playground_stop(member_id: str) -> JSONResponse:
    """Kill the remote shim and release its node reservation.

    If ``handle`` is set, use its ``stop()`` (the canonical path). If the
    launch thread hadn't reached the handle assignment yet but ``on_spawn``
    already published a ``remote_pid`` + node, ssh-kill the pid directly so
    we don't leak a 60-GB process while the registry forgets it.
    """
    with _PLAYGROUND_LOCK:
        e = _PLAYGROUND_SHIMS.get(member_id)
    if not e:
        raise HTTPException(status_code=404, detail="no shim")
    handle = e.get("handle")
    if handle is not None:
        try:
            await asyncio.to_thread(handle.stop)
        except Exception:
            logger.exception("handle.stop failed for %s", member_id)
    else:
        # Pre-handle stop: launch thread crashed or is mid-flight. If we
        # know which pid landed where, kill it directly.
        remote_pid = e.get("remote_pid")
        node_name = e.get("node")
        node_ip = e.get("host")
        if remote_pid and node_ip:
            try:
                from openweight.node_picker import Node
                from openweight.remote_launch import RemoteServerHandle
                tmp = RemoteServerHandle(
                    node=Node(name=node_name or "", ip=node_ip),
                    port=int(e.get("port") or 0),
                    base_url=e.get("base_url") or "",
                    served_model=e.get("served_model") or "",
                    remote_pid=int(remote_pid),
                    log_path=Path(e.get("log_path") or "/dev/null"),
                    tunnel_pid=int(e["tunnel_pid"]) if e.get("tunnel_pid") else None,
                    remote_port=int(e["remote_port"]) if e.get("remote_port") else None,
                )
                await asyncio.to_thread(tmp.stop)
            except Exception:
                logger.exception("ad-hoc remote kill failed for %s", member_id)
    with _PLAYGROUND_LOCK:
        _PLAYGROUND_SHIMS.pop(member_id, None)
    return JSONResponse({"member_id": member_id, "status": "stopped"})


def _post_to_shim(
    base_url: str, path: str, payload: dict[str, Any], timeout_s: float,
) -> dict[str, Any]:
    """Blocking POST to a shim endpoint (/chat/completions or /completions)."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read())


def _classify_model_role(model_path: str, kind: str, adapter_path: Optional[str]) -> str:
    """Cheap label for the context card. No network calls."""
    p = (model_path or "").lower()
    is_instruct = any(tag in p for tag in ("instruct", "chat", "-it", "-sft"))
    if kind == "base":
        return "instruct (chat-tuned)" if is_instruct else "base (next-token)"
    if kind in ("merge", "lora_merge"):
        return "merged SFT (our fine-tune)"
    if kind == "sft" and adapter_path:
        base_role = "instruct" if is_instruct else "base"
        return f"{base_role} + LoRA adapter"
    return kind or "unknown"


@app.post("/api/openweight/playground/chat")
async def playground_chat(body: dict = Body(...)) -> JSONResponse:
    """Proxy a generation request to a running member shim.

    Body: ``{member_id, mode="chat"|"completion", max_tokens=256,
    temperature=0.7, top_p=1.0}`` plus ``messages[]`` (chat mode) or
    ``prompt`` (completion mode). Chat mode hits /v1/chat/completions and
    applies the tokenizer's chat template; completion mode hits
    /v1/completions and feeds the raw prompt to the model (no template).
    """
    member_id = body.get("member_id")
    if not member_id:
        raise HTTPException(status_code=400, detail="member_id required")

    mode = body.get("mode") or "chat"
    if mode not in ("chat", "completion"):
        raise HTTPException(status_code=400, detail=f"mode must be 'chat' or 'completion', got {mode!r}")

    if mode == "chat":
        messages = body.get("messages") or []
        if not messages:
            raise HTTPException(status_code=400, detail="messages required in chat mode")
    else:
        prompt = body.get("prompt")
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt required in completion mode")

    with _PLAYGROUND_LOCK:
        e = _PLAYGROUND_SHIMS.get(member_id)
        status = (e or {}).get("status")
        base_url = (e or {}).get("base_url")
    logger.info("playground/chat: member=%s mode=%s status=%s base_url=%s", member_id, mode, status, base_url)
    if status != "ready" or not base_url:
        raise HTTPException(
            status_code=409,
            detail=f"shim for {member_id!r} is {status!r}, not ready",
        )

    shared = {
        "max_tokens": int(body.get("max_tokens", 256)),
        "temperature": float(body.get("temperature", 0.7)),
        "top_p": float(body.get("top_p", 1.0)),
    }
    if mode == "chat":
        payload = {"messages": messages, **shared}
        path = "/chat/completions"
    else:
        payload = {"prompt": prompt, **shared}
        path = "/completions"

    t0 = time.time()
    try:
        resp = await asyncio.to_thread(_post_to_shim, base_url, path, payload, 60.0)
    except urllib.error.HTTPError as ex:
        detail = ex.read().decode(errors="replace")
        logger.warning("playground/chat %s HTTPError: %s", member_id, detail)
        raise HTTPException(status_code=502, detail=f"shim {ex.code}: {detail}")
    except Exception as ex:  # noqa: BLE001
        logger.warning("playground/chat %s unreachable: %s", member_id, ex)
        raise HTTPException(status_code=502, detail=f"shim unreachable: {ex}")
    elapsed_s = time.time() - t0
    logger.info("playground/chat: member=%s elapsed=%.2fs tokens=%s", member_id, elapsed_s, resp.get("usage", {}).get("completion_tokens"))

    try:
        choice = resp["choices"][0]
        content = choice["message"]["content"] if mode == "chat" else choice["text"]
        finish_reason = choice.get("finish_reason")
    except (KeyError, IndexError) as ex:
        raise HTTPException(status_code=502, detail=f"malformed shim reply: {ex}")
    return JSONResponse({
        "member_id": member_id,
        "mode": mode,
        "content": content,
        "elapsed_s": elapsed_s,
        "usage": resp.get("usage", {}),
        "finish_reason": finish_reason,
    })


_MCQ_LETTER_RE = re.compile(r"(?:answer\s*[:\-]?\s*)?\b([ABCD])\b", re.IGNORECASE)


def _extract_mcq_letter(text: Optional[str]) -> Optional[str]:
    """Pull a final A/B/C/D out of a possibly-<think>-wrapped reply.

    Mirrors the smoke_mcq adapter's extraction: strip ``<think>...</think>``
    then find the last standalone letter in A–D. Returns None if none found
    or if the pattern is too ambiguous (e.g., no letter at all).
    """
    if not text:
        return None
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if not clean:
        return None
    # Prefer the LAST letter-looking token (models often reason then answer).
    matches = _MCQ_LETTER_RE.findall(clean)
    if not matches:
        return None
    return matches[-1].upper()


def _member_best_scores() -> dict[str, float]:
    """{member_id -> best score across all benchmarks}. Default to 1.0 if unscored."""
    rows = _query("SELECT member_id, score FROM fitness_evals WHERE score IS NOT NULL")
    out: dict[str, float] = {}
    for r in rows:
        mid = r["member_id"]
        s = float(r["score"])
        if s > out.get(mid, -1.0):
            out[mid] = s
    return out


def _aggregate_letters(
    outputs: list[dict[str, Any]], strategy: str,
) -> dict[str, Any]:
    """Weighted or plain majority over A/B/C/D letters extracted from outputs.

    Mirrors ``controller/openweight_ensemble_eval._majority_vote`` but over a
    single question rather than a whole benchmark. Ties break lexicographically
    (A wins).
    """
    from collections import Counter
    votes: Counter = Counter()
    voters: dict[str, list[str]] = {}
    for o in outputs:
        letter = o.get("answer_letter")
        if not letter or letter not in ("A", "B", "C", "D"):
            continue
        w = float(o.get("weight", 1.0)) if strategy == "weighted_majority" else 1.0
        votes[letter] += w
        voters.setdefault(letter, []).append(o["member_id"])
    if not votes:
        return {"letter": None, "strategy": strategy, "votes": {}, "voters": {}}
    max_vote = max(votes.values())
    winner = min(letter for letter, v in votes.items() if v == max_vote)
    return {
        "letter": winner,
        "strategy": strategy,
        "votes": {k: round(v, 4) for k, v in votes.items()},
        "voters": voters,
    }


@app.post("/api/openweight/playground/ensemble")
async def playground_ensemble(body: dict = Body(...)) -> JSONResponse:
    """Fan one prompt across N members per side and aggregate.

    Body: ``{a_members:[id,...], b_members:[id,...], mode, messages|prompt,
    max_tokens, temperature, top_p, strategy}``. Every listed member must
    already have a ready shim (client calls ``/playground/ensure/{id}`` per
    member first). Returns per-member outputs + a per-side aggregate vote.
    """
    a_ids = [x for x in (body.get("a_members") or []) if isinstance(x, str) and x]
    b_ids = [x for x in (body.get("b_members") or []) if isinstance(x, str) and x]
    if not a_ids or not b_ids:
        raise HTTPException(status_code=400, detail="a_members and b_members must be non-empty")

    mode = body.get("mode") or "chat"
    if mode not in ("chat", "completion"):
        raise HTTPException(status_code=400, detail=f"mode must be 'chat' or 'completion', got {mode!r}")
    if mode == "chat":
        messages = body.get("messages") or []
        if not messages:
            raise HTTPException(status_code=400, detail="messages required in chat mode")
    else:
        prompt = body.get("prompt")
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt required in completion mode")

    strategy = body.get("strategy") or "majority"
    if strategy not in ("majority", "weighted_majority"):
        raise HTTPException(status_code=400, detail=f"strategy must be 'majority' or 'weighted_majority'")

    shared = {
        "max_tokens": int(body.get("max_tokens", 256)),
        "temperature": float(body.get("temperature", 0.7)),
        "top_p": float(body.get("top_p", 1.0)),
    }
    if mode == "chat":
        payload = {"messages": messages, **shared}
        path = "/chat/completions"
    else:
        payload = {"prompt": prompt, **shared}
        path = "/completions"

    all_ids = list(dict.fromkeys(a_ids + b_ids))  # unique, preserving order
    with _PLAYGROUND_LOCK:
        base_urls: dict[str, str] = {}
        not_ready: list[dict[str, Any]] = []
        for mid in all_ids:
            e = _PLAYGROUND_SHIMS.get(mid)
            if not e or e.get("status") != "ready" or not e.get("base_url"):
                not_ready.append({"member_id": mid, "status": (e or {}).get("status", "idle")})
            else:
                base_urls[mid] = e["base_url"]
    if not_ready:
        raise HTTPException(
            status_code=409,
            detail={"message": "some shims are not ready", "members": not_ready},
        )

    scores = _member_best_scores()

    async def _call_one(mid: str) -> dict[str, Any]:
        t0 = time.time()
        try:
            resp = await asyncio.to_thread(_post_to_shim, base_urls[mid], path, payload, 120.0)
        except urllib.error.HTTPError as ex:
            body_txt = ex.read().decode(errors="replace")[:240]
            return {"member_id": mid, "error": f"{ex.code}: {body_txt}"}
        except Exception as ex:  # noqa: BLE001
            return {"member_id": mid, "error": f"unreachable: {ex}"}
        elapsed_s = time.time() - t0
        try:
            choice = resp["choices"][0]
            content = choice["message"]["content"] if mode == "chat" else choice["text"]
            finish_reason = choice.get("finish_reason")
        except (KeyError, IndexError) as ex:
            return {"member_id": mid, "error": f"malformed reply: {ex}"}
        return {
            "member_id": mid,
            "content": content,
            "elapsed_s": elapsed_s,
            "usage": resp.get("usage", {}),
            "finish_reason": finish_reason,
            "answer_letter": _extract_mcq_letter(content),
            "weight": scores.get(mid, 1.0),
        }

    t_all = time.time()
    results = await asyncio.gather(*(_call_one(mid) for mid in all_ids))
    logger.info(
        "playground/ensemble: n_unique=%d a=%d b=%d strategy=%s elapsed=%.2fs",
        len(all_ids), len(a_ids), len(b_ids), strategy, time.time() - t_all,
    )
    by_id = {r["member_id"]: r for r in results}

    a_outs = [by_id[m] for m in a_ids]
    b_outs = [by_id[m] for m in b_ids]
    return JSONResponse({
        "mode": mode,
        "strategy": strategy,
        "a": {"members": a_outs, "aggregate": _aggregate_letters(a_outs, strategy)},
        "b": {"members": b_outs, "aggregate": _aggregate_letters(b_outs, strategy)},
    })


@app.get("/api/openweight/playground/card/{member_id}")
async def playground_card(member_id: str) -> JSONResponse:
    """Compact model-context snapshot for the UI card. No shim required."""
    row = _query_one("SELECT * FROM members WHERE id = ?", (member_id,))
    if not row:
        raise HTTPException(status_code=404, detail=f"no member {member_id!r}")
    parsed = _parse_openweight_member_row(row)
    model_path, adapter = _resolve_playground_model_path(parsed)
    role = _classify_model_role(model_path or "", parsed.get("kind") or "", adapter)
    with _PLAYGROUND_LOCK:
        e = _PLAYGROUND_SHIMS.get(member_id) or {}
        shim_status = e.get("status", "idle")
        port = e.get("port")
    return JSONResponse({
        "member_id": member_id,
        "kind": parsed.get("kind"),
        "role": role,
        "base_model_id": parsed.get("base_model_id"),
        "adapter_path": adapter,
        "merged_checkpoint_path": parsed.get("merged_checkpoint_path"),
        "tokenizer_family": parsed.get("tokenizer_family"),
        "model_path": model_path,
        "shim_status": shim_status,
        "shim_port": port,
    })


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """Realtime updates via WebSocket.

    On connect, sends the full current state. Then the background poller
    pushes incremental updates as new data appears in the DB.
    """
    await manager.connect(ws)

    try:
        # Send initial state on connect
        status = _build_status()
        await ws.send_text(json.dumps({
            "type": "status_update",
            "data": status,
        }))

        # If there is a latest generation, send its data too
        if status["current_generation"] is not None:
            gen_data = _query_one(
                "SELECT * FROM generations WHERE num = ?",
                (status["current_generation"],),
            )
            if gen_data:
                await ws.send_text(json.dumps({
                    "type": "generation_update",
                    "data": gen_data,
                }))

        # Keep connection alive -- the background poller handles broadcasting.
        # We just need to read (and discard) any incoming messages to detect
        # disconnection.
        while True:
            await ws.receive_text()

    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("WebSocket error")
    finally:
        manager.disconnect(ws)


# ---------------------------------------------------------------------------
# Openweight lab page — standalone HTML served from disk
# ---------------------------------------------------------------------------

OPENWEIGHT_HTML_PATH = Path(__file__).parent / "openweight.html"


@app.get("/openweight", response_class=HTMLResponse)
async def openweight_page() -> HTMLResponse:
    if not OPENWEIGHT_HTML_PATH.exists():
        return HTMLResponse(
            "<h1>openweight.html missing</h1>", status_code=500,
        )
    return HTMLResponse(OPENWEIGHT_HTML_PATH.read_text())


# ---------------------------------------------------------------------------
# SPA static file serving
# ---------------------------------------------------------------------------

def _mount_frontend(app: FastAPI) -> None:
    """Mount the frontend dist directory with SPA fallback.

    Static files are tried first. If no matching file is found, serve
    index.html so client-side routing works.
    """
    dist = FRONTEND_DIR.resolve()
    if not dist.is_dir():
        logger.warning(
            "Frontend dist directory not found at %s -- "
            "static serving disabled. Run 'npm run build' in dashboard/frontend/.",
            dist,
        )
        return

    index_html = dist / "index.html"

    # Mount static assets (JS/CSS/images) under /assets if the directory exists.
    # Vite places bundled JS/CSS here by default.
    assets_dir = dist / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/{full_path:path}")
    async def spa_fallback(full_path: str) -> FileResponse:
        """Serve static files from the frontend build, falling back to index.html."""
        if full_path:
            requested = (dist / full_path).resolve()
            # Serve the file only if it exists and is inside the dist directory
            # (prevents path traversal)
            if requested.is_file() and str(requested).startswith(str(dist)):
                return FileResponse(str(requested))

        # SPA fallback -- serve index.html for all unmatched routes
        return FileResponse(str(index_html))


# Mount frontend as the last step so API routes take priority
_mount_frontend(app)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SIEVE dashboard server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--db",
        type=str,
        default="./results/lineage.db",
        help="Path to the lineage SQLite database",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Path to the checkpoint directory (for checkpoint stats)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8096,
        help="Port to bind the server to",
    )
    parser.add_argument(
        "--launch-dispatch",
        type=str,
        default=os.environ.get("SIEVE_LAUNCH_DISPATCH"),
        help=(
            "Shell wrapper that receives the python command to run as its "
            "last argument. Example: 'scripts/launch_openweight_on_node.sh'. "
            "If unset, launches run locally."
        ),
    )
    global DB_PATH, CHECKPOINT_DIR, LAUNCH_DISPATCH_CMD, SIEVE_ROOT
    parser.add_argument(
        "--sieve-root",
        type=str,
        default=str(SIEVE_ROOT),
        help="Root of the sieve checkout (for resolving relative paths)",
    )
    args = parser.parse_args()

    DB_PATH = args.db
    CHECKPOINT_DIR = args.checkpoint_dir
    LAUNCH_DISPATCH_CMD = args.launch_dispatch
    SIEVE_ROOT = Path(args.sieve_root).resolve()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    logger.info("Starting SIEVE dashboard on %s:%d", args.host, args.port)
    logger.info("Database: %s", DB_PATH)
    if CHECKPOINT_DIR:
        logger.info("Checkpoint dir: %s", CHECKPOINT_DIR)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        timeout_keep_alive=5,
        limit_concurrency=200,
        backlog=512,
    )


if __name__ == "__main__":
    main()
