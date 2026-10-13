"""SIEVE lab bench — FastAPI server for interactive checkpoint exploration.

Auto-discovers all .pt checkpoints under runs/*/checkpoints/, holds them
GPU-resident on demand, and serves a fan-out generation API:

  GET  /api/checkpoints       — every checkpoint with full hyperparam metadata
  GET  /api/ensembles         — saved best_ensemble.json specs per run
  GET  /api/tokenizers        — available tokenizer paths
  POST /api/complete          — fan one prompt across N models in parallel
  GET  /                      — single-page lab UI

Run from repo root:
  .venv/bin/python -m lab.server                     # default port 8099
  .venv/bin/python -m lab.server --port 8099 --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, Optional

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Repo root is the parent of this `lab/` directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from worker.checkpoint import load_checkpoint, load_model_from_checkpoint  # noqa: E402

logger = logging.getLogger("sieve.lab")


# ---------------------------------------------------------------------------
# Registry — discovers checkpoints + ensembles on startup
# ---------------------------------------------------------------------------

class CheckpointEntry(BaseModel):
    key: str                    # stable identifier: "<run>/<member_dir>"
    run: str                    # "smoke_loop", "smoke_loop_v005", etc.
    member_dir: str             # "gen-002-c884ab4b" or "step_NNN_epoch_NN" for non-loop runs
    generation: Optional[int] = None
    member_id: Optional[str] = None
    path: str                   # absolute path
    filename: str
    file_size_mb: float
    val_loss: Optional[float] = None
    train_loss: Optional[float] = None
    epoch: Optional[int] = None
    global_step: Optional[int] = None
    config: dict[str, Any] = Field(default_factory=dict)
    n_params: Optional[int] = None
    timestamp: Optional[float] = None
    tokenizer_path: Optional[str] = None  # inferred per-run from corpus metadata


class EnsembleEntry(BaseModel):
    key: str                    # "ens:<run>/<name>"
    run: str
    name: str                   # "best", "gen2-top1", etc.
    member_keys: list[str]      # checkpoint keys
    member_paths: list[str]     # absolute paths (canonical)
    val_loss: Optional[float] = None
    diversity_score: Optional[float] = None
    aggregation: Literal["logit", "prob"] = "logit"
    source: str                 # "best_ensemble.json" or "lineage.db"


def _scan_checkpoints(repo_root: Path) -> list[CheckpointEntry]:
    """Walk runs/*/checkpoints/ and pull metadata from each .pt file."""
    entries: list[CheckpointEntry] = []
    runs_dir = repo_root / "runs"
    if not runs_dir.exists():
        logger.warning("no runs/ directory at %s", runs_dir)
        return entries

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        ckpt_root = run_dir / "checkpoints"
        if not ckpt_root.exists():
            continue

        # Per-run tokenizer: try to infer from a corpus metadata file referenced
        # in any saved ensemble or just default to v002 (the manual says all
        # current runs use v002).
        tokenizer_path = str(repo_root / "data" / "tokenizers" / "v002" / "tokenizer.json")

        for pt_file in sorted(ckpt_root.rglob("*.pt")):
            if pt_file.name.endswith(".pt.tmp"):
                continue
            try:
                ckpt = torch.load(str(pt_file), map_location="cpu", weights_only=False)
            except Exception as e:
                logger.warning("skip %s: %s", pt_file, e)
                continue

            cfg = ckpt.get("config", {}) or {}
            metrics = ckpt.get("metrics", {}) or {}

            # member_dir = the parent directory of the .pt (e.g. gen-002-c884ab4b)
            # For runs without per-member subdirs (smoke_v003), use the file stem.
            if pt_file.parent == ckpt_root:
                member_dir = pt_file.stem
            else:
                member_dir = pt_file.parent.name

            # Derive generation + member_id from the dir name pattern when the
            # checkpoint metrics dict didn't include them (older smoke_loop runs
            # only store epoch/step/loss in metrics).
            derived_gen = metrics.get("generation")
            derived_member_id = metrics.get("member_id")
            import re as _re
            m = _re.match(r"gen-(\d+)-([0-9a-f]+)$", member_dir)
            if m:
                if derived_gen is None:
                    derived_gen = int(m.group(1))
                if derived_member_id is None:
                    derived_member_id = m.group(2)

            # Stable key: "<run>/<member_dir>" with file stem suffix when
            # multiple checkpoints share a member_dir (only happens for old smoke_v003).
            siblings = list(pt_file.parent.glob("*.pt"))
            key = f"{run_dir.name}/{member_dir}"
            if len(siblings) > 1 and pt_file.parent != ckpt_root:
                key = f"{key}/{pt_file.stem}"

            n_params = None
            if cfg:
                # vocab × n_embd (tied) + 12 × n_layer × n_embd² is approximate;
                # use the same calc as ModelConfig.n_params if present.
                try:
                    from model.config import ModelConfig
                    n_params = ModelConfig.from_dict(cfg).n_params
                except Exception:
                    pass

            entries.append(CheckpointEntry(
                key=key,
                run=run_dir.name,
                member_dir=member_dir,
                generation=derived_gen,
                member_id=derived_member_id,
                path=str(pt_file),
                filename=pt_file.name,
                file_size_mb=pt_file.stat().st_size / (1024 * 1024),
                val_loss=metrics.get("val_loss") or metrics.get("best_loss"),
                train_loss=metrics.get("train_loss"),
                epoch=metrics.get("epoch"),
                global_step=metrics.get("global_step"),
                config=cfg,
                n_params=n_params,
                timestamp=ckpt.get("timestamp"),
                tokenizer_path=tokenizer_path,
            ))
    return entries


def _scan_ensembles(repo_root: Path, ckpt_index: dict[str, CheckpointEntry]) -> list[EnsembleEntry]:
    """Load best_ensemble.json from each run; map members back to checkpoint keys."""
    out: list[EnsembleEntry] = []
    path_to_key = {Path(c.path).resolve(): c.key for c in ckpt_index.values()}

    for run_dir in sorted((repo_root / "runs").glob("*")):
        results = run_dir / "results"
        best = results / "best_ensemble.json"
        if not best.exists():
            continue
        try:
            spec = json.loads(best.read_text())
        except Exception as e:
            logger.warning("bad %s: %s", best, e)
            continue

        member_paths_raw = [m["checkpoint"] for m in spec.get("members", [])]
        member_paths = [str((repo_root / p).resolve()) for p in member_paths_raw]
        member_keys: list[str] = []
        for mp in member_paths:
            k = path_to_key.get(Path(mp))
            if k is not None:
                member_keys.append(k)
            else:
                # fallback: synthesize from path
                member_keys.append(f"path:{mp}")

        out.append(EnsembleEntry(
            key=f"ens:{run_dir.name}/best",
            run=run_dir.name,
            name=spec.get("ensemble_id", "best"),
            member_keys=member_keys,
            member_paths=member_paths,
            val_loss=spec.get("val_loss"),
            diversity_score=spec.get("diversity_score"),
            aggregation="logit" if spec.get("aggregation") == "logit_average" else "logit",
            source="best_ensemble.json",
        ))
    return out


# ---------------------------------------------------------------------------
# Lazy GPU-resident model cache
# ---------------------------------------------------------------------------

class ModelCache:
    """Holds GPT models in memory (GPU when available), keyed by absolute path."""

    def __init__(self, device: str) -> None:
        self.device = device
        self._cache: dict[str, torch.nn.Module] = {}

    def get(self, path: str) -> torch.nn.Module:
        abs_path = str(Path(path).resolve())
        if abs_path not in self._cache:
            t0 = time.time()
            model = load_model_from_checkpoint(abs_path, device=self.device)
            model.eval()
            logger.info("cache load: %s in %.2fs", abs_path, time.time() - t0)
            self._cache[abs_path] = model
        return self._cache[abs_path]

    def loaded_keys(self) -> list[str]:
        return list(self._cache.keys())


# ---------------------------------------------------------------------------
# Tokenizer cache
# ---------------------------------------------------------------------------

class TokenizerCache:
    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    def get(self, path: str) -> Any:
        if path not in self._cache:
            from tokenizers import Tokenizer as HFTokenizer
            self._cache[path] = HFTokenizer.from_file(path)
        return self._cache[path]


# ---------------------------------------------------------------------------
# Generation primitives
# ---------------------------------------------------------------------------

@torch.no_grad()
def _generate_single(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
) -> torch.Tensor:
    """Single-model autoregressive generation. Mirrors GPT.generate but is a
    standalone function so the ensemble path can share the inner step."""
    block_size = model.config.block_size
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


@torch.no_grad()
def _generate_ensemble(
    models: list[torch.nn.Module],
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
    aggregation: Literal["logit", "prob"],
    capture_contributions: bool,
) -> tuple[torch.Tensor, Optional[list[dict[str, Any]]]]:
    """Ensemble generation by averaging logits (or probs) across N models per step.

    If capture_contributions is True, also returns a per-step list with the
    averaged top-k tokens and each member's logprob assigned to those tokens —
    enables the "show member contributions" UI.
    """
    block_size = min(m.config.block_size for m in models)
    contributions: list[dict[str, Any]] = [] if capture_contributions else None  # type: ignore

    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        per_member_logits: list[torch.Tensor] = []
        for m in models:
            logits, _ = m(idx_cond)
            per_member_logits.append(logits[:, -1, :])  # (1, V)
        stacked = torch.stack(per_member_logits, dim=0)  # (M, 1, V)

        if aggregation == "prob":
            avg_probs = F.softmax(stacked, dim=-1).mean(dim=0)  # (1, V)
            avg_logits = torch.log(avg_probs.clamp_min(1e-30))
        else:
            avg_logits = stacked.mean(dim=0)  # (1, V)

        sample_logits = avg_logits / max(temperature, 1e-6)
        if top_k is not None:
            v, _ = torch.topk(sample_logits, min(top_k, sample_logits.size(-1)))
            sample_logits[sample_logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(sample_logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        if capture_contributions:
            # Top-5 from the averaged distribution and each member's logprob on those tokens
            k = 5
            top_logits, top_ids = torch.topk(avg_logits[0], k)
            top_ids_list = top_ids.tolist()
            avg_logprobs = F.log_softmax(avg_logits[0], dim=-1)[top_ids].tolist()
            per_member = []
            for m_logits in per_member_logits:
                m_logprobs = F.log_softmax(m_logits[0], dim=-1)[top_ids].tolist()
                per_member.append(m_logprobs)
            contributions.append({
                "top_token_ids": top_ids_list,
                "avg_logprobs": avg_logprobs,
                "per_member_logprobs": per_member,
                "chosen_token_id": int(idx_next.item()),
            })

        idx = torch.cat((idx, idx_next), dim=1)

    return idx, contributions


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ModelSpec(BaseModel):
    key: str                                          # opaque identifier echoed back
    kind: Literal["checkpoint", "ensemble"]
    paths: list[str] = Field(default_factory=list)    # 1 entry for checkpoint, N for ensemble
    aggregation: Literal["logit", "prob"] = "logit"


class CompleteRequest(BaseModel):
    models: list[ModelSpec]
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 0.8
    top_k: Optional[int] = 40
    seed: Optional[int] = None
    show_contributions: bool = False
    tokenizer_path: Optional[str] = None  # default = v002


class PaneResult(BaseModel):
    key: str
    completion: str             # generated text only (NOT including the prompt)
    full_text: str              # prompt + completion
    tokens_generated: int
    elapsed_ms: float
    contributions: Optional[list[dict[str, Any]]] = None
    error: Optional[str] = None


class CompleteResponse(BaseModel):
    prompt_token_count: int
    results: list[PaneResult]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class State:
    repo_root: Path
    device: str
    checkpoints: dict[str, CheckpointEntry]
    ensembles: dict[str, EnsembleEntry]
    model_cache: ModelCache
    tokenizer_cache: TokenizerCache
    default_tokenizer_path: str


_state = State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()
    ckpts = _scan_checkpoints(_state.repo_root)
    _state.checkpoints = {c.key: c for c in ckpts}
    enss = _scan_ensembles(_state.repo_root, _state.checkpoints)
    _state.ensembles = {e.key: e for e in enss}
    logger.info(
        "discovered %d checkpoints, %d saved ensembles in %.2fs",
        len(_state.checkpoints), len(_state.ensembles), time.time() - t0,
    )
    yield


app = FastAPI(title="SIEVE lab bench", lifespan=lifespan)


@app.get("/api/checkpoints")
def api_checkpoints() -> dict[str, Any]:
    # Group by run for the UI's sidebar.
    by_run: dict[str, list[dict[str, Any]]] = {}
    for c in _state.checkpoints.values():
        by_run.setdefault(c.run, []).append(c.model_dump())
    # Sort each run by generation, then val_loss
    for run, lst in by_run.items():
        lst.sort(key=lambda r: (r.get("generation") if r.get("generation") is not None else -1,
                                r.get("val_loss") if r.get("val_loss") is not None else 9e9))
    return {"runs": by_run, "device": _state.device, "loaded": _state.model_cache.loaded_keys()}


@app.get("/api/ensembles")
def api_ensembles() -> dict[str, Any]:
    by_run: dict[str, list[dict[str, Any]]] = {}
    for e in _state.ensembles.values():
        by_run.setdefault(e.run, []).append(e.model_dump())
    return {"runs": by_run}


@app.get("/api/tokenizers")
def api_tokenizers() -> dict[str, Any]:
    tdir = _state.repo_root / "data" / "tokenizers"
    found = []
    if tdir.exists():
        for v in sorted(tdir.iterdir()):
            tj = v / "tokenizer.json"
            if tj.exists():
                found.append({"version": v.name, "path": str(tj)})
    return {"tokenizers": found, "default": _state.default_tokenizer_path}


@app.post("/api/complete", response_model=CompleteResponse)
def api_complete(req: CompleteRequest) -> CompleteResponse:
    if not req.models:
        raise HTTPException(400, "models[] is empty")

    tok_path = req.tokenizer_path or _state.default_tokenizer_path
    tok = _state.tokenizer_cache.get(tok_path)
    prompt_ids = tok.encode(req.prompt).ids
    if not prompt_ids:
        # Fall back to a single BOS-ish token so generation has something to condition on.
        prompt_ids = [0]
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=_state.device)

    results: list[PaneResult] = []
    for spec in req.models:
        t0 = time.time()
        try:
            if req.seed is not None:
                torch.manual_seed(req.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(req.seed)

            if spec.kind == "checkpoint":
                if len(spec.paths) != 1:
                    raise ValueError("checkpoint kind requires exactly one path")
                model = _state.model_cache.get(spec.paths[0])
                out_ids = _generate_single(
                    model, prompt_tensor, req.max_new_tokens, req.temperature, req.top_k,
                )
                contributions = None
            elif spec.kind == "ensemble":
                if not spec.paths:
                    raise ValueError("ensemble kind requires at least one member path")
                models = [_state.model_cache.get(p) for p in spec.paths]
                out_ids, contributions = _generate_ensemble(
                    models, prompt_tensor, req.max_new_tokens, req.temperature, req.top_k,
                    aggregation=spec.aggregation,
                    capture_contributions=req.show_contributions,
                )
            else:
                raise ValueError(f"unknown kind: {spec.kind}")

            full_ids = out_ids[0].tolist()
            gen_ids = full_ids[len(prompt_ids):]
            full_text = tok.decode(full_ids)
            completion = tok.decode(gen_ids)

            results.append(PaneResult(
                key=spec.key,
                completion=completion,
                full_text=full_text,
                tokens_generated=len(gen_ids),
                elapsed_ms=(time.time() - t0) * 1000.0,
                contributions=contributions,
            ))
        except Exception as e:
            logger.exception("generation failed for %s", spec.key)
            results.append(PaneResult(
                key=spec.key, completion="", full_text=req.prompt,
                tokens_generated=0, elapsed_ms=(time.time() - t0) * 1000.0,
                error=str(e),
            ))

    return CompleteResponse(prompt_token_count=len(prompt_ids), results=results)


# Static files + index
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SIEVE lab bench server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument(
        "--device", default=None,
        help="torch device (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--repo-root", default=str(REPO_ROOT),
        help="path to the sieve repo root (default: parent of this file)",
    )
    parser.add_argument(
        "--tokenizer", default=None,
        help="default tokenizer.json path (default: data/tokenizers/v002/tokenizer.json)",
    )
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    _state.repo_root = Path(args.repo_root).resolve()
    _state.device = device
    _state.checkpoints = {}
    _state.ensembles = {}
    _state.model_cache = ModelCache(device=device)
    _state.tokenizer_cache = TokenizerCache()
    _state.default_tokenizer_path = (
        args.tokenizer
        or str(_state.repo_root / "data" / "tokenizers" / "v002" / "tokenizer.json")
    )

    logger.info("repo=%s device=%s", _state.repo_root, device)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
