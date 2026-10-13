#!/usr/bin/env python3
"""Live dashboard for the exp007 run (parallel Redis dispatch).

Reads three sources and re-renders every few seconds:
  1. orchestrator log  — generation boundaries, ensemble eval summaries
  2. Redis heartbeats  — which worker is training which member right now
  3. Redis result keys — per-member val_loss as jobs complete

Run with:
    .venv/bin/python scripts/watch_exp007.py
    .venv/bin/python scripts/watch_exp007.py --once        # one snapshot
    .venv/bin/python scripts/watch_exp007.py --plot        # write loss curve PNG
    .venv/bin/python scripts/watch_exp007.py --run RUN_ID  # different run

Ctrl-C to exit.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import redis
except ImportError:
    print("pip install redis", file=sys.stderr)
    sys.exit(1)

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

DEFAULT_REDIS = "redis://<node-ip>:6379/0"
DEFAULT_LOG = "/home/<user>/dev/sieve/runs/exp007_v009/logs/orchestrator.log"
# log(V) line for v009 corpus (vocab_size = 8000)
LOG_V = math.log(8000)

# Rich regex: match the per-gen summary the orchestrator prints.
# "[Generation 2] Best ensemble loss: 8.4123 | Best individual: 7.8934 | Diversity: 0.2310 | Time: 812.3s"
RE_GEN_SUMMARY = re.compile(
    r"\[Generation (\d+)\] Best ensemble loss: ([\d.]+) \| "
    r"Best individual: ([\d.]+) \| Diversity: ([\d.]+) \| Time: ([\d.]+)s"
)
RE_GEN_START = re.compile(r"\[Generation (\d+)\] Training (\d+) members")
RE_RUN_ID = re.compile(r"run_id=(\S+)")
RE_POP = re.compile(r"Population size:\s+(\d+)")
RE_GENS = re.compile(r"Max generations:\s+(\d+)")


@dataclass
class GenRow:
    generation: int
    best_ens: Optional[float] = None
    best_ind: Optional[float] = None
    diversity: Optional[float] = None
    wall_s: Optional[float] = None
    members_done: int = 0
    members_total: int = 0


@dataclass
class State:
    run_id: Optional[str] = None
    population: int = 0
    max_gens: int = 0
    current_gen: int = 0
    gens: dict[int, GenRow] = field(default_factory=dict)
    workers: dict[str, tuple[str, float]] = field(default_factory=dict)  # node -> (state, ts)
    members: dict[tuple[int, str], dict] = field(default_factory=dict)   # (gen, mid) -> result
    log_path: Path = Path(DEFAULT_LOG)


def parse_log(state: State) -> None:
    """Parse the orchestrator log for generation markers + summaries."""
    if not state.log_path.exists():
        return
    text = state.log_path.read_text(errors="ignore")
    for line in text.splitlines():
        if (m := RE_RUN_ID.search(line)) and state.run_id is None:
            state.run_id = m.group(1)
        if (m := RE_POP.search(line)) and state.population == 0:
            state.population = int(m.group(1))
        if (m := RE_GENS.search(line)) and state.max_gens == 0:
            state.max_gens = int(m.group(1))
        if m := RE_GEN_START.search(line):
            gen, total = int(m.group(1)), int(m.group(2))
            row = state.gens.setdefault(gen, GenRow(generation=gen))
            row.members_total = total
            state.current_gen = max(state.current_gen, gen)
        if m := RE_GEN_SUMMARY.search(line):
            gen = int(m.group(1))
            row = state.gens.setdefault(gen, GenRow(generation=gen))
            row.best_ens = float(m.group(2))
            row.best_ind = float(m.group(3))
            row.diversity = float(m.group(4))
            row.wall_s = float(m.group(5))


def poll_redis(r: redis.Redis, state: State) -> None:
    """Read worker heartbeats + all result keys for the current run."""
    # worker heartbeats
    state.workers.clear()
    for key in r.scan_iter(match="sieve:workers:heartbeat:*", count=200):
        raw = r.get(key)
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue
        node = key.decode().split(":")[-1].rsplit("-", 1)[0]
        state.workers[node] = (data.get("state", "?"), data.get("ts", 0.0))

    # member results for this run
    if not state.run_id:
        return
    pattern = f"sieve:results:{state.run_id}:*"
    for key in r.scan_iter(match=pattern, count=500):
        raw = r.get(key)
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue
        parts = key.decode().split(":")
        if len(parts) < 5:
            continue
        try:
            gen = int(parts[3])
        except ValueError:
            continue
        mid = parts[4]
        state.members[(gen, mid)] = data
        row = state.gens.setdefault(gen, GenRow(generation=gen))
        # Count done per generation from the results we have
    # Recompute members_done per gen from Redis results
    gen_done: dict[int, int] = {}
    for (gen, _), _ in state.members.items():
        gen_done[gen] = gen_done.get(gen, 0) + 1
    for gen, count in gen_done.items():
        state.gens.setdefault(gen, GenRow(generation=gen)).members_done = count


def sparkline(values: list[float]) -> str:
    """Unicode sparkline for a loss series."""
    if not values:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return blocks[0] * len(values)
    return "".join(blocks[int((v - lo) / (hi - lo) * (len(blocks) - 1))] for v in values)


def render(state: State) -> Group:
    # Header
    header = Text()
    header.append("SIEVE exp007 live view\n", style="bold magenta")
    if state.run_id:
        header.append(f"run_id = {state.run_id}   ", style="dim")
    header.append(f"log(V) = {LOG_V:.4f} (vocab 8000)\n", style="dim")
    header.append(f"pop={state.population}  max_gens={state.max_gens}  ")
    header.append(f"current_gen={state.current_gen}\n", style="cyan")

    # Per-generation table
    gt = Table(title="Generations", show_header=True, header_style="bold cyan", expand=False)
    gt.add_column("gen", justify="right")
    gt.add_column("members", justify="right")
    gt.add_column("best ind", justify="right")
    gt.add_column("vs log(V)", justify="right")
    gt.add_column("best ens", justify="right")
    gt.add_column("ens vs ind", justify="right")
    gt.add_column("diversity", justify="right")
    gt.add_column("wall (s)", justify="right")

    ens_series: list[float] = []
    ind_series: list[float] = []
    for gen in sorted(state.gens.keys()):
        row = state.gens[gen]
        prog = f"{row.members_done}/{row.members_total}" if row.members_total else f"{row.members_done}/?"
        bi = f"{row.best_ind:.4f}" if row.best_ind is not None else "—"
        be = f"{row.best_ens:.4f}" if row.best_ens is not None else "—"
        vs_logv = "—"
        if row.best_ind is not None:
            rel = row.best_ind / LOG_V
            style = "green" if rel < 0.9 else ("yellow" if rel < 1.0 else "red")
            vs_logv = Text(f"{rel:+.2%}", style=style) if False else f"{rel:.2f}×"
        ens_vs_ind = "—"
        if row.best_ens is not None and row.best_ind is not None:
            delta = row.best_ens - row.best_ind
            sign = "+" if delta >= 0 else ""
            ens_vs_ind = f"{sign}{delta:.4f}"
        div = f"{row.diversity:.4f}" if row.diversity is not None else "—"
        wall = f"{row.wall_s:.1f}" if row.wall_s is not None else "—"
        gt.add_row(str(gen), prog, bi, vs_logv, be, ens_vs_ind, div, wall)
        if row.best_ens is not None:
            ens_series.append(row.best_ens)
        if row.best_ind is not None:
            ind_series.append(row.best_ind)

    # Loss sparklines
    lines = Text()
    if ens_series:
        lines.append("best_ens:  ", style="cyan")
        lines.append(sparkline(ens_series), style="green")
        lines.append(f"  [{min(ens_series):.3f} … {max(ens_series):.3f}]\n", style="dim")
    if ind_series:
        lines.append("best_ind:  ", style="cyan")
        lines.append(sparkline(ind_series), style="green")
        lines.append(f"  [{min(ind_series):.3f} … {max(ind_series):.3f}]\n", style="dim")

    # Workers table
    wt = Table(title="Workers (Redis heartbeats)", show_header=True,
               header_style="bold cyan", expand=False)
    wt.add_column("node")
    wt.add_column("state")
    wt.add_column("age (s)", justify="right")
    now = time.time()
    for node in sorted(state.workers.keys()):
        wstate, ts = state.workers[node]
        age = now - ts
        style = "green" if age < 15 else ("yellow" if age < 45 else "red")
        wt.add_row(node, str(wstate), Text(f"{age:.1f}", style=style))

    # Most recent member results
    rt = Table(title="Recent member results (last 12)", show_header=True,
               header_style="bold cyan", expand=False)
    rt.add_column("gen", justify="right")
    rt.add_column("member")
    rt.add_column("worker")
    rt.add_column("status")
    rt.add_column("val_loss", justify="right")
    rt.add_column("wall (s)", justify="right")
    rt.add_column("vs log(V)", justify="right")

    items = sorted(
        state.members.items(),
        key=lambda kv: kv[1].get("wallclock_s", 0.0),
        reverse=True,
    )[:12]
    for (gen, mid), data in items:
        vl = data.get("val_loss")
        wall = data.get("wallclock_s")
        worker = (data.get("worker_id") or "?").rsplit("-", 1)[0]
        vl_str = f"{vl:.4f}" if isinstance(vl, (int, float)) else "—"
        vs_logv = "—"
        if isinstance(vl, (int, float)) and vl > 0:
            rel = vl / LOG_V
            style = "green" if rel < 0.9 else ("yellow" if rel < 1.0 else "red")
            vs_logv = Text(f"{rel:.2f}×", style=style)
        rt.add_row(
            str(gen), mid[:8], worker, str(data.get("status", "?")),
            vl_str,
            f"{wall:.1f}" if isinstance(wall, (int, float)) else "—",
            vs_logv,
        )

    return Group(Panel(header, title="exp007_v009"), gt, Panel(lines, title="loss history"), wt, rt)


def write_plot(state: State, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skip --plot", file=sys.stderr)
        return
    gens = sorted(state.gens.keys())
    ens = [state.gens[g].best_ens for g in gens]
    ind = [state.gens[g].best_ind for g in gens]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=110)
    ax.axhline(LOG_V, ls="--", color="grey", label=f"log(V)={LOG_V:.3f}")
    if any(v is not None for v in ind):
        ax.plot(gens, ind, marker="o", label="best individual")
    if any(v is not None for v in ens):
        ax.plot(gens, ens, marker="s", label="best ensemble")
    ax.set_xlabel("generation")
    ax.set_ylabel("val cross-entropy (nats)")
    ax.set_title(f"exp007_v009 — {state.run_id}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--redis", default=DEFAULT_REDIS)
    ap.add_argument("--log", default=DEFAULT_LOG, help="orchestrator log path")
    ap.add_argument("--run", default=None, help="run_id override (else auto-detected from log)")
    ap.add_argument("--once", action="store_true", help="one snapshot, no live refresh")
    ap.add_argument("--refresh", type=float, default=4.0, help="seconds between refreshes")
    ap.add_argument("--plot", type=Path, default=None, help="write loss curve PNG and exit")
    args = ap.parse_args()

    r = redis.Redis.from_url(args.redis)
    state = State(log_path=Path(args.log))
    if args.run:
        state.run_id = args.run

    def tick() -> Group:
        parse_log(state)
        poll_redis(r, state)
        return render(state)

    if args.plot:
        parse_log(state)
        poll_redis(r, state)
        write_plot(state, args.plot)
        return

    if args.once:
        Console().print(tick())
        return

    console = Console()
    try:
        with Live(tick(), console=console, refresh_per_second=1) as live:
            while True:
                time.sleep(args.refresh)
                live.update(tick())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
