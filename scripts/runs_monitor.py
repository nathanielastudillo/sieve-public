#!/usr/bin/env python3
"""Live multi-run monitor for SIEVE.

Polls run.log files across local + remote (ssh) nodes and renders a live table:
  - per-run: node, run name, gen, member i/N, last epoch, loss, tok/s, elapsed, ETA

Usage:
    python scripts/runs_monitor.py            # default set of runs
    python scripts/runs_monitor.py --once     # single snapshot instead of live

Runs are configured in the RUNS list below.
"""
from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.table import Table


@dataclass
class Run:
    label: str                  # human-friendly label (e.g. "v007 / 4K")
    node: str                   # "local" or ssh target like "node-1"
    log_path: str               # path to run.log on that node
    run_name: str = ""          # for display; defaults to log dir name


# ---------- Configure runs here ----------

RUNS: list[Run] = [
    # exp007 v009 parallel dispatch — orchestrator local, 6 worker daemons remote.
    # The orchestrator log captures gen boundaries + ensemble eval summaries;
    # the worker logs capture per-epoch training progress of whatever member
    # happens to be on that GPU right now.
    Run("exp007 orch",  "local",  "/home/<user>/dev/sieve/runs/exp007_v009/logs/orchestrator.log", "exp007_orch"),
    Run("exp007 node-4", "node-4",  "/home/<user>/sieve/runs/worker_daemon.log", "exp007_node-4"),
    Run("exp007 node-6", "node-6",  "/home/<user>/sieve/runs/worker_daemon.log", "exp007_node-6"),
    Run("exp007 node-7", "node-7",  "/home/<user>/sieve/runs/worker_daemon.log", "exp007_node-7"),
    Run("exp007 node-9", "node-9",  "/home/<user>/sieve/runs/worker_daemon.log", "exp007_node-9"),
    Run("exp007 node-10","node-10", "/home/<user>/sieve/runs/worker_daemon.log", "exp007_node-10"),
    Run("exp007 node-12","node-12", "/home/<user>/sieve/runs/worker_daemon.log", "exp007_node-12"),
]

SSH_PASS = os.environ.get("SSH_PASS", "")   # set SSH_PASS in the environment
TAIL_LINES = 400                # how many lines of log to fetch each poll
REFRESH_SEC = 5.0


# ---------- Log parsing ----------

RE_POP       = re.compile(r"Population size:\s+(\d+)")
RE_MAX_GENS  = re.compile(r"Max generations:\s+(\d+)")
RE_GEN_MARK  = re.compile(r"\[Generation (\d+)\]")
RE_TRAIN_MEM = re.compile(r"Training member (\d+)/(\d+) \[([0-9a-f]+)\] \(gen (\d+)\)")
RE_MODEL_CFG = re.compile(r"ModelConfig\(.*?vocab_size=(\d+).*?~([\d.]+M) params")
RE_EPOCH     = re.compile(r"(\d\d:\d\d:\d\d).*?epoch=(\d+)/(\d+)\s+loss=([\d.]+)\s+best=([\d.]+)\s+lr=\S+\s+tok/s=(\d+)")
RE_MEM_DONE  = re.compile(r"(\d\d:\d\d:\d\d).*?Training complete: final_loss=([\d.]+)\s+best_loss=([\d.]+)\s+time=([\d.]+)s")
RE_PROFILE   = re.compile(r"Loaded profile:\s*(\S+)")


@dataclass
class RunState:
    alive: bool = False
    profile: str = "?"
    pop_size: int = 0
    max_gens: int = 0
    cur_gen: int = 0
    cur_member: int = 0      # 1-indexed
    cur_member_total: int = 0
    vocab: str = "?"
    params: str = "?"
    last_epoch: str = ""     # e.g. "3/10"
    last_loss: Optional[float] = None
    last_tok_s: Optional[int] = None
    last_timestamp: str = ""
    member_times_s: list[float] = field(default_factory=list)  # completed member wall times
    first_log_time: Optional[datetime] = None
    last_log_time: Optional[datetime] = None
    error: str = ""


def _fetch(run: Run) -> str:
    """Return last N lines of the run log. Empty string on failure."""
    if run.node == "local":
        p = Path(run.log_path)
        if not p.exists():
            return ""
        try:
            with p.open("rb") as f:
                # read last ~64KB
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 64 * 1024))
                data = f.read().decode("utf-8", errors="replace")
            return "\n".join(data.splitlines()[-TAIL_LINES:])
        except OSError:
            return ""
    else:
        cmd = [
            "ssh", "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=4",
            "-o", "BatchMode=no",
            run.node,
            f"tail -n {TAIL_LINES} {shlex.quote(run.log_path)} 2>/dev/null || true",
        ]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
            return out.stdout
        except subprocess.TimeoutExpired:
            return ""
        except FileNotFoundError:
            return ""


def _today_dt(hms: str) -> datetime:
    """Parse HH:MM:SS into a today-local datetime (approximation for ETA calc)."""
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    try:
        h, m, s = map(int, hms.split(":"))
        return today.replace(hour=h, minute=m, second=s)
    except Exception:
        return datetime.now()


def _parse(log: str) -> RunState:
    st = RunState()
    if not log.strip():
        return st
    st.alive = True

    for line in log.splitlines():
        if m := RE_PROFILE.search(line):
            st.profile = m.group(1)
        if m := RE_POP.search(line):
            st.pop_size = int(m.group(1))
        if m := RE_MAX_GENS.search(line):
            st.max_gens = int(m.group(1))
        if m := RE_GEN_MARK.search(line):
            st.cur_gen = int(m.group(1))
        if m := RE_TRAIN_MEM.search(line):
            st.cur_member = int(m.group(1))
            st.cur_member_total = int(m.group(2))
            st.cur_gen = int(m.group(4))
        if m := RE_MODEL_CFG.search(line):
            st.vocab = m.group(1)
            st.params = m.group(2)
        if m := RE_EPOCH.search(line):
            ts, e, E, loss, _best, toks = m.groups()
            st.last_epoch = f"{e}/{E}"
            st.last_loss = float(loss)
            st.last_tok_s = int(toks)
            st.last_timestamp = ts
            dt = _today_dt(ts)
            if st.first_log_time is None:
                st.first_log_time = dt
            st.last_log_time = dt
        if m := RE_MEM_DONE.search(line):
            ts, _final, _best, t = m.groups()
            st.member_times_s.append(float(t))
            st.last_timestamp = ts
            st.last_log_time = _today_dt(ts)
            if st.first_log_time is None:
                st.first_log_time = st.last_log_time
    return st


def _eta(st: RunState) -> str:
    """Crude ETA: avg-time-per-member * members_remaining + gens_remaining * pop_size."""
    if not st.member_times_s or st.pop_size == 0:
        return "-"
    avg = sum(st.member_times_s) / len(st.member_times_s)
    members_this_gen_left = max(0, st.pop_size - st.cur_member)
    gens_left_after = max(0, (st.max_gens or 1) - st.cur_gen - 1)
    remaining_s = avg * (members_this_gen_left + gens_left_after * st.pop_size)
    if remaining_s <= 0:
        return "~done"
    eta_dt = datetime.now() + timedelta(seconds=remaining_s)
    # human-readable
    if remaining_s < 3600:
        return f"{int(remaining_s/60)}m  ({eta_dt.strftime('%H:%M')})"
    return f"{remaining_s/3600:.1f}h ({eta_dt.strftime('%H:%M')})"


def _fresh(st: RunState) -> bool:
    if not st.last_log_time:
        return False
    # last log within 3 minutes ⇒ considered live
    return (datetime.now() - st.last_log_time).total_seconds() < 180


def render(states: list[tuple[Run, RunState]]) -> Table:
    t = Table(
        title=f"SIEVE runs — {datetime.now().strftime('%H:%M:%S')}",
        expand=True,
        show_lines=False,
    )
    t.add_column("Run", style="bold cyan", no_wrap=True)
    t.add_column("Node", no_wrap=True)
    t.add_column("Profile", style="magenta")
    t.add_column("Params", justify="right")
    t.add_column("Gen", justify="right")
    t.add_column("Member", justify="right")
    t.add_column("Epoch", justify="right")
    t.add_column("Loss", justify="right")
    t.add_column("tok/s", justify="right")
    t.add_column("Avg mem time", justify="right")
    t.add_column("ETA", no_wrap=True)
    t.add_column("Status", no_wrap=True)

    for run, st in states:
        if not st.alive:
            t.add_row(run.label, run.node, "-", "-", "-", "-", "-", "-", "-", "-", "-",
                      "[red]no log[/red]")
            continue
        status = "[green]live[/green]" if _fresh(st) else "[yellow]stale[/yellow]"
        avg = f"{sum(st.member_times_s)/len(st.member_times_s):.0f}s" if st.member_times_s else "-"
        mem = (f"{st.cur_member}/{st.cur_member_total}"
               if st.cur_member_total else "-")
        gen = (f"{st.cur_gen}/{max(0, st.max_gens-1)}"
               if st.max_gens else str(st.cur_gen))
        loss = f"{st.last_loss:.4f}" if st.last_loss is not None else "-"
        tok = f"{st.last_tok_s/1000:.0f}K" if st.last_tok_s else "-"
        t.add_row(
            run.label, run.node, st.profile,
            st.params, gen, mem, st.last_epoch or "-",
            loss, tok, avg, _eta(st), status,
        )
    return t


def snapshot(console: Console) -> None:
    states = [(r, _parse(_fetch(r))) for r in RUNS]
    console.print(render(states))


def live(console: Console) -> None:
    with Live(console=console, refresh_per_second=2, screen=False) as live_view:
        while True:
            try:
                states = [(r, _parse(_fetch(r))) for r in RUNS]
                live_view.update(render(states))
                time.sleep(REFRESH_SEC)
            except KeyboardInterrupt:
                break


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="single snapshot (no live loop)")
    args = ap.parse_args()
    console = Console()
    if args.once:
        snapshot(console)
    else:
        try:
            live(console)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
