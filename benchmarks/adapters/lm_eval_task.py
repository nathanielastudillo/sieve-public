"""lm-evaluation-harness adapter.

Runs a single lm-eval task against an OpenAI-compatible endpoint (sglang
or vLLM) as a subprocess. Parses the JSON results file, extracts the
headline metric, maps to ``[0, 1]`` higher-is-better.

Task-name mapping (SIEVE → lm-eval task id):

- ``ifeval`` → ``ifeval``
- ``hellaswag`` → ``hellaswag``
- ``mmlu_pro`` → ``mmlu_pro``
- ``gpqa_diamond`` → ``gpqa_diamond_zeroshot``

Score extraction uses the ``acc``/``acc_norm`` metric for accuracy tasks
and ``prompt_level_loose_acc`` for ifeval (higher = better for both).

Requires ``lm-eval`` on the worker PATH. If it's missing, the adapter
raises so the EvalResultMsg reports ``error=ModuleNotFoundError`` — the
orchestrator can see that and route around it.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


TASK_MAP: dict[str, str] = {
    "ifeval":       "ifeval",
    "hellaswag":    "hellaswag",
    "mmlu_pro":     "mmlu_pro",
    "gpqa_diamond": "gpqa_diamond_zeroshot",
}


HEADLINE_METRIC: dict[str, str] = {
    "ifeval":       "prompt_level_loose_acc,none",
    "hellaswag":    "acc_norm,none",
    "mmlu_pro":     "acc,none",
    "gpqa_diamond": "acc_norm,none",
}


def _limit_for(task: str, proxy: bool) -> int | None:
    if not proxy:
        return None
    return {
        "ifeval":       40,
        "hellaswag":    200,
        "mmlu_pro":     200,
        "gpqa_diamond": 40,
    }.get(task, 100)


def run(
    base_url: str,
    model_name: str,
    proxy: bool = False,
    task: str = "hellaswag",
) -> tuple[float, dict[str, Any]]:
    if task not in TASK_MAP:
        raise KeyError(f"lm_eval_task: unknown task {task!r}; known: {sorted(TASK_MAP)}")

    import sys
    lm_eval = shutil.which("lm-eval") or shutil.which("lm_eval")
    if lm_eval is not None:
        lm_eval_cmd: list[str] = [lm_eval]
    else:
        # Fallback: sieve-venv inherits lm_eval as a library from story's
        # venv (via the _story_inherit.pth) but the entry-point script is
        # not on PATH. Invoke the module directly with the current python.
        try:
            import lm_eval as _lm_eval_pkg  # noqa: F401
        except ImportError as e:
            raise ModuleNotFoundError(
                "lm-eval not on PATH and lm_eval module not importable. "
                "`pip install lm-eval[api]` on the worker."
            ) from e
        lm_eval_cmd = [sys.executable, "-m", "lm_eval"]

    lm_task = TASK_MAP[task]
    with tempfile.TemporaryDirectory(prefix=f"lmeval-{task}-") as td:
        out_dir = Path(td)
        # transformers serve only exposes /v1/chat/completions, not
        # /v1/completions. Use lm-eval's chat-completions model class.
        cmd = [
            *lm_eval_cmd,
            "--model", "local-chat-completions",
            "--model_args",
                f"base_url={base_url}/chat/completions,model={model_name},"
                f"tokenizer_backend=None,num_concurrent=4,max_retries=3",
            "--tasks", lm_task,
            "--output_path", str(out_dir),
            "--log_samples",
            "--apply_chat_template",
        ]
        limit = _limit_for(task, proxy)
        if limit is not None:
            cmd += ["--limit", str(limit)]

        env = dict(os.environ)
        env.setdefault("OPENAI_API_KEY", "EMPTY")  # sglang ignores the key but lm-eval wants one set

        logger.info("lm_eval: %s", " ".join(cmd))
        proc = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=7200,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"lm-eval {task} exit={proc.returncode}; "
                f"stderr tail: {proc.stderr[-2000:]}"
            )

        # lm-eval writes results_<timestamp>.json under out_dir/<model-name-safe>/
        result_files = list(out_dir.rglob("results_*.json"))
        if not result_files:
            raise RuntimeError(
                f"lm-eval completed but no results_*.json found under {out_dir}"
            )
        results_path = sorted(result_files)[-1]
        data = json.loads(results_path.read_text())
        metric = HEADLINE_METRIC[task]
        results = data.get("results", {})
        if lm_task not in results:
            raise RuntimeError(
                f"lm-eval results missing task {lm_task}: {list(results)}"
            )
        score = results[lm_task].get(metric)
        if score is None:
            # Fall back: any float value in the task's metric dict
            candidates = [
                v for k, v in results[lm_task].items()
                if isinstance(v, (int, float)) and not k.endswith("_stderr,none")
            ]
            if not candidates:
                raise RuntimeError(
                    f"no metric {metric!r} for task {lm_task}: {results[lm_task]}"
                )
            score = candidates[0]
        return float(score), {
            "task": task, "lm_task": lm_task, "metric": metric,
            "raw": results[lm_task], "proxy": proxy, "limit": limit,
        }
