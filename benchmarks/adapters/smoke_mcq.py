"""Built-in 20-question MCQ — the dev-loop canary.

Purpose: prove the whole stack end-to-end (SFT adapter → sglang →
OpenAI-compatible eval → score back to orchestrator) without depending
on lm-eval-harness being installed. Questions are trivial; any
reasonable 7B base scores >0.8. A score of ~0.25 (chance) means the
serving stack is broken, not the model.

Not a real benchmark. Used only in smoke tests and CI.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

# Thinking models (Qwen3, SmolLM3, DeepSeek-R1, etc.) wrap reasoning in
# <think>…</think> before emitting the answer. max_tokens=4 would truncate
# inside the opening tag and score them at chance. Give enough headroom for
# reasoning; non-thinking models still stop at EOS after the letter.
MAX_TOKENS = 2048
# 60s isn't enough for 14B thinking models: ~2048 tokens at ~20-30 tok/s
# plus queueing spikes. 300s absorbs those.
TIMEOUT_S = 300.0


QUESTIONS: list[dict[str, Any]] = [
    {"q": "What is the capital of France?", "choices": ["Paris", "London", "Berlin", "Madrid"], "answer": "A"},
    {"q": "2 + 2 equals:", "choices": ["3", "4", "5", "22"], "answer": "B"},
    {"q": "Which planet is closest to the Sun?", "choices": ["Venus", "Earth", "Mercury", "Mars"], "answer": "C"},
    {"q": "Water freezes at what Celsius temperature?", "choices": ["100", "50", "32", "0"], "answer": "D"},
    {"q": "Which is a mammal?", "choices": ["Shark", "Dolphin", "Salmon", "Octopus"], "answer": "B"},
    {"q": "Shakespeare wrote:", "choices": ["War and Peace", "Hamlet", "The Odyssey", "Moby Dick"], "answer": "B"},
    {"q": "The square root of 81 is:", "choices": ["7", "8", "9", "10"], "answer": "C"},
    {"q": "Which is a primary color?", "choices": ["Green", "Purple", "Orange", "Red"], "answer": "D"},
    {"q": "Which gas do plants absorb?", "choices": ["Oxygen", "Nitrogen", "Carbon dioxide", "Helium"], "answer": "C"},
    {"q": "How many continents are there?", "choices": ["5", "6", "7", "8"], "answer": "C"},
    {"q": "H2O is:", "choices": ["Salt", "Water", "Oxygen", "Hydrogen"], "answer": "B"},
    {"q": "The Pacific is a/an:", "choices": ["Desert", "Mountain", "Ocean", "River"], "answer": "C"},
    {"q": "Which is smallest?", "choices": ["Meter", "Kilometer", "Millimeter", "Centimeter"], "answer": "C"},
    {"q": "Sun rises in the:", "choices": ["North", "South", "East", "West"], "answer": "C"},
    {"q": "DNA carries:", "choices": ["Food", "Water", "Genetic info", "Oxygen"], "answer": "C"},
    {"q": "Which is prime?", "choices": ["4", "6", "8", "11"], "answer": "D"},
    {"q": "Speed of light is approximately:", "choices": ["300 m/s", "300 km/s", "300000 km/s", "3 m/s"], "answer": "C"},
    {"q": "The Great Wall is in:", "choices": ["Japan", "China", "Korea", "India"], "answer": "B"},
    {"q": "Python is a:", "choices": ["Snake only", "Programming language", "Math concept", "Planet"], "answer": "B"},
    {"q": "How many sides does a hexagon have?", "choices": ["4", "5", "6", "8"], "answer": "C"},
]


PROMPT_TEMPLATE = (
    "Answer with only a single letter (A, B, C, or D).\n\n"
    "Question: {q}\n"
    "A. {a}\nB. {b}\nC. {c}\nD. {d}\n\n"
    "Answer:"
)


def _chat_completion(
    base_url: str, model_name: str, prompt: str, timeout_s: float = TIMEOUT_S,
) -> str:
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            body = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # Fall back to /completions for servers that don't expose /chat
        logger.debug("chat/completions failed (%s); trying /completions", e)
        return _legacy_completion(base_url, model_name, prompt, timeout_s)
    return body["choices"][0]["message"]["content"]


def _legacy_completion(
    base_url: str, model_name: str, prompt: str, timeout_s: float = TIMEOUT_S,
) -> str:
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        f"{base_url}/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        body = json.loads(r.read().decode("utf-8"))
    return body["choices"][0]["text"]


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_LETTER_RE = re.compile(r"\b([ABCD])\b")


def _extract_letter(text: str) -> str:
    # Strip complete <think>…</think> blocks before scoring. A truncated
    # thinking generation (opening tag with no close) means the model ran
    # out of budget before concluding — no answer possible, score it 0.
    cleaned = _THINK_RE.sub("", text)
    if "<think>" in cleaned.lower() and "</think>" not in cleaned.lower():
        return ""
    # Models often reason "... so the answer is D" — take the LAST
    # word-boundaried A/B/C/D, which is the conclusion.
    matches = _LETTER_RE.findall(cleaned.upper())
    return matches[-1] if matches else ""


def run(
    base_url: str,
    model_name: str,
    proxy: bool = False,
    task: str = "smoke_mcq",
) -> tuple[float, dict[str, Any]]:
    qs = QUESTIONS[:5] if proxy else QUESTIONS
    results: list[dict[str, Any]] = []
    correct = 0
    for i, item in enumerate(qs):
        prompt = PROMPT_TEMPLATE.format(
            q=item["q"], a=item["choices"][0], b=item["choices"][1],
            c=item["choices"][2], d=item["choices"][3],
        )
        try:
            text = _chat_completion(base_url, model_name, prompt)
        except Exception as e:  # noqa: BLE001
            logger.warning("smoke_mcq q%d failed: %s", i, e)
            results.append({"i": i, "error": str(e), "got": "", "answer": item["answer"]})
            continue
        got = _extract_letter(text)
        ok = got == item["answer"]
        correct += int(ok)
        results.append({"i": i, "got": got, "answer": item["answer"], "raw": text})

    score = correct / len(qs) if qs else 0.0
    raw = {"n": len(qs), "correct": correct, "proxy": proxy, "items": results}
    return score, raw
