"""Seed-model rosters for Generation 0 of the openweight track.

The Gen-0 loop the user described:

    1. Pick N SOTA open-weight 7B–13B models that each fit on one GB10.
    2. Fine-tune each in parallel on the same Lever corpus shard
       (LoRA/DoRA, one SFT job per node).
    3. Benchmark the fine-tuned ensemble.
    4. Use fitness to drive Gen-1 (mutation/crossover/immigration).

This module picks step 1: which models seed the first generation. Two
rosters are provided:

- ``LLAMA3_WITHIN_FAMILY`` — 12 Llama-3.1-tokenizer models. Use for
  Gen-0 when you want output-space ensembling to work out of the box
  (§1.3 inference-arch says same-family is the default). Fine-tunes
  stay LoRA-compatible with the base.

- ``CROSS_FAMILY`` — mixed tokenizer families. Higher pretraining
  diversity but requires EVA/DeePEn at inference time (§1.4). Use as
  Stage 2 once the within-family Gen-0 plateaus.

These are starting defaults, not a frozen roster. Refresh by pulling
the current HF Open LLM Leaderboard top-20 at 7B-13B scale before each
first-run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SeedModel:
    """A candidate base model for Gen-0 fine-tuning.

    ``why`` records the diversity rationale — why this model earns its
    slot in the roster alongside the others. Keep these short and
    capability-anchored, not marketing.
    """

    hf_id: str
    tokenizer_family: Literal[
        "llama3", "qwen2", "qwen3", "gemma2", "gemma3",
        "mistral", "phi3", "phi4", "olmo2", "smollm", "granite", "exaone",
    ]
    params_b: float
    family: str
    license: str
    why: str


# Roster 1 — Llama-3.1 tokenizer family
# All 12 share the Llama-3.1-tokenizer so logit-averaging works with
# zero alignment machinery. Diversity comes from post-training
# recipes: instruct-tuned vs base, reasoning-distilled, code-tuned,
# conversation-tuned, etc.
LLAMA3_WITHIN_FAMILY: list[SeedModel] = [
    SeedModel(
        hf_id="meta-llama/Llama-3.1-8B-Instruct",
        tokenizer_family="llama3", params_b=8.0,
        family="llama3", license="llama3.1",
        why="Strong instruct-tuned floor; canonical reference.",
    ),
    SeedModel(
        hf_id="meta-llama/Llama-3.1-8B",
        tokenizer_family="llama3", params_b=8.0,
        family="llama3", license="llama3.1",
        why="Base (no chat post-training); different prior than Instruct.",
    ),
    SeedModel(
        hf_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        tokenizer_family="llama3", params_b=8.0,
        family="llama3", license="mit",
        why="Reasoning distillation — strong chain-of-thought bias.",
    ),
    SeedModel(
        hf_id="NousResearch/Hermes-3-Llama-3.1-8B",
        tokenizer_family="llama3", params_b=8.0,
        family="llama3", license="llama3.1",
        why="Hermes tool-use / agent post-training; different alignment path.",
    ),
    SeedModel(
        hf_id="allenai/Llama-3.1-Tulu-3-8B",
        tokenizer_family="llama3", params_b=8.0,
        family="llama3", license="llama3.1",
        why="Tulu-3 DPO+PPO recipe; empirically strong on IFEval, MMLU-Pro.",
    ),
    SeedModel(
        hf_id="nvidia/Llama-3.1-Nemotron-8B-Instruct",
        tokenizer_family="llama3", params_b=8.0,
        family="llama3", license="llama3.1",
        why="NVIDIA Nemotron post-training; different alignment prior.",
    ),
    SeedModel(
        hf_id="arcee-ai/Llama-3.1-SuperNova-Lite",
        tokenizer_family="llama3", params_b=8.0,
        family="llama3", license="llama3.1",
        why="Merge-pretrained (inherits diversity from upstream merge).",
    ),
    SeedModel(
        hf_id="Sao10K/L3.1-8B-Stheno-v3.4",
        tokenizer_family="llama3", params_b=8.0,
        family="llama3", license="llama3.1",
        why="Creative/open-ended post-training — different generation distribution.",
    ),
    SeedModel(
        hf_id="Replete-AI/L3.1-8B-Replete-LUMEN-v1",
        tokenizer_family="llama3", params_b=8.0,
        family="llama3", license="llama3.1",
        why="Community mixture post-training; complements flagship recipes.",
    ),
    SeedModel(
        hf_id="MaziyarPanahi/Llama-3.1-8B-Instruct-DPO",
        tokenizer_family="llama3", params_b=8.0,
        family="llama3", license="llama3.1",
        why="DPO-only (no PPO) variant of Instruct; different preference signal.",
    ),
    SeedModel(
        hf_id="cognitivecomputations/dolphin-2.9.4-llama3.1-8b",
        tokenizer_family="llama3", params_b=8.0,
        family="llama3", license="llama3.1",
        why="Dolphin uncensored recipe — broader response distribution for diversity.",
    ),
    SeedModel(
        hf_id="Locutusque/Llama-3.1-Hermes-3-SuperNova-lumen-8B",
        tokenizer_family="llama3", params_b=8.0,
        family="llama3", license="llama3.1",
        why="Pre-merged from Hermes+SuperNova; already off the canonical manifold.",
    ),
]


# Roster 2 — cross-family
# 12 slots spanning 4 tokenizer families for maximum pretraining
# diversity. Use when Gen-0 within-family plateaus.
CROSS_FAMILY: list[SeedModel] = [
    SeedModel(
        hf_id="meta-llama/Llama-3.1-8B-Instruct",
        tokenizer_family="llama3", params_b=8.0,
        family="llama3", license="llama3.1",
        why="Llama-family anchor.",
    ),
    SeedModel(
        hf_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        tokenizer_family="llama3", params_b=8.0,
        family="llama3", license="mit",
        why="Llama-tokenizer + reasoning distillation.",
    ),
    SeedModel(
        hf_id="allenai/Llama-3.1-Tulu-3-8B",
        tokenizer_family="llama3", params_b=8.0,
        family="llama3", license="llama3.1",
        why="Llama-tokenizer + strong DPO+PPO.",
    ),
    SeedModel(
        hf_id="Qwen/Qwen2.5-7B-Instruct",
        tokenizer_family="qwen2", params_b=7.6,
        family="qwen2", license="apache-2.0",
        why="Qwen-family anchor; strong multilingual + code floor.",
    ),
    SeedModel(
        hf_id="Qwen/Qwen2.5-14B-Instruct",
        tokenizer_family="qwen2", params_b=14.7,
        family="qwen2", license="apache-2.0",
        why="Larger Qwen within GB10 LoRA budget — different capacity tier.",
    ),
    SeedModel(
        hf_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        tokenizer_family="qwen2", params_b=7.6,
        family="qwen2", license="apache-2.0",
        why="Code-pretraining bias; unique on structure/syntax tasks.",
    ),
    SeedModel(
        hf_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        tokenizer_family="qwen2", params_b=7.6,
        family="qwen2", license="mit",
        why="Reasoning distillation into Qwen tokenizer; complement to Llama R1 distill.",
    ),
    SeedModel(
        hf_id="google/gemma-2-9b-it",
        tokenizer_family="gemma2", params_b=9.2,
        family="gemma2", license="gemma",
        why="Gemma-family anchor; distinct pretraining corpus vs Llama/Qwen.",
    ),
    SeedModel(
        hf_id="google/gemma-2-9b",
        tokenizer_family="gemma2", params_b=9.2,
        family="gemma2", license="gemma",
        why="Gemma base (no instruct) — different prior than gemma-2-9b-it.",
    ),
    SeedModel(
        hf_id="mistralai/Mistral-7B-Instruct-v0.3",
        tokenizer_family="mistral", params_b=7.2,
        family="mistral", license="apache-2.0",
        why="Mistral-family anchor; different tokenizer and sliding-window attention.",
    ),
    SeedModel(
        hf_id="mistralai/Mistral-Nemo-Instruct-2407",
        tokenizer_family="mistral", params_b=12.2,
        family="mistral", license="apache-2.0",
        why="Mistral-Nemo (12B); NVIDIA+Mistral collab, different recipe vs 7B.",
    ),
    SeedModel(
        hf_id="microsoft/Phi-3.5-mini-instruct",
        tokenizer_family="phi3", params_b=3.8,
        family="phi3", license="mit",
        why="Phi-3.5 mini — smaller + different (textbook-heavy) pretraining mix.",
    ),
]


# Roster 3 — SOTA 2026-Q2 cross-family tier
# Goal: the strongest open-weight 3-14B checkpoints we can LoRA-fine-tune
# in <4h on a single 128GB GB10 (bf16, rank=16 q/k/v/o, seq_len=2048, bs=1).
# Supersedes ``CROSS_FAMILY`` for new runs — Qwen3 replaces Qwen2.5,
# Gemma-3 replaces Gemma-2, Phi-4 replaces Phi-3.5, OLMo-2-13B is new.
# Seven distinct tokenizer families for maximal output-space diversity:
# qwen3 (151k BBPE), llama3 (128k), phi4 (100k tiktoken-derived),
# gemma3 (256k SentencePiece), olmo2 (100k GPT-NeoX-style),
# smollm (128k), granite (49k StarCoder-derived).
# Ordering note: the first 8 entries are all ungated (Apache-2.0 or MIT)
# and span 5 distinct tokenizer families (qwen3, phi4, olmo2, smollm,
# granite). Gated models (Gemma license, Llama license) are tail-weighted
# so ``pick_roster(name, n=N)`` works out-of-the-box without an
# HF_TOKEN for N ≤ 8. If HF_TOKEN is set, rank up or pass the full roster.
SOTA_2026_Q2: list[SeedModel] = [
    SeedModel(
        hf_id="Qwen/Qwen3-14B",
        tokenizer_family="qwen3", params_b=14.8,
        family="qwen3", license="apache-2.0",
        why="Hybrid thinking/non-thinking toggle; SOTA 14B on MMLU-Pro/AIME.",
    ),
    SeedModel(
        hf_id="microsoft/phi-4",
        tokenizer_family="phi4", params_b=14.7,
        family="phi4", license="mit",
        why="Synthetic-heavy 9.8T-token pretrain; 84.8 MMLU; distinct tokenizer.",
    ),
    SeedModel(
        hf_id="allenai/OLMo-2-1124-13B-Instruct",
        tokenizer_family="olmo2", params_b=13.7,
        family="olmo2", license="apache-2.0",
        why="Fully open (data+code+recipes); Tulu-3 post-training; beats Qwen2.5-14B-Inst.",
    ),
    SeedModel(
        hf_id="HuggingFaceTB/SmolLM3-3B",
        tokenizer_family="smollm", params_b=3.0,
        family="smollm", license="apache-2.0",
        why="Dual-mode reasoning at 3B, 11.2T tokens — fastest LoRA iteration.",
    ),
    SeedModel(
        hf_id="ibm-granite/granite-3.3-8b-instruct",
        tokenizer_family="granite", params_b=8.2,
        family="granite", license="apache-2.0",
        why="Enterprise/RAG/FIM-optimized; small 49k vocab for tokenizer diversity.",
    ),
    SeedModel(
        hf_id="Qwen/Qwen3-8B",
        tokenizer_family="qwen3", params_b=8.2,
        family="qwen3", license="apache-2.0",
        why="Qwen3 8B — matches Qwen2.5-14B-Base on STEM at half the cost.",
    ),
    SeedModel(
        hf_id="microsoft/Phi-4-reasoning-plus",
        tokenizer_family="phi4", params_b=14.7,
        family="phi4", license="mit",
        why="RL + o3-mini-trace distillation; reasoning SOTA at 14B scale.",
    ),
    SeedModel(
        hf_id="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        tokenizer_family="qwen3", params_b=8.2,
        family="qwen3", license="mit",
        why="R1-0528 CoT distilled into Qwen3-8B; +10pt AIME over Qwen3-8B.",
    ),
    SeedModel(
        hf_id="Qwen/Qwen3-14B-Base",
        tokenizer_family="qwen3", params_b=14.8,
        family="qwen3", license="apache-2.0",
        why="Clean pretrain checkpoint — no RLHF prior to fight during SFT.",
    ),
    # Gated (require HF_TOKEN + license acceptance on HF):
    SeedModel(
        hf_id="google/gemma-3-12b-it",
        tokenizer_family="gemma3", params_b=12.0,
        family="gemma3", license="gemma",
        why="Gemma-3 12B instruct; 128K context, 140+ langs, SP-256k tokenizer.",
    ),
    SeedModel(
        hf_id="google/gemma-3-12b-pt",
        tokenizer_family="gemma3", params_b=12.0,
        family="gemma3", license="gemma",
        why="Gemma-3 12B base — 12T-token pretrain, no instruct prior.",
    ),
    SeedModel(
        hf_id="meta-llama/Llama-3.1-8B-Instruct",
        tokenizer_family="llama3", params_b=8.0,
        family="llama3", license="llama3.1",
        why="Llama-3.1 anchor — canonical instruct reference, distinct tokenizer.",
    ),
]


ROSTERS: dict[str, list[SeedModel]] = {
    "llama3_within_family": LLAMA3_WITHIN_FAMILY,
    "cross_family": CROSS_FAMILY,
    "sota_2026_q2": SOTA_2026_Q2,
}


def pick_roster(name: str, n: int | None = None) -> list[SeedModel]:
    """Return the first ``n`` models of a named roster (all by default)."""
    if name not in ROSTERS:
        raise ValueError(f"unknown roster '{name}'; have {list(ROSTERS)}")
    roster = ROSTERS[name]
    return roster if n is None else roster[:n]


def summarize(roster: list[SeedModel]) -> str:
    by_family: dict[str, int] = {}
    for m in roster:
        by_family[m.tokenizer_family] = by_family.get(m.tokenizer_family, 0) + 1
    total_gb_bf16 = sum(m.params_b * 2 for m in roster)  # bf16 weights
    parts = [
        f"{len(roster)} models",
        "tokenizer families: " + ", ".join(f"{k}×{v}" for k, v in by_family.items()),
        f"aggregate weight memory ≈ {total_gb_bf16:.0f} GB bf16",
    ]
    return " | ".join(parts)
