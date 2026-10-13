"""Fine-tuning wrapper for the openweight track.

See ``docs/architecture-openweight.md`` §"Generation loop" step 3.

Implementation note (2026-04-18): the architecture doc lists axolotl as
the primary backend. On the compute-cluster GB10 (aarch64 + CUDA 13,
torch 2.13-nightly), axolotl 0.16.1 pins torch 2.8 and imports symbols
that don't exist in the shipping accelerate on this stack. Rather than
fight the pin, this module uses **peft + trl.SFTTrainer + transformers**
directly — a thin wrapper, ~80 LOC, and every dep is already validated
on node-2. Axolotl can come back as a backend option if/when a
compatible build lands.

The controller never calls these helpers directly — the worker daemon
(``worker/finetune.py``) does. The controller only builds the
``SftConfig`` and enqueues.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from openweight.genome import SftConfig

logger = logging.getLogger(__name__)


def fine_tune(cfg: SftConfig, corpus_path: Path, out_dir: Path) -> Path:
    """Run SFT and return the adapter directory.

    ``corpus_path`` points to a jsonl file with a ``"text"`` field per
    row (matching the ``completion`` dataset convention). The
    ``datasets`` library loads it via ``load_dataset('json', ...)``.
    """
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(cfg.base_model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if cfg.precision == "bf16" else torch.float16
    # device_map={"": 0} forces the whole model onto cuda:0 without going through
    # accelerate's meta-device dispatch. "auto" was triggering
    # `Function MmBackward0 returned an invalid gradient at index 1 - expected
    # device meta but got cuda:0` on Phi-4 and OLMo-2-13B during peft LoRA
    # attachment (2026-04-23, dispatch-3). GB10 nodes are single-GPU so "auto"
    # buys us nothing.
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_id,
        torch_dtype=dtype,
        device_map={"": 0},
    )

    task_type = TaskType.CAUSAL_LM
    lora_cfg = LoraConfig(
        task_type=task_type,
        r=cfg.rank,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=cfg.target_modules,
        use_dora=(cfg.adapter_kind == "dora"),
        bias="none",
    )

    ds = load_dataset("json", data_files=str(corpus_path), split="train")

    # SIEVE_SFT_MAX_STEPS caps training at N steps regardless of epochs/dataset size.
    # Set it for diagnostic runs where a full epoch would be prohibitively long on
    # large-seq-len corpora (v009 rows are ~2k tokens each, so batch_size=1 full-epoch
    # = 22k steps = ~16h on a 3B LoRA, ~50h on 14B). Unset/<=0 → full epoch as normal.
    _max_steps_env = int(os.environ.get("SIEVE_SFT_MAX_STEPS", "0") or "0")
    _max_steps = _max_steps_env if _max_steps_env > 0 else -1

    sft_cfg = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=cfg.epochs,
        max_steps=_max_steps,
        per_device_train_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        bf16=(cfg.precision == "bf16"),
        fp16=(cfg.precision == "fp16"),
        max_length=cfg.max_seq_len,
        optim=cfg.optimizer,
        seed=cfg.seed,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )

    from openweight.trainer_callbacks import make_callback_from_env

    callbacks = []
    dashboard_cb = make_callback_from_env()
    if dashboard_cb is not None:
        callbacks.append(dashboard_cb)

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=ds,
        peft_config=lora_cfg,
        processing_class=tok,
        callbacks=callbacks or None,
    )

    logger.info(
        "fine_tune: base=%s rank=%d epochs=%d corpus=%s out=%s",
        cfg.base_model_id, cfg.rank, cfg.epochs, corpus_path, out_dir,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    return out_dir
