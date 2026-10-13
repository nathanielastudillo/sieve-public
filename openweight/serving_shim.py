"""Minimal OpenAI-compatible inference server (fallback for transformers serve).

``transformers serve`` 5.3.0 + fastapi ≥0.100 has a handler-binding bug on
``POST /v1/chat/completions`` (``body: dict`` treated as a query param). This
module is a ~120-line replacement:

- Loads one HF model at startup (``--force-model`` equivalent).
- Exposes ``/v1/models`` and ``/v1/chat/completions`` (non-stream JSON).
- Applies the tokenizer's chat template via ``apply_chat_template``.

Launched in-process via ``uvicorn.run`` when called as a script. The
subprocess launcher (``openweight/inference.py::launch_transformers_server``)
wraps this.

Scope: the minimum needed for smoke_mcq and lm-eval-harness'
``local-chat-completions`` model class. No tool-use, no streaming (lm-eval
chat-completions calls are non-streaming anyway), no vision. Add if needed.
"""

from __future__ import annotations

import argparse
import logging
import time
import uuid
from typing import Any

logger = logging.getLogger("serving_shim")


def build_app(model_path: str, dtype: str = "bfloat16"):
    import torch
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]

    logger.info("Loading tokenizer + model from %s (dtype=%s)", model_path, dtype)
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch_dtype, device_map="auto",
    )
    model.eval()
    logger.info("Model loaded on device=%s", next(model.parameters()).device)

    app = FastAPI()

    @app.get("/v1/models")
    def list_models():
        return {"object": "list", "data": [
            {"id": model_path, "object": "model", "created": int(time.time()),
             "owned_by": "local"}
        ]}

    @app.get("/health")
    def healthcheck():
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    def chat_completions(payload: dict = Body(...)):
        messages = payload.get("messages")
        if not messages:
            raise HTTPException(status_code=400, detail="messages required")

        max_new_tokens = int(payload.get("max_tokens", payload.get("max_completion_tokens", 256)))
        temperature = float(payload.get("temperature", 1.0))
        top_p = float(payload.get("top_p", 1.0))
        do_sample = temperature > 0.0

        try:
            inputs = tok.apply_chat_template(
                messages, add_generation_prompt=True,
                return_tensors="pt", return_dict=True, tokenize=True,
            ).to(model.device)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"chat template error: {e}")

        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": tok.pad_token_id,
            }
            if do_sample:
                gen_kwargs["temperature"] = max(temperature, 1e-5)
                gen_kwargs["top_p"] = top_p
            out = model.generate(**inputs, **gen_kwargs)

        gen_ids = out[0, input_len:]
        text = tok.decode(gen_ids, skip_special_tokens=True)
        completion_tokens = int(gen_ids.shape[0])
        finish_reason = "length" if completion_tokens >= max_new_tokens else "stop"

        return JSONResponse({
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.get("model", model_path),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": int(input_len),
                "completion_tokens": completion_tokens,
                "total_tokens": int(input_len) + completion_tokens,
            },
        })

    @app.post("/v1/completions")
    def completions(payload: dict = Body(...)):
        prompt = payload.get("prompt")
        if prompt is None:
            raise HTTPException(status_code=400, detail="prompt required")
        if isinstance(prompt, list):
            prompt = prompt[0]

        max_new_tokens = int(payload.get("max_tokens", 256))
        temperature = float(payload.get("temperature", 1.0))
        top_p = float(payload.get("top_p", 1.0))
        do_sample = temperature > 0.0

        inputs = tok(prompt, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": tok.pad_token_id,
            }
            if do_sample:
                gen_kwargs["temperature"] = max(temperature, 1e-5)
                gen_kwargs["top_p"] = top_p
            out = model.generate(**inputs, **gen_kwargs)

        gen_ids = out[0, input_len:]
        text = tok.decode(gen_ids, skip_special_tokens=True)
        completion_tokens = int(gen_ids.shape[0])
        finish_reason = "length" if completion_tokens >= max_new_tokens else "stop"

        return JSONResponse({
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": payload.get("model", model_path),
            "choices": [{"index": 0, "text": text, "finish_reason": finish_reason}],
            "usage": {
                "prompt_tokens": int(input_len),
                "completion_tokens": completion_tokens,
                "total_tokens": int(input_len) + completion_tokens,
            },
        })

    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    app = build_app(args.model_path, dtype=args.dtype)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
