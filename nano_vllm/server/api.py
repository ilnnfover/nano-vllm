"""P5 · FastAPI OpenAI 兼容 API：/v1/completions + /v1/chat/completions（含 streaming）。

用法:
    uvicorn nano_vllm.server.api:app --factory --model models/Qwen2.5-1.5B-Instruct
或:
    python -m nano_vllm.server.api --model models/Qwen2.5-1.5B-Instruct
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse

import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoTokenizer

from nano_vllm.engine.core import EngineCore
from nano_vllm.engine.scheduler import Scheduler
from nano_vllm.engine.sequence import SamplingParams
from nano_vllm.model_executor.runner import NanoRunner
from nano_vllm.server.async_engine import AsyncEngineCore
from nano_vllm.server.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    make_chat_chunk,
    make_chat_response,
    make_completion_chunk,
    make_completion_response,
)


def create_app(
    model_path: str,
    device: str = "cuda",
    dtype: str = "bf16",
    max_seq_len: int = 8192,
    block_size: int = 16,
    num_blocks: int = 512,
    max_num_batched_tokens: int = 2048,
    prefill_impl: str = "torch",
) -> FastAPI:
    dt = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    runner = NanoRunner(
        model_path, device=device, dtype=dt,
        max_seq_len=max_seq_len, block_size=block_size, num_blocks=num_blocks,
        attn_impl="torch", prefill_impl=prefill_impl,
    )
    scheduler = Scheduler(
        paged_cache=runner.paged_cache,
        max_num_batched_tokens=max_num_batched_tokens,
        watermark_blocks=1,
    )
    engine = EngineCore(runner, scheduler)
    async_engine = AsyncEngineCore(engine)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    app = FastAPI(title="nano-vLLM")
    app.state.async_engine = async_engine
    app.state.tokenizer = tokenizer
    app.state.runner = runner
    app.state.model_name = model_path

    def _params(req) -> SamplingParams:
        return SamplingParams(
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k if req.top_k > 0 else -1,
            max_new_tokens=req.max_tokens,
        )

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        tok = app.state.tokenizer
        ae = app.state.async_engine
        prompt_ids = tok.encode(req.prompt, add_special_tokens=True)
        params = _params(req)

        if req.stream:
            async def gen():
                async for token_id in ae.generate_stream(prompt_ids, params):
                    text = tok.decode([token_id])
                    yield f"data: {json.dumps(make_completion_chunk(text, req.model, False))}\n\n"
                yield f"data: {json.dumps(make_completion_chunk('', req.model, True))}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream")

        tokens = await ae.generate(prompt_ids, params)
        text = tok.decode(tokens)
        return JSONResponse(make_completion_response(
            text, req.model, len(prompt_ids), len(tokens),
        ))

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        tok = app.state.tokenizer
        ae = app.state.async_engine
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        prompt_text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
        params = _params(req)

        if req.stream:
            async def gen():
                async for token_id in ae.generate_stream(prompt_ids, params):
                    text = tok.decode([token_id])
                    yield f"data: {json.dumps(make_chat_chunk(text, req.model, False))}\n\n"
                yield f"data: {json.dumps(make_chat_chunk('', req.model, True))}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream")

        tokens = await ae.generate(prompt_ids, params)
        text = tok.decode(tokens)
        return JSONResponse(make_chat_response(
            text, req.model, len(prompt_ids), len(tokens),
        ))

    @app.on_event("shutdown")
    async def shutdown():
        await async_engine.aclose()

    return app


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--num-blocks", type=int, default=512)
    p.add_argument("--max-seq-len", type=int, default=8192)
    p.add_argument("--budget", type=int, default=2048)
    p.add_argument("--prefill-impl", default="torch", choices=["torch", "flashinfer"])
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    import uvicorn

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    app = create_app(
        args.model, device=device, dtype=args.dtype,
        max_seq_len=args.max_seq_len, num_blocks=args.num_blocks,
        max_num_batched_tokens=args.budget, prefill_impl=args.prefill_impl,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()