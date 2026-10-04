"""P5 · FastAPI OpenAI 兼容 API：/v1/completions + /v1/chat/completions（含 streaming）。

用法:
    uvicorn nano_vllm.server.api:app --factory --model models/Qwen2.5-1.5B-Instruct
或:
    python -m nano_vllm.server.api --model models/Qwen2.5-1.5B-Instruct
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager

import argparse
import asyncio

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
    _cid,
    _ts,
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
    num_blocks: int | None = None,
    max_num_batched_tokens: int = 2048,
    attn_impl: str = "torch",
    prefill_impl: str = "torch",
    prejoin: bool = True,
    norm_impl: str = "triton",
    prefill_mode: str = "batched",
    enable_prefix_cache: bool = True,
    max_num_seqs: int | None = None,
    long_prefill_token_threshold: int = 0,
    debug: bool | None = None,
) -> FastAPI:
    dt = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    runner = NanoRunner(
        model_path, device=device, dtype=dt,
        max_seq_len=max_seq_len, block_size=block_size, num_blocks=num_blocks,
        attn_impl=attn_impl, prefill_impl=prefill_impl,
        prejoin=prejoin, norm_impl=norm_impl,
    )
    scheduler = Scheduler(
        paged_cache=runner.paged_cache,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        long_prefill_token_threshold=long_prefill_token_threshold,
        enable_prefix_cache=enable_prefix_cache,
    )
    engine = EngineCore(runner, scheduler, prefill_mode=prefill_mode, debug=debug)
    async_engine = AsyncEngineCore(engine)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    @asynccontextmanager
    async def lifespan(app):
        yield
        await async_engine.aclose()

    app = FastAPI(title="nano-vLLM", lifespan=lifespan)
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
        prompt_ids = await asyncio.to_thread(tok.encode, req.prompt, add_special_tokens=True)
        params = _params(req)

        if req.stream:
            async def gen():
                cid = _cid("cmpl")
                created = _ts()
                count = 0
                async for token_id in ae.generate_stream(prompt_ids, params):
                    count += 1
                    text = tok.decode([token_id])
                    yield f"data: {json.dumps(make_completion_chunk(text, req.model, False, cid, created))}\n\n"
                finish = "length" if count >= req.max_tokens else "stop"
                yield f"data: {json.dumps(make_completion_chunk('', req.model, True, cid, created, finish))}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream")

        tokens = await ae.generate(prompt_ids, params)
        text = await asyncio.to_thread(tok.decode, tokens)
        finish_reason = "length" if len(tokens) >= req.max_tokens else "stop"
        return JSONResponse(make_completion_response(
            text, req.model, len(prompt_ids), len(tokens), finish_reason,
        ))

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        tok = app.state.tokenizer
        ae = app.state.async_engine
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        prompt_text = await asyncio.to_thread(
            tok.apply_chat_template, messages, add_generation_prompt=True, tokenize=False
        )
        prompt_ids = await asyncio.to_thread(tok.encode, prompt_text, add_special_tokens=False)
        params = _params(req)

        if req.stream:
            async def gen():
                cid = _cid("chatcmpl")
                created = _ts()
                count = 0
                async for token_id in ae.generate_stream(prompt_ids, params):
                    count += 1
                    text = tok.decode([token_id])
                    yield f"data: {json.dumps(make_chat_chunk(text, req.model, False, cid, created))}\n\n"
                finish = "length" if count >= req.max_tokens else "stop"
                yield f"data: {json.dumps(make_chat_chunk('', req.model, True, cid, created, finish))}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream")

        tokens = await ae.generate(prompt_ids, params)
        text = await asyncio.to_thread(tok.decode, tokens)
        finish_reason = "length" if len(tokens) >= req.max_tokens else "stop"
        return JSONResponse(make_chat_response(
            text, req.model, len(prompt_ids), len(tokens), finish_reason,
        ))


    return app


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--num-blocks", type=int, default=None)
    p.add_argument("--max-seq-len", type=int, default=8192)
    p.add_argument("--budget", type=int, default=2048)
    p.add_argument("--prefill-impl", default="torch", choices=["torch", "flashinfer"])
    p.add_argument("--attn-impl", default="torch", choices=["torch", "sdpa", "triton"])
    p.add_argument("--prefill-mode", default="batched", choices=["batched", "per-seq"])
    p.add_argument("--prejoin", action=argparse.BooleanOptionalAction, default=True,
                   help="P8 权重预拼接（QKV 3→1 / gate-up 2→1）；用 --no-prejoin 做拼接前后对照")
    p.add_argument("--norm-impl", default="triton", choices=["torch", "lib", "triton"],
                   help="RMSNorm 实现：torch（逐 op oracle）/ lib（F.rms_norm）/ triton（自研融合核）")
    p.add_argument("--no-prefix-cache", dest="enable_prefix_cache", action="store_false",
                   default=True, help="关闭 P6 前缀缓存（默认开启）")
    p.add_argument("--max-num-seqs", type=int, default=None,
                   help="A6 同批运行请求上限（缺省不限）")
    p.add_argument("--long-prefill-threshold", type=int, default=0,
                   help="A6 单请求一步最多 prefill token 数（0 = 关闭，与 vLLM 默认一致）")
    p.add_argument("--debug", action="store_true", default=None,
                   help="D4 开启 NaN/Inf 检查（shape 守卫始终开启）")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    import uvicorn

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    app = create_app(
        args.model, device=device, dtype=args.dtype,
        max_seq_len=args.max_seq_len, num_blocks=args.num_blocks,
        max_num_batched_tokens=args.budget, attn_impl=args.attn_impl,
        prefill_impl=args.prefill_impl, prefill_mode=args.prefill_mode,
        prejoin=args.prejoin, norm_impl=args.norm_impl,
        enable_prefix_cache=args.enable_prefix_cache,
        max_num_seqs=args.max_num_seqs,
        long_prefill_token_threshold=args.long_prefill_threshold,
        debug=args.debug,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()