#!/usr/bin/env python3
"""P5 · HTTP vs 直连吞吐基准。

验收标准 (Branch 4 锁定):
  HTTP 吞吐损耗 < 10%，即 http_throughput / direct_throughput >= 0.9

口径:
  HTTP 吞吐   = N 个并发请求的总输出 token 数 / 墙钟时间 (tok/s)
  直连吞吐    = engine.generate() 的总输出 token 数 / 墙钟时间 (tok/s)
  用 httpx ASGITransport 直连 FastAPI app（不起真实 HTTP server，排除网络噪声）
  两者共享同一个 runner/模型，仅路径不同

用法:
  python bench/p5_bench.py --model models/Qwen2.5-1.5B-Instruct
输出: bench/results/p5_bench_{tag}.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import asyncio
import json
import time

import httpx
import numpy as np
import torch

from nano_vllm.engine.core import EngineCore
from nano_vllm.engine.scheduler import Scheduler
from nano_vllm.engine.sequence import SamplingParams
from nano_vllm.model_executor.runner import NanoRunner
from nano_vllm.server.api import create_app


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def make_runner(model: str, num_blocks: int, max_seq_len: int, device: str, dtype: torch.dtype, prefill_impl: str = "torch") -> NanoRunner:
    return NanoRunner(
        model, device=device, dtype=dtype,
        max_seq_len=max_seq_len, block_size=16, num_blocks=num_blocks,
        attn_impl="torch",
        prefill_impl=prefill_impl,
    )


def make_engine(runner: NanoRunner, max_num_batched_tokens: int = 2048) -> EngineCore:
    scheduler = Scheduler(
        paged_cache=runner.paged_cache,
        max_num_batched_tokens=max_num_batched_tokens,
        watermark_blocks=1,
    )
    return EngineCore(runner, scheduler)


def make_prompts(tokenizer, num_requests: int, prompt_len: int) -> list[list[int]]:
    text = "Hello " * (prompt_len // 2)
    ids = tokenizer.encode(text, add_special_tokens=True)[:prompt_len]
    if len(ids) < prompt_len:
        ids.extend([0] * (prompt_len - len(ids)))
    return [list(ids) for _ in range(num_requests)]


def run_direct(
    runner: NanoRunner,
    prompts: list[list[int]],
    max_new_tokens: int,
    max_num_batched_tokens: int = 2048,
) -> dict:
    runner.paged_cache.reset()
    engine = make_engine(runner, max_num_batched_tokens)
    params = SamplingParams(temperature=0.0, max_new_tokens=max_new_tokens)

    sync(runner.device)
    t0 = time.perf_counter()
    outs = engine.generate(prompts, params)
    sync(runner.device)
    wall_ms = (time.perf_counter() - t0) * 1e3

    out_tokens = sum(len(o) for o in outs)
    throughput = out_tokens / (wall_ms / 1e3) if wall_ms > 0 else 0
    return {
        "wall_ms": wall_ms,
        "output_tokens": out_tokens,
        "throughput_tok_s": throughput,
        "num_requests": len(prompts),
    }


async def run_http(
    app,
    prompts: list[list[int]],
    max_new_tokens: int,
    model_name: str,
) -> dict:
    tokenizer = app.state.tokenizer
    prompt_texts = [tokenizer.decode(p) for p in prompts]

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://bench") as client:
        async def one_request(text: str) -> int:
            resp = await client.post("/v1/completions", json={
                "model": model_name,
                "prompt": text,
                "max_tokens": max_new_tokens,
                "temperature": 0.0,
            })
            data = resp.json()
            return data["usage"]["completion_tokens"]

        sync(app.state.runner.device)
        t0 = time.perf_counter()
        results = await asyncio.gather(*[one_request(t) for t in prompt_texts])
        sync(app.state.runner.device)
        wall_ms = (time.perf_counter() - t0) * 1e3

    out_tokens = sum(results)
    throughput = out_tokens / (wall_ms / 1e3) if wall_ms > 0 else 0
    return {
        "wall_ms": wall_ms,
        "output_tokens": out_tokens,
        "throughput_tok_s": throughput,
        "num_requests": len(prompts),
    }


def _aggregate(runs: list[dict], key: str) -> dict:
    vals = sorted(r[key] for r in runs)
    return {
        "median": float(np.median(vals)),
        "min": float(vals[0]),
        "max": float(vals[-1]),
        "all": [float(v) for v in vals],
    }


async def bench_overhead(args) -> dict:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    prompts = make_prompts(tokenizer, args.num_requests, args.prompt_len)
    max_seq_len = args.prompt_len + args.max_new_tokens
    num_blocks = (args.num_requests * args.prompt_len + args.num_requests * args.max_new_tokens) // 16 + 32

    runner = make_runner(args.model, num_blocks, max_seq_len, device, dtype, args.prefill_impl)
    app = create_app(
        args.model, device=device, dtype=args.dtype,
        max_seq_len=max_seq_len, num_blocks=num_blocks,
        max_num_batched_tokens=args.budget, prefill_impl=args.prefill_impl,
    )

    # warmup
    for _ in range(args.warmup):
        run_direct(runner, prompts, args.max_new_tokens, args.budget)
        await run_http(app, prompts, args.max_new_tokens, args.model)

    # 正式测量
    direct_runs = [run_direct(runner, prompts, args.max_new_tokens, args.budget) for _ in range(args.repeat)]
    http_runs = [await run_http(app, prompts, args.max_new_tokens, args.model) for _ in range(args.repeat)]

    direct_agg = _aggregate(direct_runs, "throughput_tok_s")
    http_agg = _aggregate(http_runs, "throughput_tok_s")
    direct_wall = _aggregate(direct_runs, "wall_ms")
    http_wall = _aggregate(http_runs, "wall_ms")

    ratio = http_agg["median"] / direct_agg["median"] if direct_agg["median"] > 0 else 0
    return {
        "direct": {
            "throughput_tok_s": direct_agg,
            "wall_ms": direct_wall,
            "output_tokens": direct_runs[0]["output_tokens"],
            "num_requests": len(prompts),
        },
        "http": {
            "throughput_tok_s": http_agg,
            "wall_ms": http_wall,
            "output_tokens": http_runs[0]["output_tokens"],
            "num_requests": len(prompts),
        },
        "overhead_ratio": ratio,
        "overhead_pct": (1 - ratio) * 100,
        "target": ">= 0.9 (overhead < 10%)",
        "pass": ratio >= 0.9,
        "warmup": args.warmup,
        "repeat": args.repeat,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--num-requests", type=int, default=12)
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--budget", type=int, default=2048, help="max_num_batched_tokens")
    p.add_argument("--prefill-impl", default="torch", choices=["torch", "flashinfer"])
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--tag", default=None)
    p.add_argument("--results-dir", default="bench/results")
    args = p.parse_args()

    tag = args.tag or "overhead"
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    results = asyncio.run(bench_overhead(args))
    results["model"] = args.model
    results["dtype"] = args.dtype

    out_path = results_dir / f"p5_bench_{tag}.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n结果已保存到 {out_path}")


if __name__ == "__main__":
    main()
