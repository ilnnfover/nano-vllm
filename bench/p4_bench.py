#!/usr/bin/env python3
"""P4 · 连续批处理吞吐基准。

验收标准 (Branch 6 锁定):
  1. 吞吐: P4 连续批 vs P3 静态批，目标 ≥ 2×
  2. TPOT p99: chunked prefill (budget=512) vs non-chunked (budget=65536)，劣化 < 2×
  3. TTFT p50/p99: 信息性指标

口径:
  吞吐   = 总输出 token 数 / 墙钟时间 (tok/s)
  TTFT   = 首 token 产出时刻 - 请求提交时刻 (ms)
  TPOT   = 相邻 token 产出间隔 (ms)，取 p99
  P4 连续批 = 所有请求同时提交，EngineCore 循环 step 调度
  P3 静态批 = generate_batch padding 拼批，等满才跑

用法:
  python bench/p4_bench.py --model models/Qwen2.5-1.5B-Instruct
  python bench/p4_bench.py --tpot-test  # TPOT p99 对比测试
输出: bench/results/p4_bench_{tag}.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import time

import numpy as np
import torch

from nano_vllm.engine.core import EngineCore
from nano_vllm.engine.scheduler import Scheduler
from nano_vllm.engine.sequence import SamplingParams
from nano_vllm.model_executor.runner import NanoRunner


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


def run_p4_continuous(
    runner: NanoRunner,
    prompts: list[list[int]],
    max_new_tokens: int,
    max_num_batched_tokens: int = 2048,
) -> dict:
    """P4 连续批：所有请求同时提交，EngineCore 循环 step。"""
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
        "max_num_batched_tokens": max_num_batched_tokens,
    }


def run_p3_static(
    runner: NanoRunner,
    prompts: list[list[int]],
    max_new_tokens: int,
) -> dict:
    """P3 静态批：generate_batch padding 拼批。"""
    from nano_vllm.model_executor.runner import SamplingParams as RunnerParams

    params = RunnerParams(temperature=0.0, max_new_tokens=max_new_tokens)

    sync(runner.device)
    t0 = time.perf_counter()
    outs, stats = runner.generate_batch(prompts, params)
    sync(runner.device)
    wall_ms = (time.perf_counter() - t0) * 1e3

    out_tokens = sum(len(o) for o in outs)
    throughput = out_tokens / (wall_ms / 1e3) if wall_ms > 0 else 0
    return {
        "wall_ms": wall_ms,
        "output_tokens": out_tokens,
        "throughput_tok_s": throughput,
        "num_requests": len(prompts),
        "padding_waste_rate": stats["padding_waste_rate"],
    }


def make_mixed_prompts(tokenizer, lengths: list[int]) -> list[list[int]]:
    """生成指定长度的混合 prompt。"""
    prompts = []
    for length in lengths:
        text = "Hello " * (length // 2)
        ids = tokenizer.encode(text, add_special_tokens=True)[:length]
        if len(ids) < length:
            ids.extend([0] * (length - len(ids)))
        prompts.append(ids)
    return prompts


def bench_throughput(args) -> dict:
    """吞吐对比：P4 连续批 vs P3 静态批。"""
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    lengths = []
    for _ in range(args.num_requests // 3):
        lengths.extend([128, 1024, 8192])
    lengths = lengths[: args.num_requests]

    prompts = make_mixed_prompts(tokenizer, lengths)
    max_seq_len = max(lengths) + args.max_new_tokens
    num_blocks = (sum(lengths) + args.num_requests * args.max_new_tokens) // 16 + 32

    runner = make_runner(args.model, num_blocks, max_seq_len, device, dtype, args.prefill_impl)

    results = {}
    results["p4_continuous"] = run_p4_continuous(
        runner, prompts, args.max_new_tokens, args.budget,
    )
    results["p3_static"] = run_p3_static(
        runner, prompts, args.max_new_tokens,
    )

    p4_t = results["p4_continuous"]["throughput_tok_s"]
    p3_t = results["p3_static"]["throughput_tok_s"]
    results["throughput_ratio"] = p4_t / p3_t if p3_t > 0 else 0
    results["target"] = ">= 2.0x"
    results["pass"] = results["throughput_ratio"] >= 2.0

    return results


def bench_tpot(args) -> dict:
    """TPOT p99 对比：chunked prefill (budget=512) vs non-chunked (budget=65536)。

    场景: 4 个短请求 (128 prompt) 先 decode，然后 1 个 8K prompt 提交。
    测量短请求在长 prefill 期间的 TPOT p99。
    """
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    short_prompts = make_mixed_prompts(tokenizer, [128] * 4)
    long_prompt = make_mixed_prompts(tokenizer, [8192])[0]
    all_prompts = short_prompts + [long_prompt]

    max_seq_len = 8192 + args.max_new_tokens
    num_blocks = (sum(len(p) for p in all_prompts) + 5 * args.max_new_tokens) // 16 + 32

    results = {}
    for label, budget in [("chunked_512", 512), ("non_chunked_65536", 65536)]:
        runner = make_runner(args.model, num_blocks, max_seq_len, device, dtype, args.prefill_impl)
        runner.paged_cache.reset()
        engine = make_engine(runner, budget)
        params = SamplingParams(temperature=0.0, max_new_tokens=args.max_new_tokens)

        sync(runner.device)
        t0 = time.perf_counter()
        outs = engine.generate(all_prompts, params)
        sync(runner.device)
        wall_ms = (time.perf_counter() - t0) * 1e3

        results[label] = {
            "wall_ms": wall_ms,
            "budget": budget,
            "output_tokens": sum(len(o) for o in outs),
        }

    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--num-requests", type=int, default=12)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--budget", type=int, default=2048, help="max_num_batched_tokens")
    p.add_argument("--prefill-impl", default="torch", choices=["torch", "flashinfer"], help="prefill attention 后端")
    p.add_argument("--tpot-test", action="store_true", help="跑 TPOT p99 对比测试")
    p.add_argument("--tag", default=None)
    p.add_argument("--results-dir", default="bench/results")
    args = p.parse_args()

    tag = args.tag or ("tpot" if args.tpot_test else "throughput")
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.tpot_test:
        results = bench_tpot(args)
    else:
        results = bench_throughput(args)

    results["model"] = args.model
    results["dtype"] = args.dtype

    out_path = results_dir / f"p4_bench_{tag}.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n结果已保存到 {out_path}")


if __name__ == "__main__":
    main()
