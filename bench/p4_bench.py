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


def _aggregate(runs: list[dict], key: str) -> dict:
    """聚合 repeat 次运行的某个指标，返回 median/min/max/all。"""
    vals = sorted(r[key] for r in runs)
    return {
        "median": float(np.median(vals)),
        "min": float(vals[0]),
        "max": float(vals[-1]),
        "all": [float(v) for v in vals],
    }


def bench_throughput(args) -> dict:
    """吞吐对比：P4 连续批 vs P3 静态批（warmup + repeat 取中位数）。"""
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

    # warmup（结果丢弃，消除 Triton JIT 编译 + CUDA context 冷启动）
    for _ in range(args.warmup):
        run_p4_continuous(runner, prompts, args.max_new_tokens, args.budget)
        run_p3_static(runner, prompts, args.max_new_tokens)

    # 正式测量：repeat 次取中位数
    p4_runs = [run_p4_continuous(runner, prompts, args.max_new_tokens, args.budget) for _ in range(args.repeat)]
    p3_runs = [run_p3_static(runner, prompts, args.max_new_tokens) for _ in range(args.repeat)]

    p4_agg = _aggregate(p4_runs, "throughput_tok_s")
    p3_agg = _aggregate(p3_runs, "throughput_tok_s")
    p4_wall = _aggregate(p4_runs, "wall_ms")
    p3_wall = _aggregate(p3_runs, "wall_ms")

    ratio = p4_agg["median"] / p3_agg["median"] if p3_agg["median"] > 0 else 0
    return {
        "p4_continuous": {
            "throughput_tok_s": p4_agg,
            "wall_ms": p4_wall,
            "output_tokens": p4_runs[0]["output_tokens"],
            "num_requests": len(prompts),
            "max_num_batched_tokens": args.budget,
        },
        "p3_static": {
            "throughput_tok_s": p3_agg,
            "wall_ms": p3_wall,
            "output_tokens": p3_runs[0]["output_tokens"],
            "num_requests": len(prompts),
            "padding_waste_rate": p3_runs[0]["padding_waste_rate"],
        },
        "throughput_ratio": ratio,
        "target": ">= 2.0x",
        "pass": ratio >= 2.0,
        "warmup": args.warmup,
        "repeat": args.repeat,
    }


def bench_tpot(args) -> dict:
    """TPOT p99 对比：chunked prefill (budget=512) vs non-chunked (budget=65536)。

    手动展开 step 循环，记录每个纯 decode step 的间隔（排除 prefill step），
    取 p99 作为 TPOT 近似。warmup 用单独 engine 实例预热 kernel。
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

        # warmup：单独 engine 跑短请求预热 kernel，不消耗正式测量额度
        for _ in range(args.warmup):
            runner.paged_cache.reset()
            we = make_engine(runner, budget)
            wp = SamplingParams(temperature=0.0, max_new_tokens=4)
            we.generate(make_mixed_prompts(tokenizer, [128]), wp)

        # 正式测量：手动 step 循环，记录纯 decode step 间隔
        runner.paged_cache.reset()
        engine = make_engine(runner, budget)
        params = SamplingParams(temperature=0.0, max_new_tokens=args.max_new_tokens)
        seqs = [engine.add_request(p, params) for p in all_prompts]

        decode_intervals: list[float] = []
        prev_t: float | None = None
        sync(runner.device)
        t0 = time.perf_counter()
        while engine.scheduler.has_requests():
            sched = engine.scheduler.schedule()
            has_prefill = any(s.is_prefill for s in sched.scheduled)
            sampled = engine._execute(sched)
            engine.scheduler.update_from_output(sched, sampled)
            sync(runner.device)
            t_now = time.perf_counter()
            if prev_t is not None and not has_prefill:
                decode_intervals.append((t_now - prev_t) * 1e3)
            prev_t = t_now
        wall_ms = (time.perf_counter() - t0) * 1e3

        out_tokens = sum(len(s.output_token_ids) for s in seqs)
        tpot_p99 = float(np.percentile(decode_intervals, 99)) if decode_intervals else 0.0
        tpot_p50 = float(np.percentile(decode_intervals, 50)) if decode_intervals else 0.0

        results[label] = {
            "wall_ms": wall_ms,
            "budget": budget,
            "output_tokens": out_tokens,
            "tpot_p50_ms": tpot_p50,
            "tpot_p99_ms": tpot_p99,
            "num_decode_steps": len(decode_intervals),
        }

    results["warmup"] = args.warmup
    results["tpot_ratio_p99"] = (
        results["chunked_512"]["tpot_p99_ms"] / results["non_chunked_65536"]["tpot_p99_ms"]
        if results["non_chunked_65536"]["tpot_p99_ms"] > 0 else 0
    )
    results["target"] = "chunked p99 < 2× non_chunked p99"
    results["pass"] = results["tpot_ratio_p99"] < 2.0
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
    p.add_argument("--warmup", type=int, default=1, help="预热轮数（结果丢弃，消除 Triton JIT 冷启动）")
    p.add_argument("--repeat", type=int, default=3, help="测量轮数（取中位数消除运行间方差）")
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
