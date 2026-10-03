#!/usr/bin/env python3
"""P7 · kernel launch 开销占比测量（roadmap 完成标准: 10–30% → ~1%）。

口径（与 docs 一致，两种都给，避免歧义）:
  - `launch_over_gpu_busy` = CPU 侧 CUDA API 时间 / GPU kernel 时间
    （roadmap 原文口径："CPU launch 时间 / GPU busy 时间"）
  - `launch_over_wall`     = CPU 侧 CUDA API 时间 / 墙钟时间
  其中 CPU 侧 CUDA API 时间 = torch profiler 中 key 以 `cuda` 开头的事件
  （cudaLaunchKernel / cudaGraphLaunch / cudaMemcpyAsync …）的 self CPU 时间之和；
  GPU kernel 时间 = 所有有 device time 的事件的 self device 时间之和。

用法:
  python bench/p7_profiler.py --model models/Qwen2.5-1.5B-Instruct --batch 8 --steps 30
输出: bench/results/p7_profiler.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import random
import time

import torch
from torch.profiler import ProfilerActivity, profile

from nano_vllm.engine.core import EngineCore
from nano_vllm.engine.scheduler import Scheduler
from nano_vllm.engine.sequence import SamplingParams
from nano_vllm.model_executor.runner import NanoRunner

VOCAB = 151936


def build_engine(runner: NanoRunner, prompts, budget=2048):
    scheduler = Scheduler(runner.paged_cache, max_num_batched_tokens=budget,
                          enable_prefix_cache=False)
    engine = EngineCore(runner, scheduler)
    params = SamplingParams(temperature=0.0, max_new_tokens=64)
    seqs = [engine.add_request(p, params) for p in prompts]
    # 先把 prefill 全部跑完（不纳入 profile 区间）
    while any(s.num_computed_tokens < s.num_prompt_tokens for s in seqs):
        engine.step()
    return engine, seqs


def profile_decode(runner: NanoRunner, prompts, steps: int) -> dict:
    engine, _ = build_engine(runner, prompts)
    # 先跑几步热身，确保 Triton 已编译 / 图已回放
    for _ in range(3):
        engine.step()
    torch.cuda.synchronize()

    # (1) 不上 profiler 的 TPOT（profiler 本身会拖慢 ~2×，比值必须用未开 profiler 的基线）
    torch.cuda.synchronize()
    tpots = []
    for _ in range(steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        engine.step()
        torch.cuda.synchronize()
        tpots.append((time.perf_counter() - t0) * 1e3)
    tpots.sort()
    tpot_p50_ms = tpots[len(tpots) // 2]

    # (2) profiler 分解 CPU 侧 CUDA API 时间
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            engine.step()
        torch.cuda.synchronize()

    def bucket(key: str) -> str:
        if key.startswith("cudaGraphLaunch"):
            return "graph_launch_us"
        if key.startswith("cudaLaunchKernel"):
            return "kernel_launch_us"
        if key.startswith("cudaMemcpy") or key.startswith("cudaMemset"):
            return "memcpy_us"
        if "Synchronize" in key:
            return "sync_us"
        return "other_api_us"

    sums: dict[str, float] = {
        "graph_launch_us": 0.0, "kernel_launch_us": 0.0, "memcpy_us": 0.0,
        "sync_us": 0.0, "other_api_us": 0.0,
    }
    gpu_busy_us = 0.0
    top_keys: dict[str, float] = {}
    for e in prof.key_averages():
        if e.key.startswith("cuda"):
            sums[bucket(e.key)] += e.self_cpu_time_total
            top_keys[e.key] = round(e.self_cpu_time_total, 1)
        if e.device_time_total > 0:
            gpu_busy_us += e.self_device_time_total

    launch_us = sums["kernel_launch_us"] + sums["graph_launch_us"]
    cpu_total_us = sum(sums.values())
    per_step = {k: v / steps for k, v in sums.items()}
    res = {
        "steps": steps,
        "unprofiled_tpot_p50_ms": tpot_p50_ms,
        "gpu_busy_us": gpu_busy_us,
        "gpu_busy_per_step_us": gpu_busy_us / steps,
        "cpu_api_total_per_step_us": cpu_total_us / steps,
        **{f"{k}_per_step": v for k, v in per_step.items()},
        "top_api_keys_us": dict(sorted(top_keys.items(), key=lambda kv: -kv[1])[:6]),
    }
    # roadmap 口径: CPU 侧 kernel launch 时间 / GPU busy 时间 与 / TPOT
    res["launch_over_gpu_busy"] = launch_us / gpu_busy_us if gpu_busy_us else None
    res["kernel_launch_per_step_us"] = sums["kernel_launch_us"] / steps
    res["launch_share_of_tpot"] = (
        (sums["kernel_launch_us"] / steps) / 1e3 / tpot_p50_ms if tpot_p50_ms else None
    )
    res["cpu_api_total_share_of_tpot"] = (
        (cpu_total_us / steps) / 1e3 / tpot_p50_ms if tpot_p50_ms else None
    )
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--num-blocks", type=int, default=4096)
    ap.add_argument("--tag", default="p7_profiler")
    args = ap.parse_args()

    assert torch.cuda.is_available()
    rng = random.Random(0)
    prompts = [
        [rng.randrange(1000, VOCAB) for _ in range(args.prompt_len)]
        for _ in range(args.batch)
    ]

    result: dict = {
        "model": args.model,
        "dtype": "bf16",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "batch": args.batch,
        "prompt_len": args.prompt_len,
        "steps": args.steps,
        "attn_impl": "triton",
    }

    # ---- 图关（P4 eager）----
    runner = NanoRunner(
        args.model, device="cuda", dtype=torch.bfloat16, block_size=16,
        attn_impl="triton", num_blocks=args.num_blocks,
        enable_cudagraph=False,
    )
    runner.paged_cache.reset()
    result["eager"] = profile_decode(runner, prompts, args.steps)
    del runner
    import gc

    gc.collect(); gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()

    # ---- 图开（P7）----
    runner = NanoRunner(
        args.model, device="cuda", dtype=torch.bfloat16, block_size=16,
        attn_impl="triton", num_blocks=args.num_blocks,
        enable_cudagraph=True,
    )
    runner.paged_cache.reset()
    result["cudagraph"] = profile_decode(runner, prompts, args.steps)
    result["cudagraph"]["capture_ms"] = runner.graph_runner.stats.capture_ms
    del runner
    gc.collect(); gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()

    e, g = result["eager"], result["cudagraph"]
    keys = (
        "unprofiled_tpot_p50_ms", "kernel_launch_per_step_us", "graph_launch_us_per_step",
        "memcpy_us_per_step", "sync_us_per_step", "cpu_api_total_per_step_us",
        "launch_over_gpu_busy", "launch_share_of_tpot", "cpu_api_total_share_of_tpot",
        "gpu_busy_per_step_us",
    )
    result["reduction"] = {
        "kernel_launch_per_step_before_us": e["kernel_launch_per_step_us"],
        "kernel_launch_per_step_after_us": g["kernel_launch_per_step_us"],
        "launch_share_of_tpot_before": e["launch_share_of_tpot"],
        "launch_share_of_tpot_after": g["launch_share_of_tpot"],
    }

    print(json.dumps(
        {"eager": {k: e[k] for k in keys}, "cudagraph": {k: g[k] for k in keys}},
        indent=2, ensure_ascii=False,
    ))

    out = Path(__file__).parent / "results" / f"{args.tag}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
