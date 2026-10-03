#!/usr/bin/env python3
"""P7 前置 · 「step 内双前向」开销量化（混合负载下 prefill 前向 + decode 前向分离计时）。

背景（docs/notes/pre-p7-fixes.md Q1）:
  当前 EngineCore._execute 把同一 step 的 prefill 批与 decode 批拆成**两次独立
  model forward**；vLLM 是单次混批前向（gpu_model_runner 把 prefill/decode token
  拼进同一 batch）。本脚本量化双前向在混合负载下的额外开销，为 P7 图捕获设计
  （decode-only 全图 vs 混批图）提供依据。

方法:
  - 混合负载: 4 条长 prompt(2048) + 4 条短 prompt(128)，budget=512 →
    长 prompt 必然分多个 chunk（512/chunk），in-progress prefill chunk 与已进入
    decode 的短请求长期同 step → 产生大量混合 step。
  - 不改引擎源码，脚本内包装 _execute / _run_prefill_batched / _run_decode /
    _run_decode_batched，逐 step 记录 (prefill 前向 ms, decode 前向 ms)，
    GPU 路径前后 torch.cuda.synchronize。

指标:
  - n_mixed: 既有 prefill 又有 decode 的 step 数（= 双前向多出的前向次数）
  - fusible_saving_upper_bound: Σ_mixed min(prefill_ms, decode_ms)
    （若混批，两次前向合并为一次，耗时近似 ≥ max(两者) → 可省 ≤ min(两者)，
    这是融合收益的**保守上界**；实际收益还包含省掉的 host 拼装与 launch，
    由 P7 的 profiler 数据补充）

用法:
  python bench/p7_pre_bench.py --model models/Qwen2.5-1.5B-Instruct
输出: bench/results/p7_pre_dual_forward.json
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

from nano_vllm.engine.core import EngineCore
from nano_vllm.engine.scheduler import Scheduler
from nano_vllm.engine.sequence import SamplingParams
from nano_vllm.model_executor.runner import NanoRunner

VOCAB = 151936  # 仅用于造随机 token id（config 驱动的模型本身不依赖此值）


def build_prompts(seed: int = 0) -> tuple[list[list[int]], dict]:
    rng = random.Random(seed)
    long_prompts = [[rng.randrange(1000, VOCAB) for _ in range(2048)] for _ in range(4)]
    short_prompts = [[rng.randrange(1000, VOCAB) for _ in range(128)] for _ in range(4)]
    return long_prompts + short_prompts, {"long": 4, "short": 4}


def instrument(engine: EngineCore, device: str):
    """包装 _execute 与三个前向入口，逐 step 记录 prefill/decode 前向耗时（ms）。"""
    records: list[dict] = []
    cur: dict = {}
    orig_execute = engine._execute
    orig_prefill = engine._run_prefill_batched
    orig_prefill_1 = engine._run_prefill
    orig_decode_1 = engine._run_decode
    orig_decode_n = engine._run_decode_batched

    def _t(fn, key, *args, **kwargs):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = fn(*args, **kwargs)
        if device == "cuda":
            torch.cuda.synchronize()
        cur[key] = cur.get(key, 0.0) + (time.perf_counter() - t0) * 1e3
        cur[key + "_calls"] = cur.get(key + "_calls", 0) + 1
        return r

    def execute(scheduler_output):
        cur.clear()
        cur["n_prefill"] = scheduler_output.num_prefills
        cur["n_decode"] = scheduler_output.num_decodes
        return orig_execute(scheduler_output)

    engine._execute = execute
    engine._run_prefill_batched = lambda sp: _t(orig_prefill, "prefill_ms", sp)
    engine._run_prefill = lambda s, n: _t(orig_prefill_1, "prefill_ms", s, n)
    engine._run_decode = lambda s: _t(orig_decode_1, "decode_ms", s)
    engine._run_decode_batched = lambda seqs: _t(orig_decode_n, "decode_ms", seqs)

    def record(step_ms: float) -> None:
        cur["step_ms"] = step_ms
        records.append(dict(cur))

    return records, record


def run_once(
    runner: NanoRunner,
    prompts: list[list[int]],
    max_new_tokens: int,
    budget: int,
    device: str,
) -> dict:
    runner.paged_cache.reset()
    scheduler = Scheduler(
        runner.paged_cache, max_num_batched_tokens=budget,
        enable_prefix_cache=False,  # 隔离变量：只测调度/前向结构，不引入命中跳过
    )
    engine = EngineCore(runner, scheduler)
    records, record = instrument(engine, device)
    params = SamplingParams(temperature=0.0, max_new_tokens=max_new_tokens)

    for p in prompts:
        engine.add_request(p, params)
    while engine.scheduler.has_requests():
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        engine.step()
        if device == "cuda":
            torch.cuda.synchronize()
        record((time.perf_counter() - t0) * 1e3)

    mixed = [r for r in records if r.get("n_prefill", 0) > 0 and r.get("n_decode", 0) > 0]
    prefill_only = [r for r in records if r.get("n_prefill", 0) > 0 and r.get("n_decode", 0) == 0]
    decode_only = [r for r in records if r.get("n_prefill", 0) == 0 and r.get("n_decode", 0) > 0]
    wall_ms = sum(r["step_ms"] for r in records)
    prefill_ms_total = sum(r.get("prefill_ms", 0.0) for r in records)
    decode_ms_total = sum(r.get("decode_ms", 0.0) for r in records)
    # 融合收益保守上界：混合 step 中较小一侧的前向时间（若混批，此侧前向被合并掉）
    fusible = sum(min(r.get("prefill_ms", 0.0), r.get("decode_ms", 0.0)) for r in mixed)

    def _mean(xs, k):
        xs = [x[k] for x in xs if x.get(k) is not None]
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "n_steps": len(records),
        "n_mixed": len(mixed),
        "n_prefill_only": len(prefill_only),
        "n_decode_only": len(decode_only),
        "wall_ms": wall_ms,
        "prefill_fwd_ms_total": prefill_ms_total,
        "decode_fwd_ms_total": decode_ms_total,
        "extra_forwards_vs_fused": len(mixed),
        "fusible_saving_upper_bound_ms": fusible,
        "fusible_saving_upper_bound_pct": (fusible / wall_ms * 100) if wall_ms else 0.0,
        "mixed_step_mean": {
            "prefill_ms": _mean(mixed, "prefill_ms"),
            "decode_ms": _mean(mixed, "decode_ms"),
            "step_ms": _mean(mixed, "step_ms"),
        },
        "mixed_samples": [
            {
                "n_prefill": r["n_prefill"], "n_decode": r["n_decode"],
                "prefill_ms": round(r.get("prefill_ms", 0.0), 3),
                "decode_ms": round(r.get("decode_ms", 0.0), 3),
                "step_ms": round(r["step_ms"], 3),
            }
            for r in mixed[:20]
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--budget", type=int, default=512)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--attn-impl", default="triton")
    ap.add_argument("--tag", default="p7_pre_dual_forward")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    runner = NanoRunner(
        args.model, device=device, dtype=dtype,
        block_size=16, attn_impl=args.attn_impl,
    )
    prompts, mix = build_prompts(seed=0)

    # warmup（不计时）
    run_once(runner, prompts, max_new_tokens=4, budget=args.budget, device=device)

    runs = [
        run_once(runner, prompts, args.max_new_tokens, args.budget, device)
        for _ in range(args.repeat)
    ]

    def med(key):
        vals = sorted(r[key] for r in runs)
        return vals[len(vals) // 2]

    summary = {
        "model": args.model,
        "device": device,
        "dtype": str(dtype),
        "attn_impl": args.attn_impl,
        "budget": args.budget,
        "max_new_tokens": args.max_new_tokens,
        "workload": {"long_prompts_2048": mix["long"], "short_prompts_128": mix["short"]},
        "repeat": args.repeat,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "torch": torch.__version__,
        "median": {
            k: med(k) for k in (
                "n_steps", "n_mixed", "n_prefill_only", "n_decode_only",
                "wall_ms", "prefill_fwd_ms_total", "decode_fwd_ms_total",
                "extra_forwards_vs_fused", "fusible_saving_upper_bound_ms",
                "fusible_saving_upper_bound_pct",
            )
        },
        "runs": runs,
    }

    out = Path(__file__).parent / "results" / f"{args.tag}.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["median"], indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
