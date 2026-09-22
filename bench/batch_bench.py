#!/usr/bin/env python3
"""P2 · 静态批吞吐基准：batch=N padding 拼批 vs 逐条串行，吞吐对比 + 浪费率落盘。

口径定义:
  吞吐   = 总输出 token 数 / 墙钟时间 (tok/s)，decode 与 prefill 时间均计入
  串行   = N 条请求逐条 generate(use_cache=True)，总时间为逐条墙钟之和
  batch  = N 条请求按 batch_size 分组 generate_batch（padding 拼批 + 静态推进）
  一致性 = greedy 下 batch 输出与串行输出逐 token 对比（附带正确性证据）
  浪费率 = padding_waste_rate（拼批 pad 浪费）+ kv_waste_rate（预分配浪费）

说明: batch 模式中已 done 的请求仍随批推进 forward（浪费算力）——这是静态批
的真实行为，也是 P4 continuous batching 的动机证据。

用法:
  python bench/batch_bench.py --batches 1 8 16 --categories short medium
输出: bench/results/{tag}.json
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import platform
import time


import numpy as np
import torch
import transformers

from nano_vllm.model_executor.runner import NanoRunner, SamplingParams


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def run_serial(runner: NanoRunner, groups: list[list[list[int]]], params: SamplingParams) -> dict:
    """逐条串行: 每条独立 generate，总时间 = 全部墙钟之和。"""
    sync(runner.device)
    t0 = time.perf_counter()
    serial_outs: list[list[int]] = []
    for group in groups:
        for ids in group:
            serial_outs.append(runner.generate(ids, params, use_cache=True))
    sync(runner.device)
    wall_ms = (time.perf_counter() - t0) * 1e3
    out_tokens = sum(len(o) for o in serial_outs)
    return {"wall_ms": wall_ms, "output_tokens": out_tokens, "serial_outs": serial_outs}


def run_batch(runner: NanoRunner, groups: list[list[list[int]]], params: SamplingParams) -> tuple[dict, list[list[int]]]:
    """按组拼批 generate_batch，返回汇总指标与全部输出（按原始顺序）。"""
    sync(runner.device)
    t0 = time.perf_counter()
    all_outs: list[list[int]] = []
    all_stats: list[dict] = []
    for group in groups:
        outs, stats = runner.generate_batch(group, params)
        all_outs.extend(outs)
        all_stats.append(stats)
    sync(runner.device)
    wall_ms = (time.perf_counter() - t0) * 1e3
    out_tokens = sum(len(o) for o in all_outs)
    agg = {
        "wall_ms": wall_ms,
        "output_tokens": out_tokens,
        "n_batches": len(groups),
        "padding_waste_rate": float(np.mean([s["padding_waste_rate"] for s in all_stats])),
        "kv_waste_rate": float(np.mean([s["kv_waste_rate"] for s in all_stats])),
        "all_outs": all_outs,
    }
    return agg, all_outs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    p.add_argument("--batches", nargs="*", type=int, default=[1, 8, 16])
    p.add_argument("--dataset", default="bench/data/datasets.json")
    p.add_argument("--categories", nargs="*", default=["short", "medium"])
    p.add_argument("--limit", type=int, default=16, help="总共最多取多少条请求")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--max-seq-len", type=int, default=None,
                   help="KV cache 预分配长度；默认按数据自动: max(prompt_len)+max_new_tokens 向上取 256 倍数")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--tag", default=None)
    p.add_argument("--results-dir", default="bench/results")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    torch.manual_seed(42)
    np.random.seed(42)

    ds = json.loads(Path(args.dataset).read_text())
    requests = [r for r in ds["requests"] if r["category"] in set(args.categories)]
    requests = requests[: args.limit]
    assert requests, "没有匹配的数据条目"

    all_ids = [r["prompt_ids"] for r in requests]
    if args.max_seq_len is not None:
        max_seq_len = args.max_seq_len
    else:
        need = max(len(p) for p in all_ids) + args.max_new_tokens
        max_seq_len = (need + 255) // 256 * 256

    print(f"[batch-bench] model={args.model} device={device} dtype={args.dtype} n={len(requests)}")
    runner = NanoRunner(args.model, device=device, dtype=dtype, max_seq_len=max_seq_len)
    print(f"[batch-bench] max_seq_len={max_seq_len} (max_prompt={max(len(p) for p in all_ids)} + gen {args.max_new_tokens})")
    if device == "cuda":
        print(
            f"[batch-bench] GPU: {torch.cuda.get_device_name(0)}, "
            f"显存 {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB"
        )

    params = SamplingParams(temperature=0.0, max_new_tokens=args.max_new_tokens)


    # 预热: 串行 1 条 + batch 最小配置 1 批
    runner.generate(all_ids[0], params, use_cache=True)
    runner.generate_batch(all_ids[: min(2, len(all_ids))], params)
    print("[batch-bench] 预热完成")

    serial = run_serial(runner, [all_ids], params)
    serial_tps = serial["output_tokens"] / (serial["wall_ms"] / 1e3)
    print(f"\n[serial] wall={serial['wall_ms']:.1f}ms  out_tok={serial['output_tokens']}  吞吐={serial_tps:.1f} tok/s")

    batch_results = []
    for bs in args.batches:
        groups = [all_ids[i : i + bs] for i in range(0, len(all_ids), bs)]
        agg, all_outs = run_batch(runner, groups, params)
        tps = agg["output_tokens"] / (agg["wall_ms"] / 1e3)
        match = all(o == s for o, s in zip(all_outs, serial["serial_outs"]))
        rec = {
            "batch_size": bs,
            "n_batches": agg["n_batches"],
            "wall_ms": agg["wall_ms"],
            "output_tokens": agg["output_tokens"],
            "tok_per_s": tps,
            "speedup_vs_serial": tps / serial_tps,
            "padding_waste_rate": agg["padding_waste_rate"],
            "kv_waste_rate": agg["kv_waste_rate"],
            "match_serial": bool(match),
        }
        batch_results.append(rec)
        print(
            f"[batch={bs:>2}] wall={agg['wall_ms']:.1f}ms  out_tok={agg['output_tokens']}  "
            f"吞吐={tps:.1f} tok/s  加速={rec['speedup_vs_serial']:.2f}x  "
            f"pad浪费={agg['padding_waste_rate']:.1%}  kv浪费={agg['kv_waste_rate']:.1%}  一致={match}"
        )

    gpu_mem = None
    if device == "cuda":
        gpu_mem = {
            "allocated_gb": torch.cuda.memory_allocated() / 2**30,
            "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
        }

    payload = {
        "tag": args.tag,
        "model_path": args.model,
        "device": device,
        "dtype": args.dtype,
        "categories": args.categories,
        "max_new_tokens": args.max_new_tokens,
        "n_requests": len(requests),
        "request_ids": [r["id"] for r in requests],
        "serial": {"wall_ms": serial["wall_ms"], "output_tokens": serial["output_tokens"], "tok_per_s": serial_tps},
        "batches": batch_results,
        "gpu_mem": gpu_mem,
        "torch": torch.__version__,
        "python": platform.python_version(),
    }
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"batch_bench_{Path(args.model).name}_{args.dtype}"
    out_path = out_dir / f"{tag}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"\n[batch-bench] 结果 -> {out_path}")


if __name__ == "__main__":
    main()