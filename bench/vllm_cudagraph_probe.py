#!/usr/bin/env python3
"""vLLM 官方实现 · CUDA Graph 捕获探针（对照 nano_vllm 的 P7 实测）。

测量目标（与 nano 侧同一组问题）:
  1. 一共捕获多少张图、按什么顺序
  2. 每次捕获用的是哪个 mempool（判断是否共享池复用）
  3. 每次捕获的内存增量（判断图之间是否复用显存）
  4. 开图 / 关图的峰值显存差（总量）

做法: patch `torch.cuda.CUDAGraph.capture_begin/end` 记录每次捕获的
(pool, 起始/结束 allocated & reserved)。**不修改 vLLM 源码**。

必须在 import vllm 之前设 `VLLM_ENABLE_V1_MULTIPROCESSING=0`，
否则 EngineCore 在子进程里，本进程观测不到。

用法:
  python bench/vllm_cudagraph_probe.py --mode eager|graph
输出: bench/results/vllm_cudagraph_probe_<mode>.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import os
from typing import Any

# 必须在 import vllm 之前设置
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch  # noqa: E402

CAPTURES: list[dict[str, Any]] = []

_orig_begin = torch.cuda.CUDAGraph.capture_begin
_orig_end = torch.cuda.CUDAGraph.capture_end


def _capture_begin(self, *args, **kwargs):  # noqa: ANN001
    rec = {
        "idx": len(CAPTURES),
        "pool_arg": str(kwargs.get("pool", args[0] if args else None)),
        "alloc_before": torch.cuda.memory_allocated(),
        "res_before": torch.cuda.memory_reserved(),
    }
    CAPTURES.append(rec)
    return _orig_begin(self, *args, **kwargs)


def _capture_end(self, *args, **kwargs):  # noqa: ANN001
    ret = _orig_end(self, *args, **kwargs)
    rec = CAPTURES[-1]
    rec["alloc_after"] = torch.cuda.memory_allocated()
    rec["res_after"] = torch.cuda.memory_reserved()
    try:
        rec["pool_actual"] = str(self.pool())
    except Exception as exc:  # pragma: no cover
        rec["pool_actual"] = f"n/a({exc})"
    return ret


torch.cuda.CUDAGraph.capture_begin = _capture_begin
torch.cuda.CUDAGraph.capture_end = _capture_end


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--mode", choices=["eager", "graph"], default="graph")
    ap.add_argument("--max-model-len", type=int, default=1024)
    ap.add_argument("--gpu-mem-util", type=float, default=0.45)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    import vllm
    from vllm import LLM, SamplingParams

    MB = 2**20
    base_res = torch.cuda.memory_reserved() / MB

    llm = LLM(
        model=args.model,
        enforce_eager=(args.mode == "eager"),
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        dtype="bfloat16",
        seed=0,
    )
    torch.cuda.synchronize()
    mem_after_init = {
        "allocated_MB": torch.cuda.memory_allocated() / MB,
        "reserved_MB": torch.cuda.memory_reserved() / MB,
        "peak_allocated_MB": torch.cuda.max_memory_allocated() / MB,
    }

    # 跑一次真实生成（确认能工作，也让可能的懒捕获发生）
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    prompts = [list(range(1000, 1000 + args.prompt_len)) for _ in range(args.batch)]
    outs = llm.generate(prompts=prompts, sampling_params=sp)
    torch.cuda.synchronize()

    # ---- 汇总捕获记录 ----
    rows = []
    prev_res = None
    for r in CAPTURES:
        rows.append({
            "idx": r["idx"],
            "pool": r.get("pool_actual", r["pool_arg"]),
            "alloc_before_MB": r["alloc_before"] / MB,
            "d_alloc_MB": (r["alloc_after"] - r["alloc_before"]) / MB,
            "d_res_MB": (r["res_after"] - r["res_before"]) / MB,
            "res_before_MB": r["res_before"] / MB,
        })

    pools = {r["pool"] for r in rows}
    summary = {
        "mode": args.mode,
        "vllm_version": vllm.__version__,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "model": args.model,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_mem_util,
        "n_captures": len(rows),
        "n_distinct_pools": len(pools),
        "pools": sorted(pools),
        "sum_d_alloc_MB": sum(r["d_alloc_MB"] for r in rows),
        "sum_d_res_MB": sum(r["d_res_MB"] for r in rows),
        "end_allocated_MB": torch.cuda.memory_allocated() / MB,
        "end_reserved_MB": torch.cuda.memory_reserved() / MB,
        "peak_allocated_MB": torch.cuda.max_memory_allocated() / MB,
        "reserved_before_llm_MB": base_res,
        "mem_after_init": mem_after_init,
        "captures": rows,
        "sample_output_len": [len(o.outputs[0].token_ids) for o in outs],
    }

    # 打印：捕获次数多时只打前 8 + 后 4
    print(f"=== mode={args.mode}  vllm={vllm.__version__} ===")
    print(f"捕获次数={len(rows)}  不同池数={len(pools)}")
    print(f"峰值 allocated = {summary['peak_allocated_MB']:.1f}MB   "
          f"结束后 allocated = {summary['end_allocated_MB']:.1f}MB")
    show = rows[:8] + ([{"idx": "...", "pool": "...", "d_alloc_MB": float('nan'),
                         "d_res_MB": float('nan')}] if len(rows) > 12 else []) + rows[-4:]
    for r in show:
        print(f"  #{r['idx']:>3} pool={str(r['pool'])[:28]:28s} "
              f"Δalloc={r['d_alloc_MB']:8.2f}MB Δres={r['d_res_MB']:8.2f}MB")

    out = Path(__file__).parent / "results" / f"vllm_cudagraph_probe_{args.mode}.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
