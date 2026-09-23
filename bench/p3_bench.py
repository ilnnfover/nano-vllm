#!/usr/bin/env python3
"""P3 · PagedAttention benchmark：block_size 扫描 + decode kernel 三实现对比。

两种模式:
  block-size: block_size ∈ {8,16,32,64} 扫描，测 TPOT + 显存峰值 + KV 浪费率
  kernel:     三实现（torch 朴素 / triton 自研 / 连续 SDPA）decode 延迟 vs seq_len 曲线

用法:
  python bench/p3_bench.py --mode block-size --model models/Qwen2.5-1.5B-Instruct
  python bench/p3_bench.py --mode kernel --seq-lens 128 512 1024 4096 8192
输出: bench/results/p3_{mode}_*.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import platform
import statistics
import time

import torch

from nano_vllm.model_executor.runner import NanoRunner


def timed(fn, device: str) -> tuple[float, object]:
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3, out


def run_block_size(args, device: str, dtype: torch.dtype) -> dict:
    """扫描 block_size：固定一条 medium prompt，测 prefill + decode。"""
    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    text = "In a shocking finding, scientists discovered a herd of unicorns. " * 40
    ids = tok(text).input_ids[: args.prompt_len]
    print(f"[block-size] prompt_len={len(ids)} decode={args.decode_len}")

    rows = []
    for bs in args.block_sizes:
        max_seq = len(ids) + args.decode_len + 8
        r = NanoRunner(
            args.model, device=device, dtype=dtype, block_size=bs,
            max_seq_len=max_seq, attn_impl=args.attn_impl,
        )
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        ttft, logits = timed(lambda: r._prefill_paged(ids), device)
        decode_ms = []
        for _ in range(args.decode_len):
            dt, logits = timed(lambda lg=logits: r._decode_paged(int(lg.argmax())), device)
            decode_ms.append(dt)
        peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0
        tpot = statistics.mean(decode_ms)
        row = {
            "block_size": bs,
            "ttft_ms": ttft,
            "tpot_ms": tpot,
            "tpot_p99_ms": float(torch.tensor(decode_ms).quantile(0.99).item()) if decode_ms else 0.0,
            "kv_waste_rate": r.paged_cache.waste_rate,
            "num_used_blocks": r.paged_cache.num_used_blocks,
            "peak_gb": peak,
        }
        rows.append(row)
        print(
            f"  bs={bs:<3} TTFT={ttft:>8.2f}ms TPOT={tpot:>7.2f}ms "
            f"waste={row['kv_waste_rate']:.4f} peak={peak:.2f}GB"
        )
    return {"mode": "block-size", "rows": rows}


def run_kernel(args, device: str, dtype: torch.dtype) -> dict:
    """三实现 decode 延迟 vs seq_len 曲线。"""
    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    base = tok("In a shocking finding, scientists discovered a herd of unicorns. " * 400).input_ids

    rows = []
    for seq_len in args.seq_lens:
        ids = base[:seq_len]
        r = NanoRunner(
            args.model, device=device, dtype=dtype, block_size=args.block_size,
            max_seq_len=seq_len + 8, attn_impl=args.attn_impl,
        )
        row = {"seq_len": seq_len}
        # torch 朴素
        r.attn_impl = "torch"
        logits = r._prefill_paged(ids)
        dts = []
        for _ in range(args.repeat):
            dt, logits = timed(lambda lg=logits: r._decode_paged(int(lg.argmax())), device)
            dts.append(dt)
        row["torch_ms"] = statistics.median(dts)
        r._free_paged()
        # triton 自研（仅 GPU）
        if device == "cuda":
            r.attn_impl = "triton"
            logits = r._prefill_paged(ids)
            dts = []
            for _ in range(args.repeat):
                dt, logits = timed(lambda lg=logits: r._decode_paged(int(lg.argmax())), device)
                dts.append(dt)
            row["triton_ms"] = statistics.median(dts)
            row["speedup_vs_torch"] = row["torch_ms"] / row["triton_ms"]
            r._free_paged()
        rows.append(row)
        line = f"  seq={seq_len:<6} torch={row['torch_ms']:>8.2f}ms"
        if "triton_ms" in row:
            line += f"  triton={row['triton_ms']:>8.2f}ms  ({row['speedup_vs_torch']:.2f}x)"
        print(line)
    return {"mode": "kernel", "rows": rows}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", default="block-size", choices=["block-size", "kernel"])
    p.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--attn-impl", default="torch", choices=["torch", "triton"])
    p.add_argument("--block-sizes", nargs="*", type=int, default=[8, 16, 32, 64])
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--prompt-len", type=int, default=1024)
    p.add_argument("--decode-len", type=int, default=64)
    p.add_argument("--seq-lens", nargs="*", type=int, default=[128, 512, 1024, 4096, 8192])
    p.add_argument("--repeat", type=int, default=8)
    p.add_argument("--tag", default=None)
    p.add_argument("--results-dir", default="bench/results")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    torch.manual_seed(42)
    print(f"[p3-bench] mode={args.mode} device={device} dtype={args.dtype} attn_impl={args.attn_impl}")
    if device == "cuda":
        print(f"[p3-bench] GPU: {torch.cuda.get_device_name(0)}")

    body = run_block_size(args, device, dtype) if args.mode == "block-size" else run_kernel(args, device, dtype)

    payload = {
        "tag": args.tag,
        "model_path": args.model,
        "device": device,
        "dtype": args.dtype,
        "attn_impl": args.attn_impl,
        "torch": torch.__version__,
        "python": platform.python_version(),
        **body,
    }
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"p3_{args.mode}_{Path(args.model).name}_{args.dtype}"
    out_path = out_dir / f"{tag}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"[p3-bench] 结果 -> {out_path}")


if __name__ == "__main__":
    main()