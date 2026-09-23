#!/usr/bin/env python3
"""P3 · 32K 单条长上下文验收：分页 prefill + decode，报告显存峰值与输出有效性。

验收标准（roadmap P3）:
  - 长上下文（32K 单条）可跑通
  - prefill 峰值 < 8 GB（16GB 显存留 50% 余量；P0-03 只算 last-token logits 后达标）
  - 输出无 NaN/Inf

用法:
  python bench/p3_longctx_check.py --model models/Qwen2.5-1.5B-Instruct --device cuda
输出: bench/results/p3_longctx_*.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import platform
import time

import torch

from nano_vllm.model_executor.runner import NanoRunner


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--attn-impl", default="triton", choices=["torch", "triton"])
    p.add_argument("--decode-len", type=int, default=8)
    p.add_argument("--prompt-len", type=int, default=None, help="缺省用 dataset shared_prefix 全量")
    p.add_argument("--tag", default=None)
    p.add_argument("--results-dir", default="bench/results")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    ds = json.loads(Path("bench/data/datasets.json").read_text())
    req = next(r for r in ds["requests"] if r["category"] == "shared_prefix")
    ids = req["prompt_ids"]
    if args.prompt_len:
        ids = ids[: args.prompt_len]
    print(f"[longctx] prompt_len={len(ids)} block_size={args.block_size} attn_impl={args.attn_impl}")

    r = NanoRunner(
        args.model, device=device, dtype=dtype, block_size=args.block_size,
        max_seq_len=len(ids) + args.decode_len + 8, attn_impl=args.attn_impl,
    )

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits = r._prefill_paged(ids)
    if device == "cuda":
        torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1e3
    prefill_peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0

    # decode
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    gen = []
    tok = int(logits.argmax())
    decode_ms = []
    for _ in range(args.decode_len):
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        logits = r._decode_paged(tok)
        if device == "cuda":
            torch.cuda.synchronize()
        decode_ms.append((time.perf_counter() - t1) * 1e3)
        tok = int(logits.argmax())
        gen.append(tok)
    decode_peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0

    finite = bool(torch.isfinite(logits).all().item())
    result = {
        "prompt_len": len(ids),
        "block_size": args.block_size,
        "attn_impl": args.attn_impl,
        "prefill_ms": prefill_ms,
        "prefill_peak_gb": prefill_peak,
        "decode_peak_gb": decode_peak,
        "decode_ms_median": float(torch.tensor(decode_ms).median().item()) if decode_ms else 0.0,
        "logits_finite": finite,
        "gen_tokens": gen,
        "kv_waste_rate": r.paged_cache.waste_rate,
        "num_used_blocks": r.paged_cache.num_used_blocks,
        "device": device,
        "dtype": args.dtype,
        "torch": torch.__version__,
        "python": platform.python_version(),
    }
    print(f"[longctx] prefill {prefill_ms:.1f}ms peak {prefill_peak:.2f}GB | "
          f"decode peak {decode_peak:.2f}GB | finite={finite} | waste={result['kv_waste_rate']:.5f}")
    print(f"[longctx] gen={gen}")

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"p3_longctx_{Path(args.model).name}_{args.dtype}"
    out_path = out_dir / f"{tag}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(f"[longctx] 结果 -> {out_path}")

    # 验收判定
    ok = finite and (device != "cuda" or prefill_peak < 8.0)
    print("[longctx] 验收: " + ("PASS ✓" if ok else "FAIL ✗"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()