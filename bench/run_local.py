#!/usr/bin/env python3
"""P0 · 本地 benchmark：对指定 backend 逐条测量 TTFT / TPOT / 吞吐。

P0 仅支持 --backend hf（HF transformers 手写 decode 循环，精确插桩）；
P1 起新 backend 注册进 BACKENDS 字典即可复用同一口径。

口径定义:
  TTFT  = prefill 一次前向的耗时 (ms)
  TPOT  = decode 阶段每 token 均耗时 (ms)，即逐条 decode 步均值
  吞吐  = output_tok/s 与 total_tok/s（含 prefill）
输出: bench/results/{tag}.json
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import transformers


def build_hf(model_path: str, device: str, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM

    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype, attn_implementation="sdpa"
        )
        .to(device)
        .eval()
    )
    return model


BACKENDS = {"hf": build_hf}


def run_one(model, prompt_ids: list[int], output_len: int, device: str) -> dict:
    torch.cuda.synchronize() if device == "cuda" else None
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    if device == "cuda":
        torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1e3

    next_tok = out.logits[0, -1].argmax()
    past = out.past_key_values
    decode_ms = []
    gen = [int(next_tok)]
    for _ in range(output_len - 1):
        t1 = time.perf_counter()
        with torch.no_grad():
            out = model(input_ids=next_tok.view(1, 1), past_key_values=past, use_cache=True)
        if device == "cuda":
            torch.cuda.synchronize()
        decode_ms.append((time.perf_counter() - t1) * 1e3)
        next_tok = out.logits[0, -1].argmax()
        past = out.past_key_values
        gen.append(int(next_tok))

    return {
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "output_len": len(gen),
        "output_ids": gen,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", default="hf", choices=sorted(BACKENDS))
    p.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    p.add_argument("--dataset", default="bench/data/datasets.json")
    p.add_argument("--categories", nargs="*", default=["short"], help="默认只跑 short")
    p.add_argument("--limit", type=int, default=None, help="每类最多跑几条")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--tag", default=None, help="结果文件名，默认自动生成")
    p.add_argument("--results-dir", default="bench/results")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    torch.manual_seed(42)
    np.random.seed(42)

    ds = json.loads(Path(args.dataset).read_text())
    requests = [
        r for r in ds["requests"] if r["category"] in set(args.categories)
    ]
    if args.limit:
        requests = requests[: args.limit]
    assert requests, "没有匹配的数据条目"

    print(f"[bench] backend={args.backend} model={args.model} device={device} dtype={args.dtype}")
    model = BACKENDS[args.backend](args.model, device, dtype)
    if device == "cuda":
        print(
            f"[bench] GPU: {torch.cuda.get_device_name(0)}, "
            f"显存 {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB"
        )

    # 预热：消除首条的首帧开销（kernel 加载 / CUDA context）
    warm = requests[0]
    run_one(model, warm["prompt_ids"][:64], 4, device)
    print(f"[bench] 预热完成, 开始 {len(requests)} 条")

    results = []
    for r in requests:
        rec = run_one(model, r["prompt_ids"], r["output_len"], device)
        rec.update(
            {
                "id": r["id"],
                "category": r["category"],
                "prompt_len": r["prompt_len"],
            }
        )
        results.append(rec)
        print(
            f"  #{r['id']:<3} {r['category']:<13} prompt={r['prompt_len']:<6} "
            f"TTFT={rec['prefill_ms']:>9.2f}ms  TPOT={statistics.mean(rec['decode_ms']):>7.2f}ms  "
            f"({1000 / statistics.mean(rec['decode_ms']):>6.1f} tok/s)"
        )

    def pct(vals: list[float], q: float) -> float:
        return float(np.percentile(vals, q))

    all_tpot = [statistics.mean(r["decode_ms"]) for r in results]
    total_out = sum(r["output_len"] for r in results)
    total_time = sum(r["prefill_ms"] + sum(r["decode_ms"]) for r in results) / 1e3
    summary = {
        "ttft_p50_ms": pct([r["prefill_ms"] for r in results], 50),
        "ttft_p99_ms": pct([r["prefill_ms"] for r in results], 99),
        "tpot_p50_ms": pct(all_tpot, 50),
        "tpot_p99_ms": pct(all_tpot, 99),
        "output_tok_per_s": total_out / total_time if total_time else 0.0,
        "n_requests": len(results),
    }

    gpu_mem = None
    if device == "cuda":
        gpu_mem = {
            "allocated_gb": torch.cuda.memory_allocated() / 2**30,
            "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
        }

    payload = {
        "tag": args.tag,
        "backend": args.backend,
        "model_path": args.model,
        "device": device,
        "dtype": args.dtype,
        "categories": args.categories,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "gpu_mem": gpu_mem,
        "summary": summary,
        "requests": results,
    }
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{args.backend}_{Path(args.model).name}_{args.dtype}"
    out_path = out_dir / f"{tag}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    print("\n[summary]")
    for k, v in summary.items():
        print(f"  {k:<20} {v:.2f}" if isinstance(v, float) else f"  {k:<20} {v}")
    print(f"[bench] 结果 -> {out_path}")


if __name__ == "__main__":
    main()