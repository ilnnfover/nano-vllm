#!/usr/bin/env python3
"""本地 benchmark：对指定 backend 逐条测量 TTFT / TPOT / 吞吐。

Backend 统一接口: prefill(prompt_ids) -> last_logits, step(tok) -> last_logits
（状态由 backend 自持: hf 用 DynamicCache, nano 用自研 KVCache, nano-eager 全量重算）。
P2 起 nano 默认走 KV cache, nano-eager 保留 P1 eager 路径作对照。

口径定义:
  TTFT  = prefill 一次前向的耗时 (ms)
  TPOT  = decode 阶段每 token 均耗时 (ms)，即逐条 decode 步均值
  吞吐  = output_tok/s（含 prefill 总时间摊入）
输出: bench/results/{tag}.json
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
from functools import partial


import numpy as np
import torch
import transformers


class HFBackend:
    def __init__(self, model_path: str, device: str, dtype: torch.dtype) -> None:
        from transformers import AutoModelForCausalLM, DynamicCache

        self.model = (
            AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, attn_implementation="sdpa")
            .to(device)
            .eval()
        )
        self._cache_cls = DynamicCache
        self.device = device
        self.cache = None

    @torch.no_grad()
    def prefill(self, prompt_ids: list[int]) -> torch.Tensor:
        self.cache = self._cache_cls()
        t = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        out = self.model(input_ids=t, past_key_values=self.cache, use_cache=True)
        return out.logits[0, -1]

    @torch.no_grad()
    def step(self, tok: int) -> torch.Tensor:
        t = torch.tensor([[tok]], dtype=torch.long, device=self.device)
        out = self.model(input_ids=t, past_key_values=self.cache, use_cache=True)
        return out.logits[0, -1]


class NanoBackend:
    """use_cache=True 走 P2 prefill+decode（KV cache），False 走 P1 eager 全量重算。"""

    def __init__(self, model_path: str, device: str, dtype: torch.dtype, use_cache: bool = True, max_seq_len: int = 1024) -> None:
        from nano_vllm.model_executor.runner import NanoRunner

        self.runner = NanoRunner(model_path, device=device, dtype=dtype, max_seq_len=max_seq_len)
        self.device = device
        self.use_cache = use_cache
        self.ids: list[int] = []

    @torch.no_grad()
    def prefill(self, prompt_ids: list[int]) -> torch.Tensor:
        if self.use_cache:
            return self.runner._prefill(prompt_ids)
        self.ids = list(prompt_ids)
        return self.runner.forward_last_logits(self.ids)

    @torch.no_grad()
    def step(self, tok: int) -> torch.Tensor:
        if self.use_cache:
            return self.runner._decode(tok)
        self.ids.append(tok)
        return self.runner.forward_last_logits(self.ids)


BACKENDS = {
    "hf": HFBackend,
    "nano": NanoBackend,
    "nano-eager": partial(NanoBackend, use_cache=False),
}


def run_one(backend, prompt_ids: list[int], output_len: int, device: str) -> dict:
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    last = backend.prefill(prompt_ids)
    if device == "cuda":
        torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1e3

    next_tok = int(last.argmax())
    decode_ms = []
    gen = [next_tok]
    for _ in range(output_len - 1):
        t1 = time.perf_counter()
        last = backend.step(next_tok)
        if device == "cuda":
            torch.cuda.synchronize()
        decode_ms.append((time.perf_counter() - t1) * 1e3)
        next_tok = int(last.argmax())
        gen.append(next_tok)

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
    p.add_argument("--max-seq-len", type=int, default=None,
                   help="KV cache 预分配长度；默认按数据自动: max(prompt_len)+max(output_len) 向上取 256 倍数")
    p.add_argument("--warmup", type=int, default=1, help="预热轮数（结果丢弃）")
    p.add_argument("--repeat", type=int, default=3, help="测量轮数（取中位数）")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    torch.manual_seed(42)
    np.random.seed(42)

    ds = json.loads(Path(args.dataset).read_text())
    requests = [r for r in ds["requests"] if r["category"] in set(args.categories)]
    if args.limit:
        requests = requests[: args.limit]
    assert requests, "没有匹配的数据条目"

    if args.max_seq_len is not None:
        max_seq_len = args.max_seq_len
    else:
        need = max(r["prompt_len"] + r["output_len"] for r in requests)
        max_seq_len = (need + 255) // 256 * 256

    print(f"[bench] backend={args.backend} model={args.model} device={device} dtype={args.dtype}")
    if args.backend.startswith("nano"):  
        backend = BACKENDS[args.backend](args.model, device, dtype, max_seq_len=max_seq_len)
        print(f"[bench] max_seq_len={max_seq_len}")
    else:
        backend = BACKENDS[args.backend](args.model, device, dtype)
    if device == "cuda":
        print(
            f"[bench] GPU: {torch.cuda.get_device_name(0)}, "
            f"显存 {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB"
        )

    warm = requests[0]
    run_one(backend, warm["prompt_ids"][:64], 4, device)
    print(f"[bench] 预热完成, warmup={args.warmup} repeat={args.repeat}")

    def pct(vals: list[float], q: float) -> float:
        return float(np.percentile(vals, q))

    def run_round() -> dict:
        """跑一轮所有请求，返回 summary 指标。"""
        results = []
        for r in requests:
            rec = run_one(backend, r["prompt_ids"], r["output_len"], device)
            rec.update({"id": r["id"], "category": r["category"], "prompt_len": r["prompt_len"]})
            results.append(rec)

        all_tpot = [statistics.mean(r["decode_ms"]) for r in results]
        total_out = sum(r["output_len"] for r in results)
        total_time = sum(r["prefill_ms"] + sum(r["decode_ms"]) for r in results) / 1e3
        return {
            "ttft_p50_ms": pct([r["prefill_ms"] for r in results], 50),
            "ttft_p99_ms": pct([r["prefill_ms"] for r in results], 99),
            "tpot_p50_ms": pct(all_tpot, 50),
            "tpot_p99_ms": pct(all_tpot, 99),
            "output_tok_per_s": total_out / total_time if total_time else 0.0,
            "n_requests": len(results),
        }

    # warmup 轮丢弃
    for _ in range(args.warmup):
        run_round()

    # repeat 轮取中位数
    summaries = [run_round() for _ in range(args.repeat)]
    keys = ["ttft_p50_ms", "ttft_p99_ms", "tpot_p50_ms", "tpot_p99_ms", "output_tok_per_s"]
    summary: dict = {}
    for k in keys:
        vals = sorted(s[k] for s in summaries)
        summary[k] = float(np.median(vals))
        summary[f"{k}_all"] = [float(v) for v in vals]
    summary["n_requests"] = summaries[0]["n_requests"]
    summary["warmup"] = args.warmup
    summary["repeat"] = args.repeat

    print(f"[bench] {args.repeat} 轮中位数:")
    for k in keys:
        print(f"  {k:<20} {summary[k]:.2f}  (all: {[f'{v:.2f}' for v in summary[f'{k}_all']]})")

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
