#!/usr/bin/env python3
"""抢占恢复对齐 vLLM · 改前/改后对照基准（单变量：monkeypatch 复刻旧口径）。

对照的两种口径（其余代码/负载/池大小完全相同）:
  - **legacy（改动前）**:
      `Sequence.reset_for_preemption` 清空 `output_token_ids`；
      `Scheduler.get_computed_blocks` 只查 prompt 且钳制 `num_prompt_tokens - 1`
      → 恢复 = 重算 prompt 尾部 + **全部已生成 token 重新生成**
  - **aligned（改动后）**:
      保留 output；查找走全流 `prompt + output` 且钳制 `num_tokens - 1`
      → 恢复 = 重算 ≤ 1 个 block，末段 logits 产出**下一个** token（不重复输出）

指标:
  - `preempt_count`: 抢占次数
  - `resume_recompute_tokens`: 恢复后为"曾被抢占的请求"重算的 token 数（核心指标）
  - `forward_tokens`: 全部前向 token 数（prefill 路径 + decode 路径）
  - `wall_ms` / `outputs_identical`（两种口径的 greedy 输出必须逐 token 一致）

用法:
  python bench/preempt_align_bench.py --model models/Qwen2.5-1.5B-Instruct
输出: bench/results/preempt_align.json

注：本脚本原文件名带 `p8_` 前缀，与 roadmap 里 P8（算子融合 + 权重预拼接）撞号。
抢占恢复对齐是「P7 前置整改」，不是阶段，故改名去掉前缀（见 docs/notes/p8-prereq.md）。
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
from nano_vllm.engine.sequence import SamplingParams, Sequence
from nano_vllm.kvmm.prefix_cache import CacheHit
from nano_vllm.model_executor.runner import NanoRunner

VOCAB = 151936

# ---------------- 旧口径的 monkeypatch（仅供对照，不参与生产路径） ----------------

_ORIG_RESET = Sequence.reset_for_preemption
_ORIG_LOOKUP = Scheduler.get_computed_blocks


def _legacy_reset(self) -> None:
    """改动前: 抢占时清空已生成 token（恢复要重新生成整段）。"""
    _ORIG_RESET(self)
    self.output_token_ids.clear()


def _legacy_lookup(self, seq: Sequence) -> CacheHit:
    """改动前: 只在 prompt 上查命中，钳制 num_prompt_tokens - 1。"""
    if self.prefix_cache is None:
        return CacheHit([], 0, False, None)
    self.prefix_lookups += 1
    self.prefix_query_tokens += seq.num_prompt_tokens
    return self.prefix_cache.find_longest_hit(
        seq.prompt_token_ids, seq.prefix_cache_extra_keys,
        max_tokens=seq.num_prompt_tokens - 1,
    )


def set_legacy(enabled: bool) -> None:
    Sequence.reset_for_preemption = _legacy_reset if enabled else _ORIG_RESET
    Scheduler.get_computed_blocks = _legacy_lookup if enabled else _ORIG_LOOKUP


# ---------------- 仪表 ----------------

class Instrument:
    def __init__(self, engine: EngineCore) -> None:
        self.prefill_tokens = 0        # 走 varlen prefill 路径的 token 数
        self.decode_tokens = 0         # 走 decode 路径的 token 数（每请求 1 token）
        self.resume_recompute = 0      # 曾被抢占的请求所处 prefill chunk 的 token 数
        self.preempt_count = 0
        self._engine = engine
        self._patch()

    def _patch(self) -> None:
        e = self._engine
        self._orig_prefill_batched = e._run_prefill_batched
        self._orig_prefill_one = e._run_prefill
        self._orig_decode_one = e._run_decode
        self._orig_decode_batch = e._run_decode_batched
        self._orig_preempt = e.scheduler._preempt

        def prefill_batched(scheduled):
            for s in scheduled:
                self.prefill_tokens += s.num_scheduled_tokens
                if getattr(s.seq, "was_preempted", False):
                    self.resume_recompute += s.num_scheduled_tokens
            return self._orig_prefill_batched(scheduled)

        def prefill_one(seq, n):
            self.prefill_tokens += n
            if getattr(seq, "was_preempted", False):
                self.resume_recompute += n
            return self._orig_prefill_one(seq, n)

        def decode_one(seq):
            self.decode_tokens += 1
            return self._orig_decode_one(seq)

        def decode_batch(seqs):
            self.decode_tokens += len(seqs)
            return self._orig_decode_batch(seqs)

        def preempt(seq):
            self.preempt_count += 1
            seq.was_preempted = True       # 标记：恢复时用于统计重算量
            return self._orig_preempt(seq)

        e._run_prefill_batched = prefill_batched
        e._run_prefill = prefill_one
        e._run_decode = decode_one
        e._run_decode_batched = decode_batch
        e.scheduler._preempt = preempt

    def restore(self) -> None:
        e = self._engine
        e._run_prefill_batched = self._orig_prefill_batched
        e._run_prefill = self._orig_prefill_one
        e._run_decode = self._orig_decode_one
        e._run_decode_batched = self._orig_decode_batch
        e.scheduler._preempt = self._orig_preempt

    def snapshot(self) -> dict:
        return {
            "preempt_count": self.preempt_count,
            "resume_recompute_tokens": self.resume_recompute,
            "prefill_tokens": self.prefill_tokens,
            "decode_tokens": self.decode_tokens,
            "forward_tokens": self.prefill_tokens + self.decode_tokens,
        }


def run_once(runner: NanoRunner, prompts, max_new: int, legacy: bool) -> dict:
    runner.paged_cache.reset()
    set_legacy(legacy)
    try:
        scheduler = Scheduler(
            runner.paged_cache, max_num_batched_tokens=1024,
            enable_prefix_cache=True,   # 抢占恢复靠前缀缓存复用自己释放的块
        )
        engine = EngineCore(runner, scheduler)
        inst = Instrument(engine)
        params = SamplingParams(temperature=0.0, max_new_tokens=max_new)
        seqs = [engine.add_request(p, params) for p in prompts]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        while engine.scheduler.has_requests():
            engine.step()
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - t0) * 1e3
        inst.restore()
        out = inst.snapshot()
        out["wall_ms"] = wall_ms
        out["outputs"] = [s.output_token_ids for s in seqs]
        out["output_tokens"] = sum(len(s.output_token_ids) for s in seqs)
        return out
    finally:
        set_legacy(False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--num-blocks", type=int, default=20, help="紧池以触发抢占")
    ap.add_argument("--prompt-len", type=int, default=48)
    ap.add_argument("--num-requests", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--tag", default="preempt_align")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    runner = NanoRunner(
        args.model, device=device, dtype=dtype, block_size=16,
        num_blocks=args.num_blocks, attn_impl="triton" if device == "cuda" else "torch",
        enable_cudagraph=False,          # 隔离变量：只测抢占记账，不掺图
    )
    rng = random.Random(0)
    prompts = [
        [rng.randrange(1000, VOCAB) for _ in range(args.prompt_len)]
        for _ in range(args.num_requests)
    ]

    runs = {"legacy": [], "aligned": []}
    for _ in range(args.repeat):
        runs["legacy"].append(run_once(runner, prompts, args.max_new_tokens, legacy=True))
        runs["aligned"].append(run_once(runner, prompts, args.max_new_tokens, legacy=False))

    def med(rows, key):
        vals = sorted(r[key] for r in rows)
        return vals[len(vals) // 2]

    keys = ("preempt_count", "resume_recompute_tokens", "prefill_tokens",
            "decode_tokens", "forward_tokens", "wall_ms", "output_tokens")
    summary = {
        "model": args.model,
        "device": device,
        "dtype": str(dtype),
        "num_blocks": args.num_blocks,
        "prompt_len": args.prompt_len,
        "num_requests": args.num_requests,
        "max_new_tokens": args.max_new_tokens,
        "repeat": args.repeat,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "torch": torch.__version__,
        "legacy": {k: med(runs["legacy"], k) for k in keys},
        "aligned": {k: med(runs["aligned"], k) for k in keys},
        "outputs_identical": all(
            lg["outputs"] == al["outputs"]
            for lg, al in zip(runs["legacy"], runs["aligned"])
        ),
    }
    legacy_rc = summary["legacy"]["resume_recompute_tokens"]
    aligned_rc = summary["aligned"]["resume_recompute_tokens"]
    summary["reduction"] = {
        "resume_recompute_tokens_before": legacy_rc,
        "resume_recompute_tokens_after": aligned_rc,
        "reduction_x": (legacy_rc / aligned_rc) if aligned_rc else None,
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    out = Path(__file__).parent / "results" / f"{args.tag}.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
