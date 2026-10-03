#!/usr/bin/env python3
"""P7 · `DecodeBuffers.fill` 微基准：单次 H2D 改造前后对照。

为什么需要单独测:
  端到端 TPOT 在单卡/WSL 上跨 run 漂移可达 ~20%（本次实测中**未改动**的 eager
  路径也漂了 21%），无法用来判断「少几次传输」这种量级的改动。所以这里在**同一
  进程内 A/B 交替**测量 `fill()` 本身，消除机器状态漂移。

对照口径（其余完全相同，只有 fill 实现不同）:
  - legacy: `torch.tensor(..., device=dev)` 建临时 GPU 张量再 `copy_`
            → 每路 1 次 H2D + 1 次 D2D + 1 次分配；padding 行用设备侧 `fill_`
  - single: host 侧构造整桶后一次 `copy_` 直达
            → 每路 1 次 H2D；padding 行随整桶送达

同时断言两种实现写出的 buffer **逐元素一致**（否则对照无意义）。

用法: python bench/p7_fill_bench.py
输出: bench/results/p7_fill_bench.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import time
from typing import Any

import torch

from nano_vllm.cudagraph.buffers import DecodeBuffers
from nano_vllm.engine.sequence import Sequence

BLOCK_SIZE = 16
SCRATCH_BLOCK = 7  # 任意固定块号


class _PagedStub:
    """只需 block_size 的最小替身（fill 只用这一个字段）。"""

    block_size = BLOCK_SIZE


def make_seqs(n: int, block_table_len: int, pos: int = 40) -> list[Sequence]:
    """构造 n 条处于稳态 decode 的请求（num_computed_tokens 落在 output 区）。"""
    seqs = []
    for i in range(n):
        seq = Sequence(seq_id=i, prompt_token_ids=list(range(1, 33)))
        seq.output_token_ids = [100 + i] * 20
        seq.num_computed_tokens = pos
        seq.block_table = [(i + 1) * 100 + b for b in range(block_table_len)]
        seqs.append(seq)
    return seqs


def legacy_fill(bufs: DecodeBuffers, seqs, paged_cache, scratch_block: int,
                 bucket: int) -> None:
    """改造前的 fill()（仅用于 A/B 对照，不参与生产路径）。"""
    n = len(seqs)
    if n > bucket:
        raise ValueError(f"请求数 {n} 超过桶 {bucket}")
    bs = paged_cache.block_size
    dev = bufs.device

    if n:
        toks, poss, slots, lens, tables = [], [], [], [], []
        for seq in seqs:
            pos = seq.num_computed_tokens
            tok = seq.input_token_ids(pos, 1)[0]
            block_table = seq.block_table
            toks.append(tok)
            poss.append(pos)
            slots.append(block_table[pos // bs] * bs + pos % bs)
            lens.append(pos + 1)
            tables.append(block_table)

        bufs.input_ids[:n].copy_(torch.tensor(toks, device=dev))
        bufs.positions[:n].copy_(torch.tensor(poss, device=dev))
        bufs.slot_mapping[:n].copy_(torch.tensor(slots, device=dev))
        bufs.seq_lens[:n].copy_(torch.tensor(lens, dtype=torch.int32, device=dev))

        max_len = max(len(t) for t in tables)
        padded = [t + [0] * (max_len - len(t)) for t in tables]
        bufs.block_table[:n].zero_()
        bufs.block_table[:n, :max_len].copy_(
            torch.tensor(padded, dtype=torch.int32, device=dev)
        )

    if n < bucket:
        bufs.input_ids[n:].fill_(0)
        bufs.positions[n:].fill_(0)
        bufs.slot_mapping[n:].fill_(scratch_block * bs)
        bufs.seq_lens[n:].fill_(1)
        bufs.block_table[n:].zero_()
        bufs.block_table[n:, 0].fill_(scratch_block)


def timeit(fn, iters: int) -> float:
    """每次调用的墙钟时间（µs）。调用间不 sync，测「纯入队 + 传输提交」成本。"""
    fn()  # 热身
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters


def check_equivalent(bufs: DecodeBuffers, seqs, paged, scratch: int,
                     bucket: int) -> bool:
    """两种实现写出的 buffer 是否逐元素一致。"""
    legacy_fill(bufs, seqs, paged, scratch, bucket)
    torch.cuda.synchronize()
    snap = {k: getattr(bufs, k).clone() for k in
            ("input_ids", "positions", "slot_mapping", "seq_lens", "block_table")}
    bufs.fill(seqs, paged, scratch, bucket)
    torch.cuda.synchronize()
    return all(
        torch.equal(getattr(bufs, k), snap[k])
        for k in ("input_ids", "positions", "slot_mapping", "seq_lens", "block_table")
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--tag", default="p7_fill_bench")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "需要 CUDA"
    paged = _PagedStub()
    dev = torch.device("cuda")

    cases = [
        ("bucket8_n8", 8, 8),    # 满桶：无 padding 行
        ("bucket8_n5", 8, 5),    # 半桶：含 padding 行
        ("bucket32_n17", 32, 17),
    ]

    out: dict[str, Any] = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "block_size": BLOCK_SIZE,
        "iters": args.iters,
        "rounds": args.rounds,
        "cases": {},
    }

    for name, bucket, n in cases:
        bufs = DecodeBuffers(max_bucket=bucket, max_blocks=64,
                             device=dev, dtype=torch.bfloat16)
        seqs = make_seqs(n, block_table_len=8)

        same = check_equivalent(bufs, seqs, paged, SCRATCH_BLOCK, bucket)

        legacy_ms, single_ms = [], []
        for _ in range(args.rounds):
            # A/B 交替：同一轮内先 legacy 后 single，机器状态漂移对两者影响接近
            legacy_ms.append(timeit(lambda: legacy_fill(bufs, seqs, paged, SCRATCH_BLOCK, bucket),
                                    args.iters))
            single_ms.append(timeit(lambda: bufs.fill(seqs, paged, SCRATCH_BLOCK, bucket),
                                    args.iters))

        def median(vals: list[float]) -> float:
            s = sorted(vals)
            return s[len(s) // 2]

        # 用**中位数**作头条口径：本机噪声下单轮 min 会被偶发低值拉偏
        # （实测 full-bucket 组 legacy 的 min 比其中位数低 17%，用 min 会得出
        #  "改后更慢"的错误结论）。min/median 都给，便于复核。
        lg, sg = median(legacy_ms), median(single_ms)
        out["cases"][name] = {
            "bucket": bucket,
            "n": n,
            "buffers_identical": same,
            "legacy_us_per_fill": round(lg, 2),
            "single_us_per_fill": round(sg, 2),
            "speedup": round(lg / sg, 3) if sg else None,
            "saved_us_per_fill": round(lg - sg, 2),
            "legacy_min_us": round(min(legacy_ms), 2),
            "single_min_us": round(min(single_ms), 2),
            "speedup_min": round(min(legacy_ms) / min(single_ms), 3),
            "legacy_rounds_us": [round(v, 2) for v in legacy_ms],
            "single_rounds_us": [round(v, 2) for v in single_ms],
        }
        print(f"{name:14s} legacy={lg:8.2f}us  single={sg:8.2f}us  "
              f"x{lg / sg:.2f}  （min 口径 x{min(legacy_ms) / min(single_ms):.2f}）"
              f"  一致={same}")

    out_path = Path(__file__).parent / "results" / f"{args.tag}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
