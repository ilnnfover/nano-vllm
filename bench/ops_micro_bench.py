#!/usr/bin/env python3
"""P8 · 自研算子单算子微基准（roadmap:150「单算子微基准（融合 vs 不融合，µs 级）」）。

覆盖 `nano_vllm/ops/` 里两个融合算子：

| 算子 | 对照（融合前） | 回答什么 |
| --- | --- | --- |
| `rms_norm` / `fused_add_rms_norm` | 逐 op / `F.rms_norm` / `aten::add` + `F.rms_norm` | 自研核 vs 库算子谁快；`num_warps` 怎么选 |
| `apply_rope` | `qwen2.apply_rotary_pos_emb`（cat + 3 个 elementwise） | 融合成 1 个 kernel 的绝对时间 vs torch 写的 ~4 个 |
| `swiglu` | `F.silu(gate) * up`（`aten::silu` + `aten::mul`） | 省掉一个 kernel 值多少 µs（本项只有 −1 kernel/层，更要看绝对时间） |

**为什么微基准值得做**：端到端只有 −2 或 −8 kernel/层 的收益，容易被整步噪声淹没；
微基准把"每个算子省了多少 µs"单独拿到，才能判断某项融合的真实性价比
（②-b 就是靠它发现"纯 norm 反而比库慢 70%"的）。

统计口径：**同进程多轮取中位数**（单轮 min 会被偶发低值拉偏，见 P7 踩坑记录 6）。

用法：
  python bench/ops_micro_bench.py                    # 全跑
  python bench/ops_micro_bench.py --only rope        # 只跑 RoPE
输出: `bench/results/<tag>.json`（默认 `ops_micro_bench`）
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as F

from nano_vllm.models.qwen2 import apply_rotary_pos_emb as rope_ref
from nano_vllm.ops.fused_norm import fused_add_rms_norm, rms_norm
from nano_vllm.ops.rope import apply_rope
from nano_vllm.ops.swiglu import swiglu

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _time_us(fn, iters: int, rounds: int) -> float:
    samples = []
    for _ in range(rounds):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e6 / iters)
    return statistics.median(samples)


def bench_norm(args, out: dict) -> None:
    print(f"\n[RMSNorm] H={args.H}（next_power_of_2 = {1 << (args.H - 1).bit_length()}）")
    print(f"  {'rows':>6} {'逐op':>10} {'F.rms_norm':>12}"
          + "".join(f"{'ours w=' + str(w):>13}" for w in args.num_warps))
    for rows in args.rows:
        x = torch.randn(rows, args.H, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(args.H, device="cuda", dtype=torch.bfloat16)
        eps = 1e-6

        def op():
            dt = x.dtype
            h = x.to(torch.float32)
            v = h.pow(2).mean(-1, keepdim=True)
            return w * (h * torch.rsqrt(v + eps)).to(dt)

        t_op = _time_us(op, args.iters, args.rounds)
        t_lib = _time_us(lambda: F.rms_norm(x, (args.H,), w, eps), args.iters, args.rounds)
        ours = {nw: _time_us(lambda nw=nw: rms_norm(x, w, eps, num_warps=nw), args.iters, args.rounds)
                for nw in args.num_warps}
        out["rms_norm"][str(rows)] = {"op_us": round(t_op, 2), "lib_us": round(t_lib, 2),
                                      "ours_us": {str(k): round(v, 2) for k, v in ours.items()}}
        print(f"  {rows:>6} {t_op:>10.2f} {t_lib:>12.2f}"
              + "".join(f"{ours[w]:>13.2f}" for w in args.num_warps)
              + f"   | 最快 {min(ours.values()):.2f} vs lib {t_lib:.2f}"
              + (" ✅" if min(ours.values()) < t_lib else ""))

    print(f"\n[融合残差加] 对照 = aten::add + F.rms_norm（lib 路径的真实开销）")
    print(f"  {'rows':>6} {'add+lib':>10}"
          + "".join(f"{'ours w=' + str(w):>13}" for w in args.num_warps))
    for rows in args.rows:
        x = torch.randn(rows, args.H, device="cuda", dtype=torch.bfloat16)
        r = torch.randn(rows, args.H, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(args.H, device="cuda", dtype=torch.bfloat16)
        eps = 1e-6

        def two_step():
            s = r + x
            return s, F.rms_norm(s, (args.H,), w, eps)

        t_two = _time_us(two_step, args.iters, args.rounds)
        ours = {nw: _time_us(lambda nw=nw: fused_add_rms_norm(x, r, w, eps, num_warps=nw),
                             args.iters, args.rounds) for nw in args.num_warps}
        out["fused_add"][str(rows)] = {"add_plus_lib_us": round(t_two, 2),
                                      "ours_us": {str(k): round(v, 2) for k, v in ours.items()}}
        print(f"  {rows:>6} {t_two:>10.2f}" + "".join(f"{ours[w]:>13.2f}" for w in args.num_warps)
              + f"   | 最快 {min(ours.values()):.2f} vs 两步 {t_two:.2f}")


def bench_rope(args, out: dict) -> None:
    """RoPE：自研融合核（1 个/qk）vs torch 参考（cat + 乘加，约 4 个/层）。"""
    H, HK, D = 12, 2, 128
    print(f"\n[RoPE] heads={H} kv_heads={HK} head_dim={D}")
    print(f"  {'(b,s)':>9} {'torch 参考':>12}"
          + "".join(f"{'ours w=' + str(w):>12}" for w in args.rope_num_warps))
    for b, s in args.rope_shapes:
        torch.manual_seed(0)
        q = torch.randn(b, H, s, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(b, HK, s, D, device="cuda", dtype=torch.bfloat16)
        cos = torch.randn(b, s, D, device="cuda", dtype=torch.bfloat16)
        sin = torch.randn(b, s, D, device="cuda", dtype=torch.bfloat16)
        t_ref = _time_us(lambda: rope_ref(q, k, cos, sin), args.iters, args.rounds)
        ours = {nw: _time_us(lambda nw=nw: apply_rope(q, k, cos, sin, num_warps=nw),
                             args.iters, args.rounds) for nw in args.rope_num_warps}
        out["rope"][f"{b}x{s}"] = {"torch_ref_us": round(t_ref, 2),
                                   "ours_us": {str(k2): round(v, 2) for k2, v in ours.items()}}
        best = min(ours.values())
        print(f"  {f'{b}x{s}':>9} {t_ref:>12.2f}"
              + "".join(f"{ours[w]:>12.2f}" for w in args.rope_num_warps)
              + f"   | 最快 {best:.2f} vs 参考 {t_ref:.2f}"
              + (f"（{t_ref / best:.2f}×）" if best < t_ref else ""))


def bench_swiglu(args, out: dict) -> None:
    """SwiGLU：自研融合核（1 个）vs `F.silu(gate) * up`（2 个 kernel）。

    用**真实布局**：gate/up 取自同一合并缓冲区（预拼接后就是这个样子）→ 非连续切片视图。
    """
    B, H = 8, args.inter
    print(f"\n[SwiGLU] intermediate={H}（gate/up 取合并缓冲区的两半，非连续视图）")
    print(f"  {'(b,s)':>9} {'torch 两步':>12}"
          + "".join(f"{'ours w=' + str(w):>12}" for w in args.swiglu_num_warps))
    for b, s in args.swiglu_shapes:
        torch.manual_seed(0)
        merged = torch.randn(b, s, 2 * H, device="cuda", dtype=torch.bfloat16)
        gate, up = merged.split(H, dim=-1)
        t_ref = _time_us(lambda: F.silu(gate) * up, args.iters, args.rounds)
        ours = {nw: _time_us(lambda nw=nw: swiglu(gate, up, num_warps=nw),
                             args.iters, args.rounds) for nw in args.swiglu_num_warps}
        out["swiglu"][f"{b}x{s}"] = {"torch_two_step_us": round(t_ref, 2),
                                     "ours_us": {str(k2): round(v, 2) for k2, v in ours.items()}}
        best = min(ours.values())
        print(f"  {f'{b}x{s}':>9} {t_ref:>12.2f}"
              + "".join(f"{ours[w]:>12.2f}" for w in args.swiglu_num_warps)
              + f"   | 最快 {best:.2f} vs 两步 {t_ref:.2f}"
              + (f"（{t_ref / best:.2f}×）" if best < t_ref else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all", choices=["all", "norm", "rope", "swiglu"])
    ap.add_argument("--H", type=int, default=1536, help="hidden 宽（1.5B 是 1536，非 2 的幂）")
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 4, 8, 16, 32, 512])
    ap.add_argument("--num-warps", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--rope-shapes", type=lambda s: tuple(int(x) for x in s.split("x")),
                    nargs="+", default=[(8, 1), (32, 1), (1, 200), (4, 64)],
                    help="RoPE 的 (batch, seq) 组合；decode 是 s=1")
    ap.add_argument("--rope-num-warps", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--inter", type=int, default=8960, help="MLP intermediate（1.5B 是 8960）")
    ap.add_argument("--swiglu-shapes", type=lambda s: tuple(int(x) for x in s.split("x")),
                    nargs="+", default=[(8, 1), (32, 1), (1, 200), (4, 64)],
                    help="SwiGLU 的 (batch, seq) 组合")
    ap.add_argument("--swiglu-num-warps", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--tag", default="ops_micro_bench")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA")

    out: dict = {"meta": {"H": args.H, "rows": args.rows, "num_warps": args.num_warps,
                          "rope_shapes": [list(x) for x in args.rope_shapes],
                          "iters": args.iters, "rounds": args.rounds,
                          "torch": torch.__version__,
                          "gpu": torch.cuda.get_device_name(0),
                          "time": time.strftime("%Y-%m-%d %H:%M:%S")},
                 "rms_norm": {}, "fused_add": {}, "rope": {}, "swiglu": {}}

    if args.only in ("all", "norm"):
        bench_norm(args, out)
    if args.only in ("all", "rope"):
        bench_rope(args, out)
    if args.only in ("all", "swiglu"):
        bench_swiglu(args, out)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{args.tag}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"\n已落盘: bench/results/{args.tag}.json")


if __name__ == "__main__":
    main()
