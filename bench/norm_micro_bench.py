#!/usr/bin/env python3
"""P8 ②-b · RMSNorm 单算子微基准（roadmap:150「单算子微基准（融合 vs 不融合，µs 级）」）。

回答三个问题：

1. **自研核 vs 库算子的绝对性能**：`_fused_add_rms_norm_kernel`（1 次 launch 出两个结果）
   vs `aten::add` + `F.rms_norm`（2 次 launch）。注意端到端只有 −2 kernel/层 的收益，
   所以这里更关心**每层省下的绝对时间**，而不是倍数。
2. **`num_warps` 怎么选**：本核一行 = 一个 program、行内归约，所以 `num_warps` 直接决定
   一行的归约有多少线程参与。行数少（decode batch=8）时并行度不足是已知取舍。
3. **H 不是 2 的幂的代价**：1536 → BLOCK_H=2048，25% 的 lane 被 mask 掉仍要占寄存器/带宽。
   与 2048（幂）对照可以看出这个代价。

用法：
  python bench/norm_micro_bench.py                 # 默认扫 num_warps 与 shape
  python bench/norm_micro_bench.py --tag xxx       # 换落盘名

输出: `bench/results/<tag>.json`（默认 `norm_micro_bench`）
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

from nano_vllm.ops.fused_norm import fused_add_rms_norm, rms_norm

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _time_us(fn, iters: int, rounds: int) -> float:
    """同进程多轮取**中位数**（单轮 min 会被偶发低值拉偏，见 P7 踩坑记录 6）。"""
    samples = []
    for _ in range(rounds):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e6 / iters)
    return statistics.median(samples)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=1536, help="hidden 宽（1.5B 是 1536，非 2 的幂）")
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 4, 8, 16, 32, 512],
                    help="行数 = batch×seq（decode 时 = batch；prefill 时可以很大）")
    ap.add_argument("--num-warps", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--tag", default="norm_micro_bench")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA")

    out: dict = {"meta": {"H": args.H, "rows": args.rows, "num_warps": args.num_warps,
                          "iters": args.iters, "rounds": args.rounds,
                          "torch": torch.__version__,
                          "gpu": torch.cuda.get_device_name(0),
                          "time": time.strftime("%Y-%m-%d %H:%M:%S")},
                 "rms_norm": {}, "fused_add": {}}

    # ---- 纯 RMSNorm：自研核 vs 逐 op vs F.rms_norm ----
    print(f"\n[纯 RMSNorm] H={args.H}（next_power_of_2 = {1 << (args.H - 1).bit_length()}）")
    print(f"  {'rows':>6} {'逐op':>10} {'F.rms_norm':>12}" +
          "".join(f"{'ours w=' + str(w):>13}" for w in args.num_warps))
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
        ours = {}
        for nw in args.num_warps:
            ours[nw] = _time_us(lambda nw=nw: rms_norm(x, w, eps, num_warps=nw),
                                args.iters, args.rounds)
        row = {"op_us": round(t_op, 2), "lib_us": round(t_lib, 2),
               "ours_us": {str(k): round(v, 2) for k, v in ours.items()}}
        out["rms_norm"][str(rows)] = row
        best = min(ours.values())
        print(f"  {rows:>6} {t_op:>10.2f} {t_lib:>12.2f}" +
              "".join(f"{ours[w]:>13.2f}" for w in args.num_warps) +
              f"   | 最快 {best:.2f} vs lib {t_lib:.2f}" + (" ✅" if best < t_lib else ""))

    # ---- 融合残差加：自研核 vs (add + F.rms_norm) ----
    print(f"\n[融合残差加] H={args.H}（对照 = aten::add + F.rms_norm，即 lib 路径的真实开销）")
    print(f"  {'rows':>6} {'add+lib':>10}" + "".join(f"{'ours w=' + str(w):>13}" for w in args.num_warps))
    for rows in args.rows:
        x = torch.randn(rows, args.H, device="cuda", dtype=torch.bfloat16)
        r = torch.randn(rows, args.H, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(args.H, device="cuda", dtype=torch.bfloat16)
        eps = 1e-6

        def two_step():
            s = r + x
            return s, F.rms_norm(s, (args.H,), w, eps)

        t_two = _time_us(two_step, args.iters, args.rounds)
        ours = {}
        for nw in args.num_warps:
            ours[nw] = _time_us(
                lambda nw=nw: fused_add_rms_norm(x, r, w, eps, num_warps=nw),
                args.iters, args.rounds,
            )
        out["fused_add"][str(rows)] = {"add_plus_lib_us": round(t_two, 2),
                                      "ours_us": {str(k): round(v, 2) for k, v in ours.items()}}
        print(f"  {rows:>6} {t_two:>10.2f}" + "".join(f"{ours[w]:>13.2f}" for w in args.num_warps) +
              f"   | 最快 {min(ours.values()):.2f} vs 两步 {t_two:.2f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{args.tag}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"\n已落盘: bench/results/{args.tag}.json")


if __name__ == "__main__":
    main()
