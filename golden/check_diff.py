#!/usr/bin/env python3
"""P0 · 对拍工具：比较两份 golden 目录，逐层报告误差。

用法:
  python golden/check_diff.py golden/A golden/B [--tol 1e-2]

P0 自检: 同一目录跑两遍 dump，diff 必须为 0。
退出码: 0=全部通过(≤tol 或完全一致), 1=超差。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HEADER = f"{'tensor':<18}{'shape':<22}{'max abs diff':>14}{'mean abs diff':>15}{'max rel diff':>14}"


def compare_npy(name: str, a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    denom = np.maximum(np.abs(b.astype(np.float64)), 1e-6)
    return float(diff.max()), float(diff.mean()), float((diff / denom).max())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dir_a", type=Path)
    p.add_argument("dir_b", type=Path)
    p.add_argument("--tol", type=float, default=1e-2, help="hidden/logits 的 max abs diff 阈值")
    args = p.parse_args()

    meta_a = json.loads((args.dir_a / "meta.json").read_text())
    meta_b = json.loads((args.dir_b / "meta.json").read_text())
    print(f"A = {args.dir_a}  (dtype={meta_a['dtype']}, device={meta_a['device']}, seed={meta_a['seed']})")
    print(f"B = {args.dir_b}  (dtype={meta_b['dtype']}, device={meta_b['device']}, seed={meta_b['seed']})")
    if meta_a["input_ids"] != meta_b["input_ids"]:
        print("[FATAL] 两份 golden 的输入 token ids 不一致，对比无意义")
        return 1

    files = sorted(f.name for f in args.dir_a.glob("*.npy"))
    print("\n" + HEADER)
    print("-" * len(HEADER))
    worst, worst_name, all_pass = 0.0, "", True
    for fname in files:
        fa, fb = args.dir_a / fname, args.dir_b / fname
        if not fb.exists():
            print(f"{fname:<18} ✗ B 侧缺失")
            all_pass = False
            continue
        a, b = np.load(fa), np.load(fb)
        if fname == "greedy_ids.npy":
            same = a.shape == b.shape and bool((a == b).all())
            status = "OK" if same else "MISMATCH"
            if not same:
                first_bad = next(
                    (i for i, (x, y) in enumerate(zip(a.flat, b.flat)) if x != y), -1
                )
                print(f"{fname:<18}{str(a.shape):<22}{'greedy 输出不一致':>14}  首个差异 idx={first_bad}")
                all_pass = False
            else:
                print(f"{fname:<18}{str(a.shape):<22}{'0 (逐 token 相同)':>14}")
            continue
        if a.shape != b.shape:
            print(f"{fname:<18} ✗ shape 不一致 {a.shape} vs {b.shape}")
            all_pass = False
            continue
        mx, mean, rel = compare_npy(fname, a, b)
        limit = args.tol if fname.startswith("hidden") else args.tol
        ok = mx <= limit
        all_pass &= ok
        if mx > worst:
            worst, worst_name = mx, fname
        print(f"{fname:<18}{str(a.shape):<22}{mx:>14.4e}{mean:>15.4e}{rel:>14.4e}  {'OK' if ok else 'FAIL'}")

    print("-" * len(HEADER))
    print(f"最差张量: {worst_name}  max abs diff = {worst:.4e}  (tol={args.tol})")
    print("结果: " + ("PASS ✓" if all_pass else "FAIL ✗"))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())