#!/usr/bin/env python3
"""回归门禁 · 正确性（硬门）+ 性能（软门，比值口径）。

## 背景

审计文档 `nano-vllm-P6前优化审计.md` / 评估报告 §4.1(3) 早就点名这个缺口：

> 10 个阶段每个都声称 TPOT 改善；无任何自动回归门禁。
> 某次重构把 P2 的 22.3ms 悄悄变回 30ms 也不会有人知道。

## 两档设计（以及为什么性能档不能做成硬门）

| 档 | 内容 | 判据 | 为什么 |
| --- | --- | --- | --- |
| **正确性** | `pytest tests/ -q` 子进程 | 退出码 ≠ 0 → **FAIL** | 确定性，与机器状态无关，可以硬门 |
| **性能** | 同进程 A/B：同一 shape 下 `eager` 与 `graph` 各测一遍 decode TPOT | **比值** `graph_speedup = eager_ms / graph_ms` 低于 baseline×(1−tol) → **FAIL**；绝对 ms 仅 WARN | 本机（WSL + 单卡）**跨 run 漂移可达 21%**——实测同一份 profiler 脚本两次运行，未被改动的 eager 路径 TPOT 从 45.28 → 35.59 ms（见 P7.md 踩坑记录 7）。绝对值做门会随机红。而 A/B **比值**受漂移影响小得多，因为两臂在同一次运行里一起漂 |

即：**绝对性能数字只记录不门禁；门禁看比值**。这是本机能给出的最可信的回归信号，
代价是「同比例地整体变快/变慢」检测不到（那需要固定频率与独占机器）。

## 用法

  python bench/regress.py                    # 跑两档并与 baseline 比对
  python bench/regress.py --update-baseline  # 重录 baseline（例如换机器/换 torch 后）
  python bench/regress.py --skip-perf        # 只跑正确性（CI 里推荐）
  python bench/regress.py --skip-correctness # 只跑性能

退出码：0 = 通过；1 = 有 FAIL 项。基线文件 `bench/results/baseline.json`。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import random
import subprocess
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "bench" / "results" / "baseline.json"


# ---------------------------------------------------------------- 正确性档

def run_correctness(args) -> dict:
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q"] + (["-x"] if args.fail_fast else []),
        cwd=ROOT, capture_output=True, text=True,
    )
    tail = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()][-1:]
    out = {
        "passed": proc.returncode == 0,
        "returncode": proc.returncode,
        "summary": tail[0] if tail else "",
        "seconds": round(time.perf_counter() - t0, 1),
    }
    print(f"[正确性] {'PASS' if out['passed'] else 'FAIL'}  {out['summary']}  ({out['seconds']}s)")
    if not out["passed"]:
        print(proc.stdout[-3000:])
    return out


# ---------------------------------------------------------------- 性能档

def _tpot_ms(args, cudagraph: bool, device: str, dtype) -> float:
    """同进程测一次 decode 的每步墙钟（ms）。"""
    from nano_vllm.engine.core import EngineCore
    from nano_vllm.engine.scheduler import Scheduler
    from nano_vllm.engine.sequence import SamplingParams
    from nano_vllm.model_executor.runner import NanoRunner

    runner = NanoRunner(
        args.model, device=device, dtype=dtype, block_size=16, num_blocks=4096,
        attn_impl="triton" if device == "cuda" else "torch",
        enable_cudagraph=cudagraph and device == "cuda",
    )
    sched = Scheduler(runner.paged_cache, max_num_batched_tokens=2048, enable_prefix_cache=False)
    engine = EngineCore(runner, sched)
    rng = random.Random(0)
    for _ in range(args.batch):
        engine.add_request(
            [rng.randrange(1000, 100000) for _ in range(args.prompt_len)],
            SamplingParams(temperature=0.0, max_new_tokens=args.steps + args.warmup + 4),
        )

    engine.step()                      # 吃掉 prefill
    for _ in range(args.warmup):       # 热身（图模式在此完成懒捕获）
        engine.step()
    torch.cuda.synchronize() if device == "cuda" else None

    t0 = time.perf_counter()
    for _ in range(args.steps):
        engine.step()
    if device == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1e3 / args.steps

    del engine, sched, runner
    import gc
    gc.collect(); gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return ms


def run_perf(args, device: str, dtype) -> dict:
    eager = _tpot_ms(args, False, device, dtype)
    graph = _tpot_ms(args, True, device, dtype) if device == "cuda" else None
    out = {
        "batch": args.batch, "prompt_len": args.prompt_len, "steps": args.steps,
        "eager_ms": round(eager, 3),
        "graph_ms": round(graph, 3) if graph else None,
        "graph_speedup": round(eager / graph, 3) if graph else None,
    }
    if graph:
        print(f"[性能]   eager {eager:.2f} ms/step   graph {graph:.2f} ms/step   "
              f"比值 {eager / graph:.2f}×")
    else:
        print(f"[性能]   eager {eager:.2f} ms/step（无 CUDA，跳过图档）")
    return out


# ---------------------------------------------------------------- 比对

def compare(base: dict, cur: dict, tol: float, tol_abs: float) -> list[dict]:
    rows: list[dict] = []

    b, c = base.get("correctness"), cur.get("correctness")
    if b and c:
        rows.append({"item": "correctness", "base": b["summary"], "now": c["summary"],
                     "level": "PASS" if c["passed"] else "FAIL"})

    bp, cp = base.get("perf"), cur.get("perf")
    if bp and cp:
        # ① 比值：门禁项
        br, cr = bp.get("graph_speedup"), cp.get("graph_speedup")
        if br and cr:
            floor = br * (1 - tol)
            rows.append({
                "item": "graph_speedup", "base": f"{br:.2f}×", "now": f"{cr:.2f}×",
                "level": "PASS" if cr >= floor else "FAIL",
                "note": f"门槛 {floor:.2f}×（基线 × (1-{tol:.0%})）",
            })
        # ② 绝对 ms：只 WARN
        for key, label in (("eager_ms", "eager_ms"), ("graph_ms", "graph_ms")):
            bv, cv = bp.get(key), cp.get(key)
            if bv and cv:
                drift = abs(cv - bv) / bv
                rows.append({
                    "item": label, "base": f"{bv:.2f} ms", "now": f"{cv:.2f} ms",
                    "level": "WARN" if drift > tol_abs else "PASS",
                    "note": f"漂移 {drift:.1%}（跨 run 漂移可达 21%，故只警告不门禁）",
                })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--steps", type=int, default=30, help="计时的 decode 步数")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--tolerance", type=float, default=0.30, help="比值门禁容差")
    ap.add_argument("--tolerance-abs", type=float, default=0.50, help="绝对 ms 的 WARN 阈值")
    ap.add_argument("--skip-correctness", action="store_true")
    ap.add_argument("--skip-perf", action="store_true")
    ap.add_argument("--fail-fast", action="store_true", help="pytest 加 -x")
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--baseline", default=str(BASELINE))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    cur: dict = {"meta": {}}
    if not args.skip_correctness:
        cur["correctness"] = run_correctness(args)
    if not args.skip_perf:
        cur["perf"] = run_perf(args, device, dtype)

    try:
        git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True).stdout.strip()
    except Exception:
        git = ""
    cur["meta"] = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git": git,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "tolerance": args.tolerance,
    }

    if args.update_baseline:
        # 合入已有基线，避免只跑一档时把另一档抹掉
        merged = {}
        bp = Path(args.baseline)
        if bp.exists():
            merged = json.loads(bp.read_text())
        merged.update(cur)
        merged.setdefault("perf", {})
        if "perf" in cur:
            merged["perf"].update(cur["perf"])
        bp.parent.mkdir(parents=True, exist_ok=True)
        bp.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n")
        print(f"\n基线已更新: {bp.relative_to(ROOT)}")
        return

    bp = Path(args.baseline)
    if not bp.exists():
        print(f"\n[错误] 基线不存在: {bp.relative_to(ROOT)}，先跑一次 --update-baseline")
        raise SystemExit(2)

    rows = compare(json.loads(bp.read_text()), cur, args.tolerance, args.tolerance_abs)
    print("\n回归比对:")
    print(f"  {'项':<16} {'基线':<16} {'本次':<16} {'结论':<6} 说明")
    for r in rows:
        print(f"  {r['item']:<16} {str(r['base']):<16} {str(r['now']):<16} {r['level']:<6} {r.get('note','')}")

    failed = [r for r in rows if r["level"] == "FAIL"]
    warn = [r for r in rows if r["level"] == "WARN"]
    print(f"\n结论: {'FAIL' if failed else 'PASS'}"
          f"（FAIL {len(failed)} / WARN {len(warn)} / 共 {len(rows)} 项）")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
