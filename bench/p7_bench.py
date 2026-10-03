#!/usr/bin/env python3
"""P7 · Decode CUDA Graph 基准（对照 P4 eager）。

三段实验（对应 roadmap §P7 的 Benchmark 清单）:
  1. `tpot_curve`: batch ∈ {1,2,4,8,16,32} 的 decode 步延迟（TPOT）曲线，
     图关（P4 eager）vs 图开（P7），并给加速比与命中率
  2. `bucket_tradeoff`: 桶数 3 档 (1,4,16) vs 6 档 (1,2,4,8,16,32) 的
     (峰值显存, 捕获耗时, TPOT, 命中率) tradeoff
  3. `hit_rate_mixed`: 混合负载（长 prompt 分 chunk + 短请求 decode）下的
     命中/回退率与 padding 浪费

口径与踩坑记录（否则数字不可信，详见 docs/stages/P7.md）:
  - TPOT = decode-only step 的墙钟时间中位数（批内每请求每步 1 token）
  - **固定 num_blocks**：`_profile_num_blocks` 按"当时剩余显存"反推会让不同配置
    拿到不同池容量（小池触发抢占 → TPOT 被重算污染）
  - **每种模式只建一个 runner 跨 batch 复用**：反复建 runner 会重复加载权重并
    叠加图池显存（16GB 卡上逼近上限 → WSL 换页 → 出现 80ms+ 的假慢步）
  - 每组先预热（含触发图捕获），再重复 `--repeat` 次取中位数

用法:
  python bench/p7_bench.py --model models/Qwen2.5-1.5B-Instruct
输出: bench/results/p7_bench_<tag>.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import gc
import json
import random
import statistics
import time

import torch

from nano_vllm.engine.core import EngineCore
from nano_vllm.engine.scheduler import Scheduler
from nano_vllm.engine.sequence import SamplingParams
from nano_vllm.model_executor.runner import NanoRunner

VOCAB = 151936
PROMPT_LEN = 128


def make_prompts(seed: int, n: int, length: int = PROMPT_LEN) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randrange(1000, VOCAB) for _ in range(length)] for _ in range(n)]


def make_runner(args, buckets, enable_graph: bool) -> NanoRunner:
    """固定 num_blocks（见文件头口径说明）。"""
    return NanoRunner(
        args.model, device="cuda", dtype=torch.bfloat16,
        block_size=16, attn_impl="triton", num_blocks=args.num_blocks,
        enable_cudagraph=enable_graph, cudagraph_buckets=buckets,
    )


def free_cuda() -> None:
    """GC（两轮收引用环）+ 同步 + 清缓存。

    注意: 调用方必须先 `del <runner>` 再调本函数——`del obj` 写在函数内只删局部名，
    调用方作用域里的引用还在，会静默保留整份权重+KV+图池（16GB 卡上直接叠加到
    OOM/换页）。踩坑记录见 docs/stages/P7.md。
    """
    gc.collect()
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def mem_state(tag: str) -> dict:
    free, total = torch.cuda.mem_get_info()
    return {
        "tag": tag,
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
        "free_gb": free / 1e9,
    }


def instrument(engine: EngineCore):
    """包 _execute 以标注每步的 prefill/decode 组成（计时在主循环）。

    返回 (state, restore)：**必须调用 restore()**——闭包 `wrapped` 持有 engine
    的 bound method，不还原会形成引用环，使 runner/EngineCore 无法及时释放，
    16GB 卡上会造成显存叠加与假慢步（踩坑记录见 docs/stages/P7.md）。
    """
    state = {"n_prefill": 0, "n_decode": 0}
    orig = engine._execute

    def wrapped(scheduler_output):
        state["n_prefill"] = scheduler_output.num_prefills
        state["n_decode"] = scheduler_output.num_decodes
        return orig(scheduler_output)

    engine._execute = wrapped

    def restore() -> None:
        engine._execute = orig

    return state, restore


def run_once(
    runner: NanoRunner,
    prompts: list[list[int]],
    max_new_tokens: int,
    use_graph: bool,
    budget: int = 2048,
) -> dict:
    """一次连续批生成，返回 decode-only step 延迟统计与图埋点（增量）。"""
    runner.paged_cache.reset()
    saved = runner.graph_runner
    runner.graph_runner = runner.graph_runner if use_graph else None
    graph = runner.graph_runner
    before = (graph.stats.replays, graph.stats.fallback_eager, graph.stats.padded_seqs) if graph else None
    try:
        scheduler = Scheduler(
            runner.paged_cache, max_num_batched_tokens=budget, enable_prefix_cache=True
        )
        engine = EngineCore(runner, scheduler)
        state, restore_instrument = instrument(engine)
        params = SamplingParams(temperature=0.0, max_new_tokens=max_new_tokens)

        torch.cuda.synchronize()
        t_wall0 = time.perf_counter()
        seqs = [engine.add_request(p, params) for p in prompts]
        decode_steps: list[float] = []
        prefill_steps: list[float] = []
        ttft_ms = None
        while engine.scheduler.has_requests():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            engine.step()
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1e3
            if state["n_decode"] > 0 and state["n_prefill"] == 0:
                decode_steps.append(dt)
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t_wall0) * 1e3
            else:
                prefill_steps.append(dt)
        wall_ms = (time.perf_counter() - t_wall0) * 1e3

        graph_stats = None
        if graph is not None and before is not None:
            s = graph.stats
            replays = s.replays - before[0]
            fallback = s.fallback_eager - before[1]
            graph_stats = {
                "replays": replays,
                "fallback_eager": fallback,
                "padded_seqs": s.padded_seqs - before[2],
                "hit_rate": replays / (replays + fallback) if replays + fallback else 0.0,
            }
        return {
            "wall_ms": wall_ms,
            "output_tokens": sum(len(s.output_token_ids) for s in seqs),
            "ttft_ms": ttft_ms,
            "tpot_p50_ms": statistics.median(decode_steps) if decode_steps else None,
            "tpot_p99_ms": (
                sorted(decode_steps)[min(len(decode_steps) - 1, int(0.99 * len(decode_steps)))]
                if decode_steps else None
            ),
            "n_decode_steps": len(decode_steps),
            "n_prefill_steps": len(prefill_steps),
            "graph": graph_stats,
        }
    finally:
        restore_instrument()
        runner.graph_runner = saved


def median_of(runs: list[dict], key: str):
    vals = sorted(r[key] for r in runs if r.get(key) is not None)
    return vals[len(vals) // 2] if vals else None


# ---------------- 段 1: TPOT 曲线 ----------------


def measure_tpot_curve(
    args, runner: NanoRunner, batches: list[int], use_graph: bool,
    static_eager: bool = False,
) -> dict:
    """同一 runner 跨 batch 复用（避免重复加载权重/叠加图池）。

    static_eager=True: 把图回放临时替换为「静态 buffer + eager 前向」
    （归因实验，其余流程不变）。
    """
    graph = runner.graph_runner
    saved_replay = None
    if static_eager:
        assert graph is not None
        saved_replay = graph.replay
        graph.replay = graph.replay_static_eager
    try:
        rows = {}
        for batch in batches:
            prompts = make_prompts(seed=100 + batch, n=batch)
            run_once(runner, prompts[:1], 4, use_graph=use_graph)  # 预热（图模式顺带触发捕获）
            torch.cuda.reset_peak_memory_stats()
            runs = [
                run_once(runner, prompts, args.max_new_tokens, use_graph=use_graph)
                for _ in range(args.repeat)
            ]
            hits = [r["graph"]["hit_rate"] for r in runs if r["graph"]]
            rows[batch] = {
                "tpot_p50_ms": median_of(runs, "tpot_p50_ms"),
                "tpot_p99_ms": median_of(runs, "tpot_p99_ms"),
                "wall_ms": median_of(runs, "wall_ms"),
                "ttft_ms": median_of(runs, "ttft_ms"),
                "tpot_runs_ms": [r["tpot_p50_ms"] for r in runs],
                "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
                "hit_rate": sum(hits) / len(hits) if hits else None,
                "padded_seqs": (runs[0]["graph"] or {}).get("padded_seqs"),
            }
        return rows
    finally:
        if saved_replay is not None:
            graph.replay = saved_replay


def tpot_curve(args, batches: list[int]) -> dict:
    buckets = tuple(args.buckets)
    # 阶段 A: eager（P4：list 寻址 + 逐层建块表张量）
    r_eager = make_runner(args, buckets, enable_graph=False)
    print(f"[mem] eager runner: {mem_state('eager')}")
    rows_eager = measure_tpot_curve(args, r_eager, batches, use_graph=False)
    del r_eager
    free_cuda()

    # 阶段 B: 图（P7）
    r_graph = make_runner(args, buckets, enable_graph=True)
    print(f"[mem] graph runner: {mem_state('graph')}")
    rows_graph = measure_tpot_curve(args, r_graph, batches, use_graph=True)
    capture_ms = r_graph.graph_runner.stats.capture_ms

    # 阶段 C: 静态 buffer + eager（归因实验：不含图回放）
    rows_static = measure_tpot_curve(
        args, r_graph, batches, use_graph=True, static_eager=True
    )
    del r_graph
    free_cuda()

    rows = []
    for batch in batches:
        e, g, s = rows_eager[batch], rows_graph[batch], rows_static[batch]
        row = {
            "batch": batch,
            "eager": {k: e[k] for k in ("tpot_p50_ms", "tpot_p99_ms", "wall_ms", "ttft_ms")},
            "graph": {k: g[k] for k in ("tpot_p50_ms", "tpot_p99_ms", "wall_ms", "ttft_ms")},
            "static_eager": {k: s[k] for k in ("tpot_p50_ms", "tpot_p99_ms")},
            "eager_tpot_runs_ms": e["tpot_runs_ms"],
            "graph_tpot_runs_ms": g["tpot_runs_ms"],
            "static_eager_tpot_runs_ms": s["tpot_runs_ms"],
            "eager_peak_gb": e["peak_gb"],
            "graph_peak_gb": g["peak_gb"],
            "graph_hit_rate": g["hit_rate"],
            "graph_padded_seqs": g["padded_seqs"],
            # 总加速：P4 eager → P7 图
            "speedup_tpot": (
                e["tpot_p50_ms"] / g["tpot_p50_ms"] if e["tpot_p50_ms"] and g["tpot_p50_ms"] else None
            ),
            # 归因: 静态 buffer 贡献（eager→static_eager）与图回放贡献（static_eager→graph）
            "speedup_from_static_buffers": (
                e["tpot_p50_ms"] / s["tpot_p50_ms"]
                if e["tpot_p50_ms"] and s["tpot_p50_ms"] else None
            ),
            "speedup_from_graph_capture": (
                s["tpot_p50_ms"] / g["tpot_p50_ms"]
                if s["tpot_p50_ms"] and g["tpot_p50_ms"] else None
            ),
            "speedup_wall": (
                e["wall_ms"] / g["wall_ms"] if g["wall_ms"] else None
            ),
        }
        rows.append(row)
        print(
            f"batch={batch:>2}  TPOT p4-eager={row['eager']['tpot_p50_ms']:7.2f}ms  "
            f"static-eager={row['static_eager']['tpot_p50_ms']:7.2f}ms  "
            f"graph={row['graph']['tpot_p50_ms']:6.2f}ms  "
            f"总加速={row['speedup_tpot']:5.2f}x（静态buffer {row['speedup_from_static_buffers']:.2f}x"
            f" × 图 {row['speedup_from_graph_capture']:.2f}x）  hit={row['graph_hit_rate']:.2f}"
        )
    return {"capture_ms": capture_ms, "batches": batches, "rows": rows}


# ---------------- 段 2: 桶数 tradeoff ----------------


def bucket_tradeoff(args, configs: list[tuple[int, ...]], batches: list[int]) -> dict:
    """同一负载下不同桶档的（峰值显存, 捕获耗时, TPOT, 命中率）。

    `batches` 里放一个"两档都能服务"的值（公平比 TPOT/显存）与一个只有 6 档能
    覆盖的值（体现桶档不足 → 该批全部回退 eager）。
    """
    rows = []
    for buckets in configs:
        runner = make_runner(args, buckets, enable_graph=True)
        run_once(runner, make_prompts(seed=1, n=1), 4, use_graph=True)  # 触发捕获
        capture_ms = runner.graph_runner.stats.capture_ms
        torch.cuda.reset_peak_memory_stats()
        entry = {"buckets": list(buckets), "capture_ms": capture_ms, "per_batch": {}}
        for batch in batches:
            prompts = make_prompts(seed=777, n=batch)
            runs = [
                run_once(runner, prompts, args.max_new_tokens, use_graph=True)
                for _ in range(args.repeat)
            ]
            hits = [r["graph"]["hit_rate"] for r in runs if r["graph"]]
            entry["per_batch"][batch] = {
                "tpot_p50_ms": median_of(runs, "tpot_p50_ms"),
                "hit_rate": sum(hits) / len(hits) if hits else None,
                "padded_seqs": (runs[0]["graph"] or {}).get("padded_seqs"),
            }
        entry["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
        rows.append(entry)
        summary = "  ".join(
            f"bs{b}: {v['tpot_p50_ms']:.2f}ms hit{v['hit_rate']:.2f}"
            for b, v in entry["per_batch"].items()
        )
        print(f"buckets={list(buckets)}  peak={entry['peak_gb']:.2f}GB  "
              f"capture={capture_ms:.0f}ms  {summary}")
        del runner
        free_cuda()
    return {"batches": batches, "rows": rows}


# ---------------- 段 3: 混合负载命中/回退 ----------------


def hit_rate_mixed(args, runner: NanoRunner, batch: int) -> dict:
    """长 prompt 分 chunk + 短请求 decode 混排 → 观察回退与 padding 浪费。"""
    rng = random.Random(9)
    n_long = max(1, batch // 4)
    prompts = [
        [rng.randrange(1000, VOCAB) for _ in range(2048)] for _ in range(n_long)
    ] + [
        [rng.randrange(1000, VOCAB) for _ in range(128)] for _ in range(batch)
    ]
    res = run_once(runner, prompts, args.max_new_tokens, use_graph=True, budget=args.budget)
    out = {
        "num_requests": len(prompts),
        "long_prompts": n_long,
        "short_prompts": batch,
        "budget": args.budget,
        "graph": res["graph"],
        "tpot_p50_ms": res["tpot_p50_ms"],
        "wall_ms": res["wall_ms"],
        "bucket_hist": dict(runner.graph_runner.stats.bucket_hist),
    }
    print(json.dumps({k: out[k] for k in ("graph", "bucket_hist")}, ensure_ascii=False))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--buckets", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--num-blocks", type=int, default=4096,
                    help="固定 KV 块池大小（避免按剩余显存反推导致跨配置不可比）")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--tag", default="p7_bench")
    ap.add_argument("--skip-tradeoff", action="store_true")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "P7 bench 需要 CUDA"
    result: dict = {
        "model": args.model,
        "dtype": "bf16",
        "device": "cuda",
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        "torch": torch.__version__,
        "attn_impl": "triton",
        "num_blocks": args.num_blocks,
        "prompt_len": PROMPT_LEN,
        "max_new_tokens": args.max_new_tokens,
        "repeat": args.repeat,
        "default_buckets": list(args.buckets),
        "mem_baseline": mem_state("start"),
    }

    print("§1 TPOT 曲线（decode-only step 中位数）")
    result["tpot_curve"] = tpot_curve(args, args.batches)

    if not args.skip_tradeoff:
        print("\n§2 桶数 tradeoff（bs=16 两档都能服务；bs=32 仅 6 档能覆盖）")
        result["bucket_tradeoff"] = bucket_tradeoff(
            args, configs=[(1, 4, 16), (1, 2, 4, 8, 16, 32)], batches=[16, 32]
        )

    print("\n§3 混合负载命中/回退（budget=%d）" % args.budget)
    r_mixed = make_runner(args, tuple(args.buckets), enable_graph=True)
    result["hit_rate_mixed"] = hit_rate_mixed(args, r_mixed, batch=8)
    del r_mixed
    free_cuda()

    out = Path(__file__).parent / "results" / f"{args.tag}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
