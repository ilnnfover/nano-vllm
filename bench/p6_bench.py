#!/usr/bin/env python3
"""P6 · 前缀缓存基准：命中率 vs TTFT 曲线 + 多轮对话 trace 重放。

两种模式:
  hit-rate  扫描共享前缀长度 → 命中率与 TTFT 的关系（开/关前缀缓存对照）
  replay    多轮对话 trace 重放（每轮 prompt = 共享 system + 逐轮增长的历史）→ 吞吐/TTFT 对照

口径:
  TTFT      = 请求提交 → 首个 output token 产出（ms）
  hit_rate  = engine.stats.prefix_cache_hit_tokens / prefix_cache_query_tokens（累计）
  提交方式  = 按序提交，前一条 prefill 完成后再提交下一条 —— 保证后续请求真能命中前缀块
              （本引擎在 forward 后才注册块 hash，并发同批提交不会命中）

用法:
  python bench/p6_bench.py --model models/Qwen2.5-1.5B-Instruct
  python bench/p6_bench.py --mode hit-rate --prefix-lens 512,1024,2048,4096
  python bench/p6_bench.py --mode replay --turns 4 --num-conversations 4
  # CPU 冒烟
  python bench/p6_bench.py --model models/Qwen2.5-0.5B-Instruct --device cpu --dtype fp32 \
      --prefix-lens 64,128 --warm-requests 2 --max-new-tokens 4 --num-conversations 2 --turns 2

输出: bench/results/p6_bench_{tag}.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import time

import numpy as np
import torch

from nano_vllm.engine.core import EngineCore
from nano_vllm.engine.scheduler import Scheduler
from nano_vllm.engine.sequence import SamplingParams, SequenceStatus
from nano_vllm.model_executor.runner import NanoRunner

BLOCK_SIZE = 16
NAN = float("nan")


# ---------------- 基建 ----------------


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def make_runner(args, device: str, dtype: torch.dtype, max_seq_len: int) -> NanoRunner:
    return NanoRunner(
        args.model, device=device, dtype=dtype, block_size=BLOCK_SIZE,
        max_seq_len=max_seq_len, num_blocks=args.num_blocks,
        attn_impl=args.attn_impl,
    )


def make_engine(runner: NanoRunner, budget: int, enable_prefix_cache: bool) -> EngineCore:
    scheduler = Scheduler(
        paged_cache=runner.paged_cache,
        max_num_batched_tokens=budget,
        watermark_blocks=1,
        enable_prefix_cache=enable_prefix_cache,
    )
    return EngineCore(runner, scheduler)


def round_to_block(n: int) -> int:
    """对齐到 block_size 整数倍，保证满块能被注册/命中。"""
    return max(BLOCK_SIZE, (n // BLOCK_SIZE) * BLOCK_SIZE)


def drive_one(runner: NanoRunner, engine: EngineCore, prompt: list[int],
              params: SamplingParams, device: str) -> tuple[float, float]:
    """提交单条请求并跑到结束。

    Returns:
        (ttft_ms, wall_ms)
    """
    seq = engine.add_request(prompt, params)
    max_steps = (len(prompt) + params.max_new_tokens) * 8 + 64

    sync(device)
    t0 = time.perf_counter()
    ttft_ms = NAN
    steps = 0
    while seq.status != SequenceStatus.FINISHED:
        engine.step()
        steps += 1
        if np.isnan(ttft_ms) and seq.output_token_ids:
            sync(device)
            ttft_ms = (time.perf_counter() - t0) * 1e3
        if steps > max_steps:
            raise RuntimeError(f"请求未在 {max_steps} 步内完成（可能发生调度死锁）")
    sync(device)
    wall_ms = (time.perf_counter() - t0) * 1e3
    return ttft_ms, wall_ms


def warmup(runner: NanoRunner, device: str, budget: int) -> None:
    """预热 kernel（Triton JIT / CUDA context），结果丢弃。"""
    params = SamplingParams(temperature=0.0, max_new_tokens=2)
    for enable in (True, False):
        runner.paged_cache.reset()
        engine = make_engine(runner, budget, enable)
        drive_one(runner, engine, list(range(64, 96)), params, device)


def _pct(values: list[float], q: float) -> float:
    clean = [v for v in values if not np.isnan(v)]
    return float(np.percentile(clean, q)) if clean else NAN


def _stats_of(engine: EngineCore) -> dict:
    s = engine.stats
    return {
        "hit_rate": s.prefix_cache_hit_rate,
        "hit_tokens": s.prefix_cache_hit_tokens,
        "query_tokens": s.prefix_cache_query_tokens,
        "lookups": s.prefix_cache_lookups,
        "preempt_count": s.total_preempt_count,
    }


# ---------------- 模式 1：命中率 vs TTFT ----------------


def make_sweep_prompts(prefix_len: int, suffix_len: int, warm_requests: int) -> tuple[list[int], list[list[int]]]:
    """1 条冷请求（触发注册）+ warm_requests 条共享前缀的暖请求。"""
    prefix = list(range(100, 100 + prefix_len))
    prompts = [prefix + list(range(20000, 20000 + suffix_len))]
    for i in range(warm_requests):
        start = 20100 + i * (suffix_len + 10)
        prompts.append(prefix + list(range(start, start + suffix_len)))
    return prefix, prompts


def run_sweep_point(runner: NanoRunner, prompts: list[list[int]], params: SamplingParams,
                    device: str, budget: int, enable_prefix_cache: bool) -> dict:
    runner.paged_cache.reset()
    engine = make_engine(runner, budget, enable_prefix_cache)
    ttfts = [drive_one(runner, engine, p, params, device)[0] for p in prompts]
    return {
        "ttfts_ms": ttfts,
        "ttft_cold_ms": ttfts[0],
        "ttft_warm_p50_ms": _pct(ttfts[1:], 50),
        "ttft_warm_p99_ms": _pct(ttfts[1:], 99),
        **_stats_of(engine),
    }


def bench_hit_rate(args, runner: NanoRunner, device: str) -> dict:
    points = []
    for prefix_len in args.prefix_lens:
        _, prompts = make_sweep_prompts(prefix_len, args.suffix_len, args.warm_requests)
        params = SamplingParams(temperature=0.0, max_new_tokens=args.max_new_tokens)

        cache = run_sweep_point(runner, prompts, params, device, args.budget, True)
        nocache = run_sweep_point(runner, prompts, params, device, args.budget, False)

        base = nocache["ttft_warm_p50_ms"]
        speedup = base / cache["ttft_warm_p50_ms"] if cache["ttft_warm_p50_ms"] > 0 else NAN
        points.append({
            "prefix_len": prefix_len,
            "prefix_len_tokens": prefix_len,
            "warm_hit_ratio": prefix_len / (prefix_len + args.suffix_len),
            "cache": cache,
            "nocache": nocache,
            "ttft_speedup_warm": speedup,
        })
        print(
            f"[hit-rate] prefix={prefix_len:<6} hit_rate={cache['hit_rate']:.3f} "
            f"ttft_cold={cache['ttft_cold_ms']:.1f}ms ttft_warm={cache['ttft_warm_p50_ms']:.1f}ms "
            f"(nocache={base:.1f}ms, {speedup:.2f}x)"
        )
    return {
        "mode": "hit-rate",
        "suffix_len": args.suffix_len,
        "warm_requests": args.warm_requests,
        "max_new_tokens": args.max_new_tokens,
        "sweep": points,
    }


# ---------------- 模式 2：多轮对话 trace 重放 ----------------


def conversation_prompt(system: list[int], turn_tokens: list[list[int]], turn: int) -> list[int]:
    """第 turn 轮 prompt = system + 前 turn+1 轮的用户 token（前缀逐轮增长）。"""
    prompt = list(system)
    for t in range(turn + 1):
        prompt += turn_tokens[t]
    return prompt


def make_replay_trace(num_conversations: int, turns: int, system_len: int,
                      turn_len: int) -> tuple[list[int], list[list[list[int]]]]:
    """构造 trace：返回 (system, per_conversation_turn_tokens)。

    per_conversation_turn_tokens[c][t] 是会话 c 第 t 轮新增的用户 token，
    长度均为 turn_len（block 对齐），保证轮间前缀可命中。
    """
    system = list(range(100, 100 + system_len))
    trace = []
    for c in range(num_conversations):
        conv_turns = []
        for t in range(turns):
            start = 30000 + c * 5000 + t * 500
            conv_turns.append(list(range(start, start + turn_len)))
        trace.append(conv_turns)
    return system, trace


def run_replay(runner: NanoRunner, system: list[int], trace: list[list[list[int]]],
               params: SamplingParams, device: str, budget: int,
               enable_prefix_cache: bool) -> dict:
    turns = len(trace[0])

    runner.paged_cache.reset()
    engine = make_engine(runner, budget, enable_prefix_cache)

    ttft_by_turn: list[list[float]] = []
    out_tokens = 0
    sync(device)
    t0 = time.perf_counter()
    for turn in range(turns):
        turn_ttfts = []
        for c in range(len(trace)):
            prompt = conversation_prompt(system, trace[c], turn)
            ttft_ms, _ = drive_one(runner, engine, prompt, params, device)
            turn_ttfts.append(ttft_ms)
            out_tokens += params.max_new_tokens
        ttft_by_turn.append(turn_ttfts)
    sync(device)
    wall_ms = (time.perf_counter() - t0) * 1e3

    all_ttfts = [v for row in ttft_by_turn for v in row]
    return {
        "wall_ms": wall_ms,
        "output_tokens": out_tokens,
        "throughput_tok_s": out_tokens / (wall_ms / 1e3) if wall_ms > 0 else 0.0,
        "ttft_p50_ms": _pct(all_ttfts, 50),
        "ttft_p99_ms": _pct(all_ttfts, 99),
        "ttft_by_turn_ms": ttft_by_turn,
        **_stats_of(engine),
    }


def bench_replay(args, runner: NanoRunner, device: str) -> dict:
    system_len = round_to_block(args.system_len)
    turn_len = round_to_block(args.turn_len)
    system, trace = make_replay_trace(
        args.num_conversations, args.turns, system_len, turn_len,
    )
    params = SamplingParams(temperature=0.0, max_new_tokens=args.max_new_tokens)

    cache = run_replay(runner, system, trace, params, device, args.budget, True)
    nocache = run_replay(runner, system, trace, params, device, args.budget, False)

    speedup = nocache["wall_ms"] / cache["wall_ms"] if cache["wall_ms"] > 0 else NAN
    ttft_speedup = (
        nocache["ttft_p50_ms"] / cache["ttft_p50_ms"] if cache["ttft_p50_ms"] > 0 else NAN
    )
    print(
        f"[replay] hit_rate={cache['hit_rate']:.3f} wall={cache['wall_ms']:.0f}ms "
        f"({nocache['wall_ms']:.0f}ms nocache, {speedup:.2f}x) "
        f"ttft_p50={cache['ttft_p50_ms']:.1f}ms ({ttft_speedup:.2f}x)"
    )
    return {
        "mode": "replay",
        "num_conversations": args.num_conversations,
        "turns": args.turns,
        "system_len": system_len,
        "turn_len": turn_len,
        "max_new_tokens": args.max_new_tokens,
        "cache": cache,
        "nocache": nocache,
        "wall_speedup": speedup,
        "ttft_p50_speedup": ttft_speedup,
    }


# ---------------- 绘图 ----------------


def plot_results(results: dict, out_path: Path) -> bool:
    """左图: 命中率 vs TTFT 曲线；右图: 多轮重放 turn-wise TTFT。matplotlib 缺失时跳过。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    ax = axes[0]
    if "hit_rate_mode" in results:
        sweep = results["hit_rate_mode"]["sweep"]
        x = [p["prefix_len"] for p in sweep]
        ax.plot(x, [p["cache"]["ttft_warm_p50_ms"] for p in sweep], "o-", label="TTFT warm (cache on)")
        ax.plot(x, [p["nocache"]["ttft_warm_p50_ms"] for p in sweep], "s--", label="TTFT warm (cache off)")
        ax.plot(x, [p["cache"]["ttft_cold_ms"] for p in sweep], "^:", label="TTFT cold (cache on)")
        ax.set_xlabel("shared prefix length (tokens)")
        ax.set_ylabel("TTFT p50 (ms)")
        ax.set_xscale("log", base=2)
        ax.grid(alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(x, [p["warm_hit_ratio"] for p in sweep], "d-", color="tab:green", label="warm hit ratio")
        ax2.plot(x, [p["cache"]["hit_rate"] for p in sweep], "v-.", color="tab:olive", label="engine hit rate")
        ax2.set_ylabel("hit rate")
        ax2.set_ylim(0, 1.05)

        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(lines, [ln.get_label() for ln in lines], fontsize=8, loc="upper left")
    ax.set_title(f"P6 hit rate vs TTFT ({results.get('model', '')})")

    ax = axes[1]
    if "replay_mode" in results:
        rep = results["replay_mode"]
        for label, key, style in (("cache on", "cache", "o-"), ("cache off", "nocache", "s--")):
            means = [float(np.mean(row)) for row in rep[key]["ttft_by_turn_ms"]]
            ax.plot(range(1, len(means) + 1), means, style, label=f"TTFT {label}")
        ax.set_xlabel("conversation turn")
        ax.set_ylabel("TTFT mean (ms)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_title(
            f"P6 multi-turn replay: hit_rate={rep['cache']['hit_rate']:.3f}, "
            f"ttft_p50_speedup={rep['ttft_p50_speedup']:.2f}x"
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


# ---------------- main ----------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--attn-impl", default="triton", choices=["torch", "triton"])
    p.add_argument("--num-blocks", type=int, default=None, help="缺省按显存反推")
    p.add_argument("--budget", type=int, default=2048, help="max_num_batched_tokens")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--mode", default="both", choices=["hit-rate", "replay", "both"])
    # hit-rate 模式
    p.add_argument("--prefix-lens", default="512,1024,2048,4096",
                   help="共享前缀长度列表（逗号分隔，自动对齐 block_size）")
    p.add_argument("--suffix-len", type=int, default=64)
    p.add_argument("--warm-requests", type=int, default=3)
    # replay 模式
    p.add_argument("--num-conversations", type=int, default=3)
    p.add_argument("--turns", type=int, default=4)
    p.add_argument("--system-len", type=int, default=512)
    p.add_argument("--turn-len", type=int, default=128)
    # 其它
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--tag", default=None)
    p.add_argument("--results-dir", default="bench/results")
    args = p.parse_args()

    args.prefix_lens = [round_to_block(int(x)) for x in args.prefix_lens.split(",") if x.strip()]

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    # max_seq_len 覆盖两种模式所需的最长序列
    need = args.max_new_tokens + 8
    if args.mode in ("hit-rate", "both"):
        need = max(need, max(args.prefix_lens) + args.suffix_len + 64 + args.max_new_tokens)
    if args.mode in ("replay", "both"):
        need = max(need, round_to_block(args.system_len) + round_to_block(args.turns * args.turn_len)
                   + args.max_new_tokens + 32)

    runner = make_runner(args, device, dtype, need)
    print(f"[p6_bench] device={device} dtype={args.dtype} model={args.model} "
          f"num_blocks={runner.paged_cache.num_blocks} max_seq_len={need}")

    if args.warmup > 0:
        for _ in range(args.warmup):
            warmup(runner, device, args.budget)

    results: dict = {
        "model": args.model,
        "dtype": args.dtype,
        "device": device,
        "block_size": BLOCK_SIZE,
        "num_blocks": runner.paged_cache.num_blocks,
        "max_seq_len": need,
        "budget": args.budget,
        "attn_impl": args.attn_impl,
        "warmup": args.warmup,
    }
    if args.mode in ("hit-rate", "both"):
        results["hit_rate_mode"] = bench_hit_rate(args, runner, device)
    if args.mode in ("replay", "both"):
        results["replay_mode"] = bench_replay(args, runner, device)

    tag = args.tag or args.mode
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"p6_bench_{tag}.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n结果已保存到 {out_path}")

    png_path = out_path.with_suffix(".png")
    if plot_results(results, png_path):
        print(f"图已保存到 {png_path}")


if __name__ == "__main__":
    main()
