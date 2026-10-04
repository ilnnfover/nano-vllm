#!/usr/bin/env python3
"""P8 验收工具 · 单层 kernel launch 计数（融合 / 预拼接前后对照）。

## 为什么需要它

roadmap P8 完成标准写的是「**单层** kernel launch 数下降 ≥30%（profiler 前后对比）」，
而 `bench/p7_profiler.py` 量的是**整步**的 launch 占比与 API 时间——粒度不够：
融合把一个 decoder layer 从「多个小 kernel」变成「少数融合 kernel」，这个变化在整步
数据里被 28 层 + 采样 + 调度摊平。

## 怎么数（减法归属，不用 profiler 的父链）

`torch.profiler` 只给扁平的 GPU kernel 列表，不给「属于哪一层」。
`KinetoEvent.cpu_parent` 本来是干这个的，但**实测在 torch 2.14 下未被填充**
（链上只有事件自身），故改用减法：

    单层 kernel 数 = (整模型一步的 kernel 数 − 全部 28 层置空后一步的 kernel 数) / 28

置空 = 把 `DecoderLayer.forward` 换成 `lambda hidden, *a, **k: hidden`（等价 identity）。
差值恰好是「28 层贡献的 kernel」，因为层外部分（embedding / rotary / final norm /
lm_head / 采样 / 调度）不受影响。

**同构性交叉校验**：再取一次「只置空第 1 层」，则 `整模型 − 只置空一层` 应当 ≈ 单层均值。
两者偏差一并落盘，用来证明「28 层同构」这个前提成立（本脚本最容易被质疑的地方）。

## 用法

  python bench/layer_kernels.py                       # P7 生产路径（triton + 图）
  python bench/layer_kernels.py --no-cudagraph        # eager 对照
  python bench/layer_kernels.py --attn-impl sdpa --no-cudagraph
  python bench/layer_kernels.py --quick               # 跳过同构性校验（省 1/3 时间）
  python bench/layer_kernels.py --tag layer_kernels_p8   # 前后对照留档

输出: `bench/results/<tag>.json`

注：只测 **decode** 步（先跑完 prefill 再 profile 下一次 step），与 P7/P8 的优化目标一致。
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
from torch.profiler import ProfilerActivity, profile

from nano_vllm.engine.core import EngineCore
from nano_vllm.engine.scheduler import Scheduler
from nano_vllm.engine.sequence import SamplingParams
from nano_vllm.model_executor.runner import NanoRunner

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# 计数口径：只算计算 kernel，排除拷贝/置位与 profiler 自身的标注事件
_EXCLUDE_PREFIX = ("Memcpy", "Memset", "record_function")


def id_patch(*args, **kwargs):
    """DecoderLayer 的 identity 替身：跳过 attention/MLP 与层内 norm，只把残差链传下去。

    延迟残差版的 forward 返回 `(hidden_states, residual)`，这里必须同样返回二元组，
    否则 `Qwen2Model` 的循环会拿到错的解包结果。`residual` 为 None（第 0 层）时补成同一个
    张量，好让链路末端的 `final norm` 仍能正常执行 —— 本替身**只用于 kernel 计数**，
    它产生的数值没有语义（可能变成 `norm(2x)`），不要去解读。
    """
    x = args[0]
    r = kwargs.get("residual")
    return x, (x if r is None else r)


def patch_layers(model, indices) -> None:
    for i in indices:
        model.model.layers[i].forward = id_patch


def count_kernels(events) -> tuple[int, dict[str, int]]:
    cuda = torch.autograd.DeviceType.CUDA
    names: dict[str, int] = {}
    for e in events:
        if e.device_type != cuda:
            continue
        if e.name.startswith(_EXCLUDE_PREFIX):
            continue
        names[e.name] = names.get(e.name, 0) + 1
    return sum(names.values()), names


def measure(args, patch: str, device: str, dtype) -> tuple[int, dict[str, int], int]:
    """跑一个 engine，返回「一步 decode」的 (kernel 数, 名称分解, 层数)。

    patch ∈ {"none", "one", "all"}：在**图捕获之前**就要打好补丁，
    因此这里每次都用全新 engine（图一旦捕获就固化了 kernel 序列，事后补丁无效）。
    """
    runner = NanoRunner(
        args.model, device=device, dtype=dtype, block_size=16,
        num_blocks=args.num_blocks,
        attn_impl=args.attn_impl if device == "cuda" else "torch",
        prejoin=args.prejoin,
        norm_impl=args.norm_impl,
        enable_cudagraph=args.cudagraph if device == "cuda" else False,
    )
    sched = Scheduler(
        runner.paged_cache, max_num_batched_tokens=args.budget, enable_prefix_cache=False
    )
    engine = EngineCore(runner, sched)
    rng = random.Random(0)
    for _ in range(args.batch):
        engine.add_request(
            [rng.randrange(1000, 100000) for _ in range(args.prompt_len)],
            SamplingParams(temperature=0.0, max_new_tokens=args.max_new_tokens),
        )

    # 1) 先跑完 prefill（层不打补丁，KV 正常写入；prefill 走 eager，与图无关）
    engine.step()
    n_layers = runner.config.num_hidden_layers
    if patch == "all":
        patch_layers(runner.model, range(n_layers))
    elif patch == "one":
        patch_layers(runner.model, [args.layer_idx])

    # 2) 热身若干 decode 步：图模式在这里完成懒捕获，捕获耗时不计入统计
    for _ in range(args.warmup_steps):
        engine.step()

    # 3) profile 恰好一步 decode
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False, with_stack=False) as prof:
        engine.step()
    torch.cuda.synchronize()

    total, names = count_kernels(prof.events())
    del engine, sched, runner
    import gc
    gc.collect()
    gc.collect()
    torch.cuda.empty_cache()
    return total, names, n_layers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--batch", type=int, default=8, help="并发请求数（= decode batch）")
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--warmup-steps", type=int, default=3)
    ap.add_argument("--layer-idx", type=int, default=1, help="同构性校验用的那一层")
    ap.add_argument("--num-blocks", type=int, default=4096)
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--attn-impl", default="triton", choices=["torch", "sdpa", "triton"])
    ap.add_argument("--cudagraph", action=argparse.BooleanOptionalAction, default=True,
                    help="是否启用 P7 图（仅 attn_impl=triton 生效）")
    ap.add_argument("--prejoin", action=argparse.BooleanOptionalAction, default=True,
                    help="P8 权重预拼接（QKV 3→1 / gate-up 2→1）；--no-prejoin 即拼接前的对照")
    ap.add_argument("--norm-impl", default="triton", choices=["torch", "lib", "triton"],
                    help="RMSNorm 实现：torch（逐 op oracle）/ lib（F.rms_norm）/ triton（自研融合核）")
    ap.add_argument("--quick", action="store_true", help="跳过同构性交叉校验")
    ap.add_argument("--top", type=int, default=15, help="名称分解打印条数")
    ap.add_argument("--tag", default="layer_kernels")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    t0 = time.perf_counter()
    total, names, n_layers = measure(args, "none", device, dtype)
    base, _, _ = measure(args, "all", device, dtype)
    per_layer = (total - base) / n_layers

    one_layer = None
    if not args.quick:
        one_total, _, _ = measure(args, "one", device, dtype)
        one_layer = total - one_total
    elapsed = time.perf_counter() - t0

    print(f"\n配置: batch={args.batch} attn_impl={args.attn_impl} "
          f"prejoin={args.prejoin} norm_impl={args.norm_impl} "
          f"cudagraph={bool(args.cudagraph and args.attn_impl == 'triton')} "
          f"prompt_len={args.prompt_len}")
    print(f"整模型一步 kernel 数     : {total}")
    print(f"28 层全置空后 kernel 数   : {base}   （层外：embedding/rotary/final norm/lm_head/采样/调度）")
    print(f"→ 28 层合计              : {total - base}")
    print(f"→ **单层 kernel 数**      : {per_layer:.1f}")
    if one_layer is not None:
        dev = abs(one_layer - per_layer) / per_layer if per_layer else 0.0
        print(f"同构性校验（只置空 1 层）: {one_layer}  "
              f"→ 与均值偏差 {dev:.1%}  {'（假设成立）' if dev < 0.15 else '（⚠ 偏差偏大，单层均值解释力弱）'}")
    print(f"\n单层 kernel 名称分解（整模型口径，按出现次数降序前 {args.top}）:")
    for name, cnt in sorted(names.items(), key=lambda kv: -kv[1])[: args.top]:
        print(f"  {cnt:5d}  {name.split('<')[0][:74]}")
    print(f"\n耗时 {elapsed:.1f}s")

    out = {
        "meta": {
            "tag": args.tag,
            "model": args.model,
            "device": device,
            "dtype": str(dtype),
            "batch": args.batch,
            "prompt_len": args.prompt_len,
            "attn_impl": args.attn_impl,
            "prejoin": bool(args.prejoin),
            "norm_impl": args.norm_impl,
            "cudagraph": bool(args.cudagraph and args.attn_impl == "triton"),
            "warmup_steps": args.warmup_steps,
            "layer_idx_for_check": args.layer_idx,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "method": "subtraction: (all_layers - no_layers) / 28",
        },
        "step_kernels_total": total,
        "step_kernels_layers_removed": base,
        "layers_total_kernels": total - base,
        "layer_kernels": round(per_layer, 2),
        "uniformity_check_one_layer": one_layer,
        "uniformity_deviation": (
            round(abs(one_layer - per_layer) / per_layer, 4)
            if one_layer is not None and per_layer else None
        ),
        "kernel_breakdown_top": dict(sorted(names.items(), key=lambda kv: -kv[1])[:40]),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{args.tag}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"已落盘: bench/results/{args.tag}.json")


if __name__ == "__main__":
    main()
