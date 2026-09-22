#!/usr/bin/env python3
"""
vLLM 冒烟测试 + 连续批处理吞吐观察。

运行:
    source ~/venvs/vllm/bin/activate
    python scripts/test_vllm.py

做三件事:
  1. 确认 vLLM 真的跑在 GPU 上(不是静默退回 CPU)
  2. 单条推理, 看输出对不对
  3. 把并发数从 1 拉到 32, 看吞吐怎么变 —— 这就是连续批处理的价值所在

⚠ 必须有 if __name__ == "__main__" 保护。
  vLLM V1 会把 EngineCore 放进独立进程; 在 WSL 下 NVML 与 fork 不兼容,
  vLLM 会强制改用 spawn。spawn 的子进程会重新 import 本模块 ——
  没有 main guard 的话, 子进程会再执行一次 LLM(), 直接报
  "An attempt has been made to start a new process before the current
   process has finished its bootstrapping phase"。
  (用 vllm serve 命令行不会遇到, 因为 CLI 自带 main guard;
   只有自己写脚本调 Python API 才需要这个保护。)
"""
from __future__ import annotations

import os
import time

MODEL = os.environ.get("TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "128"))

# ---------------------------------------------------------------------------
# WSL2 专属开关, 必须在 import vllm 之前设好
# ---------------------------------------------------------------------------
# vLLM 0.29 的 V2 model runner 依赖 UVA(统一虚拟寻址)缓冲区, 而 UVA 又要求
# pinned host memory。在 WSL2 上 vLLM 把 pin memory 默认关掉了
# (VLLM_WSL2_ENABLE_PIN_MEMORY 默认 0), 于是:
#       is_pin_memory_available() == False
#   ->  is_uva_available()        == False
#   ->  UvaBuffer() 直接 raise RuntimeError("UVA is not available")
# 而 V2 runner 仍是默认选中项, 所以 EngineCore 一起来就崩。
# 上游修复(PR #54655 / #47579)至今未合并, 0.29.0 里没有这个 fallback。
#
# 对策 A(默认): 退回 V1 model runner —— 稳定, 也是绝大多数教程/博客对应的一条路径。
# 对策 B:        保留 V2, 但手动打开 WSL2 的 pin memory。
#                想试 B 就在命令行里: VLLM_WSL2_ENABLE_PIN_MEMORY=1 python test_vllm.py
#                A、B 二选一即可。
def _in_wsl() -> bool:
    try:
        with open("/proc/version", encoding="utf-8") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


if _in_wsl() and not os.environ.get("VLLM_WSL2_ENABLE_PIN_MEMORY"):
    # 在 WSL 且用户没显式要求 pin memory -> 退回 V1 runner, 保证能跑起来。
    # setdefault: 你自己 export 过 VLLM_USE_V2_MODEL_RUNNER 就以你的为准。
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
# 不在 WSL(或用户主动开了 pin memory) -> 什么都不做, 交给 vLLM 自己决定


# ---------------------------------------------------------------------------
# FlashInfer 采样器: 没有 nvcc 就必须关掉
# ---------------------------------------------------------------------------
# vLLM 默认用 FlashInfer 做 top-k/top-p 采样。FlashInfer 是 JIT 编译型库 ——
# 首次调用时才现场编译 CUDA C++ 源码, 因此需要 nvcc + CUDA toolkit。
# 没装 toolkit 的环境会在 profile_run 阶段炸在这条链上:
#     topk_topp_sampler.forward_cuda -> flashinfer_sample
#  -> flashinfer.sampling.top_k_top_p_sampling_from_logits
#  -> jit/core.py build_and_load -> cpp_ext.get_cuda_path()
#  -> RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
# 设 0 之后走 forward_native(PyTorch 原生实现), 无需编译。
# 注意这只是【采样】环节, 不影响 attention 后端; 而且 greedy 解码(temperature=0)
# 根本不经过 top-k/top-p, 所以对本次测试的性能没有实质影响。
def _has_cuda_toolkit() -> bool:
    from shutil import which

    if which("nvcc"):
        return True
    for p in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"), "/usr/local/cuda"):
        if p and os.path.exists(os.path.join(p, "bin", "nvcc")):
            return True
    return False


if not _has_cuda_toolkit():
    # 没 nvcc -> 关掉 FlashInfer 采样器。等你装了 CUDA Toolkit 会自动重新启用。
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


def main() -> None:
    import torch
    from vllm import LLM, SamplingParams

    print(f"torch {torch.__version__} | cuda {torch.version.cuda} | available {torch.cuda.is_available()}")
    print(f"VLLM_USE_V2_MODEL_RUNNER={os.environ.get('VLLM_USE_V2_MODEL_RUNNER', '<unset>')} "
          f"| VLLM_WSL2_ENABLE_PIN_MEMORY={os.environ.get('VLLM_WSL2_ENABLE_PIN_MEMORY', '<unset>')}")
    print(f"VLLM_USE_FLASHINFER_SAMPLER={os.environ.get('VLLM_USE_FLASHINFER_SAMPLER', '<unset>')} "
          f"| nvcc 可用={_has_cuda_toolkit()}")
    assert torch.cuda.is_available(), "torch 没拿到 GPU, 别往下跑了"

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=2048,
        # 16GB 卡 + Windows 桌面侧已占 ~2.7GB, 默认 0.9 会 OOM
        gpu_memory_utilization=0.65,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)

    print("\n" + "=" * 70)
    print(" 1. 单条推理")
    print("=" * 70)
    p = "用一句话解释什么是 KV Cache。"
    t0 = time.perf_counter()
    out = llm.generate([p], sp)[0]
    dt = time.perf_counter() - t0
    n = len(out.outputs[0].token_ids)
    print(out.outputs[0].text.strip())
    print(f"\n-- {n} tokens / {dt:.2f}s = {n / dt:.1f} tok/s")

    print("\n" + "=" * 70)
    print(" 2. 并发吞吐 — 连续批处理的效果")
    print("=" * 70)
    base = "请简要说明分页显存(PagedAttention)解决了什么问题。"
    print(f"{'并发':>4}  {'总耗时':>9}  {'输出tokens':>10}  {'吞吐 tok/s':>12}")
    for n_conc in (1, 4, 16, 32):
        prompts = [f"{base} (第{i + 1}问)" for i in range(n_conc)]
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp)
        dt = time.perf_counter() - t0
        total = sum(len(o.outputs[0].token_ids) for o in outs)
        print(f"{n_conc:>4}  {dt:>8.2f}s  {total:>10}  {total / dt:>12.1f}")

    print("\n单条时吞吐低是因为 GPU 没喂饱; 并发上去后吞吐显著上升 —— 这正是连续批处理")
    print("要解决的问题。你自己的引擎最终也要拿这张表来对照。")


if __name__ == "__main__":
    main()
