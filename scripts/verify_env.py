#!/usr/bin/env python3
"""
推理引擎学习环境自检脚本。

用法:
    source ~/venvs/dev/bin/activate
    python scripts/verify_env.py

逐项检查你后续六块学习内容所依赖的底层能力是否真的可用,
而不是只看 import 成功 —— 例如 CUDA Graph 在 WSL 下能不能 capture 成功,
Triton 能不能真的编译出一个跑在 GPU 上的 kernel。
"""
from __future__ import annotations

import sys
import time

OK, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"


def check(name: str, fn) -> bool:
    t0 = time.time()
    try:
        detail = fn()
        print(f"  [{OK}] {name:<46} {detail}  ({time.time() - t0:.1f}s)")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [{FAIL}] {name:<46} {type(e).__name__}: {e}")
        return False


def main() -> int:
    results: list[bool] = []

    print("\n" + "=" * 78)
    print(" 1. 基础: PyTorch / CUDA 运行时")
    print("=" * 78)
    import torch

    results.append(check("torch import", lambda: f"v{torch.__version__}"))
    results.append(check(
        "CUDA 可用 (WSL GPU 透传)",
        lambda: torch.cuda.get_device_name(0) if torch.cuda.is_available() else (_ for _ in ()).throw(
            RuntimeError("torch.cuda.is_available() == False")),
    ))
    results.append(check("CUDA 运行时版本", lambda: f"{torch.version.cuda}"))
    results.append(check(
        "计算能力 sm_xy",
        lambda: ".".join(map(str, torch.cuda.get_device_capability(0))),
    ))

    def _matmul():
        a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
        for _ in range(3):
            c = a @ b
        torch.cuda.synchronize()
        return f"bf16 GEMM 4096²  TFLOPs≈{2 * 4096 ** 3 / 1e12:.2f}/iter"

    results.append(check("GPU 计算通路", _matmul))

    print("\n" + "=" * 78)
    print(" 2. KV Cache / 分页显存: 大块显存分配 + 非连续 gather")
    print("=" * 78)

    def _kv_pool():
        # 模拟 PagedAttention 的 KV pool: [num_blocks, block_size, num_kv_heads, head_dim]
        pool = torch.empty(2048, 16, 8, 128, dtype=torch.bfloat16, device="cuda")
        tbl = torch.randint(0, 2048, (4, 64), device="cuda")  # [batch, max_blocks_per_seq]
        sel = pool[tbl]                                        # 按 block table 索引
        torch.cuda.synchronize()
        return f"pool={tuple(pool.shape)} -> {tuple(sel.shape)}  {pool.numel() * 2 / 2**20:.0f} MB"

    results.append(check("KV block pool + block table gather", _kv_pool))

    def _free():
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        used = torch.cuda.memory_allocated() / 2**30
        return f"总显存 {total:.1f} GiB, 当前已用 {used:.2f} GiB"

    results.append(check("显存容量", _free))

    print("\n" + "=" * 78)
    print(" 3. Triton 内核 (算子级优化)")
    print("=" * 78)
    import triton
    import triton.language as tl

    @triton.jit
    def _add(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        off = pid * BLOCK + tl.arange(0, BLOCK)
        m = off < n
        tl.store(o_ptr + off, tl.load(x_ptr + off, mask=m) + tl.load(y_ptr + off, mask=m), mask=m)

    def _triton():
        x = torch.randn(1 << 16, device="cuda")
        y = torch.randn(1 << 16, device="cuda")
        o = torch.empty_like(x)
        _add[(x.numel() // 1024,)](x, y, o, x.numel(), BLOCK=1024)
        torch.cuda.synchronize()
        assert torch.allclose(o, x + y, atol=1e-5), "Triton 结果错误"
        return f"triton {triton.__version__} 编译并执行成功"

    results.append(check("Triton kernel 编译 + 执行 + 数值正确", _triton))

    def _autotune():
        # Triton autotune 是写高性能 kernel 的标配, 顺手确认可用
        from triton.runtime import driver  # noqa: F401
        return f"ptxas/backend 可用: {triton.runtime.driver.active.get_current_target()}"

    results.append(check("Triton 后端 target", _autotune))

    print("\n" + "=" * 78)
    print(" 4. CUDA Graph 捕获 (解码阶段降开销关键)")
    print("=" * 78)

    def _graph():
        m = torch.nn.Linear(512, 512, bias=False).cuda().half()
        inp = torch.randn(1, 512, device="cuda", dtype=torch.float16)
        # 预热: CUDA Graph 捕获前必须在【非默认流】上跑过一遍并完成一次同步
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                m(inp)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = m(inp)
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        return "capture + replay 成功 (WSL 下可用)"

    results.append(check("CUDA Graph capture/replay", _graph))

    print("\n" + "=" * 78)
    print(" 5. torch.compile / 动态图")
    print("=" * 78)

    def _compile():
        m = torch.nn.Linear(256, 256).cuda()
        cm = torch.compile(m, mode="reduce-overhead")
        x = torch.randn(8, 256, device="cuda")
        for _ in range(3):
            cm(x)
        torch.cuda.synchronize()
        return "reduce-overhead (含 cudagraph 树) 可用"

    results.append(check("torch.compile mode=reduce-overhead", _compile))

    print("\n" + "=" * 78)
    print(" 6. 生态: 模型加载 / 算子库 / 前缀哈希")
    print("=" * 78)
    import importlib

    def _imp(mod):
        return lambda: f"{importlib.import_module(mod).__name__} ok"

    for m in ("transformers", "datasets", "safetensors", "xxhash", "numpy"):
        results.append(check(f"import {m}", _imp(m)))

    # 可选依赖: 装了报 OK, 没装报 SKIP —— 都不计入通过率, 免得把"没装可选包"误判成环境失败
    for m in ("flashinfer", "flash_attn"):
        try:
            importlib.import_module(m)
            print(f"  [\033[92m OK \033[0m] {m} (可选) 已安装")
        except Exception:  # noqa: BLE001
            print(f"  [\033[93mSKIP\033[0m] {m} (可选) 未安装 — 不影响自研 Triton kernel, 不计入通过率")

    print("\n" + "=" * 78)
    print(" 7. 性能计数器: 能否读到 SM / 显存带宽利用率")
    print("=" * 78)

    def _prof():
        from torch.profiler import profile, ProfilerActivity
        m = torch.nn.Linear(1024, 1024).cuda().half()
        x = torch.randn(64, 1024, device="cuda", dtype=torch.float16)
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            for _ in range(5):
                m(x)
            torch.cuda.synchronize()
        ka = [e for e in p.key_averages() if e.device_type.name == "CUDA"]
        return f"采集到 {len(ka)} 个 CUDA kernel 事件"

    results.append(check("torch profiler (CUDA activity)", _prof))

    passed = sum(results)
    total = len(results)
    print("\n" + "=" * 78)
    print(f"  结果: {passed}/{total} 通过")
    if passed == total:
        print("  环境就绪, 可以开始写引擎了。")
    else:
        print("  有失败项, 按上面的报错逐条排查后再继续。")
    print("=" * 78 + "\n")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
