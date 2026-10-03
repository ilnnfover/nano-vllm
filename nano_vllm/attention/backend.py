"""P3 · attention 后端注册表：同一进程内切换 paged attention 实现。

P0-07 的问题: attention 后端选择硬编码在模型里（q.is_cuda 决定 GQA 融合 vs repeat_kv）。
P3 起用注册表 + 显式 attn_impl 开关。

decode 侧三实现对照：`torch`（einsum 朴素 oracle） / `sdpa`（F.scaled_dot_product_attention）
/ `triton`（自研 kernel，P7 CUDA Graph 唯一支持实现）。
roadmap 原写的第二实现是 flash-attn，已改为 SDPA —— 理由见 `docs/notes/p8-prereq.md`。
"""
from __future__ import annotations

from collections.abc import Callable

import torch

from nano_vllm.attention.paged_attn import paged_attention_sdpa, paged_attention_torch
from nano_vllm.attention.triton_paged_attn import paged_attention_triton
from nano_vllm.attention.varlen_prefill import varlen_prefill_torch, varlen_prefill_flashinfer

PagedAttnFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, list[int], int, int, float],
    torch.Tensor,
]

PAGED_ATTN_BACKENDS: dict[str, PagedAttnFn] = {
    "torch": paged_attention_torch,   # einsum + softmax 朴素 oracle
    "sdpa": paged_attention_sdpa,     # gather 成连续 KV 后走 F.scaled_dot_product_attention
    "triton": paged_attention_triton,  # 自研 Triton kernel（P7 图路径唯一实现）
}

PrefillAttnFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
     torch.Tensor, torch.Tensor, int, float],
    torch.Tensor,
]

PREFILL_ATTN_BACKENDS: dict[str, PrefillAttnFn] = {
    "torch": varlen_prefill_torch,
    "flashinfer": varlen_prefill_flashinfer,
}


def get_paged_attn(impl: str) -> PagedAttnFn:
    if impl not in PAGED_ATTN_BACKENDS:
        raise ValueError(f"未知 attn_impl={impl!r}, 可选: {sorted(PAGED_ATTN_BACKENDS)}")
    return PAGED_ATTN_BACKENDS[impl]


def get_prefill_attn(impl: str) -> PrefillAttnFn:
    if impl not in PREFILL_ATTN_BACKENDS:
        raise ValueError(f"未知 prefill_impl={impl!r}, 可选: {sorted(PREFILL_ATTN_BACKENDS)}")
    return PREFILL_ATTN_BACKENDS[impl]