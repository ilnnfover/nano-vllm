"""P3 · attention 后端注册表：同一进程内切换 paged attention 实现。

P0-07 的问题: attention 后端选择硬编码在模型里（q.is_cuda 决定 GQA 融合 vs repeat_kv）。
P3 起用注册表 + 显式 attn_impl 开关，P8 的"三实现对照开关"由此扩展。
"""
from __future__ import annotations

from collections.abc import Callable

import torch

from nano_vllm.attention.paged_attn import paged_attention_torch
from nano_vllm.attention.triton_paged_attn import paged_attention_triton

PagedAttnFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, list[int], int, int, float],
    torch.Tensor,
]

PAGED_ATTN_BACKENDS: dict[str, PagedAttnFn] = {
    "torch": paged_attention_torch,
    "triton": paged_attention_triton,
}


def get_paged_attn(impl: str) -> PagedAttnFn:
    if impl not in PAGED_ATTN_BACKENDS:
        raise ValueError(f"未知 attn_impl={impl!r}, 可选: {sorted(PAGED_ATTN_BACKENDS)}")
    return PAGED_ATTN_BACKENDS[impl]