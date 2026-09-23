"""P3 · 自研 Triton paged attention kernel（decode）。

与 torch 朴素版的语义差异: 朴素版先把全部 KV gather 成连续张量再算 attention；
本 kernel 按 block_table 逐块把 K/V 搬进 SRAM，用 online softmax 累加，
不物化完整的 [seq_len, head_dim]——这正是分页省显存的关键。

单条版: grid = (num_heads,)，每 program 处理 1 个 head 的 1 个 query token。
GQA: kv_head_idx = head_idx // num_queries_per_kv，Q 头共享同一组 KV。
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

# TRITON_INTERPRET=1 时 kernel 在 CPU 上解释执行（用于无 GPU 环境的逻辑验证）
_INTERPRET = bool(os.environ.get("TRITON_INTERPRET"))


@triton.jit
def _paged_attn_decode_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_table_ptr,
    seq_len_ptr,
    out_ptr,
    num_kv_heads,
    head_dim,
    block_size,
    num_queries_per_kv,
    scaling,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    head_idx = tl.program_id(0)
    kv_head_idx = head_idx // num_queries_per_kv
    seq_len = tl.load(seq_len_ptr)

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim
    q = tl.load(q_ptr + head_idx * head_dim + offs_d, mask=d_mask, other=0.0)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    num_blocks = tl.cdiv(seq_len, block_size)
    offs_n = tl.arange(0, BLOCK_N)
    for blk_i in range(num_blocks):
        physical = tl.load(block_table_ptr + blk_i)
        base = physical * block_size * num_kv_heads * head_dim + kv_head_idx * head_dim
        n_mask = (blk_i * block_size + offs_n) < seq_len
        n_mask = n_mask & (offs_n < block_size)

        k_ptrs = k_cache_ptr + base + offs_n[:, None] * (num_kv_heads * head_dim) + offs_d[None, :]
        k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
        scores = tl.sum(q[None, :] * k, axis=1) * scaling
        scores = tl.where(n_mask, scores, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)

        v_ptrs = v_cache_ptr + base + offs_n[:, None] * (num_kv_heads * head_dim) + offs_d[None, :]
        v = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    out = acc / l_i
    tl.store(out_ptr + head_idx * head_dim + offs_d, out, mask=d_mask)


def paged_attention_triton(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: list[int],
    seq_len: int,
    num_kv_heads: int,
    scaling: float,
) -> torch.Tensor:
    """Triton paged attention（decode 单条版）。

    Args:
        q: [num_heads, 1, head_dim]
        k_cache/v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
        block_table: list[int]
        seq_len: 实际 KV 长度
    Returns:
        [num_heads, 1, head_dim]
    """
    num_heads, num_tokens, head_dim = q.shape
    if num_tokens != 1:
        raise ValueError(f"P3 Triton kernel 只支持 decode 单 token, got num_tokens={num_tokens}")
    if not q.is_cuda and not _INTERPRET:
        raise RuntimeError("Triton kernel 需要 CUDA 设备（或设 TRITON_INTERPRET=1 走 CPU 解释器）")

    block_size = k_cache.shape[1]
    num_queries_per_kv = num_heads // num_kv_heads

    q_flat = q.reshape(num_heads, head_dim).contiguous()
    out = torch.empty_like(q_flat)
    bt = torch.tensor(block_table, dtype=torch.int32, device=q.device)
    seq_len_t = torch.tensor([seq_len], dtype=torch.int32, device=q.device)

    BLOCK_D = triton.next_power_of_2(head_dim)
    BLOCK_N = triton.next_power_of_2(block_size)

    _paged_attn_decode_kernel[(num_heads,)](
        q_flat,
        k_cache,
        v_cache,
        bt,
        seq_len_t,
        out,
        num_kv_heads,
        head_dim,
        block_size,
        num_queries_per_kv,
        scaling,
        BLOCK_D=BLOCK_D,
        BLOCK_N=BLOCK_N,
    )
    return out.reshape(num_heads, 1, head_dim)