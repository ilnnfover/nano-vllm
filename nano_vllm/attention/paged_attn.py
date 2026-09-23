"""P3 · paged attention 三实现对照：torch 朴素 / Triton 自研 / 连续 SDPA。

统一接口（decode 单条版）:
  q:        [num_heads, num_tokens, head_dim]   (decode 时 num_tokens=1)
  k_cache:  [num_blocks, block_size, num_kv_heads, head_dim]
  v_cache:  同上
  block_table: list[int] 逻辑块 → 物理块编号
  seq_len:  该序列实际 KV 长度（最后一块可能只用前 seq_len % block_size 个 slot）

路线 §6 红线: "两遍实现法（先对、再快）"——先写 torch 朴素版对拍通过，再写 Triton。
"""
from __future__ import annotations

import torch


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[num_kv_heads, s, head_dim] → [num_heads, s, head_dim]，GQA 展开。"""
    if n_rep == 1:
        return x
    num_kv_heads, s, head_dim = x.shape
    return x[:, None, :, :].expand(num_kv_heads, n_rep, s, head_dim).reshape(num_kv_heads * n_rep, s, head_dim)


def gather_paged_kv(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: list[int],
    seq_len: int,
    num_kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按 block_table gather 出 [num_kv_heads, seq_len, head_dim] 的连续 KV。"""
    idx = torch.tensor(block_table, dtype=torch.long, device=k_cache.device)
    k = k_cache.index_select(0, idx).reshape(-1, num_kv_heads, k_cache.shape[-1])[:seq_len]
    v = v_cache.index_select(0, idx).reshape(-1, num_kv_heads, v_cache.shape[-1])[:seq_len]
    return k.transpose(0, 1), v.transpose(0, 1)


def paged_attention_torch(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: list[int],
    seq_len: int,
    num_kv_heads: int,
    scaling: float,
) -> torch.Tensor:
    """torch 朴素 paged attention（gather 全部 KV 后做标准 attention）。

    仅用于对拍与基线，非性能路径。语义与 Triton kernel 必须逐元素一致。
    """
    num_heads = q.shape[0]
    n_rep = num_heads // num_kv_heads
    k, v = gather_paged_kv(k_cache, v_cache, block_table, seq_len, num_kv_heads)
    k = repeat_kv(k, n_rep)
    v = repeat_kv(v, n_rep)

    # [num_heads, num_tokens, seq_len]
    scores = torch.einsum("hqd,hkd->hqk", q, k) * scaling
    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    out = torch.einsum("hqk,hkd->hqd", probs, v)
    return out


def paged_attention_sdpa(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: list[int],
    seq_len: int,
    num_kv_heads: int,
    scaling: float,
) -> torch.Tensor:
    """连续 SDPA 对照：gather 成连续 KV 后走 F.scaled_dot_product_attention（is_causal=False）。"""
    k, v = gather_paged_kv(k_cache, v_cache, block_table, seq_len, num_kv_heads)
    n_rep = q.shape[0] // num_kv_heads
    if n_rep > 1:
        k = repeat_kv(k, n_rep)
        v = repeat_kv(v, n_rep)
    # SDPA 要求 [b, h, s, d]
    out = torch.nn.functional.scaled_dot_product_attention(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), scale=scaling, is_causal=False,
    )
    return out.squeeze(0)