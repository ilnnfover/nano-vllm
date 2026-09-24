"""P4 · varlen prefill 拼批 attention：多条请求的 prefill chunk 一次算完。

动机: P4 初版 prefill 逐条执行（每条一次 model forward），8K prompt 串行导致
GPU 利用率低、吞吐上不去。本模块把本 step 所有 prefill chunk 扁平拼接成一条
varlen 序列，一次 attention 算完，消除串行与 padding。

两种后端（统一接口）:
  - torch:      逐条 gather KV 做 causal attention（oracle，CPU 可测）
  - flashinfer: BatchPrefillWithPagedKVCacheWrapper（GPU，性能路径）

接口:
  q:                    [total_q, num_heads, head_dim]  扁平拼接
  k_cache/v_cache:      [num_blocks, block_size, num_kv_heads, head_dim]  (NHD)
  qo_indptr:            [batch+1] query 累计长度
  paged_kv_indptr:      [batch+1] KV block 累计数
  paged_kv_indices:     [total_blocks] 各请求物理 block 拼接
  paged_kv_last_page_len: [batch] 每请求末块有效长度（1..block_size）

causal 语义: 右对齐——query 的最后一个 token attend 到 KV 的最后一个 token。
chunked prefill 时 query 是 prompt 的后缀 [kv_len-q_len, kv_len)，右对齐正好正确。
"""
from __future__ import annotations

import torch

from nano_vllm.attention.paged_attn import repeat_kv

# flashinfer workspace 复用（128MB，按 device 缓存，避免每步重复分配）
_FLASHINFER_WORKSPACE: dict[str, torch.Tensor] = {}


def _get_workspace(device: torch.device, size_bytes: int = 128 * 1024 * 1024) -> torch.Tensor:
    key = str(device)
    if key not in _FLASHINFER_WORKSPACE:
        _FLASHINFER_WORKSPACE[key] = torch.empty(size_bytes, dtype=torch.uint8, device=device)
    return _FLASHINFER_WORKSPACE[key]


def build_paged_kv_metadata(
    block_tables: list[list[int]],
    kv_lens: list[int],
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """从 block_table 列表构造 flashinfer 分页元数据。

    Returns:
        (paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len) 均为 int32
    """
    indptr = [0]
    indices: list[int] = []
    last_page_lens: list[int] = []
    for bt, kv_len in zip(block_tables, kv_lens):
        indices.extend(bt)
        indptr.append(len(indices))
        if not bt:
            last_page_lens.append(0)
        else:
            last_page_lens.append(kv_len - (len(bt) - 1) * block_size)
    return (
        torch.tensor(indptr, dtype=torch.int32, device=device),
        torch.tensor(indices, dtype=torch.int32, device=device),
        torch.tensor(last_page_lens, dtype=torch.int32, device=device),
    )


def varlen_prefill_torch(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    qo_indptr: torch.Tensor,
    paged_kv_indptr: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    paged_kv_last_page_len: torch.Tensor,
    num_kv_heads: int,
    scaling: float,
) -> torch.Tensor:
    """torch 朴素 varlen prefill（oracle）：逐条 gather KV + 右对齐 causal attention。"""
    block_size = k_cache.shape[1]
    num_heads = q.shape[1]
    n_rep = num_heads // num_kv_heads
    batch = qo_indptr.shape[0] - 1
    head_dim = q.shape[2]

    outs: list[torch.Tensor] = []
    for i in range(batch):
        q_start, q_end = int(qo_indptr[i]), int(qo_indptr[i + 1])
        q_len = q_end - q_start
        if q_len == 0:
            continue
        blk_start, blk_end = int(paged_kv_indptr[i]), int(paged_kv_indptr[i + 1])
        blocks = paged_kv_indices[blk_start:blk_end].tolist()
        kv_len = (len(blocks) - 1) * block_size + int(paged_kv_last_page_len[i])

        q_i = q[q_start:q_end]  # [q_len, num_heads, head_dim]
        idx = torch.tensor(blocks, dtype=torch.long, device=k_cache.device)
        k = k_cache.index_select(0, idx).reshape(-1, num_kv_heads, head_dim)[:kv_len]
        v = v_cache.index_select(0, idx).reshape(-1, num_kv_heads, head_dim)[:kv_len]
        k = repeat_kv(k.transpose(0, 1), n_rep)  # [num_heads, kv_len, head_dim]
        v = repeat_kv(v.transpose(0, 1), n_rep)

        scores = torch.einsum("qhd,hkd->hqk", q_i, k) * scaling
        pos_q = torch.arange(kv_len - q_len, kv_len, device=q.device)
        pos_k = torch.arange(kv_len, device=q.device)
        mask = pos_q[:, None] >= pos_k[None, :]
        scores = scores.masked_fill(~mask, float("-inf"))
        probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        outs.append(torch.einsum("hqk,hkd->qhd", probs, v))

    if not outs:
        return torch.empty(0, num_heads, head_dim, dtype=q.dtype, device=q.device)
    return torch.cat(outs, dim=0)


def varlen_prefill_flashinfer(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    qo_indptr: torch.Tensor,
    paged_kv_indptr: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    paged_kv_last_page_len: torch.Tensor,
    num_kv_heads: int,
    scaling: float,
) -> torch.Tensor:
    """flashinfer 批量 prefill（GPU 性能路径）。

    k_cache/v_cache 直接以 tuple 传入，布局 NHD [num_blocks, block_size, num_kv_heads, head_dim]
    与 P3 PagedKVCache 一致，无需改布局或拷贝。
    """
    import flashinfer

    num_heads = q.shape[1]
    head_dim = q.shape[2]
    block_size = k_cache.shape[1]

    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        _get_workspace(q.device), "NHD"
    )
    wrapper.plan(
        qo_indptr.to(torch.int32),
        paged_kv_indptr.to(torch.int32),
        paged_kv_indices.to(torch.int32),
        paged_kv_last_page_len.to(torch.int32),
        num_heads,
        num_kv_heads,
        head_dim,
        block_size,
        causal=True,
        sm_scale=scaling,
        q_data_type=q.dtype,
        kv_data_type=k_cache.dtype,
    )
    return wrapper.run(q, (k_cache, v_cache))


def varlen_prefill_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    qo_indptr: torch.Tensor,
    paged_kv_indptr: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    paged_kv_last_page_len: torch.Tensor,
    num_kv_heads: int,
    scaling: float,
    impl: str = "torch",
) -> torch.Tensor:
    """统一入口：按 impl 分派 torch / flashinfer 后端。"""
    if impl == "flashinfer":
        return varlen_prefill_flashinfer(
            q, k_cache, v_cache, qo_indptr, paged_kv_indptr,
            paged_kv_indices, paged_kv_last_page_len, num_kv_heads, scaling,
        )
    return varlen_prefill_torch(
        q, k_cache, v_cache, qo_indptr, paged_kv_indptr,
        paged_kv_indices, paged_kv_last_page_len, num_kv_heads, scaling,
    )
