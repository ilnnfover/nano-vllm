"""P3 · 分页 KV cache：物理存储按 block 组织，每请求一张 block_table。

布局: k/v_cache = [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
token t 的物理 slot = block_table[t // block_size] * block_size + (t % block_size)

对比 P2 的连续预分配（kv_cache.py）:
  - P2 每请求占满 max_seq_len 个 slot（浪费 15%）
  - P3 按需分配 block，浪费仅最后一个 block 的尾部（< block_size 个 slot）
"""
from __future__ import annotations

import torch

from nano_vllm.core.block_pool import BlockPool


def reshape_and_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """把当前步的 K/V 按 slot_mapping 写入 paged 物理存储（gather-scatter 写入核）。

    Args:
        k_cache: [num_blocks, block_size, num_kv_heads, head_dim]（单层）
        v_cache: 同上
        key: [num_tokens, num_kv_heads, head_dim]
        value: [num_tokens, num_kv_heads, head_dim]
        slot_mapping: [num_tokens] long，每 token 的物理 slot = block_id * block_size + offset

    实现: 把 [num_blocks, block_size, ...] view 成 [num_blocks * block_size, ...]，
    用 slot_mapping 做一次高级索引 scatter。slot 索引是物理地址，天然支持乱序/跨 block。
    """
    if key.shape[0] != slot_mapping.shape[0]:
        raise ValueError(
            f"token 数不匹配: key={key.shape[0]}, slot_mapping={slot_mapping.shape[0]}"
        )
    k_flat = k_cache.view(-1, k_cache.shape[-2], k_cache.shape[-1])
    v_flat = v_cache.view(-1, v_cache.shape[-2], v_cache.shape[-1])
    k_flat[slot_mapping] = key
    v_flat[slot_mapping] = value


class PagedKVCache:
    """分页 KV cache：物理张量 + BlockPool + block_table/slot_mapping 构造。"""

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
    ) -> None:
        if block_size <= 0:
            raise ValueError(f"block_size 必须为正, got: {block_size}")
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)
        self.pool = BlockPool(num_blocks)

        shape = (num_layers, num_blocks, block_size, num_kv_heads, head_dim)
        self.k_cache = torch.zeros(shape, dtype=dtype, device=self.device)
        self.v_cache = torch.zeros(shape, dtype=dtype, device=self.device)

        self._written_tokens = 0

    @staticmethod
    def blocks_needed(seq_len: int, block_size: int) -> int:
        return (seq_len + block_size - 1) // block_size

    def allocate(self, seq_len: int) -> list[int]:
        """为长度 seq_len 的序列分配 block_table。"""
        return self.pool.allocate_n(self.blocks_needed(seq_len, self.block_size))

    def ensure_capacity(self, block_table: list[int], seq_len: int) -> list[int]:
        """确保 block_table 能容纳 seq_len 个 token，不足则追加块。"""
        need = self.blocks_needed(seq_len, self.block_size)
        if need > len(block_table):
            block_table.extend(self.pool.allocate_n(need - len(block_table)))
        return block_table

    def slot_mapping(self, block_table: list[int], start_pos: int, num_tokens: int) -> torch.Tensor:
        """构造 [start_pos, start_pos+num_tokens) 区间 token 的物理 slot 索引。"""
        slots = [
            block_table[t // self.block_size] * self.block_size + (t % self.block_size)
            for t in range(start_pos, start_pos + num_tokens)
        ]
        return torch.tensor(slots, dtype=torch.long, device=self.device)

    def write(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, slot_mapping: torch.Tensor) -> None:
        """写入单层 KV。

        Args:
            k, v: [num_kv_heads, num_tokens, head_dim]（attention 内部布局）
        """
        k_t = k.transpose(0, 1).contiguous()  # [num_tokens, num_kv_heads, head_dim]
        v_t = v.transpose(0, 1).contiguous()
        reshape_and_cache(self.k_cache[layer_idx], self.v_cache[layer_idx], k_t, v_t, slot_mapping)
        if layer_idx == 0:  # 按逻辑 token 统计一次，不按层累加
            self._written_tokens += k_t.shape[0]

    def read_blocks(self, layer_idx: int, block_table: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """按 block_table gather 出连续 KV（供 SDPA 或 torch 朴素实现用）。

        Returns:
            k, v: [num_kv_heads, len(block_table) * block_size, head_dim]
        """
        idx = torch.tensor(block_table, dtype=torch.long, device=self.device)
        k = self.k_cache[layer_idx].index_select(0, idx)  # [n_blk, block_size, kv_heads, head_dim]
        v = self.v_cache[layer_idx].index_select(0, idx)
        k = k.reshape(-1, self.num_kv_heads, self.head_dim).transpose(0, 1)
        v = v.reshape(-1, self.num_kv_heads, self.head_dim).transpose(0, 1)
        return k, v

    def free(self, block_table: list[int]) -> None:
        self.pool.free_n(block_table)

    def reset(self) -> None:
        self.pool.reset()
        self._written_tokens = 0

    @property
    def num_used_blocks(self) -> int:
        return self.pool.num_used_blocks

    @property
    def allocated_slots(self) -> int:
        return self.pool.num_used_blocks * self.block_size

    @property
    def waste_rate(self) -> float:
        """已分配 block 中未被 token 占用的 slot 占比（P3 验收 < 2%）。"""
        if self.allocated_slots == 0:
            return 1.0
        return 1.0 - self._written_tokens / self.allocated_slots