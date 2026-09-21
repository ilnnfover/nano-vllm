"""P2 · 连续预分配 KV cache（故意用浪费写法，为 P3 PagedAttention 留动机）。

布局: 每层一对 [batch, max_seq_len, num_kv_heads, head_dim] 的连续张量，
      按请求预分配整块显存，不回收、不分块——浪费率即 P3 的动机证据。
"""
from __future__ import annotations

import torch


class KVCache:
    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int = 1024,
        batch_size: int = 1,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
    ) -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.dtype = dtype
        self.device = torch.device(device)

        shape = (num_layers, batch_size, max_seq_len, num_kv_heads, head_dim)
        self.k_cache = torch.zeros(shape, dtype=dtype, device=self.device)
        self.v_cache = torch.zeros(shape, dtype=dtype, device=self.device)
        self.seq_len = 0

    def reset(self) -> None:
        self.seq_len = 0

    @property
    def capacity(self) -> int:
        return self.max_seq_len * self.batch_size

    @property
    def waste_rate(self) -> float:
        if self.seq_len == 0:
            return 1.0
        return 1.0 - self.seq_len / self.max_seq_len

    @torch.no_grad()
    def write(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, start_pos: int) -> None:
        s = k.shape[2]
        self.k_cache[layer_idx, :, start_pos : start_pos + s] = k.transpose(1, 2)
        self.v_cache[layer_idx, :, start_pos : start_pos + s] = v.transpose(1, 2)

    @torch.no_grad()
    def read(self, layer_idx: int, total_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        k = self.k_cache[layer_idx, :, :total_len].transpose(1, 2)
        v = self.v_cache[layer_idx, :, :total_len].transpose(1, 2)
        return k, v
