"""P3 · AttentionMetadata：把裸参数收进结构体，调用点只传一个入参。

P0-06 的问题: is_prefill / cache_seq_len / attn_mask 三个裸参数穿透 4 层
（models/qwen2.py → runner.py → bench/* → tests/*），P3 起每次签名变更都要穿一整套。

P3 起统一走 metadata；P2 路径保留旧参数以维持对拍基线（kv_cache=连续 cache 时 metadata=None）。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AttentionMetadata:
    """分页路径的 attention 元数据（P3 单条 + P4 批量 decode）。"""

    is_prefill: bool = True
    slot_mapping: torch.Tensor | None = None
    block_table: list[int] | None = None
    seq_len: int = 0
    attn_impl: str = "torch"
    # P4 批量 decode: 多条序列各自的 block_table 和 seq_len
    block_tables: list[list[int]] | None = None
    seq_lens: list[int] | None = None