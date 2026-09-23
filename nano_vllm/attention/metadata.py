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
    """单条分页路径的 attention 元数据（P3 decode/prefill 共用）。"""

    # prefill 走 SDPA is_causal=True；decode 走 paged attention
    is_prefill: bool = True
    # 当前步 token 的物理写入位置 [num_tokens]（reshape_and_cache 用）
    slot_mapping: torch.Tensor | None = None
    # 逻辑块 → 物理块编号（decode 读 KV 用）
    block_table: list[int] | None = None
    # 该序列当前 KV 总长度（decode attention 的 softmax 范围）
    seq_len: int = 0
    # decode attention 后端: "torch"（朴素对照）| "triton"（自研 kernel）
    attn_impl: str = "torch"