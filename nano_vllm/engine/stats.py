"""F1 · EngineCore 每步统计（step-level metrics）。

vLLM 参考: vllm/v1/metrics/stats.py::SchedulerStats
最小版: 每步快照 + 累计计数，bench 落盘 JSON。Prometheus 可延后。

P6: prefix_cache_* 字段由 Scheduler 的累计命中统计填充（命中率埋点）。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class EngineCoreStats:
    """引擎每步统计快照 + 累计计数。"""

    # ---- 本步快照（每步覆盖）----
    num_batched_tokens: int = 0
    num_running: int = 0
    num_waiting: int = 0
    num_prefills: int = 0
    num_decodes: int = 0
    cache_usage: float = 0.0
    waste_rate: float = 0.0

    # ---- 累计计数（单调递增）----
    total_steps: int = 0
    total_preempt_count: int = 0
    total_tokens_generated: int = 0

    # ---- P6 前缀缓存（累计，由 Scheduler 填充）----
    prefix_cache_hit_rate: float = 0.0    # 命中率 = 命中 token / 查询 prompt token
    prefix_cache_lookups: int = 0         # 累计 lookup 次数
    prefix_cache_query_tokens: int = 0    # 累计查询 prompt token
    prefix_cache_hit_tokens: int = 0      # 累计命中 token
    prefix_cache_hit_blocks: int = 0      # 累计命中块数

    # ---- P7 decode CUDA Graph（累计，由 DecodeGraphRunner 填充）----
    cudagraph_replays: int = 0            # 图回放次数（命中）
    cudagraph_fallback_eager: int = 0     # 回退 eager 次数（超桶 / 未启用图）
    cudagraph_padded_seqs: int = 0        # 累计 padding 行数（桶对齐浪费）
    cudagraph_capture_ms: float = 0.0     # 捕获总耗时（含 warmup）
    cudagraph_buckets: int = 0            # 已捕获桶数

    @property
    def cudagraph_hit_rate(self) -> float:
        """图命中率 = 回放 / (回放 + 回退)。"""
        total = self.cudagraph_replays + self.cudagraph_fallback_eager
        return self.cudagraph_replays / total if total else 0.0

    def update(
        self,
        num_batched_tokens: int,
        num_running: int,
        num_waiting: int,
        num_prefills: int,
        num_decodes: int,
        cache_usage: float,
        waste_rate: float,
        preempt_count: int,
        tokens_generated: int,
        prefix_cache_hit_rate: float = 0.0,
        prefix_cache_lookups: int = 0,
        prefix_cache_query_tokens: int = 0,
        prefix_cache_hit_tokens: int = 0,
        prefix_cache_hit_blocks: int = 0,
        cudagraph_replays: int = 0,
        cudagraph_fallback_eager: int = 0,
        cudagraph_padded_seqs: int = 0,
        cudagraph_capture_ms: float = 0.0,
        cudagraph_buckets: int = 0,
    ) -> None:
        """每步更新：快照字段覆盖，累计字段累加。"""
        self.num_batched_tokens = num_batched_tokens
        self.num_running = num_running
        self.num_waiting = num_waiting
        self.num_prefills = num_prefills
        self.num_decodes = num_decodes
        self.cache_usage = cache_usage
        self.waste_rate = waste_rate
        self.total_steps += 1
        self.total_preempt_count += preempt_count
        self.total_tokens_generated += tokens_generated
        self.prefix_cache_hit_rate = prefix_cache_hit_rate
        self.prefix_cache_lookups = prefix_cache_lookups
        self.prefix_cache_query_tokens = prefix_cache_query_tokens
        self.prefix_cache_hit_tokens = prefix_cache_hit_tokens
        self.prefix_cache_hit_blocks = prefix_cache_hit_blocks
        self.cudagraph_replays = cudagraph_replays
        self.cudagraph_fallback_eager = cudagraph_fallback_eager
        self.cudagraph_padded_seqs = cudagraph_padded_seqs
        self.cudagraph_capture_ms = cudagraph_capture_ms
        self.cudagraph_buckets = cudagraph_buckets

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)