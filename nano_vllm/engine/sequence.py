"""P4 · Sequence：请求的生命周期数据类。

对应 vLLM 的 Request + RequestStatus，但大幅简化：
  - 三态状态机 WAITING / RUNNING / FINISHED（vLLM 有 PREEMPTED/SWAPPED 等）
  - 抢占回 WAITING 从头 recompute（Branch 2），不需单独 PREEMPTED 态
  - 无 spec decoding / encoder inputs / mm features 等

核心字段:
  - num_computed_tokens: prefill chunk 进度（0 → len(prompt) 表示 prefill 完成）
  - block_table: 逻辑块 → 物理块编号（P3 PagedKVCache 用）
  - output_token_ids: 已生成的 token（decode 产出）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class SequenceStatus(IntEnum):
    WAITING = 0
    RUNNING = 1
    FINISHED = 2


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    max_new_tokens: int = 128


@dataclass
class Sequence:
    """单个请求的完整生命周期状态。"""

    seq_id: int
    prompt_token_ids: list[int]
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    eos_token_id: int | None = None

    status: SequenceStatus = SequenceStatus.WAITING
    block_table: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0
    output_token_ids: list[int] = field(default_factory=list)

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_tokens(self) -> int:
        return self.num_prompt_tokens + len(self.output_token_ids)

    @property
    def num_new_tokens(self) -> int:
        """待计算的 token 数（num_tokens - num_computed_tokens）。"""
        return self.num_tokens - self.num_computed_tokens

    @property
    def is_prefill(self) -> bool:
        """是否在 prefill 阶段（尚未算完所有 prompt token）。"""
        return self.num_computed_tokens < self.num_prompt_tokens

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    @property
    def is_running(self) -> bool:
        return self.status == SequenceStatus.RUNNING

    def append_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)

    def reset_for_preemption(self) -> None:
        """抢占后重置：从头 recompute（Branch 2）。"""
        self.num_computed_tokens = 0
        self.output_token_ids.clear()
        self.block_table.clear()
        self.status = SequenceStatus.WAITING

    def finish(self) -> None:
        self.status = SequenceStatus.FINISHED

    def __repr__(self) -> str:
        return (
            f"Sequence(id={self.seq_id}, status={self.status.name}, "
            f"prompt={self.num_prompt_tokens}, output={len(self.output_token_ids)}, "
            f"computed={self.num_computed_tokens})"
        )