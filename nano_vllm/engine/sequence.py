"""P4/P6 · Sequence：请求的生命周期数据类。

对应 vLLM 的 Request + RequestStatus，但大幅简化：
  - 三态状态机 WAITING / RUNNING / FINISHED（vLLM 有 PREEMPTED/SWAPPED 等）
  - 抢占回 WAITING 从头 recompute（Branch 2），不需单独 PREEMPTED 态
  - 无 spec decoding / encoder inputs / mm features 等

核心字段:
  - num_computed_tokens: prefill chunk 进度（0 → len(prompt) 表示 prefill 完成）
  - block_table: 逻辑块 → 物理块编号（P3 PagedKVCache 用）
  - output_token_ids: 已生成的 token（decode 产出）

P6 前缀缓存扩展:
  - prefix_cache_done: 是否已做前缀查找（首次 schedule 做一次，抢占重置后重做）
  - num_registered_blocks: 滚动链已注册 hash 的块数（增量注册，避免重复算 hash）
  - last_block_hash: 滚动链最后一块的 hash（register_blocks 的 prev_hash 参数）
  - prefix_cache_extra_keys: 请求级额外 hash key（预留 LoRA adapter id / 多模态 hash 等
    「token 相同但 KV 不该共享」的标识；默认空 = 无额外约束）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from nano_vllm.kvmm.prefix_cache import ExtraKeys


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

    # P6 前缀缓存：滚动链状态 + 查找标记
    prefix_cache_done: bool = False
    num_registered_blocks: int = 0
    last_block_hash: bytes | None = None
    # P6 前缀缓存：请求级额外 hash key，需**顺序稳定**（预留 LoRA / 多模态等场景）
    prefix_cache_extra_keys: ExtraKeys = ()

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

    # ---------------- 统一 token 流（抢占恢复对齐 vLLM 的基础） ----------------
    #
    # 现状: "输入 token 从哪取"按阶段二分——prefill 取 prompt 切片、decode 取
    # output[-1]。这使"重算 prompt+output 区间"无法表达（抢占恢复只能清空 output
    # 从 prompt 重跑，见 docs/notes/preempt-recompute-align.md）。
    # 统一后: 请求只有一条 token 流 T = prompt + output，一切按**绝对位置**取。

    @property
    def all_token_ids(self) -> list[int]:
        """完整 token 流 T = prompt_token_ids + output_token_ids。"""
        return self.prompt_token_ids + self.output_token_ids

    def input_token_ids(self, start: int, num_tokens: int) -> list[int]:
        """位置 [start, start+num_tokens) 的输入 token（位置式取法）。

        - 位置 p < len(T)：取真实 token `T[p]`（追赶期 = 重算已知 token 的 KV）
        - 位置 p >= len(T)：重喂最后一个 token `T[-1]`（decode 的经典动作：
          用最后一个已知 token 的 logits 产出下一个 token）

        现状两条路径都是本公式的特例：
          prefill: start..start+k 全在 prompt 内 → `T[p] == prompt_token_ids[p]`
          decode : start == num_computed_tokens == len(T)-1 且落在 output 内
                   → `T[-1] == output_token_ids[-1]`
        """
        if num_tokens <= 0:
            return []
        T = self.all_token_ids
        if not T:
            raise ValueError("空 token 流：请求至少要有 1 个 prompt token")
        last = len(T) - 1
        return [T[p] if p < last else T[last] for p in range(start, start + num_tokens)]

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    @property
    def is_running(self) -> bool:
        return self.status == SequenceStatus.RUNNING

    def append_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)

    def reset_for_preemption(self) -> None:
        """抢占后重置：丢掉 KV、保留已生成 token，从头重算（对齐 vLLM）。

        **保留 `output_token_ids`**（vLLM `_preempt_request` 只把
        `num_computed_tokens` 归零，不动 output）：恢复时按位置式取法重算
        prompt+output 区间的 KV，命中上界为 `num_tokens - 1` → 重算量 ≤ 1 个 block；
        且末段的 logits 用于产出**下一个** token，不会重复输出已发过的 token。

        P6: 前缀缓存状态重置——重调度时重新查找命中（命中自己刚释放的块）。
        """
        self.num_computed_tokens = 0
        self.block_table.clear()
        self.status = SequenceStatus.WAITING
        self.prefix_cache_done = False
        self.num_registered_blocks = 0
        self.last_block_hash = None

    def finish(self) -> None:
        self.status = SequenceStatus.FINISHED

    def __repr__(self) -> str:
        return (
            f"Sequence(id={self.seq_id}, status={self.status.name}, "
            f"prompt={self.num_prompt_tokens}, output={len(self.output_token_ids)}, "
            f"computed={self.num_computed_tokens}, "
            f"registered={self.num_registered_blocks})"
        )