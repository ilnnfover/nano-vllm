"""P4 · Scheduler：连续批处理调度器（iteration-level scheduling）。

核心算法（Branch 2/3/5 锁定）:
  1. 先调度 RUNNING 请求（decode 优先 — Branch 5）
     - decode: 每请求 1 token
     - in-progress prefill: chunk 受剩余 budget 限制
     - block 不足时抢占自身（free → reset → 放回 waiting 队首）
  2. 再调度 WAITING 请求（prefill chunk，用剩余 budget）
     - block 不足时 LIFO 抢占 running 中最新请求（Branch 3）
     - 无 running 可抢占则停止调度
  3. token budget 统一记账（Branch 3）：Σ(prefill chunk) + Σ(decode 1) ≤ max_num_batched_tokens
  4. watermark 预留（Branch 3）：空闲块 - 需求 < watermark 时不分配
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from nano_vllm.core.paged_kv_cache import PagedKVCache
from nano_vllm.engine.sequence import Sequence, SequenceStatus


@dataclass
class ScheduledSeq:
    """单个请求在一个 step 内的调度决策。"""

    seq: Sequence
    num_tokens: int
    is_prefill: bool

    @property
    def is_last_prefill_chunk(self) -> bool:
        """是否为 prefill 的最后一块（算完这块 prefill 就结束）。"""
        return self.is_prefill and (
            self.seq.num_computed_tokens + self.num_tokens >= self.seq.num_prompt_tokens
        )


@dataclass
class SchedulerOutput:
    scheduled: list[ScheduledSeq] = field(default_factory=list)
    preempted_seq_ids: list[int] = field(default_factory=list)

    @property
    def num_batched_tokens(self) -> int:
        return sum(s.num_tokens for s in self.scheduled)

    @property
    def num_prefills(self) -> int:
        return sum(1 for s in self.scheduled if s.is_prefill)

    @property
    def num_decodes(self) -> int:
        return sum(1 for s in self.scheduled if not s.is_prefill)


class Scheduler:
    def __init__(
        self,
        paged_cache: PagedKVCache,
        max_num_batched_tokens: int = 2048,
        watermark_blocks: int = 0,
    ) -> None:
        self.paged_cache = paged_cache
        self.block_size = paged_cache.block_size
        self.max_num_batched_tokens = max_num_batched_tokens
        self.watermark_blocks = watermark_blocks

        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self._next_seq_id = 0

    @property
    def num_free_blocks(self) -> int:
        return self.paged_cache.pool.num_free_blocks

    def blocks_needed(self, seq_len: int) -> int:
        return (seq_len + self.block_size - 1) // self.block_size

    def add_request(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def has_requests(self) -> bool:
        return bool(self.waiting) or bool(self.running)

    def _try_ensure_capacity(self, seq: Sequence, new_seq_len: int) -> bool:
        """检查并分配 block 使 block_table 能容纳 new_seq_len 个 token。

        Returns:
            True 如果分配成功（或不需要分配），False 如果 block 不足。
        """
        need = self.blocks_needed(new_seq_len) - len(seq.block_table)
        if need <= 0:
            return True
        if self.num_free_blocks - need < self.watermark_blocks:
            return False
        seq.block_table.extend(self.paged_cache.pool.allocate_n(need))
        return True

    def _free_seq_blocks(self, seq: Sequence) -> None:
        if seq.block_table:
            self.paged_cache.free(seq.block_table)
            seq.block_table.clear()

    def _preempt(self, seq: Sequence) -> None:
        """抢占一个请求：free block → reset → 放回 waiting 队首。"""
        self._free_seq_blocks(seq)
        seq.reset_for_preemption()
        if seq in self.running:
            self.running.remove(seq)
        self.waiting.appendleft(seq)

    def schedule(self) -> SchedulerOutput:
        token_budget = self.max_num_batched_tokens
        scheduled: list[ScheduledSeq] = []
        preempted_ids: list[int] = []

        # ---- 1. 调度 RUNNING（decode 优先 + in-progress prefill chunk）----
        for seq in list(self.running):
            if token_budget <= 0:
                break

            if seq.is_prefill:
                num_new = min(seq.num_new_tokens, token_budget)
                is_prefill = True
            else:
                num_new = 1
                is_prefill = False

            new_seq_len = seq.num_computed_tokens + num_new
            if not self._try_ensure_capacity(seq, new_seq_len):
                self._preempt(seq)
                preempted_ids.append(seq.seq_id)
                continue

            scheduled.append(ScheduledSeq(seq, num_new, is_prefill))
            token_budget -= num_new

        # ---- 2. 调度 WAITING（新 prefill chunk）----
        while self.waiting and token_budget > 0:
            seq = self.waiting[0]
            num_new = min(seq.num_new_tokens, token_budget)
            new_seq_len = seq.num_computed_tokens + num_new

            if not self._try_ensure_capacity(seq, new_seq_len):
                if self.running:
                    victim = self.running[-1]
                    for i, s in enumerate(scheduled):
                        if s.seq is victim:
                            token_budget += s.num_tokens
                            scheduled.pop(i)
                            break
                    self._preempt(victim)
                    preempted_ids.append(victim.seq_id)
                    continue
                break

            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled.append(ScheduledSeq(seq, num_new, True))
            token_budget -= num_new

        return SchedulerOutput(scheduled, preempted_ids)

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        sampled_tokens: dict[int, int],
    ) -> list[Sequence]:
        """执行后更新状态：推进 computed_tokens、追加生成 token、检查完成。

        Args:
            sampled_tokens: {seq_id: token_id}，仅含产生输出的请求
                （decode 或 last prefill chunk）

        Returns:
            已完成的 Sequence 列表。
        """
        finished: list[Sequence] = []
        for s in scheduler_output.scheduled:
            seq = s.seq
            seq.num_computed_tokens += s.num_tokens

            if seq.seq_id in sampled_tokens:
                token = sampled_tokens[seq.seq_id]
                seq.append_token(token)

                should_stop = (
                    token == seq.eos_token_id
                    or len(seq.output_token_ids) >= seq.sampling_params.max_new_tokens
                )
                if should_stop:
                    seq.finish()
                    self._free_seq_blocks(seq)
                    if seq in self.running:
                        self.running.remove(seq)
                    finished.append(seq)

        return finished