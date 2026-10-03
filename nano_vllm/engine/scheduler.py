"""P4 · Scheduler：连续批处理调度器（iteration-level scheduling）。

核心算法（Branch 2/3/5 锁定; R2/P8 修订见下）:
  0. **统一记账（对齐 vLLM，P8）**：请求只有一条 token 流
     `T = prompt + output`，没有 prefill/decode 阶段之分；
     `num_new_tokens = num_tokens - num_computed_tokens` —— 稳态 decode 恒为 1，
     抢占恢复后的追赶段为整段（受 budget / 节流阀截断）。
     采样判据 = chunk 终点到达 T 末尾（`ScheduledSeq.samples_this_step`）。
  1. 先调度 RUNNING 请求
     - 单 token 且位置已过 prompt 区 → decode 路径（可进 P7 图）
     - 多 token chunk / prompt 区单 token → varlen prefill 路径
     - block 不足时抢占自身（free → reset → 放回 waiting 队首）——唯一抢占路径
  2. 再调度 WAITING 请求（首个 chunk，用剩余 budget）
     - block 不足时停止准入（**不抢占 running**，R2 修订，对齐 vLLM：
       抢占只发生在 RUNNING 段；waiting 靠 running 完成后自然释放的块入场）
  3. token budget 统一记账（Branch 3）：Σ(chunk) ≤ max_num_batched_tokens
  4. watermark 预留（Branch 3）：空闲块 - 需求 < watermark 时不分配
  5. R2 准入守卫：add_request 拒绝「独占全池也无法完成」的请求（防 livelock）

A6 节流阀（对齐 vLLM scheduler 标配）:
  - `long_prefill_token_threshold`：单请求一步最多吃多少 prefill token（0 = 关闭，vLLM 默认）。
    开启可防止一条 8K prompt 连续多个 step 独占预算、把同批短请求的 TPOT 打炸；
    实测代价见 scheduler.__init__ 注释与 docs/stages/P4.md。
  - `max_num_seqs`：同批运行请求上限（None = 不限）。
  - `watermark_blocks` 缺省改为**按比例**（`watermark_ratio`，vLLM 默认 0.01）：
    块池按显存反推后（B1）有上万块，固定 1 块不再有保护意义。

P6 前缀缓存接口（对齐 vLLM `KVCacheManager`）:
  - `get_computed_blocks(seq)` : 查命中前缀（不改状态）
  - `_attach_hit_blocks(seq, hit)` : 挂载命中块 + touch + 推进 num_computed_tokens
  - `allocate_slots(seq, num_new_tokens)` : 容量检查与分配
  （vLLM 把后两者的决策合在 `allocate_slots` 内；本实现拆成两步，便于对拍与抢占处理）
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from nano_vllm.kvmm.paged_kv_cache import PagedKVCache
from nano_vllm.kvmm.prefix_cache import CacheHit, PrefixCache
from nano_vllm.engine.sequence import Sequence, SequenceStatus


@dataclass
class ScheduledSeq:
    """单个请求在一个 step 内的调度决策。

    统一记账（抢占恢复对齐 vLLM）后，`num_scheduled_tokens` 不再依赖
    "prefill/decode 阶段"，而恒等于 `num_new_tokens = num_tokens - num_computed_tokens`
    （被 token budget / 节流阀截断）——稳态 decode 时它天然等于 1，
    抢占恢复后的追赶段则可能 > 1。
    """

    seq: Sequence
    num_scheduled_tokens: int
    is_prefill: bool
    """**路由标记**（不是阶段标记）：True = 本 chunk 走 varlen prefill 路径。

    判据见 `Scheduler._use_prefill_route`：多 token chunk，或单 token 但位置
    仍在 prompt 区（末尾 prompt token）。其余走 decode 路径（可进 P7 的图）。
    """

    @property
    def samples_this_step(self) -> bool:
        """本 chunk 是否要采样：其终点到达**已知 token 流末尾** `seq.num_tokens`。

        - 稳态 decode：`computed == num_tokens - 1`，+1 恰好到达 → 采样（产出下一个）
        - 抢占恢复追赶：只有最后一段到达末尾才采样（中间段纯重算 KV，不采样）
        - 因此"重算 output 区间"不会重复产出 token
        """
        return (
            self.seq.num_computed_tokens + self.num_scheduled_tokens
            >= self.seq.num_tokens
        )


@dataclass
class SchedulerOutput:
    scheduled: list[ScheduledSeq] = field(default_factory=list)
    preempted_seq_ids: list[int] = field(default_factory=list)

    @property
    def num_batched_tokens(self) -> int:
        return sum(s.num_scheduled_tokens for s in self.scheduled)

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
        watermark_blocks: int | None = None,
        watermark_ratio: float = 0.01,
        max_num_seqs: int | None = None,
        long_prefill_token_threshold: int = 0,
        enable_prefix_cache: bool = True,
        prefix_cache_salt: int | str = 0,
    ) -> None:
        self.paged_cache = paged_cache
        self.block_size = paged_cache.block_size
        self.max_num_batched_tokens = max_num_batched_tokens
        # A6 · watermark：显式传值优先；否则按比例预留（vLLM 默认 0.01）。
        # 固定 1 块在块池按显存反推（B1，上万块）后已失去保护意义。
        self.watermark_blocks = (
            watermark_blocks
            if watermark_blocks is not None
            else max(1, int(paged_cache.pool.num_blocks * watermark_ratio))
        )
        # A6 · 长 prefill 节流：单请求一步最多吃多少 prefill token。
        # 缺省 0 = 关闭，**与 vLLM 一致**（`SchedulerConfig.long_prefill_token_threshold` 默认 0）。
        # 实测代价：开启（在 1.5B/bf16 上取 1024）会用约 17% 吞吐换约 12% TPOT 尾延迟改善，
        # 故不默认开启，由调用方按负载 opt-in。见 docs/stages/P4.md 的 A6 复测小节。
        self.long_prefill_token_threshold = (
            long_prefill_token_threshold if long_prefill_token_threshold > 0 else None
        )
        # A6 · 同批并发上限（None = 不限制；vLLM 默认 256，我们留给调用方显式开启）
        self.max_num_seqs = max_num_seqs
        self.enable_prefix_cache = enable_prefix_cache
        self.prefix_cache = (
            PrefixCache(paged_cache.pool, self.block_size, salt=prefix_cache_salt)
            if enable_prefix_cache else None
        )

        # P6 前缀缓存累计统计（命中率埋点，供 F1 metrics 消费）
        self.prefix_lookups = 0        # 累计 lookup 次数（含抢占后重查）
        self.prefix_query_tokens = 0   # 累计查询的 prompt token 数（命中率分母）
        self.prefix_hit_tokens = 0     # 累计因命中跳过的 token 数（命中率分子）
        self.prefix_hit_blocks = 0     # 累计命中块数

        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self._next_seq_id = 0

    @property
    def num_free_blocks(self) -> int:
        return self.paged_cache.pool.num_free_blocks

    @property
    def num_running(self) -> int:
        return len(self.running)

    @property
    def num_waiting(self) -> int:
        return len(self.waiting)

    def blocks_needed(self, seq_len: int) -> int:
        return (seq_len + self.block_size - 1) // self.block_size

    def add_request(self, seq: Sequence) -> None:
        """请求入队（R2 守卫：单请求自身可行性检查）。

        请求最大长度（prompt + max_new_tokens）所需的块数超过**全池总量**时，
        即使抢占掉所有其他请求、独占整池也永远无法完成——入队只会导致
        调度循环里「抢占 → 重算 → 再抢占」的 livelock。对齐 vLLM：请求无法
        被调度时最终会以失败结束而不是无限空转，这里在准入时直接拒绝。
        """
        need = self.blocks_needed(seq.num_prompt_tokens + seq.sampling_params.max_new_tokens)
        if need > self.paged_cache.pool.num_blocks:
            raise ValueError(
                f"请求 #{seq.seq_id} 最大长度 "
                f"{seq.num_prompt_tokens}(prompt)+{seq.sampling_params.max_new_tokens}(max_new)"
                f" = {seq.num_prompt_tokens + seq.sampling_params.max_new_tokens} token"
                f" 需 {need} 块，超过块池总量 {self.paged_cache.pool.num_blocks} 块——"
                f"独占全池也无法完成，拒绝入队（防 livelock）。"
                f"可增大 num_blocks 或减小 max_new_tokens。"
            )
        self.waiting.append(seq)

    def has_requests(self) -> bool:
        return bool(self.waiting) or bool(self.running)

    def abort(self, seq_id: int) -> None:
        """中止请求：从 running/waiting 移除 + 释放 block（C1）。"""
        for seq in self.running:
            if seq.seq_id == seq_id:
                self._free_seq_blocks(seq)
                self.running.remove(seq)
                return
        for seq in list(self.waiting):
            if seq.seq_id == seq_id:
                self._free_seq_blocks(seq)
                self.waiting.remove(seq)
                return

    @property
    def waste_rate(self) -> float:
        """实时浪费率：1 - Σ(seq_len) / (num_used_blocks * block_size)（B3）。"""
        total_tokens = sum(seq.num_tokens for seq in self.running)
        num_used = self.paged_cache.pool.num_blocks - self.paged_cache.pool.num_free_blocks
        if num_used == 0:
            return 0.0
        return 1.0 - total_tokens / (num_used * self.block_size)

    @property
    def prefix_cache_hit_rate(self) -> float:
        """累计前缀命中率 = 命中 token / 查询 prompt token（无查询返回 0）。"""
        if self.prefix_query_tokens == 0:
            return 0.0
        return self.prefix_hit_tokens / self.prefix_query_tokens

    def reset_prefix_stats(self) -> None:
        """清零前缀缓存统计（bench repeat 间隔离用）。"""
        self.prefix_lookups = 0
        self.prefix_query_tokens = 0
        self.prefix_hit_tokens = 0
        self.prefix_hit_blocks = 0

    def get_computed_blocks(self, seq: Sequence) -> CacheHit:
        """查询 seq 可复用的前缀命中块（**不改 seq 状态**，仅累计查询统计）。

        对齐 vLLM `KVCacheManager.get_computed_blocks(request)`。差异说明：本实现把
        「命中块挂载」拆到 `_attach_hit_blocks`、把「容量分配」拆到 `allocate_slots` 两步，
        vLLM 在 `allocate_slots` 内合并决策。

        使用 `seq.prefix_cache_extra_keys` 作为额外 hash key，保证「token 相同但 KV 不该
        共享」（LoRA adapter 等）的请求查不到对方的块。

        R1 钳制（对齐 vLLM `kv_cache_manager.get_computed_blocks` 的 `num_tokens - 1`）:
        命中最多覆盖 `num_tokens - 1` 个 token，保证至少重算最后一个 token 以产出
        logits。否则同 prompt 重放且长度恰为 block_size 整数倍时，命中覆盖整个 prompt
        → `num_new_tokens == 0` → 空前向 / 永不产出首 token。

        注: 抢占恢复对齐后，token 流是 **prompt + 已生成**（`Sequence.all_token_ids`），
        故钳制用 `num_tokens - 1`（新请求 output 为空时与 `num_prompt_tokens - 1` 等价，
        恢复时能复用已生成 token 的块 → 重算量 ≤ 1 个 block）。
        """
        if self.prefix_cache is None:
            return CacheHit([], 0, False, None)
        self.prefix_lookups += 1
        self.prefix_query_tokens += seq.num_tokens
        return self.prefix_cache.find_longest_hit(
            seq.all_token_ids, seq.prefix_cache_extra_keys,
            max_tokens=seq.num_tokens - 1,
        )

    def _attach_hit_blocks(self, seq: Sequence, hit: CacheHit) -> None:
        """把命中块挂到 seq.block_table 并推进 num_computed_tokens（touch 共享块）。

        对齐 vLLM 纯 Transformer 默认行为后，partial 块不注册 hash（见 prefix_cache
        B1 修订），故 find_longest_hit 只返回满块、`hit.is_partial` 恒 False。下方
        partial COW 分支**保留作设计参考**（理解 vLLM partial 命中 + COW 机制用），
        实际不进入；若将来启用 fine-grained hash 则复用。vLLM 侧同样只在 partial 命中
        时走 COW，且要求 Mamba align group —— 证据见 `docs/notes/prefix-cache-partial-hits.md`。
        """
        if not hit.block_ids:
            seq.prefix_cache_done = True
            return
        for bid in hit.block_ids:
            self.paged_cache.pool.touch(bid)
        seq.block_table.extend(hit.block_ids)
        seq.num_computed_tokens = hit.num_tokens
        if hit.is_partial:
            assert self.prefix_cache is not None, "命中来自 prefix_cache，必然非 None"
            if self.paged_cache.pool.num_free_blocks <= self.watermark_blocks:
                for bid in hit.block_ids:
                    self.paged_cache.pool.free(bid)
                seq.block_table.clear()
                seq.num_computed_tokens = 0
                seq.prefix_cache_done = True
                return
            old_last = seq.block_table[-1]
            new_last = self.prefix_cache.cow_for_write(self.paged_cache, old_last)
            seq.block_table[-1] = new_last
        # 命中生效（partial 回滚分支已提前 return，不计入）
        self.prefix_hit_tokens += hit.num_tokens
        self.prefix_hit_blocks += hit.num_blocks
        seq.num_registered_blocks = hit.num_full_blocks
        seq.last_block_hash = hit.last_full_hash
        seq.prefix_cache_done = True

    def allocate_slots(self, seq: Sequence, num_new_tokens: int) -> bool:
        """确保 block_table 能容纳 `num_computed_tokens + num_new_tokens`，不足则分配。

        对齐 vLLM `KVCacheManager.allocate_slots(request, num_new_tokens)`。

        Returns:
            True 分配成功（或不需要分配），False 块不足（触发抢占/等待）。
        """
        new_seq_len = seq.num_computed_tokens + num_new_tokens
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

        # ---- 1. 调度 RUNNING（统一记账：追赶 chunk 与 decode 同一公式）----
        for seq in list(self.running):
            if token_budget <= 0:
                break

            # num_new_tokens = num_tokens - num_computed_tokens：
            #   稳态 decode（computed == num_tokens-1）→ 1
            #   抢占恢复追赶（computed 落后）→ 整段（受 budget / 节流阀截断）
            num_new = min(seq.num_new_tokens, token_budget)
            if self.long_prefill_token_threshold is not None:
                num_new = min(num_new, self.long_prefill_token_threshold)
            is_prefill = self._use_prefill_route(seq, num_new)

            if not self.allocate_slots(seq, num_new):
                self._preempt(seq)
                preempted_ids.append(seq.seq_id)
                continue

            scheduled.append(ScheduledSeq(seq, num_new, is_prefill))
            token_budget -= num_new

        # ---- 2. 调度 WAITING（新 prefill chunk）----
        while self.waiting and token_budget > 0:
            if self.max_num_seqs is not None and len(self.running) >= self.max_num_seqs:
                break
            seq = self.waiting[0]
            if self.prefix_cache is not None and not seq.prefix_cache_done:
                self._attach_hit_blocks(seq, self.get_computed_blocks(seq))
            num_new = min(seq.num_new_tokens, token_budget)
            if self.long_prefill_token_threshold is not None:
                num_new = min(num_new, self.long_prefill_token_threshold)

            if not self.allocate_slots(seq, num_new):
                # R2 修订（对齐 vLLM）: waiting 请求**不抢占** running。
                # vLLM 中抢占只发生在 RUNNING 调度段（allocate_slots 失败时），waiting 段
                # 块不足直接停止准入、等 running 自然释放（vllm scheduler: 本步发生抢占
                # 后不再调度 waiting）。原实现「waiting 抢占 running 最新者」在组合需求
                # 超过池容量时会互相 ping-pong（A 抢 B → B 重算 → B 抢 A → …），
                # 双方都到不了 max_new_tokens → livelock。代价是长 running 会暂时
                # 饿死 waiting（有界：running 必在 max_new_tokens 内完成并释放）。
                break

            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled.append(
                ScheduledSeq(seq, num_new, self._use_prefill_route(seq, num_new))
            )
            token_budget -= num_new

        return SchedulerOutput(scheduled, preempted_ids)

    @staticmethod
    def _use_prefill_route(seq: Sequence, num_new: int) -> bool:
        """本 chunk 是否走 varlen prefill 路径（对齐 vLLM 的"按 token 数路由"）。

        | 情形 | 路由 | 理由 |
        | --- | --- | --- |
        | `num_new > 1`（新请求 prefill / 抢占恢复追赶段） | prefill | 多 token 需 chunk 内 causal |
        | `num_new == 1` 且位置在 **prompt 区**（末尾 prompt token） | prefill | **保持与改动前一致的 kernel 选择**，避免数值路径变化 |
        | `num_new == 1` 且位置在 output 区（稳态 decode / 尾部追赶） | decode | 单 token + 全历史 KV，可进 P7 图 |
        """
        return num_new > 1 or seq.num_computed_tokens < seq.num_prompt_tokens

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
            seq.num_computed_tokens += s.num_scheduled_tokens

            new_token = sampled_tokens.get(seq.seq_id)
            if new_token is not None:
                seq.append_token(new_token)

            if self.prefix_cache is not None:
                num_full = seq.num_computed_tokens // self.block_size
                if num_full > seq.num_registered_blocks:
                    all_tokens = seq.prompt_token_ids + seq.output_token_ids
                    seq.num_registered_blocks, seq.last_block_hash = (
                        self.prefix_cache.register_blocks(
                            all_tokens, seq.block_table, num_full,
                            seq.num_registered_blocks, seq.last_block_hash,
                            seq.prefix_cache_extra_keys,
                        )
                    )

            if new_token is not None:
                should_stop = (
                    new_token == seq.eos_token_id
                    or len(seq.output_token_ids) >= seq.sampling_params.max_new_tokens
                )
                if should_stop:

                    seq.finish()
                    self._free_seq_blocks(seq)
                    if seq in self.running:
                        self.running.remove(seq)
                    finished.append(seq)

        return finished