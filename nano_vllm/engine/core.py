"""P4 · EngineCore：引擎主循环（schedule → execute → update）。

对应 vLLM 的 EngineCore.step()，但同步单线程（async 留 P5）。

执行策略（初始版，正确性优先）:
  - prefill: 逐条执行（每条一次 model forward），chunked prefill 读历史 KV + 混合 mask
  - decode:  逐条执行（每条一次 model forward），走 P3 paged attention
  - 后续优化: prefill varlen 拼批 + decode batched（Branch 4 手写 Triton kernel）
"""
from __future__ import annotations

import os

import torch

from nano_vllm.attention.metadata import AttentionMetadata
from nano_vllm.attention.varlen_prefill import build_paged_kv_metadata
from nano_vllm.engine.scheduler import Scheduler, SchedulerOutput
from nano_vllm.engine.sequence import Sequence, SamplingParams
from nano_vllm.engine.stats import EngineCoreStats
from nano_vllm.model_executor.runner import NanoRunner


def validate_batch_metadata(
    *,
    num_input_tokens: int,
    num_slot_mappings: int,
    kv_lens: list[int],
    block_tables: list[list[int]],
    block_size: int,
    qo_indptr_last: int | None = None,
    max_position_embeddings: int | None = None,
) -> None:
    """校验一个 step 的批元数据自洽（D4 守卫）。

    背景: P3 踩坑 1/2 —— 块表 padding 读到脏块、末块半满未截断 → 直接 NaN。
    P6 引入块共享（ref_cnt>1）后，越界写会**污染另一个请求的 KV**，症状是
    「别人的输出变了」而不报错、不 NaN。这里把长度/容量不自洽变成立刻 raise，
    把静默算错转成显式失败。

    Raises:
        ValueError: 任一项不自洽。
    """
    if num_slot_mappings != num_input_tokens:
        raise ValueError(
            f"slot_mapping 长度 {num_slot_mappings} != input token 数 {num_input_tokens}"
        )
    if qo_indptr_last is not None and qo_indptr_last != num_input_tokens:
        raise ValueError(
            f"qo_indptr 末值 {qo_indptr_last} != input token 数 {num_input_tokens}"
        )
    if len(kv_lens) != len(block_tables):
        raise ValueError(
            f"kv_lens 条数 {len(kv_lens)} != block_tables 条数 {len(block_tables)}"
        )
    for i, (kv_len, table) in enumerate(zip(kv_lens, block_tables)):
        capacity = len(table) * block_size
        if kv_len > capacity:
            raise ValueError(
                f"请求 #{i} 的 kv_len={kv_len} 超出块表容量 {capacity}"
                f"（{len(table)} 块 × {block_size}），会读到脏块/越界写"
            )
    if max_position_embeddings is not None and kv_lens:
        longest = max(kv_lens)
        if longest > max_position_embeddings:
            raise ValueError(
                f"kv_len 最大值 {longest} 超出 max_position_embeddings {max_position_embeddings}"
            )


def check_finite(tensor: torch.Tensor, what: str) -> None:
    """NaN/Inf 守卫：在数值越界产生处暴露，而不是等到采样出垃圾 token。"""
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        raise RuntimeError(
            f"{what} 出现 NaN/Inf，优先怀疑块表越界或末块未截断（P3 踩坑 1/2）"
        )


class EngineCore:
    def __init__(
        self,
        runner: NanoRunner,
        scheduler: Scheduler,
        prefill_mode: str = "batched",
        debug: bool | None = None,
    ) -> None:
        self.runner = runner
        self.scheduler = scheduler
        self.device = runner.device
        self.dtype = runner.dtype
        self._next_seq_id = 0
        self.prefill_mode = prefill_mode
        # D4：debug 下额外做 logits NaN/Inf 检查（shape 断言始终开启，代价可忽略）
        self.debug = bool(os.environ.get("NANO_VLLM_DEBUG")) if debug is None else debug
        self.stats = EngineCoreStats()

    def add_request(
        self,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams | None = None,
    ) -> Sequence:
        seq = Sequence(
            seq_id=self._next_seq_id,
            prompt_token_ids=list(prompt_token_ids),
            sampling_params=sampling_params or SamplingParams(),
            eos_token_id=next(iter(self.runner.eos_ids)) if self.runner.eos_ids else None,
        )
        self._next_seq_id += 1
        self.scheduler.add_request(seq)
        return seq

    def abort(self, seq_id: int) -> None:
        """中止请求（C1）。"""
        self.scheduler.abort(seq_id)

    def step(self) -> tuple[list[Sequence], dict[int, int]]:
        """执行一次调度+前向，返回 (完成的 sequences, 本步产出的 {seq_id: token_id})。"""
        if not self.scheduler.has_requests():
            return [], {}
        scheduler_output = self.scheduler.schedule()
        sampled = self._execute(scheduler_output)
        finished = self.scheduler.update_from_output(scheduler_output, sampled)
        self.stats.update(
            num_batched_tokens=scheduler_output.num_batched_tokens,
            num_running=self.scheduler.num_running,
            num_waiting=self.scheduler.num_waiting,
            num_prefills=scheduler_output.num_prefills,
            num_decodes=scheduler_output.num_decodes,
            cache_usage=self.runner.paged_cache.cache_usage,
            waste_rate=self.scheduler.waste_rate,
            preempt_count=len(scheduler_output.preempted_seq_ids),
            tokens_generated=len(sampled),
            prefix_cache_hit_rate=self.scheduler.prefix_cache_hit_rate,
            prefix_cache_lookups=self.scheduler.prefix_lookups,
            prefix_cache_query_tokens=self.scheduler.prefix_query_tokens,
            prefix_cache_hit_tokens=self.scheduler.prefix_hit_tokens,
            prefix_cache_hit_blocks=self.scheduler.prefix_hit_blocks,
        )
        return finished, sampled

    def _execute(self, scheduler_output: SchedulerOutput) -> dict[int, int]:
        """执行模型前向，返回 {seq_id: sampled_token_id}。

        采样判据统一为 `ScheduledSeq.samples_this_step`（chunk 终点到达已知 token 流
        末尾才采样）——前向对**全部**被调度的请求执行（KV 必须写），只是不一定采样：
        抢占恢复的中间追赶段只重算 KV，末段才产出下一个 token。
        """
        sampled: dict[int, int] = {}
        prefill_seqs = [s for s in scheduler_output.scheduled if s.is_prefill]
        decode_seqs = [s for s in scheduler_output.scheduled if not s.is_prefill]

        if prefill_seqs:
            if self.prefill_mode == "per-seq":
                for s in prefill_seqs:
                    logits = self._run_prefill(s.seq, s.num_scheduled_tokens)
                    if s.samples_this_step:
                        sampled[s.seq.seq_id] = self._sample(s.seq, logits)
            else:
                logits_map = self._run_prefill_batched(prefill_seqs)
                for s in prefill_seqs:
                    if s.samples_this_step:
                        seq = s.seq
                        sampled[seq.seq_id] = self._sample(seq, logits_map[seq.seq_id])

        if len(decode_seqs) == 1:
            s = decode_seqs[0]
            logits = self._run_decode(s.seq)
            if s.samples_this_step:
                sampled[s.seq.seq_id] = self._sample(s.seq, logits)
        elif len(decode_seqs) > 1:
            seqs = [s.seq for s in decode_seqs]
            logits_list = self._run_decode_batched(seqs)
            # 仅对"到达流末尾"的请求采样（budget=1 时追赶段可能只走 1 token 且未到末尾）
            idx = [i for i, s in enumerate(decode_seqs) if s.samples_this_step]
            sample_seqs = [seqs[i] for i in idx]
            sample_logits = [logits_list[i] for i in idx]
            # A2: 批量 greedy 采样（一次 argmax 替代逐条 int() 的 N 次同步）
            all_greedy = all(seq.sampling_params.temperature <= 0.0 for seq in sample_seqs)
            if all_greedy:
                logits_batch = torch.stack(sample_logits)
                token_ids = self.runner.sampler.batch_sample_greedy(logits_batch)
                for seq, tid in zip(sample_seqs, token_ids):
                    sampled[seq.seq_id] = tid
            else:
                for seq, logits in zip(sample_seqs, sample_logits):
                    sampled[seq.seq_id] = self._sample(seq, logits)

        return sampled

    def _sample(self, seq: Sequence, logits: torch.Tensor) -> int:
        return self.runner.sampler.sample(
            logits,
            temperature=seq.sampling_params.temperature,
            top_k=seq.sampling_params.top_k,
            top_p=seq.sampling_params.top_p,
        )

    @torch.no_grad()
    def _run_prefill(self, seq: Sequence, num_tokens: int) -> torch.Tensor:
        """执行 prefill chunk：处理位置 [num_computed, num_computed+num_tokens)。

        token 取法为**位置式**（`Sequence.input_token_ids`）——当前 prefill 阶段的
        chunk 恒落在 prompt 内，故与 `prompt_token_ids[chunk_start:...]` 等价；
        统一取法是为抢占恢复能重算 output 区间做准备。
        """
        chunk_start = seq.num_computed_tokens
        chunk_ids = seq.input_token_ids(chunk_start, num_tokens)
        total_seq_len = chunk_start + num_tokens

        input_ids = torch.tensor([chunk_ids], dtype=torch.long, device=self.device)
        position_ids = torch.arange(
            chunk_start, total_seq_len, device=self.device
        ).unsqueeze(0)
        slot_mapping = self.runner.paged_cache.slot_mapping(
            seq.block_table, chunk_start, num_tokens
        )
        validate_batch_metadata(
            num_input_tokens=len(chunk_ids),
            num_slot_mappings=slot_mapping.numel(),
            kv_lens=[total_seq_len],
            block_tables=[seq.block_table],
            block_size=self.runner.paged_cache.block_size,
            max_position_embeddings=self.runner.config.max_position_embeddings,
        )
        metadata = AttentionMetadata(
            is_prefill=True,
            slot_mapping=slot_mapping,
            block_table=seq.block_table,
            seq_len=total_seq_len,
            prefill_impl="torch",
        )
        logits, _ = self.runner.model(
            input_ids,
            paged_cache=self.runner.paged_cache,
            metadata=metadata,
            position_ids=position_ids,
        )
        if self.debug:
            check_finite(logits, "per-seq prefill logits")
        return logits[0, -1]

    @torch.no_grad()
    def _run_prefill_batched(self, scheduled_prefills) -> dict[int, torch.Tensor]:
        """varlen 拼批 prefill：多条请求的 chunk 扁平拼接，一次 model forward。

        Returns:
            {seq_id: last_token_logits}，仅含产生输出的请求（last prefill chunk）
        """
        seqs = [s.seq for s in scheduled_prefills]
        num_tokens_list = [s.num_scheduled_tokens for s in scheduled_prefills]

        input_ids: list[int] = []
        position_ids: list[int] = []
        slot_mappings: list[torch.Tensor] = []
        qo_indptr = [0]
        block_tables: list[list[int]] = []
        kv_lens: list[int] = []

        for seq, num_tokens in zip(seqs, num_tokens_list):
            chunk_start = seq.num_computed_tokens
            input_ids.extend(seq.input_token_ids(chunk_start, num_tokens))
            position_ids.extend(range(chunk_start, chunk_start + num_tokens))
            slot_mappings.append(self.runner.paged_cache.slot_mapping(
                seq.block_table, chunk_start, num_tokens
            ))
            qo_indptr.append(qo_indptr[-1] + num_tokens)
            block_tables.append(seq.block_table)
            kv_lens.append(chunk_start + num_tokens)

        input_ids_t = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        position_ids_t = torch.tensor([position_ids], device=self.device)
        slot_mapping = torch.cat(slot_mappings)
        qo_indptr_t = torch.tensor(qo_indptr, dtype=torch.int32, device=self.device)
        validate_batch_metadata(
            num_input_tokens=len(input_ids),
            num_slot_mappings=slot_mapping.numel(),
            kv_lens=kv_lens,
            block_tables=block_tables,
            block_size=self.runner.paged_cache.block_size,
            qo_indptr_last=qo_indptr[-1],
            max_position_embeddings=self.runner.config.max_position_embeddings,
        )
        paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len = build_paged_kv_metadata(
            block_tables, kv_lens, self.runner.paged_cache.block_size, self.device,
        )
        metadata = AttentionMetadata(
            is_prefill=True,
            slot_mapping=slot_mapping,
            qo_indptr=qo_indptr_t,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
            prefill_impl=self.runner.prefill_impl,
        )
        # 只取每条请求最后一个 token 的 logits（避免算全部 [total_q, vocab]）
        last_idx = torch.tensor(
            [qo_indptr[i + 1] - 1 for i in range(len(seqs))],
            dtype=torch.long, device=self.device,
        )
        logits, _ = self.runner.model(
            input_ids_t,
            paged_cache=self.runner.paged_cache,
            metadata=metadata,
            position_ids=position_ids_t,
            last_idx=last_idx,
        )
        if self.debug:
            check_finite(logits, "batched prefill logits")
        return {seq.seq_id: logits[i] for i, seq in enumerate(seqs)}

    @torch.no_grad()
    def _run_decode(self, seq: Sequence) -> torch.Tensor:
        """执行 decode：处理位置 `num_computed_tokens`（稳态即"最后生成的 token"）。

        位置式取法与 `output_token_ids[-1]` 等价（稳态 `computed == num_tokens - 1`
        且该位置落在 output 内），但前者也能覆盖"位置落在 prompt 内"的单 token chunk
        （末尾 prompt token 的追赶），为抢占恢复铺路。
        """
        pos = seq.num_computed_tokens
        token_id = seq.input_token_ids(pos, 1)[0]
        new_seq_len = pos + 1

        input_ids = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
        position_ids = torch.tensor([[pos]], device=self.device)
        slot_mapping = self.runner.paged_cache.slot_mapping(
            seq.block_table, pos, 1
        )
        validate_batch_metadata(
            num_input_tokens=1,
            num_slot_mappings=slot_mapping.numel(),
            kv_lens=[new_seq_len],
            block_tables=[seq.block_table],
            block_size=self.runner.paged_cache.block_size,
            max_position_embeddings=self.runner.config.max_position_embeddings,
        )
        metadata = AttentionMetadata(
            is_prefill=False,
            slot_mapping=slot_mapping,
            block_table=seq.block_table,
            seq_len=new_seq_len,
            attn_impl=self.runner.attn_impl,
        )
        logits, _ = self.runner.model(
            input_ids,
            paged_cache=self.runner.paged_cache,
            metadata=metadata,
            position_ids=position_ids,
        )
        if self.debug:
            check_finite(logits, "single decode logits")
        return logits[0, -1]

    @torch.no_grad()
    def _run_decode_batched(self, seqs: list[Sequence]) -> list[torch.Tensor]:
        """批量 decode：多条序列一次 model forward，attention 内部逐条 gather KV。"""
        b = len(seqs)
        input_ids = torch.tensor(
            [[seq.input_token_ids(seq.num_computed_tokens, 1)[0]] for seq in seqs],
            dtype=torch.long, device=self.device,
        )
        position_ids = torch.tensor(
            [[seq.num_computed_tokens] for seq in seqs],
            device=self.device,
        )
        slot_mappings = []
        for seq in seqs:
            pos = seq.num_computed_tokens
            slot_mappings.append(self.runner.paged_cache.slot_mapping(
                seq.block_table, pos, 1
            ))
        slot_mapping = torch.cat(slot_mappings)
        block_tables = [seq.block_table for seq in seqs]
        seq_lens = [seq.num_computed_tokens + 1 for seq in seqs]
        validate_batch_metadata(
            num_input_tokens=b,
            num_slot_mappings=slot_mapping.numel(),
            kv_lens=seq_lens,
            block_tables=block_tables,
            block_size=self.runner.paged_cache.block_size,
            max_position_embeddings=self.runner.config.max_position_embeddings,
        )
        metadata = AttentionMetadata(
            is_prefill=False,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            seq_lens=seq_lens,
            attn_impl=self.runner.attn_impl,
        )
        logits, _ = self.runner.model(
            input_ids,
            paged_cache=self.runner.paged_cache,
            metadata=metadata,
            position_ids=position_ids,
        )
        if self.debug:
            check_finite(logits, "batched decode logits")
        return [logits[i, -1] for i in range(b)]

    def generate(
        self,
        prompts: list[list[int]],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[int]]:
        """连续批处理生成：提交所有请求，循环 step 直到全部完成。"""
        seqs = [self.add_request(p, sampling_params) for p in prompts]
        outputs: dict[int, list[int]] = {}
        while self.scheduler.has_requests():
            finished, _ = self.step()
            for seq in finished:
                outputs[seq.seq_id] = seq.output_token_ids
        return [outputs[seq.seq_id] for seq in seqs]
