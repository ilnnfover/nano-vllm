"""P4 · EngineCore：引擎主循环（schedule → execute → update）。

对应 vLLM 的 EngineCore.step()，但同步单线程（async 留 P5）。

执行策略（初始版，正确性优先）:
  - prefill: 逐条执行（每条一次 model forward），chunked prefill 读历史 KV + 混合 mask
  - decode:  逐条执行（每条一次 model forward），走 P3 paged attention
  - 后续优化: prefill varlen 拼批 + decode batched（Branch 4 手写 Triton kernel）
"""
from __future__ import annotations

import torch

from nano_vllm.attention.metadata import AttentionMetadata
from nano_vllm.attention.varlen_prefill import build_paged_kv_metadata
from nano_vllm.engine.scheduler import Scheduler, SchedulerOutput
from nano_vllm.engine.sequence import Sequence, SamplingParams
from nano_vllm.model_executor.runner import NanoRunner


class EngineCore:
    def __init__(
        self,
        runner: NanoRunner,
        scheduler: Scheduler,
    ) -> None:
        self.runner = runner
        self.scheduler = scheduler
        self.device = runner.device
        self.dtype = runner.dtype
        self._next_seq_id = 0

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

    def step(self) -> tuple[list[Sequence], dict[int, int]]:
        """执行一次调度+前向，返回 (完成的 sequences, 本步产出的 {seq_id: token_id})。"""
        if not self.scheduler.has_requests():
            return [], {}
        scheduler_output = self.scheduler.schedule()
        sampled = self._execute(scheduler_output)
        finished = self.scheduler.update_from_output(scheduler_output, sampled)
        return finished, sampled

    def _execute(self, scheduler_output: SchedulerOutput) -> dict[int, int]:
        """执行模型前向，返回 {seq_id: sampled_token_id}。"""
        sampled: dict[int, int] = {}
        prefill_seqs = [s for s in scheduler_output.scheduled if s.is_prefill]
        decode_seqs = [s for s in scheduler_output.scheduled if not s.is_prefill]

        if prefill_seqs:
            logits_map = self._run_prefill_batched(prefill_seqs)
            for s in prefill_seqs:
                if s.is_last_prefill_chunk:
                    seq = s.seq
                    sampled[seq.seq_id] = self._sample(seq, logits_map[seq.seq_id])

        if len(decode_seqs) == 1:
            seq = decode_seqs[0].seq
            logits = self._run_decode(seq)
            sampled[seq.seq_id] = self._sample(seq, logits)
        elif len(decode_seqs) > 1:
            seqs = [s.seq for s in decode_seqs]
            logits_list = self._run_decode_batched(seqs)
            for seq, logits in zip(seqs, logits_list):
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
        """执行 prefill chunk：处理 prompt[num_computed : num_computed+num_tokens]。"""
        chunk_start = seq.num_computed_tokens
        chunk_ids = seq.prompt_token_ids[chunk_start : chunk_start + num_tokens]
        total_seq_len = chunk_start + num_tokens

        input_ids = torch.tensor([chunk_ids], dtype=torch.long, device=self.device)
        position_ids = torch.arange(
            chunk_start, total_seq_len, device=self.device
        ).unsqueeze(0)
        slot_mapping = self.runner.paged_cache.slot_mapping(
            seq.block_table, chunk_start, num_tokens
        )
        metadata = AttentionMetadata(
            is_prefill=True,
            slot_mapping=slot_mapping,
            block_table=seq.block_table,
            seq_len=total_seq_len,
            attn_impl="torch",
        )
        logits, _ = self.runner.model(
            input_ids,
            paged_cache=self.runner.paged_cache,
            metadata=metadata,
            position_ids=position_ids,
        )
        return logits[0, -1]

    @torch.no_grad()
    def _run_prefill_batched(self, scheduled_prefills) -> dict[int, torch.Tensor]:
        """varlen 拼批 prefill：多条请求的 chunk 扁平拼接，一次 model forward。

        Returns:
            {seq_id: last_token_logits}，仅含产生输出的请求（last prefill chunk）
        """
        seqs = [s.seq for s in scheduled_prefills]
        num_tokens_list = [s.num_tokens for s in scheduled_prefills]

        input_ids: list[int] = []
        position_ids: list[int] = []
        slot_mappings: list[torch.Tensor] = []
        qo_indptr = [0]
        block_tables: list[list[int]] = []
        kv_lens: list[int] = []

        for seq, num_tokens in zip(seqs, num_tokens_list):
            chunk_start = seq.num_computed_tokens
            chunk_ids = seq.prompt_token_ids[chunk_start : chunk_start + num_tokens]
            input_ids.extend(chunk_ids)
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
            attn_impl=self.runner.prefill_impl,
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
        return {seq.seq_id: logits[i] for i, seq in enumerate(seqs)}

    @torch.no_grad()
    def _run_decode(self, seq: Sequence) -> torch.Tensor:
        """执行 decode：处理最后生成的 1 个 token。"""
        token_id = seq.output_token_ids[-1]
        pos = seq.num_computed_tokens
        new_seq_len = pos + 1

        input_ids = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
        position_ids = torch.tensor([[pos]], device=self.device)
        slot_mapping = self.runner.paged_cache.slot_mapping(
            seq.block_table, pos, 1
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
        return logits[0, -1]

    @torch.no_grad()
    def _run_decode_batched(self, seqs: list[Sequence]) -> list[torch.Tensor]:
        """批量 decode：多条序列一次 model forward，attention 内部逐条 gather KV。"""
        b = len(seqs)
        input_ids = torch.tensor(
            [[seq.output_token_ids[-1]] for seq in seqs],
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
