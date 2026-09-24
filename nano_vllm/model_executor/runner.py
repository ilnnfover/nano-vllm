"""P1/P2/P3 · 执行器：use_cache=False 走 P1 eager 重算（最差基线），use_cache=True 走 P2 prefill+decode。

P3 起新增 use_paged=True：走分页 KV cache（BlockPool + block_table + slot_mapping），
decode attention 可选 torch 朴素 / 自研 Triton kernel（attn_impl）。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from nano_vllm.attention.metadata import AttentionMetadata
from nano_vllm.config import Qwen2Config
from nano_vllm.core.paged_kv_cache import PagedKVCache
from nano_vllm.kv_cache import KVCache
from nano_vllm.models.qwen2 import Qwen2ForCausalLM
from nano_vllm.sample.sampler import Sampler


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    max_new_tokens: int = 128


class NanoRunner:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        seed: int | None = None,
        max_seq_len: int = 1024,
        block_size: int = 16,
        num_blocks: int | None = None,
        attn_impl: str = "torch",
        prefill_impl: str = "torch",
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.block_size = block_size
        self.attn_impl = attn_impl
        self.prefill_impl = prefill_impl
        self.config = Qwen2Config.from_json(Path(model_path) / "config.json")
        self.model = Qwen2ForCausalLM(self.config).to(device=device, dtype=dtype).eval()
        self.model.load_weights(model_path)
        self.sampler = Sampler(device)
        if seed is not None:
            self.sampler.seed(seed)
        self.kv_cache = KVCache(
            num_layers=self.config.num_hidden_layers,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            max_seq_len=max_seq_len,
            batch_size=1,
            dtype=dtype,
            device=device,
        )
        # P3 分页 KV cache：num_blocks 缺省时按 max_seq_len 推导（单条够用即可）
        self.paged_cache = PagedKVCache(
            num_layers=self.config.num_hidden_layers,
            num_blocks=num_blocks or PagedKVCache.blocks_needed(max_seq_len, block_size) + 1,
            block_size=block_size,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            dtype=dtype,
            device=device,
        )
        self._block_table: list[int] | None = None
        self._paged_seq_len = 0
        self.pad_id = 0

    @property
    def eos_ids(self) -> set[int]:
        return self.config.eos_ids

    @torch.no_grad()
    def forward_last_logits(self, ids: list[int]) -> torch.Tensor:
        t = torch.tensor([ids], dtype=torch.long, device=self.device)
        last_idx = torch.tensor([len(ids) - 1], device=self.device)
        logits, _ = self.model(t, last_idx=last_idx)
        return logits[0]

    @torch.no_grad()
    def _prefill(self, prompt_ids: list[int]) -> torch.Tensor:
        self.kv_cache.reset()
        t = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        last_idx = torch.tensor([len(prompt_ids) - 1], device=self.device)
        logits, _ = self.model(t, kv_cache=self.kv_cache, is_prefill=True, cache_seq_len=0, last_idx=last_idx)
        self.kv_cache.seq_len = len(prompt_ids)
        return logits[0]

    @torch.no_grad()
    def _decode(self, tok: int) -> torch.Tensor:
        pos = self.kv_cache.seq_len
        t = torch.tensor([[tok]], dtype=torch.long, device=self.device)
        logits, _ = self.model(t, kv_cache=self.kv_cache, is_prefill=False, cache_seq_len=pos)
        self.kv_cache.seq_len = pos + 1
        return logits[0, -1]

    # ---------------- P3 分页路径 ----------------

    @torch.no_grad()
    def _prefill_paged(self, prompt_ids: list[int]) -> torch.Tensor:
        seq_len = len(prompt_ids)
        self.paged_cache._written_tokens = 0
        self._block_table = self.paged_cache.allocate(seq_len)
        self._paged_seq_len = seq_len
        slot_mapping = self.paged_cache.slot_mapping(self._block_table, 0, seq_len)
        metadata = AttentionMetadata(
            is_prefill=True, slot_mapping=slot_mapping, block_table=self._block_table,
            seq_len=seq_len, attn_impl=self.attn_impl,
        )
        t = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)
        last_idx = torch.tensor([seq_len - 1], device=self.device)
        logits, _ = self.model(
            t, paged_cache=self.paged_cache, metadata=metadata,
            position_ids=position_ids, last_idx=last_idx,
        )
        return logits[0]

    @torch.no_grad()
    def _decode_paged(self, tok: int) -> torch.Tensor:
        pos = self._paged_seq_len
        new_seq_len = pos + 1
        self.paged_cache.ensure_capacity(self._block_table, new_seq_len)
        slot_mapping = self.paged_cache.slot_mapping(self._block_table, pos, 1)
        metadata = AttentionMetadata(
            is_prefill=False, slot_mapping=slot_mapping, block_table=self._block_table,
            seq_len=new_seq_len, attn_impl=self.attn_impl,
        )
        t = torch.tensor([[tok]], dtype=torch.long, device=self.device)
        position_ids = torch.tensor([[pos]], device=self.device)
        logits, _ = self.model(
            t, paged_cache=self.paged_cache, metadata=metadata, position_ids=position_ids,
        )
        self._paged_seq_len = new_seq_len
        return logits[0, -1]

    def _free_paged(self) -> None:
        if self._block_table is not None:
            self.paged_cache.free(self._block_table)
            self._block_table = None
        self._paged_seq_len = 0

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: list[int],
        params: SamplingParams | None = None,
        use_cache: bool = True,
        use_paged: bool = False,
    ) -> list[int]:
        params = params or SamplingParams()
        ids = list(prompt_ids)
        out: list[int] = []

        if use_paged:
            last_logits = self._prefill_paged(prompt_ids)
            for _ in range(params.max_new_tokens):
                tok = self.sampler.sample(
                    last_logits, temperature=params.temperature,
                    top_k=params.top_k, top_p=params.top_p,
                )
                if tok in self.eos_ids:
                    break
                out.append(tok)
                last_logits = self._decode_paged(tok)
            self._free_paged()
            return out

        if use_cache:
            last_logits = self._prefill(prompt_ids)
            for _ in range(params.max_new_tokens):
                tok = self.sampler.sample(
                    last_logits, temperature=params.temperature,
                    top_k=params.top_k, top_p=params.top_p,
                )
                if tok in self.eos_ids:
                    break
                out.append(tok)
                last_logits = self._decode(tok)
        else:
            for _ in range(params.max_new_tokens):
                logits = self.forward_last_logits(ids)
                tok = self.sampler.sample(
                    logits, temperature=params.temperature,
                    top_k=params.top_k, top_p=params.top_p,
                )
                if tok in self.eos_ids:
                    break
                out.append(tok)
                ids.append(tok)
        return out
    def _build_prefill_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        b, s = attention_mask.shape
        causal = torch.tril(torch.ones(s, s, device=attention_mask.device, dtype=torch.bool))
        full = causal[None, None, :, :] & attention_mask[:, None, None, :].bool()
        mask = torch.zeros(b, 1, s, s, device=attention_mask.device, dtype=self.dtype)
        mask.masked_fill_(~full, float("-inf"))
        return mask

    def _build_decode_mask(self, valid_mask: torch.Tensor) -> torch.Tensor:
        b, total = valid_mask.shape
        mask = torch.zeros(b, 1, 1, total, device=valid_mask.device, dtype=self.dtype)
        mask.masked_fill_(~valid_mask[:, None, None, :].bool(), float("-inf"))
        return mask

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: list[list[int]],
        params: SamplingParams | None = None,
    ) -> tuple[list[list[int]], dict]:
        params = params or SamplingParams()
        b = len(prompts)
        prompt_lens = [len(p) for p in prompts]
        max_prompt = max(prompt_lens)

        input_ids = torch.full((b, max_prompt), self.pad_id, dtype=torch.long, device=self.device)
        attn = torch.zeros(b, max_prompt, device=self.device, dtype=self.dtype)
        for i, p in enumerate(prompts):
            input_ids[i, : len(p)] = torch.tensor(p, dtype=torch.long, device=self.device)
            attn[i, : len(p)] = 1.0

        batch_cache = KVCache(
            num_layers=self.config.num_hidden_layers,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            max_seq_len=self.kv_cache.max_seq_len,
            batch_size=b,
            dtype=self.dtype,
            device=self.device,
        )
        batch_cache.reset()

        prefill_pos = torch.zeros(b, max_prompt, dtype=torch.long, device=self.device)
        for i, l in enumerate(prompt_lens):
            prefill_pos[i, :l] = torch.arange(l, device=self.device)

        prefill_mask = self._build_prefill_mask(attn)
        last_idx = torch.tensor([l - 1 for l in prompt_lens], device=self.device)
        logits, _ = self.model(
            input_ids, kv_cache=batch_cache, is_prefill=True,
            cache_seq_len=0, attn_mask=prefill_mask, position_ids=prefill_pos,
            last_idx=last_idx,
        )
        batch_cache.seq_len = max_prompt

        last_logits = logits

        valid_mask = attn.clone()
        seq_lens = list(prompt_lens)
        outs: list[list[int]] = [[] for _ in range(b)]
        done = [False] * b
        total_tokens = sum(prompt_lens)

        for _ in range(params.max_new_tokens):
            tokens = []
            for i in range(b):
                if done[i]:
                    tokens.append(self.pad_id)
                    continue
                tok = self.sampler.sample(
                    last_logits[i], temperature=params.temperature,
                    top_k=params.top_k, top_p=params.top_p,
                )
                if tok in self.eos_ids:
                    done[i] = True
                else:
                    outs[i].append(tok)
                tokens.append(tok if not done[i] else self.pad_id)

            if all(done):
                break

            t = torch.tensor([tokens], dtype=torch.long, device=self.device).T
            decode_pos = torch.tensor([seq_lens], dtype=torch.long, device=self.device).T
            new_valid = torch.tensor(
                [[0 if done[i] else 1] for i in range(b)],
                device=self.device, dtype=self.dtype,
            )
            valid_mask = torch.cat([valid_mask, new_valid], dim=1)
            total_tokens += sum(0 if d else 1 for d in done)

            decode_mask = self._build_decode_mask(valid_mask)
            logits, _ = self.model(
                t, kv_cache=batch_cache, is_prefill=False,
                cache_seq_len=batch_cache.seq_len, attn_mask=decode_mask,
                position_ids=decode_pos,
            )
            batch_cache.seq_len += 1
            for i in range(b):
                if not done[i]:
                    seq_lens[i] += 1
            last_logits = logits[:, -1]

        waste = 1.0 - total_tokens / (b * batch_cache.seq_len) if batch_cache.seq_len > 0 else 1.0
        stats = {
            "batch_size": b,
            "max_prompt_len": max_prompt,
            "total_tokens": total_tokens,
            "batch_slots": b * batch_cache.seq_len,
            "padding_waste_rate": waste,
            "kv_waste_rate": batch_cache.waste_rate,
        }
        return outs, stats
