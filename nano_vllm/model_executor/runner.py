"""P3+ · 执行器：分页 KV cache（BlockPool + block_table + slot_mapping），
decode attention 可选 torch 朴素 / sdpa / 自研 Triton kernel（attn_impl，见 attention/backend.py）。
其中 CUDA Graph（P7）只支持 `triton`；缺省 `enable_cudagraph` 据此自动决定。

P1 eager / P2 连续 cache / P2 静态批的历史入口已拆到 legacy_api.py（E5），
本模块保留薄委托转发以维持向后兼容。
"""
from __future__ import annotations

from pathlib import Path

import torch

from nano_vllm.attention.metadata import AttentionMetadata
from nano_vllm.config import Qwen2Config
from nano_vllm.kvmm.paged_kv_cache import PagedKVCache
from nano_vllm.engine.sequence import SamplingParams  # re-export for backward compat
from nano_vllm.kv_cache import KVCache
from nano_vllm.model_executor.legacy_api import (
    forward_last_logits as _fwd_last_logits,
    generate_batch as _gen_batch,
    generate_cached as _gen_cached,
    generate_eager as _gen_eager,
)
from nano_vllm.models.qwen2 import Qwen2ForCausalLM
from nano_vllm.sample.sampler import Sampler



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
        prejoin: bool = True,
        fused_norm: bool = True,
        enable_cudagraph: bool | None = None,
        cudagraph_buckets: tuple[int, ...] | None = None,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.block_size = block_size
        self.attn_impl = attn_impl
        self.prefill_impl = prefill_impl
        self.prejoin = prejoin
        self.fused_norm = fused_norm
        self.enable_cudagraph = (
            (device == "cuda" and torch.cuda.is_available() and attn_impl == "triton")
            if enable_cudagraph is None
            else enable_cudagraph
        )
        self.cudagraph_buckets = cudagraph_buckets
        self.config = Qwen2Config.from_json(Path(model_path) / "config.json")
        self.model = Qwen2ForCausalLM(
            self.config, prejoin=prejoin, fused_norm=fused_norm
        ).to(device=device, dtype=dtype).eval()
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
        # P3 分页 KV cache：num_blocks 缺省时按显存反推（B1）
        if num_blocks is None:
            num_blocks = self._profile_num_blocks(block_size, dtype, device)
        self.paged_cache = PagedKVCache(
            num_layers=self.config.num_hidden_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            dtype=dtype,
            device=device,
        )
        self._block_table: list[int] | None = None
        self._paged_seq_len = 0
        self.pad_id = 0
        # P7 · decode CUDA Graph（懒捕获：首次回放时才 capture，见 DecodeGraphRunner）
        self.graph_runner = None
        if self.enable_cudagraph:
            from nano_vllm.cudagraph import DEFAULT_BUCKETS, DecodeGraphRunner

            self.graph_runner = DecodeGraphRunner(
                self.model,
                self.paged_cache,
                self.config,
                buckets=cudagraph_buckets or DEFAULT_BUCKETS,
                attn_impl=attn_impl,
                device=device,
                dtype=dtype,
            )

    def _profile_num_blocks(self, block_size: int, dtype: torch.dtype, device: str) -> int:
        """按剩余显存反推 num_blocks（B1）。CPU 回退到单条够用的默认值。"""
        fallback = PagedKVCache.blocks_needed(self.config.max_position_embeddings, block_size) + 1
        if device != "cuda" or not torch.cuda.is_available():
            return fallback
        free_bytes, _ = torch.cuda.mem_get_info()
        per_token_kv = (
            self.config.num_hidden_layers * 2 * self.config.num_key_value_heads
            * self.config.head_dim * dtype.itemsize
        )
        per_block = block_size * per_token_kv
        num_blocks = int(free_bytes * 0.5 / per_block)
        return max(num_blocks, fallback)

    @property
    def eos_ids(self) -> set[int]:
        return self.config.eos_ids

    @torch.no_grad()
    def forward_last_logits(self, ids: list[int]) -> torch.Tensor:
        return _fwd_last_logits(self.model, ids, self.device)

    @torch.no_grad()
    def _prefill(self, prompt_ids: list[int]) -> torch.Tensor:
        """P2 连续 cache prefill（chunked prefill 测试用）。"""
        self.kv_cache.reset()
        t = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        last_idx = torch.tensor([len(prompt_ids) - 1], device=self.device)
        logits, _ = self.model(t, kv_cache=self.kv_cache, is_prefill=True, cache_seq_len=0, last_idx=last_idx)
        self.kv_cache.seq_len = len(prompt_ids)
        return logits[0]

    @torch.no_grad()
    def _decode(self, tok: int) -> torch.Tensor:
        """P2 连续 cache decode（chunked prefill 测试用）。"""
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
            seq_len=seq_len, prefill_impl=self.prefill_impl,
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

        if use_paged:
            out: list[int] = []
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
            return _gen_cached(self, prompt_ids, params)
        return _gen_eager(self, prompt_ids, params)

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: list[list[int]],
        params: SamplingParams | None = None,
    ) -> tuple[list[list[int]], dict]:
        params = params or SamplingParams()
        return _gen_batch(self, prompts, params)
