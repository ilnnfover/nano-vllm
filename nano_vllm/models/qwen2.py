"""P1 · Qwen2 模型实现（eager，无 KV cache），数值路径逐行对齐 HF modeling_qwen2.py。

对拍约定: hidden_states[0]=embedding 输出, [1..N]=各 decoder layer 输出（final norm 前）,
与 HF output_hidden_states 的索引语义一致, 供 golden/check_diff.py 使用。
权重命名与 HF 完全同名（model.* / lm_head.weight）, 加载近乎恒等映射。
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from nano_vllm.attention.backend import get_paged_attn
from nano_vllm.attention.metadata import AttentionMetadata
from nano_vllm.config import Qwen2Config


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, rope_theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return x.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class Attention(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = config.num_kv_groups
        self.head_dim = config.head_dim
        self.scaling = config.head_dim**-0.5
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * config.head_dim, config.hidden_size, bias=False)

    def _sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None,
        is_causal: bool,
    ) -> torch.Tensor:
        """SDPA 封装：CUDA + GQA + head_dim<=256 时走 enable_gqa 融合，否则手动 repeat_kv。"""
        if self.num_kv_groups > 1 and q.is_cuda and self.head_dim <= 256:
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask,
                scale=self.scaling, is_causal=is_causal, enable_gqa=True,
            )
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            scale=self.scaling, is_causal=is_causal,
        )

    def _forward_paged(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        paged_cache,
        layer_idx: int,
        metadata: AttentionMetadata | None,
        b: int,
        s: int,
    ) -> torch.Tensor:
        """P3 分页路径（单条 b=1）。

        prefill: K/V 按 slot_mapping 写入物理块，attention 用当前 chunk + SDPA is_causal=True
        decode:  新 K/V 写入物理块，attention 用 block_table 间接寻址读全部历史 KV
        """
        if metadata is None:
            raise ValueError("分页路径必须传 AttentionMetadata")
        if b != 1:
            raise ValueError(f"P3 分页路径只支持单条 b=1, got b={b}")
        paged_cache.write(layer_idx, k[0], v[0], metadata.slot_mapping)
        if metadata.is_prefill:
            out = self._sdpa(q, k, v, attn_mask=None, is_causal=True)
        else:
            attn_fn = get_paged_attn(metadata.attn_impl)
            out = attn_fn(
                q[0],
                paged_cache.k_cache[layer_idx],
                paged_cache.v_cache[layer_idx],
                metadata.block_table,
                metadata.seq_len,
                self.num_kv_heads,
                self.scaling,
            ).unsqueeze(0)
        return self.o_proj(out.transpose(1, 2).reshape(b, s, -1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings,
        kv_cache=None,
        layer_idx: int = 0,
        is_prefill: bool = True,
        cache_seq_len: int = 0,
        attn_mask: torch.Tensor | None = None,
        paged_cache=None,
        metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        b, s, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if paged_cache is not None:
            return self._forward_paged(q, k, v, paged_cache, layer_idx, metadata, b, s)

        # ---- P2 连续 cache 路径（对拍基线，保持原逻辑）----
        if kv_cache is not None:
            kv_cache.write(layer_idx, k, v, cache_seq_len)
            total_len = cache_seq_len + s
            if is_prefill:
                k_attn, v_attn = k, v
            else:
                k_attn, v_attn = kv_cache.read(layer_idx, total_len)
        else:
            k_attn, v_attn = k, v

        use_mask = attn_mask is not None
        is_causal = is_prefill and not use_mask
        out = self._sdpa(q, k_attn, v_attn, attn_mask, is_causal)
        out = out.transpose(1, 2).reshape(b, s, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings,
        kv_cache=None,
        layer_idx: int = 0,
        is_prefill: bool = True,
        cache_seq_len: int = 0,
        attn_mask: torch.Tensor | None = None,
        paged_cache=None,
        metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, position_embeddings,
            kv_cache=kv_cache, layer_idx=layer_idx,
            is_prefill=is_prefill, cache_seq_len=cache_seq_len,
            attn_mask=attn_mask, paged_cache=paged_cache, metadata=metadata,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen2Model(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.rope_theta)

    def forward(
        self,
        input_ids: torch.Tensor,
        output_hidden_states: bool = False,
        kv_cache=None,
        is_prefill: bool = True,
        cache_seq_len: int = 0,
        attn_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        paged_cache=None,
        metadata: AttentionMetadata | None = None,
    ):
        b, s = input_ids.shape
        if position_ids is None:
            start = 0 if is_prefill else cache_seq_len
            position_ids = torch.arange(start, start + s, device=input_ids.device).unsqueeze(0).expand(b, -1)
        hidden_states = self.embed_tokens(input_ids)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden = [hidden_states] if output_hidden_states else None
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states, position_embeddings,
                kv_cache=kv_cache, layer_idx=layer_idx,
                is_prefill=is_prefill, cache_seq_len=cache_seq_len,
                attn_mask=attn_mask, paged_cache=paged_cache, metadata=metadata,
            )
            if output_hidden_states:
                all_hidden.append(hidden_states)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden[-1] = hidden_states
        return hidden_states, all_hidden


class Qwen2ForCausalLM(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        output_hidden_states: bool = False,
        kv_cache=None,
        is_prefill: bool = True,
        cache_seq_len: int = 0,
        attn_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        last_idx: torch.Tensor | None = None,
        paged_cache=None,
        metadata: AttentionMetadata | None = None,
    ):
        hidden_states, all_hidden = self.model(
            input_ids, output_hidden_states,
            kv_cache=kv_cache, is_prefill=is_prefill,
            cache_seq_len=cache_seq_len, attn_mask=attn_mask,
            position_ids=position_ids,
            paged_cache=paged_cache, metadata=metadata,
        )
        if last_idx is not None:
            b = hidden_states.shape[0]
            selected = hidden_states[torch.arange(b, device=hidden_states.device), last_idx]
            logits = self.lm_head(selected)
        else:
            logits = self.lm_head(hidden_states)
        return logits, all_hidden

    @torch.no_grad()
    def load_weights(self, model_dir: str) -> None:
        from safetensors.torch import load_file

        weights = load_file(f"{model_dir}/model.safetensors")
        if self.config.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
            weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
        missing, unexpected = self.load_state_dict(weights, strict=False)
        if unexpected:
            raise KeyError(f"权重文件中存在未知键: {unexpected[:5]}")
        if missing:
            raise KeyError(f"权重文件缺少键: {missing[:5]}")