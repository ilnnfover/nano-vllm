"""P1 · Qwen2 模型实现（eager，无 KV cache），数值路径逐行对齐 HF modeling_qwen2.py。

对拍约定: hidden_states[0]=embedding 输出, [1..N]=各 decoder layer 输出（final norm 前）,
与 HF output_hidden_states 的索引语义一致, 供 golden/check_diff.py 使用。
权重命名**默认**与 HF 完全同名（model.* / lm_head.weight），加载近乎恒等映射；
P8 的 `prejoin=True` 会把加载改成「同名 + 三处合并」，见 `_prejoin_state_dict`。
"""
from __future__ import annotations

import re

import torch
from torch import nn
import torch.nn.functional as F

from nano_vllm.attention.backend import get_paged_attn
from nano_vllm.attention.metadata import AttentionMetadata
from nano_vllm.attention.triton_paged_attn import (
    paged_attention_triton_batch,
    paged_attention_triton_batch_tensor,
)
from nano_vllm.attention.varlen_prefill import varlen_prefill_attention
from nano_vllm.config import Qwen2Config
from nano_vllm.ops.fused_norm import fused_add_rms_norm, rms_norm
from nano_vllm.ops.rope import apply_rope


# ------------------------------------------------------- P8 · 权重预拼接（load 期）

_QKV_KEY = re.compile(
    r"^(?P<prefix>.+\.self_attn)\.(?P<proj>[qkv])_proj\.(?P<param>weight|bias)$"
)
_GATE_UP_KEY = re.compile(r"^(?P<prefix>.+\.mlp)\.(?P<proj>gate|up)_proj\.weight$")


def _prejoin_state_dict(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """P8 · 把 HF 的 q/k/v_proj 与 gate/up_proj 权重合并成 qkv_proj / gate_up_proj。

    **合并顺序必须与 `Attention._project_qkv` / `MLP.forward` 的切分一致**：

        qkv     = cat([q, k, v], dim=0)   ← 模型侧 split([q, k, v], dim=-1)
        gate_up = cat([gate, up], dim=0)  ← 模型侧 split(intermediate, dim=-1)

    两个显式校验（而不是静默跳过）：
      * Qwen2 的 QKV **带 bias**（Llama 没有，见 roadmap 附录坑点 1），weight 与 bias
        必须都拼齐，缺一个就报错；
      * 拼错顺序**不会崩，只会静默算错**，所以这里校验三件套齐全，绝不吞掉异常。

    不参与合并的键（embed_tokens / lm_head / norm 等）原样透传。
    """
    out: dict[str, torch.Tensor] = {}
    qkv: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    gate_up: dict[str, dict[str, torch.Tensor]] = {}

    for key, val in weights.items():
        m = _QKV_KEY.match(key)
        if m:
            qkv.setdefault((m["prefix"], m["param"]), {})[m["proj"]] = val
            continue
        m = _GATE_UP_KEY.match(key)
        if m:
            gate_up.setdefault(m["prefix"], {})[m["proj"]] = val
            continue
        out[key] = val

    for (prefix, param), parts in qkv.items():
        if set(parts) != {"q", "k", "v"}:
            raise KeyError(f"{prefix}.{param}: q/k/v 不齐，实际拿到 {sorted(parts)}")
        out[f"{prefix}.qkv_proj.{param}"] = torch.cat(
            [parts["q"], parts["k"], parts["v"]], dim=0
        )
    for prefix, parts in gate_up.items():
        if set(parts) != {"gate", "up"}:
            raise KeyError(f"{prefix}.gate_up_proj.weight: gate/up 不齐，实际拿到 {sorted(parts)}")
        out[f"{prefix}.gate_up_proj.weight"] = torch.cat([parts["gate"], parts["up"]], dim=0)
    return out


class RMSNorm(nn.Module):
    """三实现对拍（`impl` 由 `NanoRunner(norm_impl=...)` 选择）：

    | impl | 实现 | 每层 norm 相关 kernel |
    | --- | --- | --- |
    | `torch` | 逐 op（与 HF `Qwen2RMSNorm` 逐行对齐）—— **oracle**，不参与性能路径 | 8/次 |
    | `lib` | `F.rms_norm`（库融合核） | 1/次 |
    | `triton` | `nano_vllm/ops/fused_norm.py` 自研核，**并把残差加一起融掉** | 1/次（含加） |
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, impl: str = "triton") -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.impl = impl

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """纯 RMSNorm（没有可融合的残差加）。第 0 层的 `input_layernorm` 用它。"""
        if self.impl == "triton":
            return rms_norm(hidden_states, self.weight, self.variance_epsilon)
        if self.impl == "lib":
            return F.rms_norm(
                hidden_states, (self.weight.numel(),), self.weight, self.variance_epsilon
            )
        return self._op_norm(hidden_states)

    def forward_with_residual(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """融合「残差加」与随后的 RMSNorm，返回 `(norm(x), x)`，其中 `x = residual + hidden_states`。

        **延迟残差设计的落点**：第二个返回值要作为下一个残差段继续往下传。本层欠的
        `y = residual + mlp_out` 这个加法，被推迟到下一层（或 `Qwen2Model.norm`）的 norm 里
        顺手做掉 —— 于是每层的两次「加 + 归一」各变成**一次**调用，而不是两次。
        """
        if self.impl == "triton":
            # 一个核同时算 x+residual、把 x 存回、并对它做归一
            return fused_add_rms_norm(
                hidden_states, residual, self.weight, self.variance_epsilon
            )
        x = residual + hidden_states
        if self.impl == "lib":
            return (
                F.rms_norm(x, (self.weight.numel(),), self.weight, self.variance_epsilon),
                x,
            )
        return self._op_norm(x), x

    def _op_norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """逐 op 版（oracle）：与 HF `Qwen2RMSNorm` 逐行对齐，供另外两条路径对拍。

        注意它在**加法之后才转 bf16**：`x` 先落成 bf16、再读回来算方差，比融合核多一次舍入
        （融合核让加法与归约共享同一份 fp32 中间值）。所以对拍时它是"精度较低的一方"，
        详见 `docs/notes/p8-triton-norm.md`。
        """
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
        # 显式注解赋值：静态检查器看不见 register_buffer 建立的动态属性（否则 inv_freq 被当成
        # Module）。该名字已在 `_buffers` 中，故 `Module.__setattr__` 仍写回缓冲区 →
        # 设备迁移 / state_dict 行为不变（persistent=False 本就不进 state_dict）。
        self.inv_freq: torch.Tensor = inv_freq

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
    def __init__(
        self, config: Qwen2Config, prejoin: bool = True, rope_impl: str = "triton"
    ) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = config.num_kv_groups
        self.head_dim = config.head_dim
        self.scaling = config.head_dim**-0.5
        self.prejoin = prejoin
        self.rope_impl = rope_impl
        if prejoin:
            # P8 · 预拼接：3 次 GEMM → 1 次。输出维按 [q, k, v] 排列，与
            # `_prejoin_state_dict` 的 cat 顺序、`_project_qkv` 的 split 顺序三者一致。
            self.qkv_proj = nn.Linear(
                config.hidden_size,
                (config.num_attention_heads + 2 * config.num_key_value_heads) * config.head_dim,
                bias=True,
            )
        else:
            self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=True)
            self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=True)
            self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * config.head_dim, config.hidden_size, bias=False)

    def _project_qkv(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """QKV 投影 → `[b, heads, s, head_dim]` 三件套（预拼接 / 原始三投影两条路径）。

        预拼接路径用 `unflatten` 而**不是** `view`：`split` 出来的切片不是连续张量
        （行 stride 仍是合并后的总宽），`view` 会直接报错；而 `unflatten` 只作用于
        最后一维（该维 stride 恒为 1），所以仍是**零拷贝视图**，不会引入额外 copy kernel
        ——否则「3 GEMM 合 1」省下的开销会被切片的拷贝 kernel 吃回去。
        """
        if self.prejoin:
            qkv = self.qkv_proj(hidden_states)
            q, k, v = qkv.split(
                (
                    self.num_heads * self.head_dim,
                    self.num_kv_heads * self.head_dim,
                    self.num_kv_heads * self.head_dim,
                ),
                dim=-1,
            )
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
        return (
            q.unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2),
            k.unflatten(-1, (self.num_kv_heads, self.head_dim)).transpose(1, 2),
            v.unflatten(-1, (self.num_kv_heads, self.head_dim)).transpose(1, 2),
        )

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
        """P3/P4 分页路径（单条 b=1）。

        prefill 首块: K/V 写入物理块，SDPA is_causal=True
        prefill 续块 (P4 chunked): K/V 写入物理块，读全部 KV（历史+当前），混合 mask
        decode:  新 K/V 写入物理块，block_table 间接寻址读全部历史 KV
        """
        if metadata is None:
            raise ValueError("分页路径必须传 AttentionMetadata")
        if metadata.is_prefill and b > 1:
            raise ValueError(f"分页 prefill 暂只支持 b=1, got b={b}")
        if (
            not metadata.is_prefill
            and b > 1
            and metadata.block_tables is None
            and metadata.block_table_tensor is None
        ):
            # P7: 批量 decode 可用 list 寻址（block_tables）或张量寻址（block_table_tensor）
            raise ValueError("批量 decode 必须传 block_tables 或 block_table_tensor")
        paged_cache.write(layer_idx, k[0] if b == 1 else k.squeeze(2).transpose(0, 1),
                          v[0] if b == 1 else v.squeeze(2).transpose(0, 1),
                          metadata.slot_mapping)
        if metadata.is_prefill:
            if metadata.qo_indptr is not None:
                if (
                    metadata.paged_kv_indptr is None
                    or metadata.paged_kv_indices is None
                    or metadata.paged_kv_last_page_len is None
                ):
                    raise ValueError(
                        "varlen prefill 必须同时提供 qo_indptr 与三个 paged_kv_* 元数据"
                    )
                # P4 varlen 拼批 prefill: flat 拼接多条请求，一次算完
                q_flat = q[0].transpose(0, 1).contiguous()  # [total_q, num_heads, head_dim]
                out_flat = varlen_prefill_attention(
                    q_flat,
                    paged_cache.k_cache[layer_idx],
                    paged_cache.v_cache[layer_idx],
                    metadata.qo_indptr,
                    metadata.paged_kv_indptr,
                    metadata.paged_kv_indices,
                    metadata.paged_kv_last_page_len,
                    self.num_kv_heads,
                    self.scaling,
                    impl=metadata.prefill_impl,
                )
                out = out_flat.transpose(0, 1).unsqueeze(0)  # [1, num_heads, total_q, head_dim]
            elif metadata.seq_len > s:
                k_all, v_all = paged_cache.read_blocks(layer_idx, metadata.block_table)
                k_all = k_all[:, :metadata.seq_len, :].unsqueeze(0)
                v_all = v_all[:, :metadata.seq_len, :].unsqueeze(0)
                cache_seq_len = metadata.seq_len - s
                mask = torch.zeros(s, metadata.seq_len, device=q.device, dtype=q.dtype)
                mask[:, cache_seq_len:] = torch.triu(
                    torch.full((s, s), float("-inf"), device=q.device, dtype=q.dtype),
                    diagonal=1,
                )
                out = self._sdpa(q, k_all, v_all, attn_mask=mask.unsqueeze(0).unsqueeze(0), is_causal=False)
            else:
                out = self._sdpa(q, k, v, attn_mask=None, is_causal=True)
        else:
            # P7: 张量寻址路径（图捕获）——block_table/seq_lens 来自静态 buffer
            if metadata.block_table_tensor is not None:
                if metadata.seq_lens_tensor is None:
                    raise ValueError("张量寻址路径必须同时提供 seq_lens_tensor")
                return self.o_proj(
                    paged_attention_triton_batch_tensor(
                        q,
                        paged_cache.k_cache[layer_idx],
                        paged_cache.v_cache[layer_idx],
                        metadata.block_table_tensor,
                        metadata.seq_lens_tensor,
                        self.num_kv_heads,
                        self.scaling,
                    ).transpose(1, 2).reshape(b, s, -1)
                )
            attn_fn = get_paged_attn(metadata.attn_impl)
            if b == 1:
                if metadata.block_table is None:
                    raise ValueError("单条 decode 必须传 block_table")
                out = attn_fn(
                    q[0],
                    paged_cache.k_cache[layer_idx],
                    paged_cache.v_cache[layer_idx],
                    metadata.block_table,
                    metadata.seq_len,
                    self.num_kv_heads,
                    self.scaling,
                ).unsqueeze(0)
            else:
                block_tables, seq_lens = metadata.block_tables, metadata.seq_lens
                if block_tables is None or seq_lens is None:
                    raise ValueError("批量 decode 必须传 block_tables 与 seq_lens")
                if metadata.attn_impl == "triton":
                    out = paged_attention_triton_batch(
                        q,
                        paged_cache.k_cache[layer_idx],
                        paged_cache.v_cache[layer_idx],
                        block_tables,
                        seq_lens,
                        self.num_kv_heads,
                        self.scaling,
                    )
                else:
                    outs = []
                    for i in range(b):
                        outs.append(attn_fn(
                            q[i],
                            paged_cache.k_cache[layer_idx],
                            paged_cache.v_cache[layer_idx],
                            block_tables[i],
                            seq_lens[i],
                            self.num_kv_heads,
                            self.scaling,
                        ))
                    out = torch.stack(outs)
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
        q, k, v = self._project_qkv(hidden_states)

        cos, sin = position_embeddings
        if self.rope_impl == "triton":
            # P8 ③ · 自研融合核：整段 RoPE 一个 kernel（torch 参考要 ~4 个/层）
            q, k = apply_rope(q, k, cos, sin)
        else:
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if paged_cache is not None:
            return self._forward_paged(q, k, v, paged_cache, layer_idx, metadata, b, s)

        # ---- P2 连续 cache 路径（对拍基线，保持原逻辑）----
        if kv_cache is not None:
            kv_cache.write(layer_idx, k, v, cache_seq_len)
            total_len = cache_seq_len + s
            if is_prefill and cache_seq_len > 0:
                # P0-01 修复: chunked prefill 续段——读全部 KV，构造混合 mask
                # 历史 KV 全可见 + 当前 chunk 内 causal
                k_attn, v_attn = kv_cache.read(layer_idx, total_len)
                chunk_mask = torch.zeros(s, total_len, device=q.device, dtype=q.dtype)
                chunk_mask[:, cache_seq_len:] = torch.triu(
                    torch.full((s, s), float("-inf"), device=q.device, dtype=q.dtype),
                    diagonal=1,
                )
                attn_mask = chunk_mask.unsqueeze(0).unsqueeze(0)
            elif is_prefill:
                k_attn, v_attn = k, v
            else:
                k_attn, v_attn = kv_cache.read(layer_idx, total_len)
        else:
            k_attn, v_attn = k, v

        use_mask = attn_mask is not None
        is_causal = is_prefill and cache_seq_len == 0 and not use_mask
        out = self._sdpa(q, k_attn, v_attn, attn_mask, is_causal)
        out = out.transpose(1, 2).reshape(b, s, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, config: Qwen2Config, prejoin: bool = True) -> None:
        super().__init__()
        self.prejoin = prejoin
        self.intermediate_size = config.intermediate_size
        if prejoin:
            # P8 · 预拼接：2 次 GEMM → 1 次。输出维按 [gate, up] 排列。
            self.gate_up_proj = nn.Linear(
                config.hidden_size, 2 * config.intermediate_size, bias=False
            )
        else:
            self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.prejoin:
            # 与 `_prejoin_state_dict` 的 cat([gate, up]) 顺序对应
            gate, up = self.gate_up_proj(x).split(self.intermediate_size, dim=-1)
        else:
            gate, up = self.gate_proj(x), self.up_proj(x)
        return self.down_proj(self.act_fn(gate) * up)


class DecoderLayer(nn.Module):
    """**延迟残差（deferred residual）** 版 decoder layer。

    与教科书写法（每层自己收口两次 `x = residual + sublayer(x)`）的区别：

        教科书:  residual = x; h = norm(x);  h = attn(h);  x = residual + h      ← 独立 add
                 residual = x; h = norm(x);  h = mlp(h);   x = residual + h      ← 独立 add
        本实现:  (h, x) = fused_add_norm(attn_out, residual)   ← 加+归一并成一个核
                 (h, x) = fused_add_norm(mlp_out,  residual)   ← 同上
                 最后把「欠的」最后一次加交给下一层（或 final norm）收口

    每层因此少两次独立 `aten::add`（以及它带的一次显存往返）。对齐 vLLM
    `LlamaDecoderLayer.forward` 的做法：`residual` 作为**入参**传进来。
    """

    def __init__(
        self,
        config: Qwen2Config,
        prejoin: bool = True,
        norm_impl: str = "triton",
        rope_impl: str = "triton",
    ) -> None:
        super().__init__()
        self.self_attn = Attention(config, prejoin, rope_impl)
        self.mlp = MLP(config, prejoin)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, norm_impl)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps, norm_impl
        )
        # 延迟残差只有在 norm 能融合「加」时才省得下来；torch 逐 op 路径下等价于原写法
        self.deferred_residual = norm_impl in ("triton", "lib")

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
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """返回 `(hidden_states, residual)`；调用方负责最后的 `residual + hidden_states`。

        `residual is None` 只出现在第 0 层：那时还没有欠账的残差，`input_layernorm`
        无从融合（这一层的 norm 只能单独跑）——对齐 vLLM 同处的分支。
        """
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm.forward_with_residual(
                hidden_states, residual
            )

        hidden_states = self.self_attn(
            hidden_states, position_embeddings,
            kv_cache=kv_cache, layer_idx=layer_idx,
            is_prefill=is_prefill, cache_seq_len=cache_seq_len,
            attn_mask=attn_mask, paged_cache=paged_cache, metadata=metadata,
        )
        hidden_states, residual = self.post_attention_layernorm.forward_with_residual(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        # 「y = residual + mlp_out」不在这里做：留给下一层的 norm（或 final norm）收口
        return hidden_states, residual


class Qwen2Model(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        prejoin: bool = True,
        norm_impl: str = "triton",
        rope_impl: str = "triton",
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            DecoderLayer(config, prejoin, norm_impl, rope_impl)
            for _ in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, norm_impl)
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

        all_hidden: list[torch.Tensor] | None = (
            [hidden_states] if output_hidden_states else None
        )
        # 延迟残差：residual 逐层往下传，每层「欠」的那次加法由下一层（最后一层由 final norm）收口
        residual: torch.Tensor | None = None
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states, position_embeddings,
                kv_cache=kv_cache, layer_idx=layer_idx,
                is_prefill=is_prefill, cache_seq_len=cache_seq_len,
                attn_mask=attn_mask, paged_cache=paged_cache, metadata=metadata,
                residual=residual,
            )
            if all_hidden is not None:
                # 本层的真实输出是「收口后」的值 = residual + hidden_states。
                # 注意：只有开 output_hidden_states（对拍用，非性能路径）时才会付这一次 add。
                all_hidden.append(
                    hidden_states if residual is None else residual + hidden_states
                )

        # 最后一层欠的加法与 final norm 一起收口
        hidden_states = self.norm.forward_with_residual(hidden_states, residual)[0]
        if all_hidden is not None:
            all_hidden[-1] = hidden_states
        return hidden_states, all_hidden


class Qwen2ForCausalLM(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        prejoin: bool = True,
        norm_impl: str = "triton",
        rope_impl: str = "triton",
    ) -> None:
        super().__init__()
        self.config = config
        self.prejoin = prejoin
        self.norm_impl = norm_impl
        self.rope_impl = rope_impl
        self.model = Qwen2Model(config, prejoin, norm_impl, rope_impl)
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
        skip_lm_head: bool = False,
    ):
        hidden_states, all_hidden = self.model(
            input_ids, output_hidden_states,
            kv_cache=kv_cache, is_prefill=is_prefill,
            cache_seq_len=cache_seq_len, attn_mask=attn_mask,
            position_ids=position_ids,
            paged_cache=paged_cache, metadata=metadata,
        )
        if skip_lm_head:
            # P1 · 图捕获路径：图只到 final norm 后的 hidden_states，logits 在图外算
            # （见 `compute_logits`）。理由：**静态形状下图内躲不掉整桶计算**——
            # 即便传 last_idx 也必须是固定索引，而 padding 行就落在 [0, bucket) 里，
            # 照样白算 lm_head。移出图后只算有效行 n。
            return hidden_states, all_hidden
        if last_idx is not None:
            b = hidden_states.shape[0]
            selected = hidden_states[torch.arange(b, device=hidden_states.device), last_idx]
            logits = self.lm_head(selected)
        else:
            logits = self.lm_head(hidden_states)
        return logits, all_hidden

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """P1 · 图外 logits 投影：`[n, hidden_size]` → `[n, vocab_size]`。

        对齐 vLLM `self.model.compute_logits(sample_hidden_states)`
        （`gpu_model_runner.py:4509-4510`，其前置是 `hidden_states[logits_indices]`）。

        与图内版本数值同源（同一个 `lm_head`），区别只在**行数**：图内受静态形状
        限制必须按整桶算，这里只算有效行。
        """
        return self.lm_head(hidden_states)

    @torch.no_grad()
    def load_weights(self, model_dir: str) -> None:
        from safetensors.torch import load_file

        weights = load_file(f"{model_dir}/model.safetensors")
        if self.config.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
            weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
        if self.prejoin:
            # P8 · 键重映射必须在 `load_state_dict` **之前**：下方对 unexpected 键直接
            # raise，原始的 q/k/v_proj（3×28 层 ×weight+bias）若不在这里被消费掉，
            # 会被判为「权重文件中存在未知键」而加载失败。
            weights = _prejoin_state_dict(weights)
        missing, unexpected = self.load_state_dict(weights, strict=False)
        if unexpected:
            raise KeyError(f"权重文件中存在未知键: {unexpected[:5]}")
        if missing:
            raise KeyError(f"权重文件缺少键: {missing[:5]}")