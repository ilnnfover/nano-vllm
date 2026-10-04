"""P8 ① 权重预拼接的单测（**免模型**：不加载真实权重、不需要 CUDA）。

为什么单独写：
  * 拼接**顺序**错了不会崩、只会静默算错——这是本项最危险的点，必须用测试锁住；
  * Qwen2 的 QKV **带 bias**（Llama 没有），weight 与 bias 必须走同一套顺序，缺一即报错；
  * 键重映射发生在 `load_state_dict` 之前，一旦漏消费原始键，加载会因「未知键」失败。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from nano_vllm.config import Qwen2Config
from nano_vllm.models.qwen2 import MLP, Attention, _prejoin_state_dict


def tiny_config(**over) -> Qwen2Config:
    base = dict(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        intermediate_size=64, vocab_size=128, rms_norm_eps=1e-6, rope_theta=1e4,
        hidden_act="silu", tie_word_embeddings=False, max_position_embeddings=128,
        head_dim=8, eos_token_id=0, bos_token_id=None,
    )
    base.update(over)
    return Qwen2Config(**base)


class TestPrejoinStateDict(unittest.TestCase):
    """`_prejoin_state_dict`：键重映射与顺序。"""

    def _hf_weights(self) -> dict[str, torch.Tensor]:
        torch.manual_seed(0)
        return {
            "model.embed_tokens.weight": torch.randn(128, 32),
            "model.layers.0.self_attn.q_proj.weight": torch.full((32, 32), 1.0),
            "model.layers.0.self_attn.k_proj.weight": torch.full((16, 32), 2.0),
            "model.layers.0.self_attn.v_proj.weight": torch.full((16, 32), 3.0),
            "model.layers.0.self_attn.q_proj.bias": torch.full((32,), 4.0),
            "model.layers.0.self_attn.k_proj.bias": torch.full((16,), 5.0),
            "model.layers.0.self_attn.v_proj.bias": torch.full((16,), 6.0),
            "model.layers.0.self_attn.o_proj.weight": torch.randn(32, 32),
            "model.layers.0.mlp.gate_proj.weight": torch.full((64, 32), 7.0),
            "model.layers.0.mlp.up_proj.weight": torch.full((64, 32), 8.0),
            "model.layers.0.mlp.down_proj.weight": torch.randn(32, 64),
        }

    def test_join_order_and_bias(self) -> None:
        """顺序必须是 [q, k, v] / [gate, up]；weight 与 bias 都要拼。"""
        out = _prejoin_state_dict(self._hf_weights())

        qkv_w = out["model.layers.0.self_attn.qkv_proj.weight"]
        self.assertEqual(tuple(qkv_w.shape), (64, 32))
        self.assertTrue(torch.all(qkv_w[:32] == 1.0))   # q
        self.assertTrue(torch.all(qkv_w[32:48] == 2.0))  # k
        self.assertTrue(torch.all(qkv_w[48:] == 3.0))    # v

        qkv_b = out["model.layers.0.self_attn.qkv_proj.bias"]
        self.assertEqual(tuple(qkv_b.shape), (64,))
        self.assertTrue(torch.all(qkv_b[:32] == 4.0))
        self.assertTrue(torch.all(qkv_b[32:48] == 5.0))
        self.assertTrue(torch.all(qkv_b[48:] == 6.0))

        gu = out["model.layers.0.mlp.gate_up_proj.weight"]
        self.assertEqual(tuple(gu.shape), (128, 32))
        self.assertTrue(torch.all(gu[:64] == 7.0))   # gate
        self.assertTrue(torch.all(gu[64:] == 8.0))   # up

    def test_original_keys_consumed(self) -> None:
        """原始 q/k/v、gate/up 键必须被消费掉，否则 load_state_dict 会报「未知键」。"""
        out = _prejoin_state_dict(self._hf_weights())
        for stale in (
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.bias",
            "model.layers.0.mlp.gate_proj.weight",
        ):
            self.assertNotIn(stale, out)

    def test_unrelated_keys_passthrough(self) -> None:
        out = _prejoin_state_dict(self._hf_weights())
        for keep in (
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
        ):
            self.assertIn(keep, out)

    def test_missing_bias_raises(self) -> None:
        """Qwen2 的 QKV 带 bias：缺一个就必须显式失败，绝不静默跳过。"""
        w = self._hf_weights()
        del w["model.layers.0.self_attn.k_proj.bias"]
        with self.assertRaises(KeyError) as cm:
            _prejoin_state_dict(w)
        self.assertIn("q/k/v 不齐", str(cm.exception))

    def test_missing_gate_raises(self) -> None:
        w = self._hf_weights()
        del w["model.layers.0.mlp.up_proj.weight"]
        with self.assertRaises(KeyError) as cm:
            _prejoin_state_dict(w)
        self.assertIn("gate/up 不齐", str(cm.exception))


class TestPrejoinForwardEquivalence(unittest.TestCase):
    """`prejoin=True` 的前向必须与三段投影/两段投影**数值等价**（CPU fp32，非 bf16）。"""

    def setUp(self) -> None:
        self.cfg = tiny_config()

    def test_attention_projection_matches_three_linears(self) -> None:
        torch.manual_seed(0)
        ref = Attention(self.cfg, prejoin=False)
        new = Attention(self.cfg, prejoin=True)
        with torch.no_grad():
            new.qkv_proj.weight.copy_(
                torch.cat([ref.q_proj.weight, ref.k_proj.weight, ref.v_proj.weight], dim=0)
            )
            new.qkv_proj.bias.copy_(
                torch.cat([ref.q_proj.bias, ref.k_proj.bias, ref.v_proj.bias], dim=0)
            )
            x = torch.randn(2, 5, self.cfg.hidden_size)
            expect = (ref.q_proj(x), ref.k_proj(x), ref.v_proj(x))
            q, k, v = new._project_qkv(x)
            got = (
                q.transpose(1, 2).reshape(2, 5, -1),
                k.transpose(1, 2).reshape(2, 5, -1),
                v.transpose(1, 2).reshape(2, 5, -1),
            )
        for name, e, g in zip("qkv", expect, got):
            self.assertEqual(e.shape, g.shape, name)
            # 切片是**视图**（unflatten 而非 view/copy）：fp32 下应逐位相同或仅差 GEMM 形状的舍入
            self.assertTrue(torch.allclose(e, g, atol=1e-5, rtol=0), f"{name}: {(e - g).abs().max()}")

    def test_attention_projection_needs_no_copy(self) -> None:
        """预拼接路径的 q/k/v 必须是**切片视图**而非拷贝，否则省下的 GEMM 会被拷贝 kernel 吃回去。

        判据用 stride：合并缓冲区形状 `[b, s, total]`（total = q+k+v 宽），
        `unflatten` + `transpose` 之后仍是它的视图，故 **s 维 stride == total**；
        若实现改用了 `contiguous()` / `.reshape()`（会拷贝），s 维 stride 会退化成 `head_dim`。
        """
        cfg = self.cfg
        new = Attention(cfg, prejoin=True)
        x = torch.randn(1, 3, cfg.hidden_size)
        with torch.no_grad():
            q, k, v = new._project_qkv(x)
        s = x.shape[1]
        total = (cfg.num_attention_heads + 2 * cfg.num_key_value_heads) * cfg.head_dim
        for name, t in (("q", q), ("k", k), ("v", v)):
            self.assertEqual(t.shape, (1, cfg.num_attention_heads if name == "q"
                                       else cfg.num_key_value_heads, s, cfg.head_dim), name)
            self.assertEqual(t.stride(3), 1, name)
            self.assertEqual(t.stride(2), total, f"{name}: s 维 stride 应为合并后行宽（视图）")
            self.assertEqual(t.stride(0), s * total, name)

    def test_mlp_matches_two_linears(self) -> None:
        torch.manual_seed(0)
        ref = MLP(self.cfg, prejoin=False)
        new = MLP(self.cfg, prejoin=True)
        with torch.no_grad():
            new.gate_up_proj.weight.copy_(
                torch.cat([ref.gate_proj.weight, ref.up_proj.weight], dim=0)
            )
            new.down_proj.weight.copy_(ref.down_proj.weight)
            x = torch.randn(2, 5, self.cfg.hidden_size)
            expect = ref.down_proj(ref.act_fn(ref.gate_proj(x)) * ref.up_proj(x))
            got = new(x)
        self.assertTrue(torch.allclose(expect, got, atol=1e-5, rtol=0), (expect - got).abs().max())


if __name__ == "__main__":
    unittest.main()
