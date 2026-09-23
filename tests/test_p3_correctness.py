"""P3 · 正确性测试：分页 KV cache / paged attention / 自研 Triton kernel。

覆盖:
  1. BlockPool 分配/回收/耗尽守卫/ref_cnt
  2. PagedKVCache reshape_and_cache 写入-读取一致性 + slot_mapping 跨块正确
  3. paged attention 三路对拍: torch 朴素 vs SDPA vs Triton（interpreter 或 GPU）
  4. 分页 generate vs 连续 cache generate 逐 token 一致（多 block_size + 跨块边界）
  5. waste_rate 验收: 分页 < 2%（对比 P2 连续 15%）

运行:
  python tests/test_p3_correctness.py                 # CPU 跑 1/2/4/5 + Triton skip
  TRITON_INTERPRET=1 python tests/test_p3_correctness.py  # CPU 解释器跑 Triton kernel
  python tests/test_p3_correctness.py                 # GPU 上跑真实 Triton kernel
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest

import torch

from nano_vllm.attention.paged_attn import paged_attention_sdpa, paged_attention_torch
from nano_vllm.core.block_pool import BlockPool
from nano_vllm.core.paged_kv_cache import PagedKVCache

MODEL = "models/Qwen2.5-0.5B-Instruct"
INTERPRET = bool(os.environ.get("TRITON_INTERPRET"))


class TestBlockPool(unittest.TestCase):
    def test_allocate_free_roundtrip(self):
        p = BlockPool(4)
        self.assertEqual(p.num_free_blocks, 4)
        b = p.allocate_n(3)
        self.assertEqual(b, [0, 1, 2])  # FIFO，分配顺序确定
        self.assertEqual(p.num_used_blocks, 3)
        p.free(b[0])
        self.assertEqual(p.num_used_blocks, 2)
        self.assertEqual(p.num_free_blocks, 2)

    def test_exhaustion_guard(self):
        p = BlockPool(2)
        with self.assertRaises(ValueError):
            p.allocate_n(3)
        self.assertEqual(p.num_free_blocks, 2, "分配失败不能产生半分配")

    def test_double_free_raises(self):
        p = BlockPool(1)
        b = p.allocate()
        p.free(b)
        with self.assertRaises(ValueError):
            p.free(b)

    def test_reset(self):
        p = BlockPool(3)
        p.allocate_n(3)
        p.reset()
        self.assertEqual(p.num_free_blocks, 3)


class TestPagedKVCache(unittest.TestCase):
    def _make(self, block_size=16):
        return PagedKVCache(
            num_layers=2, num_blocks=64, block_size=block_size,
            num_kv_heads=2, head_dim=8, dtype=torch.float32, device="cpu",
        )

    def test_write_read_roundtrip(self):
        kv = self._make()
        seq_len = 37  # 跨 3 个 block（block_size=16）
        bt = kv.allocate(seq_len)
        sm = kv.slot_mapping(bt, 0, seq_len)
        k = torch.randn(2, seq_len, 8)
        v = torch.randn(2, seq_len, 8)
        kv.write(0, k, v, sm)
        rk, rv = kv.read_blocks(0, bt)
        self.assertTrue(torch.allclose(rk[:, :seq_len], k, atol=1e-6))
        self.assertTrue(torch.allclose(rv[:, :seq_len], v, atol=1e-6))

    def test_slot_mapping_cross_block(self):
        kv = self._make(block_size=4)
        bt = kv.allocate(10)  # [0,1,2]
        sm = kv.slot_mapping(bt, 0, 10)
        # token 0..3 → block 0 slot 0..3；token 4..7 → block 1；token 8,9 → block 2
        self.assertEqual(sm.tolist(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        # 从 pos=3 起写 2 个 token（跨块 0→1）
        sm2 = kv.slot_mapping(bt, 3, 2)
        self.assertEqual(sm2.tolist(), [3, 4])

    def test_append_grows_block_table(self):
        kv = self._make(block_size=4)
        bt = kv.allocate(4)  # 恰好 1 块
        self.assertEqual(len(bt), 1)
        kv.ensure_capacity(bt, 5)  # 增长到 5 → 需要 2 块
        self.assertEqual(len(bt), 2)

    def test_waste_rate(self):
        kv = self._make(block_size=16)
        seq_len = 1024  # 恰好整除 block_size → 零浪费
        bt = kv.allocate(seq_len)
        sm = kv.slot_mapping(bt, 0, seq_len)
        kv.write(0, torch.randn(2, seq_len, 8), torch.randn(2, seq_len, 8), sm)
        self.assertLess(kv.waste_rate, 0.02, "整除 block_size 时分页浪费应 < 2%")


class TestPagedAttention(unittest.TestCase):
    def _inputs(self, num_heads, num_kv_heads, head_dim, block_size, num_blocks, seq_len, seed):
        g = torch.Generator().manual_seed(seed)
        k_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, generator=g)
        v_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, generator=g)
        q = torch.randn(num_heads, 1, head_dim, generator=g)
        n_need = (seq_len + block_size - 1) // block_size
        return q, k_cache, v_cache, list(range(n_need))

    def test_torch_vs_sdpa(self):
        for seq_len, bs in [(1, 16), (5, 16), (17, 16), (100, 16), (100, 8), (33, 32)]:
            q, k, v, bt = self._inputs(12, 2, 128, bs, 128, seq_len, 42)
            a = paged_attention_torch(q, k, v, bt, seq_len, 2, 128**-0.5)
            b = paged_attention_sdpa(q, k, v, bt, seq_len, 2, 128**-0.5)
            self.assertLess((a - b).abs().max().item(), 1e-4, f"seq={seq_len} bs={bs}")

    def test_triton_vs_torch(self):
        """Triton kernel 对拍（CPU 解释器或 GPU）。无 CUDA 且非 interpret 时 skip。"""
        if not torch.cuda.is_available() and not INTERPRET:
            self.skipTest("需要 CUDA 或 TRITON_INTERPRET=1")
        from nano_vllm.attention.triton_paged_attn import paged_attention_triton

        for seq_len, bs in [(1, 16), (5, 16), (17, 16), (100, 16), (65, 64)]:
            q, k, v, bt = self._inputs(12, 2, 128, bs, 128, seq_len, 7)
            a = paged_attention_triton(q, k, v, bt, seq_len, 2, 128**-0.5)
            b = paged_attention_torch(q, k, v, bt, seq_len, 2, 128**-0.5)
            self.assertLess((a - b).abs().max().item(), 1e-4, f"seq={seq_len} bs={bs}")


class TestPagedGenerate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import transformers

        from nano_vllm.model_executor.runner import NanoRunner

        cls.tok = transformers.AutoTokenizer.from_pretrained(MODEL)
        cls.params = None  # 延迟到各 test 构造
        cls.runner = NanoRunner(MODEL, device="cpu", dtype=torch.float32, block_size=16, max_seq_len=256)

    def test_paged_matches_cache(self):
        from nano_vllm.model_executor.runner import SamplingParams

        params = SamplingParams(temperature=0.0, max_new_tokens=16)
        texts = [
            "The meaning of life is",
            "Hi",
            "In a shocking finding, scientists discovered a herd of unicorns living in a remote valley",
        ]
        for text in texts:
            ids = self.tok(text).input_ids
            cache = self.runner.generate(ids, params, use_cache=True)
            paged = self.runner.generate(ids, params, use_paged=True)
            self.assertEqual(cache, paged, f"分页与连续 cache 分叉于 len={len(ids)}")

    def test_paged_matches_cache_across_blocks(self):
        from nano_vllm.model_executor.runner import NanoRunner, SamplingParams

        params = SamplingParams(temperature=0.0, max_new_tokens=20)
        ids = self.tok(
            "In a shocking finding, scientists discovered a herd of unicorns. " * 3
        ).input_ids
        self.assertGreater(len(ids), 16, "测试需跨 block 边界")
        for bs in [8, 16, 32, 64]:
            r = NanoRunner(MODEL, device="cpu", dtype=torch.float32, block_size=bs, max_seq_len=256)
            cache = r.generate(ids, params, use_cache=True)
            paged = r.generate(ids, params, use_paged=True)
            self.assertEqual(cache, paged, f"block_size={bs} 分页与连续 cache 分叉")


if __name__ == "__main__":
    unittest.main(verbosity=2)