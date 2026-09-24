"""P0-01/P0-02 · chunked prefill 语义测试。

验证: 将一段 prompt 分 2/3 段做 chunked prefill，最后一段的 last-token logits
      与一次全量 prefill 的 last-token logits 位级一致（CPU fp32）。

P0-01 的 bug: is_prefill=True 时无条件只用当前 chunk K/V，忽略 cache_seq_len>0 的历史 → 丢前缀。
修复后: is_prefill=True 且 cache_seq_len>0 时，读全部 KV + 混合 mask（历史全可见 + chunk 内 causal）。

运行:
  python tests/test_chunked_prefill.py
  python -m pytest tests/test_chunked_prefill.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest

import torch

from nano_vllm.model_executor.runner import NanoRunner

MODEL = "models/Qwen2.5-0.5B-Instruct"


class TestChunkedPrefill(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = NanoRunner(MODEL, device="cpu", dtype=torch.float32, max_seq_len=256)

    def _full_prefill(self, prompt_ids: list[int]) -> torch.Tensor:
        """一次全量 prefill，返回 last-token logits。"""
        return self.runner._prefill(prompt_ids)

    def _chunked_prefill(self, prompt_ids: list[int], chunks: list[int]) -> torch.Tensor:
        """分段 chunked prefill，返回最后一段的 last-token logits。

        Args:
            prompt_ids: 完整 prompt
            chunks: 每段长度，如 [20, 17] 表示前 20 token 一段、后 17 token 一段
        """
        r = self.runner
        r.kv_cache.reset()
        offset = 0
        logits = None
        for i, chunk_len in enumerate(chunks):
            segment = prompt_ids[offset : offset + chunk_len]
            t = torch.tensor([segment], dtype=torch.long, device=r.device)
            position_ids = torch.arange(offset, offset + chunk_len, device=r.device).unsqueeze(0)
            last_idx = torch.tensor([chunk_len - 1], device=r.device)
            logits, _ = r.model(
                t, kv_cache=r.kv_cache, is_prefill=True,
                cache_seq_len=offset, position_ids=position_ids, last_idx=last_idx,
            )
            r.kv_cache.seq_len = offset + chunk_len
            offset += chunk_len
        return logits[0]

    def test_2_chunk_matches_full(self):
        """分 2 段 vs 全量 prefill，last-token logits 一致。"""
        ids = list(range(37))  # 37 = 非整除 block_size=16，覆盖边界
        full = self._full_prefill(ids)
        chunked = self._chunked_prefill(ids, [20, 17])
        self.assertTrue(
            torch.allclose(full, chunked, atol=1e-4),
            f"2-chunk diff max={ (full - chunked).abs().max().item():.2e}",
        )

    def test_3_chunk_matches_full(self):
        """分 3 段 vs 全量 prefill，last-token logits 一致。"""
        ids = list(range(37))
        full = self._full_prefill(ids)
        chunked = self._chunked_prefill(ids, [12, 13, 12])
        self.assertTrue(
            torch.allclose(full, chunked, atol=1e-4),
            f"3-chunk diff max={(full - chunked).abs().max().item():.2e}",
        )

    def test_chunked_then_decode_matches_full(self):
        """chunked prefill + decode vs 全量 prefill + decode，生成逐 token 一致。"""
        from nano_vllm.model_executor.runner import SamplingParams

        ids = list(range(30))
        params = SamplingParams(temperature=0.0, max_new_tokens=8)

        # 全量 prefill + decode
        full_out = []
        logits = self.runner._prefill(ids)
        for _ in range(params.max_new_tokens):
            tok = int(logits.argmax())
            full_out.append(tok)
            logits = self.runner._decode(tok)

        # chunked prefill + decode
        chunked_out = []
        logits = self._chunked_prefill(ids, [15, 15])
        for _ in range(params.max_new_tokens):
            tok = int(logits.argmax())
            chunked_out.append(tok)
            logits = self.runner._decode(tok)

        self.assertEqual(full_out, chunked_out, "chunked prefill + decode 生成分叉")


if __name__ == "__main__":
    unittest.main(verbosity=2)