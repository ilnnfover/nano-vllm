"""P2 · 正确性测试：KV cache 路径与 eager 路径逐 token 一致。运行: python tests/test_p2_correctness.py

覆盖:
  1. batch=1 use_cache=True vs use_cache=False 逐 token 一致
  2. batch=N generate_batch vs 逐个 generate 一致（含极端长度差 padding）
  3. HF 对照：nano cache greedy 与 HF use_cache=False greedy 一致（0.5B 快速）
  4. max_seq_len 越界防护
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest

import torch
import transformers

from nano_vllm.model_executor.runner import NanoRunner, SamplingParams

MODEL = "models/Qwen2.5-0.5B-Instruct"


def make_runner() -> NanoRunner:
    return NanoRunner(MODEL, device="cpu", dtype=torch.float32)


def make_prompts() -> list[list[int]]:
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    texts = [
        "The meaning of life is",
        "Hi",
        "Python is a programming language that",
        "In a shocking finding, scientists discovered a herd of unicorns living in a remote valley",
    ]
    return [tok(t).input_ids for t in texts]


class TestP2Correctness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = make_runner()
        cls.prompts = make_prompts()
        cls.params = SamplingParams(temperature=0.0, max_new_tokens=16)

    def test_cache_matches_eager(self):
        """batch=1: KV cache 路径与 eager 重算路径逐 token 一致"""
        for ids in self.prompts:
            eager = self.r.generate(ids, self.params, use_cache=False)
            cache = self.r.generate(ids, self.params, use_cache=True)
            self.assertEqual(eager, cache, f"分叉于 prompt_len={len(ids)}")

    def test_batch_matches_serial(self):
        """batch=N: padding 拼批与逐个生成逐 token 一致（含极端长度差）"""
        lens = [len(p) for p in self.prompts]
        self.assertLess(min(lens) / max(lens), 0.2, "测试数据需含极端长度差")

        serial = [self.r.generate(p, self.params, use_cache=True) for p in self.prompts]
        batch_outs, stats = self.r.generate_batch(self.prompts, self.params)
        for i, (s, b) in enumerate(zip(serial, batch_outs)):
            self.assertEqual(s, b, f"req{i} (len={lens[i]}) 批内与串行不一致")
        self.assertIn("padding_waste_rate", stats)
        self.assertIn("kv_waste_rate", stats)

    def test_hf_greedy_match(self):
        """nano cache greedy 与 HF generate(use_cache=False, 纯 greedy) 一致"""
        tok = transformers.AutoTokenizer.from_pretrained(MODEL)
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
        ids = self.prompts[0]
        ref = hf_model.generate(
            torch.tensor([ids]), max_new_tokens=16, do_sample=False,
            use_cache=False, repetition_penalty=1.0,
        )[0].tolist()
        hf_out = ref[len(ids):]
        # HF generate 会在 eos 停; 去掉尾部 eos
        eos = {151645, 151643}
        while hf_out and hf_out[-1] in eos:
            hf_out.pop()

        nano_out = self.r.generate(ids, self.params, use_cache=True)
        self.assertEqual(nano_out, hf_out)

    def test_seq_len_overflow_raises(self):
        """序列超过 max_seq_len 应报错而非静默写错位置"""
        r = NanoRunner(
            MODEL, device="cpu", dtype=torch.float32, max_seq_len=8,
        )
        ids = self.prompts[0]  # len > 8
        with self.assertRaises(ValueError):
            r.generate(ids, SamplingParams(temperature=0.0, max_new_tokens=4), use_cache=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)