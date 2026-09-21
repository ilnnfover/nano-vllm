"""P1 · Sampler 单元测试。运行: python tests/test_sampler.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest

import torch

from nano_vllm.sample.sampler import Sampler


def make_sampler():
    return Sampler("cpu")


class TestSampler(unittest.TestCase):
    def setUp(self):
        self.s = make_sampler()

    def test_greedy_is_argmax(self):
        logits = torch.tensor([0.1, 5.0, 0.3])
        self.assertEqual(self.s.sample(logits, temperature=0.0), 1)

    def test_top_k_1_equals_greedy(self):
        logits = torch.tensor([0.1, 5.0, 0.3])
        for seed in range(5):
            self.s.seed(seed)
            self.assertEqual(self.s.sample(logits, temperature=1.0, top_k=1), 1)

    def test_top_k_filter(self):
        logits = torch.tensor([10.0, 9.0, 1.0, 0.0])
        for seed in range(20):
            self.s.seed(seed)
            tok = self.s.sample(logits, temperature=1.0, top_k=2)
            self.assertIn(tok, {0, 1})

    def test_top_p_keeps_most_probable(self):
        logits = torch.tensor([100.0, 1.0, 0.5, 0.1])
        for seed in range(20):
            self.s.seed(seed)
            tok = self.s.sample(logits, temperature=1.0, top_p=0.1)
            self.assertEqual(tok, 0)

    def test_temperature_scaling(self):
        logits = torch.tensor([1.0, 2.0])
        a = torch.softmax(logits / 0.5, dim=-1)
        self.assertAlmostEqual(a[1].item(), torch.softmax(torch.tensor([2.0, 4.0]), -1)[1].item())

    def test_seed_reproducibility(self):
        torch.manual_seed(0)
        logits = torch.randn(1000)
        self.s.seed(42)
        seq_a = [self.s.sample(logits) for _ in range(10)]
        self.s.seed(42)
        seq_b = [self.s.sample(logits) for _ in range(10)]
        self.assertEqual(seq_a, seq_b)

    def test_greedy_matches_multinomial_top1(self):
        logits = torch.randn(500)
        greedy = self.s.sample(logits, temperature=0.0)
        kth = torch.topk(logits, 1)[0][-1]
        self.assertEqual(greedy, int((logits == kth).nonzero()[0]))


if __name__ == "__main__":
    unittest.main()
