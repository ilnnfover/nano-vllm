"""P4 · 连续批处理正确性测试。

验证 P4 EngineCore 输出与 P1 eager（use_cache=False）逐 token 一致。
对拍基准: P1 eager 串行 greedy（Branch 6 锁定）。

测试项:
  1. 单请求: P4 单条 == P1 单条
  2. 多请求: P4 多条并发 == P1 逐条串行
  3. chunked prefill: max_num_batched_tokens=8 强制分块，P4 == P1
  4. 抢占恢复: num_blocks=8 容不下 3 个请求同时运行，触发抢占→recompute，P4 == P1

运行:
  python tests/test_p4_correctness.py
  python -m pytest tests/test_p4_correctness.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest

import torch

from nano_vllm.core.paged_kv_cache import PagedKVCache
from nano_vllm.engine.core import EngineCore
from nano_vllm.engine.scheduler import Scheduler
from nano_vllm.engine.sequence import SamplingParams
from nano_vllm.model_executor.runner import NanoRunner

MODEL = "models/Qwen2.5-0.5B-Instruct"
GREEDY = SamplingParams(temperature=0.0, max_new_tokens=20)


def make_runner(num_blocks: int = 64, max_seq_len: int = 256) -> NanoRunner:
    return NanoRunner(
        MODEL, device="cpu", dtype=torch.float32,
        max_seq_len=max_seq_len, block_size=16, num_blocks=num_blocks,
    )


def make_engine(runner: NanoRunner, max_num_batched_tokens: int = 2048) -> EngineCore:
    scheduler = Scheduler(
        paged_cache=runner.paged_cache,
        max_num_batched_tokens=max_num_batched_tokens,
        watermark_blocks=0,
    )
    return EngineCore(runner, scheduler)


def p1_greedy(runner: NanoRunner, prompt: list[int], max_new_tokens: int = 20) -> list[int]:
    return runner.generate(
        prompt, SamplingParams(temperature=0.0, max_new_tokens=max_new_tokens),
        use_cache=False,
    )


class TestP4Correctness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = make_runner()

    def setUp(self):
        self.runner.paged_cache.reset()

    def test_single_request(self):
        prompt = list(range(30))
        engine = make_engine(self.runner)
        p4_out = engine.generate([prompt], GREEDY)[0]
        p1_out = p1_greedy(self.runner, prompt, GREEDY.max_new_tokens)
        self.assertEqual(p4_out, p1_out, f"P4={p4_out} != P1={p1_out}")

    def test_multi_request(self):
        prompts = [list(range(25)), list(range(10, 40)), list(range(20, 50))]
        engine = make_engine(self.runner)
        p4_outs = engine.generate(prompts, GREEDY)
        for prompt, p4_out in zip(prompts, p4_outs):
            p1_out = p1_greedy(self.runner, prompt, GREEDY.max_new_tokens)
            self.assertEqual(p4_out, p1_out, f"P4={p4_out} != P1={p1_out}")

    def test_chunked_prefill(self):
        prompt = list(range(30))
        engine = make_engine(self.runner, max_num_batched_tokens=8)
        p4_out = engine.generate([prompt], GREEDY)[0]
        p1_out = p1_greedy(self.runner, prompt, GREEDY.max_new_tokens)
        self.assertEqual(p4_out, p1_out, f"chunked P4={p4_out} != P1={p1_out}")

    def test_preemption(self):
        runner = make_runner(num_blocks=8, max_seq_len=128)
        prompts = [list(range(20)), list(range(10, 30)), list(range(20, 40))]
        params = SamplingParams(temperature=0.0, max_new_tokens=15)
        engine = make_engine(runner)
        p4_outs = engine.generate(prompts, params)
        for prompt, p4_out in zip(prompts, p4_outs):
            p1_out = p1_greedy(runner, prompt, params.max_new_tokens)
            self.assertEqual(p4_out, p1_out, f"preempt P4={p4_out} != P1={p1_out}")


if __name__ == "__main__":
    unittest.main()