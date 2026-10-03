"""抢占恢复对齐 vLLM · 分步实施测试（S1 起）。

背景文档: `docs/notes/preempt-recompute-align.md`

S1（本文件第一批用例）: `Sequence` 的统一 token 流访问器
  - `all_token_ids` = prompt + output
  - `input_token_ids(start, k)` 位置式取法，覆盖三种位置：
      ① prompt 区（= 原 prefill 切片）
      ② output 区（= 原 decode 取 output[-1] 的推广：可重算**任意已生成位置**）
      ③ 流末尾/越界（重喂 `T[-1]`，用于产出下一个 token）

S3（后续加入）: 抢占恢复只重算 ≤1 块 + output 不重复 + 与 P1 eager 一致

运行: ~/venvs/dev/bin/python -m pytest tests/test_preempt_align.py -v
"""
import gc
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch

from nano_vllm.engine.core import EngineCore
from nano_vllm.engine.scheduler import Scheduler
from nano_vllm.engine.sequence import SamplingParams, Sequence
from nano_vllm.model_executor.runner import NanoRunner


def make_seq(prompt: list[int], output: list[int] | None = None,
             computed: int = 0) -> Sequence:
    seq = Sequence(
        seq_id=0, prompt_token_ids=list(prompt),
        sampling_params=SamplingParams(max_new_tokens=8),
    )
    seq.output_token_ids = list(output or [])
    seq.num_computed_tokens = computed
    return seq


# ---------------- S1: 统一 token 流 ----------------


class TestUnifiedTokenStream:
    def test_all_token_ids_is_prompt_plus_output(self):
        seq = make_seq([10, 11], [20, 21, 22])
        assert seq.all_token_ids == [10, 11, 20, 21, 22]
        assert seq.num_tokens == 5
        assert seq.num_prompt_tokens == 2

    def test_prompt_region_matches_prefill_slice(self):
        """位置落在 prompt 内 → 与原 prefill 切片 `prompt[start:start+k]` 完全一致。"""
        seq = make_seq([10, 11, 12, 13], output=[])
        assert seq.input_token_ids(0, 4) == [10, 11, 12, 13]
        assert seq.input_token_ids(2, 2) == [12, 13]
        assert seq.input_token_ids(0, 4) == seq.prompt_token_ids[0:4]

    def test_output_region_generalizes_decode_token(self):
        """位置落在 output 内 → 可取任意已生成位置（这是现状做不到的能力）。"""
        seq = make_seq([10, 11], [20, 21, 22])
        assert seq.input_token_ids(2, 3) == [20, 21, 22]
        assert seq.input_token_ids(3, 1) == [21]
        # 与原 decode 取法（output[-1]）在稳态位置一致：last = 4
        assert seq.input_token_ids(4, 1) == [seq.output_token_ids[-1]]

    def test_spanning_prompt_and_output(self):
        """跨 prompt/output 边界的连续 chunk（抢占恢复后的常见形态）。"""
        seq = make_seq([10, 11, 12], [20, 21])
        assert seq.input_token_ids(1, 4) == [11, 12, 20, 21]

    def test_beyond_stream_refeeds_last_token(self):
        """位置 >= len(T) → 重喂最后一个 token（decode 用于产出下一个）。"""
        seq = make_seq([10], [20, 21])
        assert seq.input_token_ids(3, 1) == [21]   # p == len(T)
        assert seq.input_token_ids(5, 1) == [21]   # p > len(T)（防御性）

    def test_zero_or_negative_length_returns_empty(self):
        seq = make_seq([10, 11])
        assert seq.input_token_ids(0, 0) == []
        assert seq.input_token_ids(1, -1) == []

    def test_empty_stream_raises(self):
        seq = Sequence(seq_id=0, prompt_token_ids=[])
        with pytest.raises(ValueError, match="空 token 流"):
            seq.input_token_ids(0, 1)

    def test_stream_is_a_copy_not_a_view(self):
        """返回的是新列表：调用方改动不会污染请求状态。"""
        seq = make_seq([10, 11], [20])
        toks = seq.input_token_ids(0, 3)
        toks[0] = 999
        assert seq.prompt_token_ids[0] == 10
        assert seq.output_token_ids == [20]


class TestNumNewTokensUnified:
    """`num_new_tokens` 是统一记账的核心量：稳态 decode 恒为 1，抢占后为整段。"""

    def test_steady_decode_is_one(self):
        # 稳态不变量: num_computed_tokens == num_tokens - 1
        seq = make_seq([10, 11, 12], [20, 21], computed=4)
        assert seq.num_new_tokens == 1

    def test_after_preemption_is_whole_stream(self):
        seq = make_seq([10, 11, 12], [20, 21], computed=0)
        assert seq.num_new_tokens == 5

    def test_prop_tokens_before_any_output(self):
        seq = make_seq([10, 11, 12], output=[], computed=0)
        assert seq.num_new_tokens == 3


# ---------------- S3: 抢占恢复只重算 ≤1 块 · output 不重复 ----------------


class TestResumeReuseKV(unittest.TestCase):
    """S3 核心：抢占后保留 output，恢复时按 num_tokens-1 命中 → 重算 ≤ 1 个 block。

    现状（改动前）：`reset_for_preemption` 清空 output → 恢复 = 重算 prompt 尾部 +
    **全部已生成 token 重新生成**；本用例断言的是对齐后的行为。
    """

    BLOCK = 16
    NUM_BLOCKS = 8

    @classmethod
    def setUpClass(cls):
        cls.runner = NanoRunner(
            "models/Qwen2.5-0.5B-Instruct", device="cpu", dtype=torch.float32,
            max_seq_len=256, block_size=cls.BLOCK, num_blocks=cls.NUM_BLOCKS,
        )

    @classmethod
    def tearDownClass(cls):
        del cls.runner
        gc.collect()

    def _scheduler(self, **kw) -> Scheduler:
        return Scheduler(
            self.runner.paged_cache, max_num_batched_tokens=2048,
            watermark_blocks=0, enable_prefix_cache=True, **kw,
        )

    def test_preemption_keeps_output_tokens(self):
        """reset_for_preemption 保留 output_token_ids（对齐 vLLM）。"""
        seq = Sequence(seq_id=0, prompt_token_ids=list(range(20)),
                       sampling_params=SamplingParams(max_new_tokens=8))
        seq.output_token_ids = [7, 8, 9]
        seq.num_computed_tokens = 22
        seq.block_table = [1, 2]
        seq.reset_for_preemption()
        self.assertEqual(seq.output_token_ids, [7, 8, 9], "output 被清空（未对齐 vLLM）")
        self.assertEqual(seq.num_computed_tokens, 0)
        self.assertEqual(seq.block_table, [])

    def test_resume_recompute_bounded_by_one_block(self):
        """恢复时重算量 = num_tokens - floor((num_tokens-1)/bs)*bs ≤ block_size。

        用纯调度器构造（不跑模型）：先让一个请求"已生成若干 token 并被抢占"，
        再重调度，检查它这一步被调度的 token 数。
        """
        self.runner.paged_cache.reset()
        sched = self._scheduler()
        prompt = list(range(100, 100 + 32))          # 2 个满块
        seq = Sequence(seq_id=0, prompt_token_ids=prompt + list(range(200, 216)),
                       sampling_params=SamplingParams(max_new_tokens=64))
        seq.output_token_ids = list(range(500, 500 + 19))   # 已生成 19 个
        sched.add_request(seq)

        # 第一次调度：新请求（无命中），一次 chunk 算到流末尾
        out = sched.schedule()
        s = out.scheduled[0]
        self.assertTrue(s.is_prefill)
        self.assertEqual(s.num_scheduled_tokens, seq.num_tokens)
        self.assertTrue(s.samples_this_step)
        sched.update_from_output(out, {0: 600})       # 采样 → 追加 1 个 token
        self.assertEqual(seq.num_computed_tokens, seq.num_tokens - 1)
        n_before = seq.num_tokens
        self.assertGreaterEqual(n_before, 48)

        # 抢占（保留 output），再重调度：命中自己的前缀块 → 只重算尾部
        outputs_before_preempt = list(seq.output_token_ids)
        sched._preempt(seq)
        self.assertEqual(seq.output_token_ids, outputs_before_preempt, "output 未保留")
        out = sched.schedule()
        s = out.scheduled[0]
        recompute = s.num_scheduled_tokens
        expected = n_before - (n_before - 1) // self.BLOCK * self.BLOCK
        self.assertEqual(recompute, expected)
        self.assertLessEqual(
            recompute, self.BLOCK,
            f"恢复重算 {recompute} token 超过 1 个 block（{self.BLOCK}）",
        )
        # 恢复后能命中自己已生成 token 的块（命中 token 数 > prompt 长度）
        self.assertGreater(
            sched.prefix_hit_tokens, seq.num_prompt_tokens,
            "恢复未复用已生成 token 的 KV（说明查找仍在 prompt 上截断）",
        )

    def test_resume_does_not_duplicate_output(self):
        """恢复后 output 的既有前缀不变（不会被重新 append 一遍）。"""
        self.runner.paged_cache.reset()
        sched = self._scheduler()
        seq = Sequence(seq_id=0, prompt_token_ids=list(range(100, 121)),
                       sampling_params=SamplingParams(max_new_tokens=64))
        sched.add_request(seq)
        while len(seq.output_token_ids) < 24:      # 先正常生成到 output 区出现过满块
            out = sched.schedule()
            sched.update_from_output(out, {0: 700})

        before = list(seq.output_token_ids)
        self.assertEqual(len(before), 24)
        sched._preempt(seq)

        # 只跑到"恢复后第一次采样"为止
        guard = 0
        while len(seq.output_token_ids) == len(before) and guard < 64:
            guard += 1
            out = sched.schedule()
            s = out.scheduled[0]
            # 中间追赶段（未到达流末尾）不得采样，否则会重复产出
            sched.update_from_output(out, {0: 701} if s.samples_this_step else {})

        self.assertEqual(seq.output_token_ids[:len(before)], before,
                         "恢复后既有 output 前缀被改动（重复产出）")
        self.assertEqual(len(seq.output_token_ids), len(before) + 1)


class TestResumeMatchesEagerNano(unittest.TestCase):
    """S3 端到端：抢占-恢复后输出仍与 P1 eager 逐 token 一致（0.5B/CPU）。"""

    @classmethod
    def setUpClass(cls):
        cls.runner = NanoRunner(
            "models/Qwen2.5-0.5B-Instruct", device="cpu", dtype=torch.float32,
            max_seq_len=128, block_size=16, num_blocks=8,
        )

    @classmethod
    def tearDownClass(cls):
        del cls.runner
        gc.collect()

    def test_preemption_with_outputs_matches_eager(self):
        prompts = [list(range(20)), list(range(10, 30)), list(range(20, 40))]
        params = SamplingParams(temperature=0.0, max_new_tokens=15)

        self.runner.paged_cache.reset()
        scheduler = Scheduler(
            self.runner.paged_cache, max_num_batched_tokens=2048,
            watermark_blocks=0, enable_prefix_cache=True,
        )
        engine = EngineCore(self.runner, scheduler)
        seqs = [engine.add_request(p, params) for p in prompts]
        while engine.scheduler.has_requests():
            engine.step()

        self.assertGreater(engine.stats.total_preempt_count, 0, "未触发抢占，用例无效")
        for prompt, seq in zip(prompts, seqs):
            ref = self.runner.generate(prompt, params, use_cache=False)
            self.assertEqual(seq.output_token_ids, ref,
                             f"抢占恢复输出 != P1 eager\nseq={seq.output_token_ids}\nref={ref}")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
