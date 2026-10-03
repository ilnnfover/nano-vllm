"""A6 · 调度节流阀（P6 前置遗留）。

覆盖 vLLM scheduler 的三个标配节流阀:
  - `long_prefill_token_threshold`：单请求一步最多吃多少 prefill token
  - `max_num_seqs`：同批运行请求上限
  - watermark 由「固定块数」改为「按比例预留」（块池按显存反推后固定 1 块失去意义）

全部为无模型纯 CPU 测试（直接驱动 Scheduler）。
运行: python -m pytest tests/test_p6_throttle.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch

from nano_vllm.engine.scheduler import Scheduler
from nano_vllm.engine.sequence import Sequence
from nano_vllm.kvmm.paged_kv_cache import PagedKVCache

BLOCK_SIZE = 16


def make_paged(num_blocks: int = 64) -> PagedKVCache:
    return PagedKVCache(
        num_layers=1, num_blocks=num_blocks, block_size=BLOCK_SIZE,
        num_kv_heads=1, head_dim=2, dtype=torch.float32, device="cpu",
    )


def make_scheduler(num_blocks: int = 64, budget: int = 64, **kwargs) -> Scheduler:
    return Scheduler(
        make_paged(num_blocks),
        max_num_batched_tokens=budget,
        enable_prefix_cache=False,
        **kwargs,
    )


class TestLongPrefillThreshold:
    def test_caps_single_request_chunk(self):
        """budget=64 + threshold=16：一条 prompt=64 的请求一步只吃 16 个 token。"""
        sched = make_scheduler(budget=64, long_prefill_token_threshold=16)
        sched.add_request(Sequence(seq_id=0, prompt_token_ids=list(range(64))))

        out = sched.schedule()
        assert len(out.scheduled) == 1
        assert out.scheduled[0].num_scheduled_tokens == 16

    def test_chunking_progresses_over_steps(self):
        """连续 3 步各吃 16，prefill 进度逐 step 推进。"""
        sched = make_scheduler(budget=64, long_prefill_token_threshold=16)
        seq = Sequence(seq_id=0, prompt_token_ids=list(range(64)))
        sched.add_request(seq)

        for expected in (16, 32, 48):
            out = sched.schedule()
            sched.update_from_output(out, {})
            assert seq.num_computed_tokens == expected
            assert seq.is_prefill

    def test_zero_disables_throttle(self):
        """threshold<=0 → 关闭节流，一步吃满 budget。"""
        sched = make_scheduler(budget=64, long_prefill_token_threshold=0)
        assert sched.long_prefill_token_threshold is None
        sched.add_request(Sequence(seq_id=0, prompt_token_ids=list(range(64))))
        out = sched.schedule()
        assert out.scheduled[0].num_scheduled_tokens == 64

    def test_default_is_disabled(self):
        """缺省关闭，与 vLLM `SchedulerConfig.long_prefill_token_threshold=0` 一致。"""
        sched = make_scheduler(budget=64)
        assert sched.long_prefill_token_threshold is None

    def test_negative_also_disables(self):
        sched = make_scheduler(budget=64, long_prefill_token_threshold=-8)
        assert sched.long_prefill_token_threshold is None


class TestMaxNumSeqs:
    def test_caps_running_batch(self):
        """max_num_seqs=1：两条同时到达，只放一条进 running，另一条留在 waiting。"""
        sched = make_scheduler(budget=256, max_num_seqs=1)
        for i in range(2):
            sched.add_request(Sequence(seq_id=i, prompt_token_ids=list(range(8))))

        out = sched.schedule()
        assert len(out.scheduled) == 1
        assert sched.num_running == 1
        assert sched.num_waiting == 1

    def test_none_means_unlimited(self):
        sched = make_scheduler(budget=256, max_num_seqs=None)
        for i in range(3):
            sched.add_request(Sequence(seq_id=i, prompt_token_ids=list(range(8))))
        out = sched.schedule()
        assert len(out.scheduled) == 3
        assert sched.num_waiting == 0


class TestWatermarkRatio:
    def test_ratio_default_scales_with_pool(self):
        """1000 块 × 0.01 → 预留 10 块（固定 1 块在上万块池下已无保护意义）。"""
        sched = make_scheduler(num_blocks=1000)
        assert sched.watermark_blocks == 10

    def test_ratio_has_floor_of_one(self):
        """小池下比例取整为 0 时至少预留 1 块。"""
        sched = make_scheduler(num_blocks=8)
        assert sched.watermark_blocks == 1

    def test_explicit_value_overrides_ratio(self):
        sched = make_scheduler(num_blocks=1000, watermark_blocks=1)
        assert sched.watermark_blocks == 1

    def test_explicit_zero_overrides_ratio(self):
        """显式传 0 必须保持 0（既有测试与抢占用例依赖确定性紧余量）。"""
        sched = make_scheduler(num_blocks=1000, watermark_blocks=0)
        assert sched.watermark_blocks == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
