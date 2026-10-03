"""P7 前置整改 · 回归测试（R1 / R2 + evict 别名守卫）。

背景（docs/notes/pre-p7-fixes.md）:
  - R1: 同 prompt 重放且长度恰为 block_size 整数倍时，前缀命中覆盖整个 prompt
       → num_new_tokens == 0 → 空前向 / 永不产出首 token。
       修复: get_computed_blocks 钳制命中到 num_prompt_tokens - 1（对齐 vLLM
       kv_cache_manager.get_computed_blocks 的 num_tokens - 1）。
  - R2a: prompt+max_new_tokens 超过全池容量的请求独占全池也无法完成，
       入队必然 livelock → add_request 直接拒绝。
  - R2b: waiting 请求抢占 running 会在组合需求 > 池容量时互相 ping-pong
       （A 抢 B → B 重算 → B 抢 A → … 双方都到不了 max_new_tokens）。
       修复: waiting 不抢占 running（对齐 vLLM：抢占只发生在 RUNNING 段）。
  - 附带: _evict_hashes 别名守卫——同内容块重算后 hash 索引指向新块，
       旧块被重新分配时不得误删新块的注册。

运行:
  python -m pytest tests/test_pre_p7_fixes.py -v
"""
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest

import pytest
import torch

from nano_vllm.engine.core import EngineCore
from nano_vllm.engine.scheduler import Scheduler
from nano_vllm.engine.sequence import SamplingParams, Sequence, SequenceStatus
from nano_vllm.kvmm.block_pool import BlockPool
from nano_vllm.kvmm.paged_kv_cache import PagedKVCache
from nano_vllm.model_executor.runner import NanoRunner

BLOCK_SIZE = 16


def make_paged(num_blocks: int = 64) -> PagedKVCache:
    return PagedKVCache(
        num_layers=1, num_blocks=num_blocks, block_size=BLOCK_SIZE,
        num_kv_heads=1, head_dim=2, dtype=torch.float32, device="cpu",
    )


def make_scheduler(paged: PagedKVCache, **kw) -> Scheduler:
    kw.setdefault("watermark_blocks", 0)
    return Scheduler(paged, **kw)


def register_full_blocks(sched: Scheduler, tokens: list[int], n_blocks: int) -> None:
    """模拟历史请求注册的前 n_blocks 个满块 hash（块分配后释放，hash 保留在索引）。"""
    pc = sched.prefix_cache
    assert pc is not None
    blocks = sched.paged_cache.pool.allocate_n(n_blocks)
    pc.register_blocks(
        tokens, blocks, num_full_blocks=n_blocks,
        start_block=0, prev_hash=None,
    )
    sched.paged_cache.pool.free_n(blocks)


# ---------------- R1: 全 prompt 命中钳制 ----------------


class TestR1FullPromptHitClamp:
    def test_full_prompt_hit_clamped(self):
        """prompt=32（block_size 整数倍）且两块全部命中：命中必须钳制到 16，
        保证 num_new_tokens ≥ 1（修复前 = 0 → 空前向/死循环）。"""
        paged = make_paged(num_blocks=16)
        sched = make_scheduler(paged, enable_prefix_cache=True)
        prompt = list(range(100, 132))  # 32 = 2 × 16，整除
        register_full_blocks(sched, prompt, 2)

        seq = Sequence(seq_id=0, prompt_token_ids=prompt)
        sched.add_request(seq)
        out = sched.schedule()

        assert len(out.scheduled) == 1
        s = out.scheduled[0]
        assert s.num_scheduled_tokens >= 1, "全 prompt 命中时 num_new_tokens 为 0（R1 回归）"
        assert seq.num_computed_tokens == 16  # 命中钳制到 31//16=1 块
        assert s.num_scheduled_tokens == 16

    def test_non_multiple_prompt_unaffected(self):
        """prompt=33（非整除）：钳制到 32 与修复前行为一致，不多剪。"""
        paged = make_paged(num_blocks=16)
        sched = make_scheduler(paged, enable_prefix_cache=True)
        prompt = list(range(100, 133))  # 33
        register_full_blocks(sched, prompt, 2)  # 前 2 个满块

        seq = Sequence(seq_id=0, prompt_token_ids=prompt)
        sched.add_request(seq)
        out = sched.schedule()

        assert out.scheduled[0].num_scheduled_tokens == 1  # 33 - 32
        assert seq.num_computed_tokens == 32

    def test_find_longest_hit_max_tokens_boundary(self):
        """max_tokens 边界：块覆盖 token 数超过 max_tokens 即停。"""
        paged = make_paged(num_blocks=16)
        sched = make_scheduler(paged, enable_prefix_cache=True)
        pc = sched.prefix_cache
        assert pc is not None
        tokens = list(range(64))  # 4 个满块
        register_full_blocks(sched, tokens, 4)

        assert pc.find_longest_hit(tokens).num_tokens == 64        # 不钳制
        assert pc.find_longest_hit(tokens, max_tokens=64).num_tokens == 64
        assert pc.find_longest_hit(tokens, max_tokens=63).num_tokens == 48
        assert pc.find_longest_hit(tokens, max_tokens=32).num_tokens == 32
        assert pc.find_longest_hit(tokens, max_tokens=31).num_tokens == 16
        assert pc.find_longest_hit(tokens, max_tokens=15).num_tokens == 0
        assert pc.find_longest_hit(tokens, max_tokens=0).num_tokens == 0


# ---------------- 附带: _evict_hashes 别名守卫 ----------------


class TestEvictHashAliasGuard:
    def test_old_block_realloc_does_not_evict_new_registration(self):
        """同 hash 先注册到旧块、再注册到新块（同 prompt 重算）后，
        旧块被重新分配（触发 _evict_hashes）时不得删掉指向新块的索引项。"""
        from nano_vllm.kvmm.prefix_cache import compute_block_hash

        pool = BlockPool(8)
        h = compute_block_hash([1, 2, 3], None)
        old_id = pool.allocate()
        new_id = pool.allocate()
        pool.register_hash(old_id, h, num_tokens=3)
        assert pool.get_block_id_by_hash(h) == old_id

        # 同内容重算注册到新块（R1 场景下被重算的满块 hash 会重指向新块）
        pool.register_hash(new_id, h, num_tokens=3)
        assert pool.get_block_id_by_hash(h) == new_id

        # 旧块释放后被重新分配 → allocate 内部调 _evict_hashes(old_id)
        pool.free(old_id)
        pool._evict_hashes(old_id)
        assert pool.get_block_id_by_hash(h) == new_id, "旧块重新分配误删了新块的 hash 注册"
        assert not pool.has_hash(old_id)
        assert pool.has_hash(new_id)


# ---------------- R2a: add_request 容量守卫 ----------------


class TestR2CapacityGuard:
    def test_infeasible_request_rejected(self):
        """prompt+max_new 独占全池也放不下 → ValueError（修复前 livelock）。"""
        paged = make_paged(num_blocks=4)  # 4 × 16 = 64 token
        sched = make_scheduler(paged, enable_prefix_cache=False)
        seq = Sequence(
            seq_id=0, prompt_token_ids=list(range(49)),  # 49 + 16 = 65 token → 5 块 > 4
            sampling_params=SamplingParams(max_new_tokens=16),
        )
        with pytest.raises(ValueError, match="拒绝入队"):
            sched.add_request(seq)
        assert sched.num_waiting == 0

    def test_exactly_full_pool_accepted(self):
        """恰好占满全池（= 池容量）的请求合法——边界不误伤。"""
        paged = make_paged(num_blocks=4)  # 64 token
        sched = make_scheduler(paged, enable_prefix_cache=False)
        seq = Sequence(
            seq_id=0, prompt_token_ids=list(range(48)),  # 48 + 16 = 64 → 4 块 = 4
            sampling_params=SamplingParams(max_new_tokens=16),
        )
        sched.add_request(seq)
        assert sched.num_waiting == 1

    def test_feasible_request_accepted(self):
        paged = make_paged(num_blocks=4)
        sched = make_scheduler(paged, enable_prefix_cache=False)
        sched.add_request(Sequence(
            seq_id=0, prompt_token_ids=list(range(16)),
            sampling_params=SamplingParams(max_new_tokens=16),  # 16+16=32 → 2 块 ≤ 4
        ))
        assert sched.num_waiting == 1


# ---------------- R2b: waiting 不抢占 running + ping-pong 终止 ----------------


class TestR2WaitingDoesNotPreempt:
    def test_waiting_stops_when_blocks_short(self):
        """running 已占块 → waiting 请求留在队列、不触发任何抢占（对齐 vLLM）。"""
        paged = make_paged(num_blocks=4)
        sched = make_scheduler(paged, enable_prefix_cache=False)
        r0 = Sequence(seq_id=0, prompt_token_ids=list(range(48)),
                      sampling_params=SamplingParams(max_new_tokens=4))   # 3 块
        w0 = Sequence(seq_id=1, prompt_token_ids=list(range(100, 148)),
                      sampling_params=SamplingParams(max_new_tokens=4))   # 3 块
        sched.add_request(r0)
        sched.add_request(w0)

        out = sched.schedule()
        assert [s.seq.seq_id for s in out.scheduled] == [0]
        assert out.preempted_seq_ids == []  # 关键：waiting 不再抢占 running
        assert sched.num_waiting == 1
        assert sched.num_running == 1

    def test_running_self_preempt_still_works(self):
        """RUNNING 段的自身抢占（修复后唯一抢占路径）不受影响。"""
        paged = make_paged(num_blocks=6)
        sched = make_scheduler(paged, enable_prefix_cache=False)
        # 两条 prompt=48（3 块）同时 running（6 块用满），decode 到 49 token 各需第 4 块
        params = SamplingParams(max_new_tokens=8)
        a = Sequence(seq_id=0, prompt_token_ids=list(range(48)), sampling_params=params)
        b = Sequence(seq_id=1, prompt_token_ids=list(range(100, 148)), sampling_params=params)
        sched.add_request(a)
        sched.add_request(b)

        out = sched.schedule()  # 双双准入
        assert len(out.scheduled) == 2
        sched.update_from_output(out, {0: 7, 1: 7})  # 各 +1 output → 49 token

        out = sched.schedule()  # 各需第 4 块，free=0 → 列表首位 a 自身抢占
        assert out.preempted_seq_ids == [0]
        assert sched.num_running == 1
        assert sched.num_waiting == 1

    def test_pingpong_terminates(self):
        """组合需求 > 池容量时（修复前 waiting 抢 running 会 ping-pong livelock），
        两条请求都应在有界步数内完成。"""
        paged = make_paged(num_blocks=6)  # 96 token
        sched = make_scheduler(paged, max_num_batched_tokens=128, enable_prefix_cache=False)
        params = SamplingParams(max_new_tokens=8)  # 48+8=56 token → 4 块；2×4 > 6
        seqs = [
            Sequence(seq_id=0, prompt_token_ids=list(range(48)), sampling_params=params),
            Sequence(seq_id=1, prompt_token_ids=list(range(100, 148)), sampling_params=params),
        ]
        for s in seqs:
            sched.add_request(s)

        max_steps = 300  # 修复前此场景互相抢占、output 被反复 reset，永不完成
        for _ in range(max_steps):
            if not sched.has_requests():
                break
            out = sched.schedule()
            sampled = {
                s.seq.seq_id: 7
                for s in out.scheduled
                if s.samples_this_step
            }
            sched.update_from_output(out, sampled)
        else:
            pytest.fail(f"ping-pong 未在 {max_steps} 步内完成（livelock 回归）")

        assert all(s.status == SequenceStatus.FINISHED for s in seqs)
        assert [len(s.output_token_ids) for s in seqs] == [8, 8]


# ---------------- E2E: R1 同 prompt 重放（0.5B / CPU） ----------------


class TestR1SamePromptReplayE2E(unittest.TestCase):
    """同 prompt（长度为 block_size 整数倍）连续两次生成：第二次全 prompt 命中，
    修复前会在 prefill 空批上崩溃/挂起，修复后输出与第一次完全一致。"""

    @classmethod
    def setUpClass(cls):
        cls.runner = NanoRunner(
            "models/Qwen2.5-0.5B-Instruct", device="cpu", dtype=torch.float32,
            max_seq_len=256, block_size=BLOCK_SIZE, num_blocks=64,
        )

    @classmethod
    def tearDownClass(cls):
        del cls.runner
        gc.collect()

    def test_replay_identical_output(self):
        self.runner.paged_cache.reset()
        scheduler = Scheduler(
            self.runner.paged_cache, max_num_batched_tokens=2048,
            watermark_blocks=0, enable_prefix_cache=True,
        )
        engine = EngineCore(self.runner, scheduler)

        prompt = list(range(100, 132))  # 32 = 2 × 16，整除
        params = SamplingParams(temperature=0.0, max_new_tokens=8)

        out1 = engine.generate([prompt], params)[0]
        out2 = engine.generate([prompt], params)[0]  # 全 prompt 命中场景
        self.assertEqual(out1, out2, f"重放输出分叉: {out1} != {out2}")
        self.assertEqual(len(out2), 8)
        self.assertGreater(
            scheduler.prefix_hit_tokens, 0, "第二次重放未发生前缀命中，测试无效"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
