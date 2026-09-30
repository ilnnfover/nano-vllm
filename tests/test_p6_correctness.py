"""P6 · 前缀缓存端到端正确性测试。

覆盖（对应 roadmap §P6 完成标准）:
  1. 共享前缀场景：开/关前缀缓存 greedy 输出**逐 token 一致**（正确性红线）
  2. 命中确实**跳过 prefill 重算**：prefill 处理 token 数之差 == 命中 token 数
  3. 抢占 × 前缀缓存：紧显存触发抢占，输出仍与 P1 eager 一致，且抢占后重新命中
  4. 调度器级：抢占 reset 后重新命中自己的前缀块（不依赖模型）

对拍基准:
  - 关闭前缀缓存的同一条 P4 连续批路径（同设备/同 dtype/同 attention 后端 → 同路径逐 token 一致）
  - P1 eager 串行 greedy

内存说明: 本文件主 runner 全程复用；抢占用例用的 5 块小池 runner 用完即释放；
tearDownClass 主动释放权重，避免整仓测试套件内存叠加超限。

运行:
  python -m pytest tests/test_p6_correctness.py -v
"""
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest

import torch

from nano_vllm.engine.core import EngineCore
from nano_vllm.engine.scheduler import Scheduler
from nano_vllm.engine.sequence import SamplingParams, Sequence, SequenceStatus
from nano_vllm.kvmm.paged_kv_cache import PagedKVCache
from nano_vllm.model_executor.runner import NanoRunner

MODEL = "models/Qwen2.5-0.5B-Instruct"
BLOCK_SIZE = 16
PREFIX_LEN = 32            # 2 个满块
SUFFIX_LEN = 16            # 1 个块
NUM_BLOCKS = 64


def make_runner(num_blocks: int = NUM_BLOCKS, max_seq_len: int = 256) -> NanoRunner:
    return NanoRunner(
        MODEL, device="cpu", dtype=torch.float32,
        max_seq_len=max_seq_len, block_size=BLOCK_SIZE, num_blocks=num_blocks,
    )


def make_engine(
    runner: NanoRunner,
    budget: int = 2048,
    enable_prefix_cache: bool = True,
    watermark_blocks: int = 0,
) -> EngineCore:
    scheduler = Scheduler(
        paged_cache=runner.paged_cache,
        max_num_batched_tokens=budget,
        watermark_blocks=watermark_blocks,
        enable_prefix_cache=enable_prefix_cache,
    )
    return EngineCore(runner, scheduler)


def prefix_prompts(tag_base: int, count: int, suffix_len: int = SUFFIX_LEN):
    """构造 count 条共享 PREFIX_LEN 前缀、各自 suffix_len 独立后缀的 prompt。"""
    prefix = list(range(100, 100 + PREFIX_LEN))
    prompts = [
        prefix + list(range(tag_base + i * suffix_len, tag_base + (i + 1) * suffix_len))
        for i in range(count)
    ]
    return prefix, prompts


def drive_interleaved(
    runner: NanoRunner,
    prompts: list[list[int]],
    params: SamplingParams,
    enable_prefix_cache: bool = True,
) -> tuple[list[list[int]], EngineCore]:
    """顺序提交：前一条 prefill 完成后才提交下一条，使后续请求能命中前缀块。"""
    runner.paged_cache.reset()
    engine = make_engine(runner, enable_prefix_cache=enable_prefix_cache)
    seqs: list[Sequence] = []
    for prompt in prompts:
        seq = engine.add_request(prompt, params)
        seqs.append(seq)
        while seq.num_computed_tokens < seq.num_prompt_tokens:
            engine.step()
    while engine.scheduler.has_requests():
        engine.step()
    return [s.output_token_ids for s in seqs], engine


class TestP6PrefixCacheE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = make_runner()

    @classmethod
    def tearDownClass(cls):
        del cls.runner
        gc.collect()

    def setUp(self):
        self.runner.paged_cache.reset()

    def test_shared_prefix_output_identical(self):
        """开/关前缀缓存：共享前缀场景 greedy 输出逐 token 一致。"""
        _, prompts = prefix_prompts(tag_base=2000, count=2)
        params = SamplingParams(temperature=0.0, max_new_tokens=8)

        out_cache, eng_cache = drive_interleaved(self.runner, prompts, params, enable_prefix_cache=True)
        self.assertGreater(
            eng_cache.stats.prefix_cache_hit_tokens, 0,
            "开缓存路径未发生前缀命中，测试无效",
        )

        out_nocache, eng_nocache = drive_interleaved(self.runner, prompts, params, enable_prefix_cache=False)
        self.assertEqual(eng_nocache.stats.prefix_cache_hit_tokens, 0)
        self.assertEqual(eng_nocache.stats.prefix_cache_lookups, 0)

        self.assertEqual(
            out_cache, out_nocache,
            f"开缓存输出 != 关缓存输出\ncache={out_cache}\nnocache={out_nocache}",
        )

    def test_hit_skips_prefill_recompute(self):
        """命中跳过的重算量：prefill 处理 token 数之差 == 命中的前缀 token 数。"""
        _, prompts = prefix_prompts(tag_base=3000, count=2)

        n_cache = self._count_prefill_tokens(prompts, enable_prefix_cache=True)
        n_nocache = self._count_prefill_tokens(prompts, enable_prefix_cache=False)

        self.assertEqual(n_nocache, 2 * (PREFIX_LEN + SUFFIX_LEN))
        self.assertEqual(n_cache, (PREFIX_LEN + SUFFIX_LEN) + SUFFIX_LEN)
        self.assertEqual(n_nocache - n_cache, PREFIX_LEN)

    def _count_prefill_tokens(self, prompts: list[list[int]], enable_prefix_cache: bool) -> int:
        """统计进入 prefill forward 的 token 总数（包装 _run_prefill_batched）。"""
        self.runner.paged_cache.reset()
        engine = make_engine(self.runner, enable_prefix_cache=enable_prefix_cache)
        counter = {"n": 0}
        orig = engine._run_prefill_batched

        def wrapped(scheduled_prefills):
            counter["n"] += sum(s.num_tokens for s in scheduled_prefills)
            return orig(scheduled_prefills)

        engine._run_prefill_batched = wrapped

        params = SamplingParams(temperature=0.0, max_new_tokens=8)
        for prompt in prompts:
            seq = engine.add_request(prompt, params)
            while seq.num_computed_tokens < seq.num_prompt_tokens:
                engine.step()
        while engine.scheduler.has_requests():
            engine.step()
        return counter["n"]

    def test_preemption_with_prefix_cache(self):
        """小池触发抢占；抢占后仍能命中前缀，且输出与 P1 eager 逐 token 一致。

        池 5 块；每条 prompt = 32 共享前缀 + 4 独立 = 36 token（3 块），
        decode 至 51 token 需 4 块 → 两条并发放不下 → 触发 LIFO 抢占。
        注: 用独立小池 runner（而非 watermark）——watermark 语义是"全局预留"，
        与"小物理池"不等价，会让单条请求也放不下而互相抢占成死循环。
        """
        runner = make_runner(num_blocks=5, max_seq_len=128)
        try:
            _, prompts = prefix_prompts(tag_base=4000, count=3, suffix_len=4)
            params = SamplingParams(temperature=0.0, max_new_tokens=15)

            runner.paged_cache.reset()
            engine = make_engine(runner, enable_prefix_cache=True)

            # 第 1 条跑完（注册前缀块 hash 后释放），再提交后两条制造块竞争 → 抢占
            s0 = engine.add_request(prompts[0], params)
            while s0.status != SequenceStatus.FINISHED:
                engine.step()
            seqs = [s0]
            for prompt in prompts[1:]:
                seqs.append(engine.add_request(prompt, params))
            while engine.scheduler.has_requests():
                engine.step()

            self.assertGreater(engine.stats.total_preempt_count, 0, "未触发抢占，测试配置无效")
            self.assertGreater(engine.stats.prefix_cache_hit_tokens, 0, "抢占场景未发生前缀命中")

            for prompt, seq in zip(prompts, seqs):
                ref = runner.generate(
                    prompt, SamplingParams(temperature=0.0, max_new_tokens=params.max_new_tokens),
                    use_cache=False,
                )
                self.assertEqual(
                    seq.output_token_ids, ref,
                    f"preempt+cache 输出 != P1 eager\nseq={seq.output_token_ids}\nref={ref}",
                )
        finally:
            del runner
            gc.collect()


class TestP6PreemptionRelookup(unittest.TestCase):
    """调度器级：抢占 reset 后重新命中自己的前缀块（不依赖模型）。"""

    def test_relookup_after_preemption_hits(self):
        paged = PagedKVCache(
            num_layers=1, num_blocks=16, block_size=BLOCK_SIZE,
            num_kv_heads=1, head_dim=2, dtype=torch.float32, device="cpu",
        )
        sched = Scheduler(
            paged, max_num_batched_tokens=64, watermark_blocks=0,
            enable_prefix_cache=True,
        )
        prefix = list(range(100, 100 + PREFIX_LEN))

        # 手动注册前缀满块（模拟历史请求留下的缓存）
        pc = sched.prefix_cache
        assert pc is not None, "本用例需开启前缀缓存"
        cached_blocks = paged.pool.allocate_n(PREFIX_LEN // BLOCK_SIZE)
        pc.register_blocks(
            prefix, cached_blocks, num_full_blocks=PREFIX_LEN // BLOCK_SIZE,
            start_block=0, prev_hash=None,
        )
        paged.pool.free_n(cached_blocks)

        seq = Sequence(
            seq_id=0,
            prompt_token_ids=prefix + list(range(200, 200 + SUFFIX_LEN)),
        )
        sched.add_request(seq)
        sched.schedule()
        self.assertEqual(sched.prefix_hit_tokens, PREFIX_LEN)
        self.assertEqual(seq.num_computed_tokens, PREFIX_LEN)

        # 抢占：释放块 + 重置前缀状态
        sched._preempt(seq)
        self.assertEqual(seq.num_computed_tokens, 0)
        self.assertFalse(seq.prefix_cache_done)
        self.assertEqual(seq.block_table, [])

        # 重新调度：应再次命中（被释放的块 hash 仍在索引中）
        sched.schedule()
        self.assertEqual(sched.prefix_lookups, 2)
        self.assertEqual(sched.prefix_hit_tokens, 2 * PREFIX_LEN)
        self.assertEqual(seq.num_computed_tokens, PREFIX_LEN)


if __name__ == "__main__":
    unittest.main()
