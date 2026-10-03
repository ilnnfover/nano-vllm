"""P7 · Decode CUDA Graph 正确性测试。

覆盖（对应 roadmap §P7 完成标准）:
  1. 张量寻址路径 == list 寻址路径（图捕获的前置改造，逐元素一致）
  2. 图回放输出 == eager 输出（逐 token 一致），含**边界 batch size**：
     n=1（桶 1）/ n=3（桶 4，含 padding 行）/ n=5（超过最大桶 → 回退 eager）
  3. uniform decode 全部命中图（纯 decode 负载 fallback == 0）
  4. padding 行不污染真实请求的 KV（由 2 的输出一致性 + scratch 块预留共同保证）
  5. 埋点：replays / fallback / padded_seqs / capture_ms 有值

运行:
  ~/venvs/dev/bin/python -m pytest tests/test_p7_correctness.py -v
  （无 CUDA 时整体 skip；图捕获只支持 attn_impl="triton"）
"""
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest

import pytest
import torch

from nano_vllm.attention.metadata import AttentionMetadata
from nano_vllm.engine.core import EngineCore
from nano_vllm.engine.scheduler import Scheduler
from nano_vllm.engine.sequence import SamplingParams, Sequence
from nano_vllm.model_executor.runner import NanoRunner

MODEL = "models/Qwen2.5-0.5B-Instruct"
BLOCK_SIZE = 16
BUCKETS = (1, 2, 4)
GPU = torch.cuda.is_available()
skip_no_gpu = pytest.mark.skipif(not GPU, reason="图捕获需要 CUDA 设备")


def make_runner(num_blocks: int = 128, buckets=BUCKETS) -> NanoRunner:
    return NanoRunner(
        MODEL, device="cuda", dtype=torch.bfloat16,
        max_seq_len=512, block_size=BLOCK_SIZE, num_blocks=num_blocks,
        attn_impl="triton", enable_cudagraph=True, cudagraph_buckets=buckets,
    )


def make_engine(runner: NanoRunner, budget: int = 2048) -> EngineCore:
    scheduler = Scheduler(
        paged_cache=runner.paged_cache,
        max_num_batched_tokens=budget,
        watermark_blocks=0,
        enable_prefix_cache=False,
    )
    return EngineCore(runner, scheduler)


def run_generate(runner: NanoRunner, prompts: list[list[int]], max_new_tokens: int,
                 use_graph: bool) -> tuple[list[list[int]], EngineCore]:
    """跑一次连续批生成；use_graph 控制是否走图（同池、同 dtype、同后端）。"""
    runner.paged_cache.reset()
    engine = make_engine(runner)
    saved, runner.graph_runner = runner.graph_runner, (runner.graph_runner if use_graph else None)
    try:
        outs = engine.generate(prompts, SamplingParams(temperature=0.0,
                                                       max_new_tokens=max_new_tokens))
    finally:
        runner.graph_runner = saved
    return outs, engine


@skip_no_gpu
class TestP7TensorPath(unittest.TestCase):
    """第 1 步：静态张量寻址与 list 寻址必须逐元素一致。"""

    @classmethod
    def setUpClass(cls):
        cls.runner = make_runner(num_blocks=64)
        cls.runner.graph_runner = None  # 本用例手搓 metadata，不需要图

    @classmethod
    def tearDownClass(cls):
        del cls.runner
        gc.collect()

    def _prefill(self, prompts: list[list[int]]) -> list[Sequence]:
        runner = self.runner
        paged = runner.paged_cache
        seqs: list[Sequence] = []
        for i, prompt in enumerate(prompts):
            seq = Sequence(seq_id=i, prompt_token_ids=list(prompt),
                           sampling_params=SamplingParams(temperature=0.0, max_new_tokens=2))
            seq.block_table = paged.allocate(len(prompt))
            md = AttentionMetadata(
                is_prefill=True,
                slot_mapping=paged.slot_mapping(seq.block_table, 0, len(prompt)),
                block_table=seq.block_table,
                seq_len=len(prompt),
                prefill_impl="torch",
            )
            t = torch.tensor([prompt], dtype=torch.long, device=runner.device)
            pos = torch.arange(len(prompt), device=runner.device).unsqueeze(0)
            last = torch.tensor([len(prompt) - 1], device=runner.device)
            logits, _ = runner.model(
                t, paged_cache=paged, metadata=md, position_ids=pos, last_idx=last,
            )
            seq.num_computed_tokens = len(prompt)
            seq.output_token_ids = [int(logits[0].argmax())]
            seqs.append(seq)
        return seqs

    def _decode_logits(self, seqs: list[Sequence], tensor_path: bool) -> torch.Tensor:
        runner = self.runner
        paged = runner.paged_cache
        n = len(seqs)
        input_ids = torch.tensor(
            [[s.output_token_ids[-1]] for s in seqs], dtype=torch.long, device=runner.device
        )
        positions = torch.tensor(
            [[s.num_computed_tokens] for s in seqs], device=runner.device
        )
        slot_mapping = torch.cat([
            paged.slot_mapping(s.block_table, s.num_computed_tokens, 1) for s in seqs
        ])
        if tensor_path:
            max_blocks = min(paged.num_blocks,
                             runner.config.max_position_embeddings // BLOCK_SIZE + 1)
            bt = torch.zeros((n, max_blocks), dtype=torch.int32, device=runner.device)
            for i, s in enumerate(seqs):
                bt[i, :len(s.block_table)] = torch.tensor(
                    s.block_table, dtype=torch.int32, device=runner.device
                )
            md = AttentionMetadata(
                is_prefill=False,
                slot_mapping=slot_mapping,
                block_table_tensor=bt,
                seq_lens_tensor=torch.tensor(
                    [s.num_computed_tokens + 1 for s in seqs],
                    dtype=torch.int32, device=runner.device,
                ),
                attn_impl="triton",
            )
        else:
            md = AttentionMetadata(
                is_prefill=False,
                slot_mapping=slot_mapping,
                block_tables=[s.block_table for s in seqs],
                seq_lens=[s.num_computed_tokens + 1 for s in seqs],
                attn_impl="triton",
            )
        logits, _ = runner.model(
            input_ids, paged_cache=paged, metadata=md, position_ids=positions,
        )
        return logits[0, -1] if n == 1 else logits[0]

    def test_tensor_path_equals_list_path(self):
        prompts = [
            list(range(100, 121)),
            list(range(200, 231)),
            list(range(300, 313)),
        ]
        seqs = self._prefill(prompts)

        out_list = self._decode_logits(seqs, tensor_path=False)
        out_tensor = self._decode_logits(seqs, tensor_path=True)
        self.assertTrue(
            torch.equal(out_list, out_tensor),
            f"张量寻址与 list 寻址结果不一致 (max diff "
            f"{(out_list - out_tensor).abs().max().item():.3e})",
        )


def snapshot_graph_stats(stats) -> dict:
    return {
        "replays": stats.replays,
        "fallback_eager": stats.fallback_eager,
        "padded_seqs": stats.padded_seqs,
    }


def delta(before: dict, after: dict) -> dict:
    return {k: after[k] - before[k] for k in before}


@skip_no_gpu
class TestP7GraphCorrectness(unittest.TestCase):
    """第 2/3 步：图回放 == eager，边界 batch size 全覆盖。

    注意: graph.stats 是 **runner 级累计值**，跨用例不清零，故断言全部用增量。
    """

    @classmethod
    def setUpClass(cls):
        cls.runner = make_runner()
        # 预热一次触发懒捕获，使后续用例的增量统计只包含自身回放
        run_generate(cls.runner, [list(range(100, 120))], 2, use_graph=True)
        assert cls.runner.graph_runner.captured

    @classmethod
    def tearDownClass(cls):
        del cls.runner
        gc.collect()

    def _assert_graph_equals_eager(self, prompts: list[list[int]], max_new_tokens: int = 6):
        """跑 eager / 图两条路径并比对输出；返回（图路径 outputs, 统计增量）。"""
        before = snapshot_graph_stats(self.runner.graph_runner.stats)
        eager_outs, _ = run_generate(self.runner, prompts, max_new_tokens, use_graph=False)
        graph_outs, engine = run_generate(self.runner, prompts, max_new_tokens, use_graph=True)
        after = snapshot_graph_stats(self.runner.graph_runner.stats)
        self.assertEqual(
            graph_outs, eager_outs,
            f"图回放输出 != eager 输出\nprompts={[len(p) for p in prompts]}\n"
            f"graph={graph_outs}\neager={eager_outs}",
        )
        return graph_outs, delta(before, after), engine

    def test_bucket_1(self):
        """n=1 → 桶 1（边界 batch size）。"""
        _, d, _ = self._assert_graph_equals_eager([list(range(100, 120))])
        self.assertGreater(d["replays"], 0, "n=1 未命中图")
        self.assertEqual(d["fallback_eager"], 0)

    def test_bucket_padding(self):
        """n=3 → 桶 4（含 1 行 padding），输出仍与 eager 一致。"""
        prompts = [list(range(100, 120)), list(range(200, 220)), list(range(300, 320))]
        _, d, _ = self._assert_graph_equals_eager(prompts)
        self.assertGreater(d["replays"], 0)
        self.assertGreater(d["padded_seqs"], 0, "未观察到 padding 行（测试配置无效）")
        self.assertEqual(d["fallback_eager"], 0)

    def test_exact_bucket_no_padding(self):
        """n=4（恰好满桶）→ padding 行数应为 0。"""
        prompts = [list(range(100 + 10 * i, 120 + 10 * i)) for i in range(4)]
        _, d, _ = self._assert_graph_equals_eager(prompts)
        self.assertEqual(d["padded_seqs"], 0, "恰好满桶时不应有 padding 行")

    def test_over_max_bucket_falls_back(self):
        """n=5 > 最大桶 4 → 回退 eager，输出仍逐 token 一致。"""
        prompts = [list(range(100 + 10 * i, 120 + 10 * i)) for i in range(5)]
        _, d, _ = self._assert_graph_equals_eager(prompts)
        self.assertGreater(d["fallback_eager"], 0, "超桶未回退 eager")
        self.assertEqual(d["replays"], 0, "5 条等长请求全程超桶，不应有图命中")

    def test_uniform_decode_all_hits(self):
        """纯 decode 负载（等长 prompt + n ≤ 最大桶）：uniform decode 全部命中图。"""
        prompts = [list(range(100 + 10 * i, 120 + 10 * i)) for i in range(3)]
        _, d, _ = self._assert_graph_equals_eager(prompts, max_new_tokens=8)
        self.assertGreater(d["replays"], 0)
        self.assertEqual(
            d["fallback_eager"], 0,
            "uniform decode 出现回退（未全部命中图）",
        )

    def test_mixed_replays_and_fallback_in_one_run(self):
        """同一轮内既有命中（≤4 条 decode）又有回退（加入第 5 条后）。"""
        runner = self.runner
        runner.paged_cache.reset()
        engine = make_engine(runner)
        before = snapshot_graph_stats(runner.graph_runner.stats)
        params = SamplingParams(temperature=0.0, max_new_tokens=12)

        for i in range(4):
            engine.add_request(list(range(100 + 10 * i, 120 + 10 * i)), params)
        for _ in range(3):  # 4 条 decode → 命中桶 4
            engine.step()
        engine.add_request(list(range(500, 520)), params)  # 第 5 条入场
        while engine.scheduler.has_requests():
            engine.step()

        d = delta(before, snapshot_graph_stats(runner.graph_runner.stats))
        self.assertGreater(d["replays"], 0, "≤4 条 decode 时应命中图")
        self.assertGreater(d["fallback_eager"], 0, "5 条 decode 时应回退 eager")

    def test_capture_stats_recorded(self):
        """捕获埋点：桶数 / 捕获耗时 / 图缓存 / 输出 buffer 落在图池内。"""
        graph = self.runner.graph_runner
        self.assertIsNotNone(graph)
        self.assertTrue(graph.captured)
        self.assertEqual(graph.stats.captured_buckets, len(BUCKETS))
        self.assertGreater(graph.stats.capture_ms, 0.0)
        self.assertEqual(sorted(graph.graphs), list(BUCKETS))
        # 输出 buffer 必须落在图池里；P1 起图输出是 hidden_states（logits 在图外算）
        for bucket, out in graph.outputs.items():
            self.assertEqual(
                tuple(out.shape), (bucket, self.runner.config.hidden_size)
            )
        # P2: 输入只有**一套**按最大桶分配的缓冲，各桶用前缀视图共享
        self.assertEqual(graph.buffers.max_bucket, max(BUCKETS))
        self.assertEqual(tuple(graph.buffers.input_ids.shape), (max(BUCKETS),))
        self.assertEqual(
            tuple(graph.buffers.block_table.shape),
            (max(BUCKETS), graph.max_blocks),
        )
        # 前缀视图共享同一基地址（这正是 P2 的安全前提）
        for bucket in BUCKETS:
            self.assertEqual(
                graph.buffers.input_ids[:bucket].data_ptr(),
                graph.buffers.input_ids.data_ptr(),
            )

    def test_scratch_block_reserved_across_reset(self):
        """padding 落点块必须永久预留：不在 free 队列、reset 后仍是同一块。"""
        graph = self.runner.graph_runner
        scratch = graph.scratch_block
        pool = self.runner.paged_cache.pool
        self.assertNotIn(scratch, pool.free_block_ids)
        self.runner.paged_cache.reset()
        self.assertNotIn(scratch, pool.free_block_ids)
        self.assertEqual(graph.scratch_block, scratch)
        self.assertEqual(pool.blocks[scratch].ref_cnt, 1)


@skip_no_gpu
class TestP7GraphVsEagerLongerSequence(unittest.TestCase):
    """较长上下文下的图/非图一致性（覆盖 seq_len 跨多个块、decode 增长换块）。"""

    @classmethod
    def setUpClass(cls):
        cls.runner = make_runner(num_blocks=256)

    @classmethod
    def tearDownClass(cls):
        del cls.runner
        gc.collect()

    def test_long_prompt_multi_block(self):
        prompts = [list(range(1000, 1000 + 200)), list(range(3000, 3000 + 150))]
        eager_outs, _ = run_generate(self.runner, prompts, 8, use_graph=False)
        graph_outs, engine = run_generate(self.runner, prompts, 8, use_graph=True)
        self.assertEqual(graph_outs, eager_outs)
        self.assertGreater(engine.stats.cudagraph_replays, 0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
