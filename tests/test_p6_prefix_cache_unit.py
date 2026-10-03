"""P6 模块 2 单元测试：prefix_cache.py + PagedKVCache.copy_block。

覆盖:
  - compute_block_hash: 确定性 / 滚动链 / 不同 prev_hash 产生不同 hash
  - CacheHit: dataclass 基本属性
  - PrefixCache.find_longest_hit: 空 / 全命中 / partial 命中 / 未命中 / 部分命中后断链
  - PrefixCache.register_blocks: 注册满块 hash / 返回滚动链状态 / 跳过已注册块
  - PrefixCache.cow_for_write: 复制块 / ref_cnt 变化 / 新块无 hash / 原块 hash 保留
  - PagedKVCache.copy_block: 跨层 KV 复制正确性
  - Scheduler 命中率埋点: lookup 计数 / 命中 token / 跳过 prefill / 开关
  - extra keys / salt: token 相同但 KV 不该共享时不得命中（LoRA 位预留语义）
"""
from __future__ import annotations

import pytest
import torch

from nano_vllm.engine.scheduler import Scheduler
from nano_vllm.engine.sequence import SamplingParams, Sequence
from nano_vllm.engine.stats import EngineCoreStats
from nano_vllm.kvmm.block_pool import BlockPool
from nano_vllm.kvmm.paged_kv_cache import PagedKVCache
from nano_vllm.kvmm.prefix_cache import (
    CacheHit,
    PrefixCache,
    compute_block_hash,
)


# ---------------- compute_block_hash ----------------


class TestComputeBlockHash:
    def test_deterministic(self):
        tokens = [1, 2, 3]
        assert compute_block_hash(tokens, None) == compute_block_hash(tokens, None)

    def test_different_tokens(self):
        assert compute_block_hash([1, 2, 3], None) != compute_block_hash([1, 2, 4], None)

    def test_prev_hash_changes_result(self):
        tokens = [1, 2, 3]
        h0 = compute_block_hash(tokens, None)
        h1 = compute_block_hash(tokens, h0)
        assert h0 != h1

    def test_rolling_chain_order_matters(self):
        """hash(a||b) != hash(b||a) 当 prev_hash 不同时。"""
        tokens = [3, 4, 5]
        h_a = compute_block_hash([1, 2], None)
        h_b = compute_block_hash([2, 1], None)
        assert compute_block_hash(tokens, h_a) != compute_block_hash(tokens, h_b)

    def test_returns_bytes(self):
        h = compute_block_hash([1], None)
        assert isinstance(h, bytes)
        assert len(h) == 32  # sha256 digest


# ---------------- CacheHit ----------------


class TestCacheHit:
    def test_empty_hit(self):
        hit = CacheHit([], 0, False)
        assert hit.num_blocks == 0
        assert hit.num_tokens == 0
        assert not hit.is_partial

    def test_full_hit(self):
        hit = CacheHit([0, 1], 32, False)
        assert hit.num_blocks == 2
        assert hit.num_tokens == 32

    def test_partial_hit(self):
        hit = CacheHit([0, 1], 20, True)
        assert hit.is_partial
        assert hit.num_blocks == 2


# ---------------- PrefixCache.find_longest_hit ----------------


class TestFindLongestHit:
    def test_empty_tokens(self):
        pool = BlockPool(8)
        pc = PrefixCache(pool, block_size=16)
        hit = pc.find_longest_hit([])
        assert hit.block_ids == []
        assert hit.num_tokens == 0
        assert not hit.is_partial

    def test_no_hit(self):
        pool = BlockPool(8)
        pc = PrefixCache(pool, block_size=16)
        hit = pc.find_longest_hit([1, 2, 3])
        assert hit.block_ids == []
        assert hit.num_tokens == 0

    def test_full_hit(self):
        """注册 2 个满块，查找相同 token 应全命中。"""
        pool = BlockPool(8)
        pc = PrefixCache(pool, block_size=4)
        tokens = [10, 11, 12, 13, 20, 21, 22, 23]
        # 分配 2 块并注册 hash
        block_ids = pool.allocate_n(2)
        pc.register_blocks(tokens, block_ids, num_full_blocks=2, start_block=0, prev_hash=None)
        # 释放回缓存（ref_cnt=0，但 hash 保留在 _hash_index）
        pool.free_n(block_ids)
        # 查找
        hit = pc.find_longest_hit(tokens)
        assert hit.block_ids == block_ids
        assert hit.num_tokens == 8
        assert not hit.is_partial

    def test_partial_hit(self):
        """注册 1 满块 + 1 partial 块，查找应命中 partial。"""
        pool = BlockPool(8)
        pc = PrefixCache(pool, block_size=4)
        tokens = [10, 11, 12, 13, 20, 21]  # 1 满块 + 2 token partial
        block_ids = pool.allocate_n(2)
        # 注册满块
        pc.register_blocks(tokens, block_ids, num_full_blocks=1, start_block=0, prev_hash=None)
        # 手动注册 partial 块 hash
        h0 = compute_block_hash(tokens[0:4], None)
        h1 = compute_block_hash(tokens[4:6], h0)
        pool.register_hash(block_ids[1], h1, num_tokens=2)
        pool.free_n(block_ids)
        # 查找
        hit = pc.find_longest_hit(tokens)
        assert hit.block_ids == block_ids
        assert hit.num_tokens == 6
        assert hit.is_partial

    def test_partial_miss_breaks_chain(self):
        """第 2 块未命中时，只返回第 1 块命中。"""
        pool = BlockPool(8)
        pc = PrefixCache(pool, block_size=4)
        tokens = [10, 11, 12, 13, 20, 21, 22, 23]
        block_ids = pool.allocate_n(1)
        pc.register_blocks(tokens, block_ids, num_full_blocks=1, start_block=0, prev_hash=None)
        pool.free_n(block_ids)
        # 查找 8 个 token，只有前 4 命中
        hit = pc.find_longest_hit(tokens)
        assert hit.num_blocks == 1
        assert hit.num_tokens == 4
        assert not hit.is_partial  # 第 1 块是满块

    def test_default_config_never_yields_partial(self):
        """B1 修订口径固化：默认只注册满块 → 未注册的尾块命中不了，`is_partial` 恒 False。

        这是「纯 Transformer 下 COW 永不触发」的前提，也是 roadmap P6 验收口径修订的依据。
        """
        pool = BlockPool(8)
        pc = PrefixCache(pool, block_size=4)
        tokens = [1, 2, 3, 4, 5, 6]  # 1 个满块 + 1 个 2-token 尾块
        block_ids = pool.allocate_n(2)
        pc.register_blocks(tokens, block_ids, num_full_blocks=1, start_block=0, prev_hash=None)
        pool.free_n(block_ids)

        hit = pc.find_longest_hit(tokens)
        assert hit.block_ids == block_ids[:1]
        assert hit.num_tokens == 4
        assert hit.is_partial is False
        assert hit.num_full_blocks == 1


# ---------------- PrefixCache.register_blocks ----------------


class TestRegisterBlocks:
    def test_register_full_blocks(self):
        pool = BlockPool(8)
        pc = PrefixCache(pool, block_size=4)
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        block_ids = pool.allocate_n(2)
        new_start, last_hash = pc.register_blocks(
            tokens, block_ids, num_full_blocks=2, start_block=0, prev_hash=None
        )
        assert new_start == 2
        assert last_hash is not None
        # 验证 hash 已注册
        assert pool.has_hash(block_ids[0])
        assert pool.has_hash(block_ids[1])

    def test_skip_already_registered(self):
        """start_block > 0 时跳过已注册块。"""
        pool = BlockPool(8)
        pc = PrefixCache(pool, block_size=4)
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        block_ids = pool.allocate_n(2)
        # 先注册第 0 块
        _, h0 = pc.register_blocks(
            tokens, block_ids, num_full_blocks=1, start_block=0, prev_hash=None
        )
        # 再从第 1 块开始注册
        new_start, last_hash = pc.register_blocks(
            tokens, block_ids, num_full_blocks=2, start_block=1, prev_hash=h0
        )
        assert new_start == 2
        assert last_hash is not None
        assert pool.has_hash(block_ids[1])

    def test_nothing_to_register(self):
        """num_full_blocks == start_block 时无操作。"""
        pool = BlockPool(8)
        pc = PrefixCache(pool, block_size=4)
        tokens = [1, 2, 3, 4]
        block_ids = pool.allocate_n(1)
        new_start, last_hash = pc.register_blocks(
            tokens, block_ids, num_full_blocks=1, start_block=1, prev_hash=None
        )
        assert new_start == 1
        assert last_hash is None



# ---------------- PrefixCache.cow_for_write ----------------


class TestCowForWrite:
    def _make_paged_cache(self):
        return PagedKVCache(
            num_layers=2,
            num_blocks=8,
            block_size=4,
            num_kv_heads=2,
            head_dim=3,
            dtype=torch.float32,
            device="cpu",
        )

    def test_cow_basic(self):
        """COW: 新块 Y 拿到 KV 副本，原块 X ref_cnt-1。"""
        paged = self._make_paged_cache()
        pc = PrefixCache(paged.pool, block_size=4)
        # 分配块 X 并写入一些 KV
        x = paged.pool.allocate()
        paged.k_cache[0, x, 0, 0, 0] = 42.0
        paged.v_cache[1, x, 1, 1, 1] = 99.0
        # 再分配一个块占位，确保 COW 分配的 Y != X
        _ = paged.pool.allocate()
        # COW
        y = pc.cow_for_write(paged, x)
        assert y != x
        # Y 拿到了 X 的 KV 副本
        assert paged.k_cache[0, y, 0, 0, 0] == 42.0
        assert paged.v_cache[1, y, 1, 1, 1] == 99.0
        # Y ref_cnt == 1（allocate 设的）
        assert paged.pool.blocks[y].ref_cnt == 1
        # X ref_cnt == 0（free 了）
        assert paged.pool.blocks[x].ref_cnt == 0

    def test_cow_preserves_hash_on_original(self):
        """COW 后原块 X 的 hash 保留在 _hash_index（可被后续请求命中）。"""
        paged = self._make_paged_cache()
        pc = PrefixCache(paged.pool, block_size=4)
        x = paged.pool.allocate()
        # 给 X 注册 hash
        h = compute_block_hash([1, 2, 3, 4], None)
        paged.pool.register_hash(x, h, num_tokens=4)
        _ = paged.pool.allocate()
        # COW
        y = pc.cow_for_write(paged, x)
        # X 的 hash 仍在索引中
        assert paged.pool.get_block_id_by_hash(h) == x
        # Y 无 hash
        assert not paged.pool.has_hash(y)

    def test_cow_with_shared_block(self):
        """X 被 2 个请求引用（ref_cnt=2），COW 后 X ref_cnt=1，另一引用者不受影响。"""
        paged = self._make_paged_cache()
        pc = PrefixCache(paged.pool, block_size=4)
        x = paged.pool.allocate()
        paged.pool.touch(x)  # ref_cnt = 2
        _ = paged.pool.allocate()
        y = pc.cow_for_write(paged, x)
        # X ref_cnt = 1（另一引用者仍持有）
        assert paged.pool.blocks[x].ref_cnt == 1
        assert paged.pool.blocks[y].ref_cnt == 1


# ---------------- PagedKVCache.copy_block ----------------


class TestCopyBlock:
    def test_copy_preserves_data(self):
        paged = PagedKVCache(
            num_layers=3, num_blocks=4, block_size=2,
            num_kv_heads=2, head_dim=4, dtype=torch.float32, device="cpu",
        )
        src = paged.pool.allocate()
        dst = paged.pool.allocate()
        # 写入已知数据
        for layer in range(3):
            paged.k_cache[layer, src] = torch.randn(2, 2, 4)
            paged.v_cache[layer, src] = torch.randn(2, 2, 4)
        paged.copy_block(src, dst)
        for layer in range(3):
            assert torch.equal(paged.k_cache[layer, src], paged.k_cache[layer, dst])
            assert torch.equal(paged.v_cache[layer, src], paged.v_cache[layer, dst])

    def test_copy_does_not_alias(self):
        """copy 后修改 dst 不影响 src（非别名）。"""
        paged = PagedKVCache(
            num_layers=1, num_blocks=4, block_size=2,
            num_kv_heads=1, head_dim=2, dtype=torch.float32, device="cpu",
        )
        src = paged.pool.allocate()
        dst = paged.pool.allocate()
        paged.k_cache[0, src, 0, 0, 0] = 1.0
        paged.copy_block(src, dst)
        paged.k_cache[0, dst, 0, 0, 0] = 999.0
        assert paged.k_cache[0, src, 0, 0, 0] == 1.0  # src 不受影响


# ---------------- Scheduler 命中率埋点（步骤 1） ----------------


class TestSchedulerPrefixStats:
    """调度器级命中率埋点：直接驱动 Scheduler（无需模型前向）。"""

    BLOCK_SIZE = 4

    def _make_scheduler(self, num_blocks: int = 16, budget: int = 8, enable: bool = True):
        paged = PagedKVCache(
            num_layers=1, num_blocks=num_blocks, block_size=self.BLOCK_SIZE,
            num_kv_heads=1, head_dim=2, dtype=torch.float32, device="cpu",
        )
        sched = Scheduler(
            paged, max_num_batched_tokens=budget, watermark_blocks=0,
            enable_prefix_cache=enable,
        )
        return paged, sched

    def _seed_cached_block(self, paged: PagedKVCache, tokens: list[int]) -> int:
        """在块池中注册一个已满块的 hash，并释放（模拟历史请求留下的缓存）。"""
        bid = paged.pool.allocate()
        paged.pool.register_hash(bid, compute_block_hash(tokens, None), num_tokens=len(tokens))
        paged.pool.free(bid)
        return bid

    def test_hit_accounting_and_prefill_skip(self):
        """命中时：查询/命中 token 与块数被计数，且 num_computed_tokens 跳过命中部分。"""
        paged, sched = self._make_scheduler()
        cached = [1, 2, 3, 4]
        bid = self._seed_cached_block(paged, cached)

        seq = Sequence(
            seq_id=0, prompt_token_ids=cached + [5, 6, 7, 8],
            sampling_params=SamplingParams(max_new_tokens=8),  # R2 守卫: 显式 max_new 适配小池
        )
        sched.add_request(seq)
        sched.schedule()

        assert sched.prefix_lookups == 1
        assert sched.prefix_query_tokens == 8
        assert sched.prefix_hit_tokens == 4
        assert sched.prefix_hit_blocks == 1
        assert sched.prefix_cache_hit_rate == pytest.approx(0.5)
        # 命中生效：跳过前 4 token，首块复用缓存块
        assert seq.num_computed_tokens == self.BLOCK_SIZE
        assert seq.block_table[0] == bid

    def test_no_hit_not_counted_as_hit(self):
        """未命中：查询计数增加但命中为 0。"""
        _, sched = self._make_scheduler()

        seq = Sequence(
            seq_id=0, prompt_token_ids=[100, 101, 102, 103, 104, 105, 106, 107],
            sampling_params=SamplingParams(max_new_tokens=8),
        )
        sched.add_request(seq)
        sched.schedule()

        assert sched.prefix_lookups == 1
        assert sched.prefix_query_tokens == 8
        assert sched.prefix_hit_tokens == 0
        assert sched.prefix_cache_hit_rate == 0.0
        assert seq.num_computed_tokens == 0

    def test_disabled_prefix_cache_no_lookup(self):
        """关闭开关：不构造 PrefixCache，不产生任何查找计数。"""
        _, sched = self._make_scheduler(enable=False)

        assert sched.prefix_cache is None
        seq = Sequence(
            seq_id=0, prompt_token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            sampling_params=SamplingParams(max_new_tokens=8),
        )
        sched.add_request(seq)
        sched.schedule()

        assert sched.prefix_lookups == 0
        assert sched.prefix_query_tokens == 0
        assert sched.prefix_cache_hit_rate == 0.0
        assert seq.num_computed_tokens == 0

    def test_reset_prefix_stats(self):
        """reset_prefix_stats 清零全部计数。"""
        paged, sched = self._make_scheduler()
        self._seed_cached_block(paged, [1, 2, 3, 4])
        seq = Sequence(
            seq_id=0, prompt_token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            sampling_params=SamplingParams(max_new_tokens=8),
        )
        sched.add_request(seq)
        sched.schedule()
        assert sched.prefix_hit_tokens == 4

        sched.reset_prefix_stats()
        assert sched.prefix_lookups == 0
        assert sched.prefix_query_tokens == 0
        assert sched.prefix_hit_tokens == 0
        assert sched.prefix_hit_blocks == 0
        assert sched.prefix_cache_hit_rate == 0.0


# ---------------- EngineCoreStats P6 字段（步骤 1） ----------------


class TestEngineCoreStatsPrefixFields:
    def test_prefix_fields_round_trip(self):
        """stats.update 写入 P6 字段并出现在 to_dict 中。"""
        stats = EngineCoreStats()
        stats.update(
            num_batched_tokens=8, num_running=1, num_waiting=0,
            num_prefills=1, num_decodes=0, cache_usage=0.5, waste_rate=0.1,
            preempt_count=0, tokens_generated=1,
            prefix_cache_hit_rate=0.75, prefix_cache_lookups=2,
            prefix_cache_query_tokens=16, prefix_cache_hit_tokens=12,
            prefix_cache_hit_blocks=3,
        )
        d = stats.to_dict()
        assert d["prefix_cache_hit_rate"] == pytest.approx(0.75)
        assert d["prefix_cache_lookups"] == 2
        assert d["prefix_cache_query_tokens"] == 16
        assert d["prefix_cache_hit_tokens"] == 12
        assert d["prefix_cache_hit_blocks"] == 3


# ---------------- extra keys / salt（步骤 4A） ----------------


class TestExtraKeysAndSalt:
    def test_extra_keys_change_hash(self):
        tokens = [1, 2, 3, 4]
        assert compute_block_hash(tokens, None, ("lora-a",)) != compute_block_hash(tokens, None)
        assert compute_block_hash(tokens, None, ("lora-a",)) != compute_block_hash(tokens, None, ("lora-b",))

    def test_extra_keys_isolate_sharing(self):
        """同 token 但 extra_keys 不同 ⇒ 不命中；相同 ⇒ 命中。"""
        pool = BlockPool(8)
        pc = PrefixCache(pool, block_size=4)
        tokens = [1, 2, 3, 4]
        bids = pool.allocate_n(1)
        pc.register_blocks(
            tokens, bids, num_full_blocks=1, start_block=0, prev_hash=None,
            extra_keys=("lora-a",),
        )
        pool.free_n(bids)

        assert pc.find_longest_hit(tokens, extra_keys=("lora-a",)).block_ids == bids
        assert pc.find_longest_hit(tokens, extra_keys=("lora-b",)).block_ids == []
        assert pc.find_longest_hit(tokens).block_ids == []

    def test_salt_isolates_namespace(self):
        """不同 salt 的 PrefixCache 即便共享同一 BlockPool 也互不命中（跨实例隔离）。"""
        pool = BlockPool(8)
        pc_a = PrefixCache(pool, block_size=4, salt="tenant-a")
        pc_b = PrefixCache(pool, block_size=4, salt="tenant-b")
        pc_none = PrefixCache(pool, block_size=4)
        tokens = [1, 2, 3, 4]
        bids = pool.allocate_n(1)
        pc_a.register_blocks(tokens, bids, num_full_blocks=1, start_block=0, prev_hash=None)
        pool.free_n(bids)

        assert pc_a.find_longest_hit(tokens).block_ids == bids
        assert pc_b.find_longest_hit(tokens).block_ids == []
        assert pc_none.find_longest_hit(tokens).block_ids == []

    def test_default_salt_and_keys_match_legacy_hash(self):
        """salt=0 且 extra_keys 为空时 hash 与旧口径完全一致（向后兼容）。"""
        tokens = [7, 8, 9, 10]
        assert compute_block_hash(tokens, None, ()) == compute_block_hash(tokens, None)
        assert PrefixCache(BlockPool(2), 4)._seed is None


class TestSchedulerExtraKeysIsolation:
    """调度器级：extra_keys 不同的请求不共享前缀块（LoRA 位预留的语义验证）。"""

    BLOCK_SIZE = 4

    def _make_scheduler(self):
        paged = PagedKVCache(
            num_layers=1, num_blocks=16, block_size=self.BLOCK_SIZE,
            num_kv_heads=1, head_dim=2, dtype=torch.float32, device="cpu",
        )
        sched = Scheduler(
            paged, max_num_batched_tokens=64, watermark_blocks=0, enable_prefix_cache=True,
        )
        return paged, sched

    def test_extra_keys_gate_sharing(self):
        paged, sched = self._make_scheduler()
        pc = sched.prefix_cache
        assert pc is not None
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]

        bids = paged.pool.allocate_n(2)
        pc.register_blocks(
            tokens, bids, num_full_blocks=2, start_block=0, prev_hash=None,
            extra_keys=("lora-a",),
        )
        paged.pool.free_n(bids)

        seq_a = Sequence(
            seq_id=0, prompt_token_ids=tokens,
            sampling_params=SamplingParams(max_new_tokens=8),
        )
        seq_a.prefix_cache_extra_keys = ("lora-a",)
        sched.add_request(seq_a)
        sched.schedule()
        # R1 钳制: prompt=8 是 block_size=4 的整数倍，全 prompt 命中被钳到 8-1=7
        # → 只命中 1 个满块（4 token），保证至少重算最后一个 prompt token。
        assert seq_a.num_computed_tokens == 4
        assert sched.prefix_hit_tokens == 4

        seq_b = Sequence(
            seq_id=1, prompt_token_ids=tokens,
            sampling_params=SamplingParams(max_new_tokens=8),
        )
        seq_b.prefix_cache_extra_keys = ("lora-b",)
        sched.add_request(seq_b)
        sched.schedule()
        assert seq_b.num_computed_tokens == 0, "不同 extra_keys 不应命中对方的前缀块"
        assert sched.prefix_hit_tokens == 4, "未命中不应增加命中计数"

    def test_scheduler_salt_inherited_from_prefix_cache(self):
        """Scheduler 的 prefix_cache_salt 透传到 PrefixCache。"""
        paged = PagedKVCache(
            num_layers=1, num_blocks=8, block_size=self.BLOCK_SIZE,
            num_kv_heads=1, head_dim=2, dtype=torch.float32, device="cpu",
        )
        sched = Scheduler(
            paged, max_num_batched_tokens=64, watermark_blocks=0,
            enable_prefix_cache=True, prefix_cache_salt="ns-1",
        )
        assert sched.prefix_cache is not None
        assert sched.prefix_cache.salt == "ns-1"
        assert sched.prefix_cache._seed is not None