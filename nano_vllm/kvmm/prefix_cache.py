"""P6 · 前缀缓存逻辑：滚动 hash + 最长命中查找 + 写满块注册 + COW。

对照 vLLM v1/core/kv_cache_utils.py + KVCacheManager，裁剪项见下。

设计决策（grill-me 2026-09-27/28 锁定，2026-09-28 修订 B1）:
  - B1 hash: 纯 token 滚动 sha256, `hash[i] = sha256(prev_hash || tokens_i)`。
  - B1 修订（对齐 vLLM 纯 Transformer 默认行为）: **只注册满块 hash，partial 块不注册**。
    vLLM 的 `enable_partial_hash_hits` 默认 False、仅 Mamba(SSM) group 才开启
    （kv_cache_coordinator.py:69/693）；纯 Transformer 下 `cache_blocks` 的
    `num_full_blocks = num_tokens // block_size` 只注册满块。理由: partial 块注册后
    永远是 partial（COW 后请求写的是新块），每次命中都要 COW 复制整块，`remainder`
    小时代价超过收益。故对齐 vLLM，partial 块不注册、`is_partial` 恒 False。
  - B4 COW: 仅 partial 命中触发；采用「复制给请求」分支（新块 Y 给请求、block_table[k]=Y，
    共享块 X 原封不动留给缓存与其余引用者）。vLLM 的「复制给缓存」分支是为绕开其 worker
    端 BlockTable 的 append-only 约束（增量下发协议），自研 block_table 是可变 list 无此约束。
    **注**: 对齐 vLLM 后 partial 块不注册 → Transformer 下 COW 永不触发，`cow_for_write` /
    `CacheHit.is_partial` 作为设计参考保留（理解 vLLM COW 机制用），非死代码清理对象。
  - B5 extra keys / salt（对齐 vLLM `generate_block_hash_extra_keys`）:
    hash 除 token 外还纳入 `extra_keys` 与引擎级 `salt`，区分「token 相同但 KV 不该共享」的场景
    （LoRA adapter / 多模态 / prompt_embeds / 命名空间隔离）。否则不同 adapter 会复用同一份
    KV，失败形态是不报错、不 NaN、只静默算错。
    取舍: 自研只支持**请求级统一 extra_keys**（覆盖 LoRA/salt 语义）；vLLM 的多模态按块内
    偏移取 key（同一图片落在块内不同位置需区分）本实现不覆盖，留给将来扩展。
    salt 仅在「链首」生效（后续块经 parent hash 继承），与 vLLM 把 cache_salt 只加在
    start_token_idx==0 一致。

滚动链状态（num_registered_blocks / last_hash）由调用方（Sequence）持有，本模块无状态，
保证多 hash 场景下滚动链不受块字段歧义影响。
"""
from __future__ import annotations

import hashlib
from array import array
from dataclasses import dataclass
from typing import Any

from nano_vllm.kvmm.block_pool import BlockPool

# 请求级额外 hash key（LoRA adapter id / 多模态 hash / 命名空间等）：内容对引擎**不透明**，
# 只需**顺序稳定**且可 repr（见 `compute_block_hash`），故用 `Any` 不做过窄约束。
ExtraKeys = tuple[Any, ...]


def compute_block_hash(
    token_ids: list[int],
    prev_hash: bytes | None,
    extra_keys: ExtraKeys = (),
) -> bytes:
    """滚动块 hash: sha256(prev_hash || token_ids 字节 || extra_keys)。

    token_ids 为该块实际覆盖的 token（partial 块可少于 block_size）。

    extra_keys 用于区分「token 相同但 KV 不该共享」的场景（LoRA adapter id、salt 等）——
    否则不同 adapter 会用同一份 KV 而静默算错。对齐 vLLM:
    `kv_cache_utils.generate_block_hash_extra_keys` + `hash_block_tokens`。

    约定: extra_keys 必须**顺序稳定**（同一逻辑 key 每次序列化结果一致）。
    """
    h = hashlib.sha256()
    if prev_hash is not None:
        h.update(prev_hash)
    h.update(array("i", token_ids).tobytes())
    if extra_keys:
        h.update(repr(tuple(extra_keys)).encode())
    return h.digest()


@dataclass
class CacheHit:
    """最长命中前缀结果。

    注: 对齐 vLLM 纯 Transformer 默认行为（partial 块不注册 hash）后，
    `is_partial` 恒为 False，字段保留作设计参考（理解 vLLM partial 命中 + COW 机制）。
    """

    block_ids: list[int]   # 命中的块 id（按序，首块到末块）
    num_tokens: int        # 命中 token 数（末块 partial 时非 block_size 倍数）
    is_partial: bool       # 末块是否 partial（覆盖 < block_size，请求继续写该块前需 COW）；恒 False
    last_full_hash: bytes | None = None  # 最后一个**满块**的滚动 hash（partial 块前那个；无满块时 None）

    @property
    def num_blocks(self) -> int:
        return len(self.block_ids)

    @property
    def num_full_blocks(self) -> int:
        """命中中的满块数（不含 partial 末块）。"""
        return len(self.block_ids) - (1 if self.is_partial else 0)


class PrefixCache:
    """前缀缓存逻辑：命中查找 / 写满块注册 / COW（COW 保留作设计参考）。无状态，滚动链进度由调用方维护。

    salt: 引擎级命名空间。非 0 时作为滚动链种子参与首块 hash，使不同实例/命名空间的
    缓存互不命中（对应 vLLM 的 `request.cache_salt`，只是我们放在引擎级）。
    """

    def __init__(self, block_pool: BlockPool, block_size: int, salt: int | str = 0) -> None:
        self.pool = block_pool
        self.block_size = block_size
        self.salt = salt
        # salt 为 0（假值）时不做隔离，保持与无 salt 时的 hash 完全一致（向后兼容）
        self._seed: bytes | None = (
            hashlib.sha256(f"nano-vllm-prefix-salt:{salt}".encode()).digest() if salt else None
        )

    def find_longest_hit(
        self,
        token_ids: list[int],
        extra_keys: ExtraKeys = (),
        max_tokens: int | None = None,
    ) -> CacheHit:
        """逐块算滚动 hash 查索引，返回最长命中前缀。

        只命中注册过 hash 的**满块**（partial 块不注册，故 is_partial 恒 False）。
        保留 partial 判断逻辑作为设计参考（若将来启用 fine-grained hash 则复用）。

        extra_keys 必须与注册时一致，否则查不到（这正是「token 相同但不该共享」的机制）。

        max_tokens: 命中最多覆盖的 token 数（R1 钳制，对齐 vLLM
        `kv_cache_manager.get_computed_blocks` 的 `num_tokens - 1`）。调用方传
        `num_prompt_tokens - 1`，保证至少重算最后一个 prompt token 以产出首 token
        logits——否则同 prompt 重放且长度恰为 block_size 整数倍时，命中会覆盖整个
        prompt，`num_new_tokens == 0`，出现空前向 / 永不产出首 token。
        """
        bs = self.block_size
        if not token_ids:
            return CacheHit([], 0, False, None)
        block_ids: list[int] = []
        num_tokens = 0
        is_partial = False
        prev: bytes | None = self._seed
        last_full_hash: bytes | None = None
        n_blocks = (len(token_ids) + bs - 1) // bs
        for i in range(n_blocks):
            start = i * bs
            end = min(start + bs, len(token_ids))
            if max_tokens is not None and end > max_tokens:
                break
            prev = compute_block_hash(token_ids[start:end], prev, extra_keys)
            bid = self.pool.get_block_id_by_hash(prev)
            if bid is None:
                break
            block_ids.append(bid)
            num_tokens = end
            is_partial = (end - start) < bs
            if not is_partial:
                last_full_hash = prev
        return CacheHit(block_ids, num_tokens, is_partial, last_full_hash)

    def register_blocks(
        self,
        token_ids: list[int],
        block_table: list[int],
        num_full_blocks: int,
        start_block: int,
        prev_hash: bytes | None,
        extra_keys: ExtraKeys = (),
    ) -> tuple[int, bytes | None]:
        """为块 [start_block, num_full_blocks) 注册满块 hash（对齐 vLLM cache_blocks）。

        Args:
            token_ids: 请求完整 token 序列（prompt + 已生成）。
            block_table: 请求块表（前缀命中块在前，新分配块在后）。
            num_full_blocks: 已写满的块数 = num_computed_tokens // block_size。
            start_block: 下一个待注册块号（已注册块数），避免重复计算。
            prev_hash: 块 start_block-1 的滚动 hash（start_block==0 时为 None，映射到引擎 salt 种子）。
            extra_keys: 与该请求查找时一致的额外 key（LoRA id 等）。

        Returns:
            (新 start_block, 最后注册块的滚动 hash)。
        """
        bs = self.block_size
        i = start_block
        h = prev_hash if prev_hash is not None else self._seed
        while i < num_full_blocks:
            h = compute_block_hash(token_ids[i * bs : (i + 1) * bs], h, extra_keys)
            self.pool.register_hash(block_table[i], h, num_tokens=bs)
            i += 1
        return i, h


    def cow_for_write(self, paged_cache, block_id: int) -> int:
        """COW（复制给请求分支）：为请求分配私有副本 Y，返回 Y 供 block_table 替换。

        共享块 X 原封不动：hash 保留、其余引用者不受影响；请求对 X 的引用被移除
        （ref_cnt--，到 0 则回 free_queue 仍可被命中）。Y ref_cnt=1、无 hash（请求独占写）。

        **当前无调用者**：B1 修订后 partial 块不注册 hash → 本分支不可达。对应 vLLM 的
        `SingleTypeKVCacheManager._apply_cow`，vLLM 里同样只在 partial 命中时触发
        （且要求 Mamba align group）。源码证据与复刻路径见
        `docs/notes/prefix-cache-partial-hits.md`。
        """
        y = self.pool.allocate()
        paged_cache.copy_block(block_id, y)
        self.pool.free(block_id)
        return y