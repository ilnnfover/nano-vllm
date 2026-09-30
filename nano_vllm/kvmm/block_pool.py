"""P3/P6 · BlockPool：分页 KV cache 的物理块分配器 + 前缀缓存索引。

P3 设计取舍:
  - vLLM 的 FreeKVCacheBlockQueue 用双向链表 + fake head/tail，为的是 O(1) 中段删除
    （prefix cache 淘汰时要摘除任意块）。
  - block_id 即物理存储 [num_blocks, block_size, ...] 的第一维下标。

P6 前缀缓存改造（对齐 vLLM v1/core/block_pool.py）:
  - 双结构分离:
      * `_free_queue`  : OrderedDict，只装 ref_cnt==0 的块，按 LRU 序（头部最久未使用、尾部最近使用）；
                         分配来源 + 淘汰顺序。命中时 touch 会把块从队列摘除（O(1)）。
      * `_hash_index`  : dict[hash -> block_id]，含**所有有 hash 的块，不论 ref_cnt**，
                         仅用于命中查找。被引用块保留索引 → 并发共享前缀都能命中。
  - 多 hash: 主 hash 存 KVCacheBlock.block_hash（+ num_tokens），额外 hash 存 `_block_hashes[block_id]`。
  - ref_cnt > 1: 多个请求可共享同一命中块，最后一个释放才回 free_queue。
  - 淘汰不清零 KV: torch `[:seq_len]` / triton `offs_n` 掩码 / varlen `last_page_len` 三机制兜底。
"""
from __future__ import annotations

from collections import OrderedDict


class KVCacheBlock:
    """物理块的元数据。block_id 对应物理 KV 存储的第一维下标。"""

    __slots__ = ("block_id", "ref_cnt", "block_hash", "block_hash_num_tokens")

    def __init__(self, block_id: int) -> None:
        self.block_id = block_id
        self.ref_cnt = 0
        self.block_hash: bytes | None = None
        self.block_hash_num_tokens: int | None = None

    def __repr__(self) -> str:
        return (
            f"KVCacheBlock(block_id={self.block_id}, ref_cnt={self.ref_cnt}, "
            f"hash={self.block_hash.hex()[:8] if self.block_hash else None}, "
            f"num_tokens={self.block_hash_num_tokens})"
        )


class BlockPool:
    """管理 num_blocks 个物理块 + 前缀缓存 hash 索引 + LRU 淘汰。"""

    def __init__(self, num_blocks: int) -> None:
        if num_blocks <= 0:
            raise ValueError(f"num_blocks 必须为正, got: {num_blocks}")
        self.num_blocks = num_blocks
        self.blocks = [KVCacheBlock(i) for i in range(num_blocks)]
        # 只装 ref_cnt==0 的块；头部 LRU（优先淘汰/分配），尾部最近使用
        self._free_queue: OrderedDict[int, None] = OrderedDict(
            (i, None) for i in range(num_blocks)
        )
        # hash -> block_id，含被引用块（命中查找用）
        self._hash_index: dict[bytes, int] = {}
        # block_id -> 额外 hash 集合（主 hash 在 KVCacheBlock.block_hash）
        self._block_hashes: dict[int, set[bytes]] = {}

    # ---------------- 容量查询 ----------------

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_queue)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - len(self._free_queue)

    @property
    def free_block_ids(self) -> list[int]:
        return list(self._free_queue)

    # ---------------- 分配 / 释放 ----------------

    def allocate(self) -> int:
        """从 LRU 头部取一个空闲块（若该块有 hash 则淘汰其缓存），ref_cnt=1。"""
        if not self._free_queue:
            raise ValueError(
                f"BlockPool 耗尽: {self.num_blocks} 个块已全部分配。"
                f"增大 num_blocks 或减小 batch/序列长度。"
            )
        block_id, _ = self._free_queue.popitem(last=False)
        self._evict_hashes(block_id)
        self.blocks[block_id].ref_cnt = 1
        return block_id

    def allocate_n(self, n: int) -> list[int]:
        """分配 n 个块。原子性: 不足 n 个时先校验再分配，不产生半分配。"""
        if n > len(self._free_queue):
            raise ValueError(f"BlockPool 剩余 {len(self._free_queue)} 块, 无法分配 {n} 块")
        return [self.allocate() for _ in range(n)]

    def free(self, block_id: int) -> None:
        """ref_cnt 减 1；减到 0 时块回 free_queue 尾部（刷最近使用），hash 索引保留。"""
        if not 0 <= block_id < self.num_blocks:
            raise ValueError(f"非法 block_id={block_id}, 范围 [0, {self.num_blocks})")
        blk = self.blocks[block_id]
        if blk.ref_cnt <= 0:
            raise ValueError(f"释放未分配的块 block_id={block_id} (ref_cnt={blk.ref_cnt})")
        blk.ref_cnt -= 1
        if blk.ref_cnt == 0:
            self._free_queue[block_id] = None
            self._free_queue.move_to_end(block_id)

    def free_n(self, block_ids: list[int]) -> None:
        for block_id in block_ids:
            self.free(block_id)

    def touch(self, block_id: int) -> None:
        """命中共享块：ref_cnt +1 并从 free_queue 摘除（被引用期间不可淘汰）。"""
        blk = self.blocks[block_id]
        blk.ref_cnt += 1
        self._free_queue.pop(block_id, None)

    def reset(self) -> None:
        """全部回收 + 清空前缀缓存索引（跨请求复用池时调用）。"""
        self._free_queue = OrderedDict((i, None) for i in range(self.num_blocks))
        self._hash_index.clear()
        self._block_hashes.clear()
        for blk in self.blocks:
            blk.ref_cnt = 0
            blk.block_hash = None
            blk.block_hash_num_tokens = None

    # ---------------- 前缀缓存 hash 管理 ----------------

    def register_hash(self, block_id: int, block_hash: bytes, num_tokens: int | None) -> None:
        """给块注册一个 hash（主 hash 或额外 hash），并写入 hash 索引。"""
        blk = self.blocks[block_id]
        if blk.block_hash is None:
            blk.block_hash = block_hash
            blk.block_hash_num_tokens = num_tokens
        else:
            self._block_hashes.setdefault(block_id, set()).add(block_hash)
        self._hash_index[block_hash] = block_id

    def get_block_id_by_hash(self, block_hash: bytes) -> int | None:
        """按 hash 查命中块（含被引用块）。未命中返回 None。"""
        return self._hash_index.get(block_hash)

    def has_hash(self, block_id: int) -> bool:
        """块是否挂有任意 hash（主 hash 或额外 hash）。"""
        return (
            self.blocks[block_id].block_hash is not None
            or block_id in self._block_hashes
        )

    def num_hash_tokens(self, block_id: int) -> int:
        """块主 hash 覆盖的 token 数（未注册返回 0）。"""
        n = self.blocks[block_id].block_hash_num_tokens
        return n or 0

    def move_hashes(self, src_block_id: int, dst_block_id: int) -> None:
        """COW: 把 src 块的所有 hash 重指向 dst 块（prefix cache 保留私有副本在 dst）。

        调用后 src 块不再挂任何 hash（可被请求继续写入），dst 成为缓存载体。

        对应 vLLM 的 `BlockPool.move_block_hashes`（唯一调用点是 MambaManager 的 partial
        tail offload）。**当前无调用者**：纯 Transformer + 只注册满块的配置下 partial 命中
        不可达，故「复制给缓存」分支不会被走到。见 `docs/notes/prefix-cache-partial-hits.md`。
        """
        src = self.blocks[src_block_id]
        dst = self.blocks[dst_block_id]
        if dst.block_hash is not None or dst_block_id in self._block_hashes:
            raise ValueError(f"dst 块 {dst_block_id} 已挂 hash，不可作为 COW 载体")
        if src.block_hash is not None:
            dst.block_hash = src.block_hash
            dst.block_hash_num_tokens = src.block_hash_num_tokens
            self._hash_index[src.block_hash] = dst_block_id
            src.block_hash = None
            src.block_hash_num_tokens = None
        extra = self._block_hashes.pop(src_block_id, None)
        if extra:
            self._block_hashes.setdefault(dst_block_id, set()).update(extra)
            for h in extra:
                self._hash_index[h] = dst_block_id

    def _evict_hashes(self, block_id: int) -> None:
        """块被重新分配（覆写）前，从 hash 索引移除它的所有 hash。"""
        blk = self.blocks[block_id]
        if blk.block_hash is not None:
            self._hash_index.pop(blk.block_hash, None)
            blk.block_hash = None
            blk.block_hash_num_tokens = None
        for h in self._block_hashes.pop(block_id, ()):
            self._hash_index.pop(h, None)
