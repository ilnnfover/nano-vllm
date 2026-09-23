"""P3 · BlockPool：分页 KV cache 的物理块分配器（free list）。

设计取舍:
  - vLLM 的 FreeKVCacheBlockQueue 用双向链表 + fake head/tail，为的是 O(1) 中段删除
    （prefix cache 淘汰时要摘除任意块）。P3 无淘汰需求，用 FIFO deque 即可，分配顺序确定。
  - ref_cnt 字段为 P6 的 copy-on-write / 多请求共享块预留；P3 只用 0（空闲）/1（占用）。
  - block_id 即物理存储 [num_blocks, block_size, ...] 的第一维下标。
"""
from __future__ import annotations

from collections import deque


class KVCacheBlock:
    """物理块的元数据。block_id 对应物理 KV 存储的第一维下标。"""

    __slots__ = ("block_id", "ref_cnt")

    def __init__(self, block_id: int) -> None:
        self.block_id = block_id
        self.ref_cnt = 0

    def __repr__(self) -> str:
        return f"KVCacheBlock(block_id={self.block_id}, ref_cnt={self.ref_cnt})"


class BlockPool:
    """管理 num_blocks 个物理块的分配与回收。

    FIFO free list：按 block_id 升序初始，释放的块追加到队尾。
    分配顺序确定（同 seed 同输入必然得到相同 block_id 序列），便于对拍复现。
    """

    def __init__(self, num_blocks: int) -> None:
        if num_blocks <= 0:
            raise ValueError(f"num_blocks 必须为正, got: {num_blocks}")
        self.num_blocks = num_blocks
        self.blocks = [KVCacheBlock(i) for i in range(num_blocks)]
        self._free: deque[int] = deque(range(num_blocks))

    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - len(self._free)

    @property
    def free_block_ids(self) -> list[int]:
        return list(self._free)

    def allocate(self) -> int:
        """分配一个块，返回 block_id。池耗尽时 raise。"""
        if not self._free:
            raise ValueError(
                f"BlockPool 耗尽: {self.num_blocks} 个块已全部分配。"
                f"增大 num_blocks 或减小 batch/序列长度。"
            )
        block_id = self._free.popleft()
        self.blocks[block_id].ref_cnt = 1
        return block_id

    def allocate_n(self, n: int) -> list[int]:
        """分配 n 个块。原子性: 不足 n 个时先校验再分配，不产生半分配。"""
        if n > len(self._free):
            raise ValueError(f"BlockPool 剩余 {len(self._free)} 块, 无法分配 {n} 块")
        return [self.allocate() for _ in range(n)]

    def free(self, block_id: int) -> None:
        """释放一个块。ref_cnt 减到 0 才真正回到 free list。"""
        if not 0 <= block_id < self.num_blocks:
            raise ValueError(f"非法 block_id={block_id}, 范围 [0, {self.num_blocks})")
        blk = self.blocks[block_id]
        if blk.ref_cnt <= 0:
            raise ValueError(f"释放未分配的块 block_id={block_id} (ref_cnt={blk.ref_cnt})")
        blk.ref_cnt -= 1
        if blk.ref_cnt == 0:
            self._free.append(block_id)

    def free_n(self, block_ids: list[int]) -> None:
        for block_id in block_ids:
            self.free(block_id)

    def reset(self) -> None:
        """全部回收（跨请求复用池时调用）。"""
        self._free = deque(range(self.num_blocks))
        for blk in self.blocks:
            blk.ref_cnt = 0