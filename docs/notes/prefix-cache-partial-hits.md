# 调研 · Prefix Cache 的 partial（fine-grained）命中与 COW

> 日期：2026-09-30
> 对照源码：`vllm-src/`（本地 vLLM 快照，只读）
> 起因：roadmap §P6 原完成标准写着「COW 触发时输出仍逐 token 一致（正确性红线）」，
> 但纯 Transformer + 默认配置下 COW **永不触发**，该标准不可执行。本笔记记录 vLLM 的
> 真实机制、触发条件与代价，作为 P6 验收口径修订的依据。

---

## 0. TL;DR

1. **COW 在 vLLM V1 仍在使用**，但只在 **partial（sub-block / fine-grained）前缀命中**时触发。
2. partial 命中的总开关 `enable_partial_hash_hits` 要求 **存在 Mamba `align` group**；
   纯 Transformer 下恒为 `False` → **COW 不触发**。
3. V0 的 `vllm/core/block/`（老 `BlockSpaceManager` 那套 COW）目录**已删除**；
   现在的 COW 是 V1 重写的、触发面窄得多的版本。
4. 对自研（纯 Transformer，roadmap 已裁剪 Mamba）而言，COW 属**设计参考**，
   不是可端到端验收的路径；正确性由单测覆盖。

---

## 1. 触发链路（按源码顺序）

### 1.1 总开关：必须有 Mamba align group

```678:695:vllm-src/vllm/v1/core/kv_cache_coordinator.py
        # Fine-grained hash hits require Mamba "align" and compatible cache
        # managers in every group. ...
        has_partial_mamba_group = any(
            isinstance(g.kv_cache_spec, MambaSpec)
            and g.kv_cache_spec.mamba_cache_mode == "align"
            and (
                (dcp_world_size == 1 and g.kv_cache_spec.block_size > hash_block_size)
                or ...
            )
            for g in kv_cache_config.kv_cache_groups
        )
        self.enable_partial_hash_hits = (
            allow_partial_hash_hits and has_partial_mamba_group
        )
```

即：**没有 Mamba `align` group，partial 命中直接不可能**，后面的 COW 链路全部不可达。

### 1.2 判定「命中尾部落在块内部」

```168:178:vllm-src/vllm/v1/core/single_type_kv_cache_manager.py
    def _has_partial_local_hit(
        self,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
    ) -> bool:
        # The local prefix-cache hit ends inside one of this manager's
        # blocks: the shared tail block needs CoW.
        return (
            len(new_computed_blocks) > 0
            and num_local_computed_tokens % self.block_size != 0
        )
```

命中长度不是块对齐的 → 共享尾块里还有空间被「别人」继续写 → 需要私有副本。

### 1.3 登记待 COW 的请求

```318:324:vllm-src/vllm/v1/core/single_type_kv_cache_manager.py
        if self._has_partial_local_hit(new_computed_blocks, num_local_computed_tokens):
            block_idx = num_local_computed_tokens // self.block_size
            self._partial_hit_reqs[request_id] = (block_idx, new_computed_blocks[-1])
            self.num_cached_block[request_id] = block_idx
```

### 1.4 下一次分配时执行 COW

```384:394:vllm-src/vllm/v1/core/single_type_kv_cache_manager.py
        cow_blocks: list[KVCacheBlock] = []
        if request_id in self._partial_hit_reqs:
            # Partial hit: redirect the shared tail to a private CoW block.
            block_idx, source_block = self._partial_hit_reqs.pop(request_id)
            cow_block = self.block_pool.get_new_blocks(1)[0]
            self._apply_cow(request_id, block_idx, source_block, cow_block)
            self.new_block_ids.append(cow_block.block_id)
            cow_blocks.append(cow_block)
```

### 1.5 COW 本体（两个方向的分支都存在）

**分支 A · 复制给请求**（共享块 X 保留给缓存与其它引用者，请求改用私有块 Y）：

```451:471:vllm-src/vllm/v1/core/single_type_kv_cache_manager.py
    def _apply_cow(
        self, request_id: str, block_idx: int,
        source_block: KVCacheBlock, cow_block: KVCacheBlock,
    ) -> None:
        """Redirect a partial prefix-cache hit to a private CoW block.
        ...
        """
        req_blocks = self.req_to_blocks[request_id]
        assert req_blocks[block_idx] is source_block
        assert not source_block.is_null and source_block.ref_cnt > 0
        req_blocks[block_idx] = cow_block
        self._pending_cow_copies.append((source_block, cow_block))
        cow_block.ref_cnt += 1
```

真正的张量拷贝不在这里发生：`(source, cow)` 交给 worker 执行（`take_pending_cow_copies`），
且两端都保持引用，避免同一步内被回收。

**分支 B · 复制给缓存**（hash 迁到私有副本，请求继续写原块）—— 只在 `MambaManager` 的
partial tail offload 路径：

```1912:1915:vllm-src/vllm/v1/core/single_type_kv_cache_manager.py
                        assert req_blocks[block_idx] is source_block
                        self.block_pool.move_block_hashes(source_block, cow_block)
                        self._pending_cow_copies.append((source_block, cow_block))
                        source_block.ref_cnt += 1
```

对应 `BlockPool.move_block_hashes`：

```642:652:vllm-src/vllm/v1/core/block_pool.py
    def move_block_hashes(self, src_block: KVCacheBlock, dst_block: KVCacheBlock) -> None:
        """Re-point ``src_block``'s prefix-cache entries to ``dst_block``.

        Used when the request owning ``src_block`` keeps writing into it
        : the prefix cache holds a private copy (``dst_block``)
        under the same hashes instead. Entries stay live; no events emitted.
        """
```

### 1.6 partial 边界如何进缓存

```447:461:vllm-src/vllm/v1/core/block_pool.py
    def cache_partial_block(self, request, block, num_tokens, kv_cache_group_id,
                            block_size, replace_existing_hashes=False) -> BlockHashWithGroupId | None:
        """Register a partial prefix-cache entry for an existing block.

        Prefix-cache keys normally identify full cache blocks. A partial entry
        makes an existing cache block reachable from a fine-grained prefix
        boundary inside that block without allocating or copying a new
        ``KVCacheBlock``.
```

即：**同一个物理块可以挂多个 hash**（主 hash = 满块，partial entry = 块内更细的边界）。

### 1.7 调度器侧还有一层门

```371:373:vllm-src/vllm/v1/core/sched/scheduler.py
        self.mamba_partial_cache_hit = (
            self.need_mamba_block_aligned_split
            and self.hash_block_size < self.block_size
            and self.kv_cache_manager.coordinator.enable_partial_hash_hits
        )
```

`kv_cache_manager.py:187-195` 的 `mamba_fine_grained_prefix_cache` 还额外要求有 eagle group
且 `num_reprefillable_tokens == 0`。所以 partial 命中在 vLLM 里是**多重 opt-in**。

---

## 2. 为什么纯 Transformer 下关闭（理解，非源码原文）

- **收益侧**：SSM（Mamba）的 recurrent state 有「时序边界」语义 —— 必须在正确的位置
  checkpoint 才能恢复，所以 vLLM 需要**比物理块更细的 hash 粒度**（`hash_block_size < block_size`）。
  partial 命中是为这个需求服务的。
- **代价侧**：partial 命中意味着「共享块的尾部边界被我方复用」，请求要继续往里写就必须
  先整块复制（`_apply_cow`）。**命中长度越短，复制的相对浪费越大**。
- 纯注意力没有边界状态语义，partial 命中带来的只是「多一次整块拷贝」，收益为负。
  因此 vLLM 把它绑死在 Mamba `align` 上（`has_partial_mamba_group`）。

> 这一条与本仓库的 B1 修订一致：`prefix_cache.py` 里我们选择「只注册满块 hash」，
> 理由与 vLLM 默认行为相同 —— partial 块注册后永远是 partial（COW 后请求写的是新块），
> 每次命中都要 COW 复制整块，`remainder` 小时代价超过收益。

---

## 3. 与自研实现的对应关系

| vLLM | 自研（nano_vllm） | 状态 |
|---|---|---|
| `block_pool.cache_partial_block` | `BlockPool.register_hash(block_id, h, num_tokens)`（支持一块多 hash） | 能力具备；B1 修订后不注册 partial |
| `_has_partial_local_hit` | `CacheHit.is_partial` / `num_full_blocks` | 恒 `False` |
| `_apply_cow`（复制给请求） | `PrefixCache.cow_for_write`（`scheduler._attach_hit_blocks` 调用点） | 有实现，**当前无调用者** |
| `move_block_hashes`（复制给缓存） | `BlockPool.move_hashes` | 有实现，**当前无调用者** |
| `block_pool.touch` / `free`（ref_cnt） | `BlockPool.touch` / `free` / `free_n` | 已对齐并已接入 |
| pending COW 交给 worker 拷贝 | 无需拆分（单进程同步执行，`copy_block` 直接拷） | 架构差异，非缺陷 |

差异说明：vLLM 的「复制给缓存」分支是为绕开其 worker 端 BlockTable 的 **append-only 约束**
（BlockTable 增量下发协议），自研 `block_table` 是可变 list，没有这个约束，所以主线用
「复制给请求」分支即可。

---

## 4. 若将来要启用 partial 命中（复刻路径）

1. `PrefixCache.register_blocks` 增加 partial 块注册（用 `pool.register_hash(..., num_tokens=<可整除 hash 粒度>)`）；
2. `find_longest_hit` 允许返回 partial 末块（现在已保留该逻辑，只是永不进入）；
3. `_attach_hit_blocks` 的 COW 分支解除「恒不可达」—— 需要同时定义**回滚策略**
   （余量不足时放弃命中）与 `pinned` 语义（COW 期间两端不可被回收）；
4. 新增验收：`enable_partial_hash_hits` 开/关下输出逐 token 一致 + COW 次数可统计；
5. 需要先确认收益为正 —— 对纯 Transformer，参考第 2 节的代价分析，**默认不建议开**。

---

## 5. 参考文件

| 主题 | 位置 |
|---|---|
| 总开关 / Mamba 判定 | `vllm-src/vllm/v1/core/kv_cache_coordinator.py:678-726` |
| partial 判定与 COW 执行 | `vllm-src/vllm/v1/core/single_type_kv_cache_manager.py:168-178, 318-324, 384-394, 451-471, 1912-1930` |
| partial 边界注册 / hash 迁移 | `vllm-src/vllm/v1/core/block_pool.py:447-486, 642-658` |
| 调度器侧门控 | `vllm-src/vllm/v1/core/sched/scheduler.py:371-385`、`vllm-src/vllm/v1/core/kv_cache_manager.py:187-195` |
| 自研实现 | `nano_vllm/kvmm/prefix_cache.py`、`nano_vllm/kvmm/block_pool.py`、`nano_vllm/engine/scheduler.py` |
