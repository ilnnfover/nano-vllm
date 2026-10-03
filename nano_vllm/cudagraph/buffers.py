"""P7/P2 · Decode 图捕获的静态输入缓冲（地址固定，跨 step 复用）。

P2 改造：**一套按最大桶分配的缓冲，各桶通过前缀视图共享**——对齐 vLLM 的
persistent buffers（`gpu_model_runner.py:812-825`：按 `max_num_tokens` 分配一次，
每步只写 `[:num_tokens_padded]` 切片）。改造前是"每桶一套"，6 个桶 6 套。

**为什么共享是安全的**：每张图捕获时传的是 `buf[:bucket]` **前缀视图**，其
`data_ptr()` 与整条 buffer 相同、stride 也相同；而 kernel 只触碰 `[0, bucket)` 行，
所以桶 b 的图永远不会读到桶 b' 的行。且回放是**顺序**的（一个 step 只回放一张图），
不存在并发别名。这与 scratch 块/"顺序回放无别名风险"是同一套论证。

| 张量 | 形状 | 说明 |
| --- | --- | --- |
| `input_ids` | `[max_bucket]` int64 | decode 输入 token（padding 行填 0） |
| `positions` | `[max_bucket]` int64 | 每请求当前 position |
| `slot_mapping` | `[max_bucket]` int64 | K/V 写入的物理 slot |
| `block_table` | `[max_bucket, max_blocks]` int32 | 间接寻址表（模型内核直接读） |
| `seq_lens` | `[max_bucket]` int32 | 每请求 KV 长度 |

**输出不在这里**：`hidden_states`（P1 起图输出到 hidden，logits 移到图外）是
**捕获的产物**，每桶一块，由 `DecodeGraphRunner.outputs` 持有——因为图的输出地址
必须由捕获那一刻的池分配决定，无法预先共享一条。

> `is_padding` 掩码**故意不做**：vLLM 里它只服务 MoE 专家路由跳过 padding token
> （`VLLM_MOE_SKIP_PADDING`，消费者全是 deepseek/kimi 系 MoE 模型），dense 模型
> 没有消费者。nano 的 padding 隔离由 scratch 块完成。
"""
from __future__ import annotations

import torch


class DecodeBuffers:
    """**一套**静态输入缓冲，按最大桶分配；各桶用 `[:bucket]` 前缀视图取用。"""

    def __init__(
        self,
        max_bucket: int,
        max_blocks: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if max_bucket <= 0:
            raise ValueError(f"max_bucket 必须为正, got: {max_bucket}")
        self.max_bucket = max_bucket
        self.max_blocks = max_blocks
        self.device = torch.device(device)
        self.dtype = dtype
        self.input_ids = torch.zeros(max_bucket, dtype=torch.int64, device=self.device)
        self.positions = torch.zeros(max_bucket, dtype=torch.int64, device=self.device)
        self.slot_mapping = torch.zeros(max_bucket, dtype=torch.int64, device=self.device)
        self.block_table = torch.zeros(
            (max_bucket, max_blocks), dtype=torch.int32, device=self.device
        )
        self.seq_lens = torch.zeros(max_bucket, dtype=torch.int32, device=self.device)

    # ---------------- 填充 ----------------

    def fill_padding_only(self, scratch_block: int, block_size: int, bucket: int) -> None:
        """把 `[:bucket]` 全部按 padding 配置填充（warmup / capture 用）。

        padding 行只读/写 **预留的 scratch 块**，因此热身与捕获不会碰任何请求的 KV。
        """
        self.input_ids[:bucket].fill_(0)
        self.positions[:bucket].fill_(0)
        self.slot_mapping[:bucket].fill_(scratch_block * block_size)
        self.seq_lens[:bucket].fill_(1)
        self.block_table[:bucket].zero_()
        self.block_table[:bucket, 0].fill_(scratch_block)

    def fill(self, seqs, paged_cache, scratch_block: int, bucket: int) -> None:
        """把 `len(seqs)` 条 decode 请求写入 `[:n]`，`[n, bucket)` 写 padding 配置。

        **每类数据一次 H2D 直达静态 buffer**：整桶内容（真实行 + padding 行）先在
        host 侧一次构造完整，再 `copy_` 进 buffer——源张量留在 CPU，由 `copy_` 自身
        完成 host→device。相比"`torch.tensor(..., device=dev)` 建临时 GPU 张量再
        `copy_`"省掉 **N 次分配 + N 次 D2D**；同时 padding 行随整桶一起送达，
        不再需要设备侧 `fill_`（省 6 次 kernel launch/step）。

        `[bucket, max_bucket)` 行**不动**：没有任何图会读它们（见模块 docstring）。
        padding 行: 读写都落在预留的 scratch 块上（永不与真实请求冲突）。
        """
        n = len(seqs)
        if not 0 < n <= bucket <= self.max_bucket:
            raise ValueError(
                f"非法批大小: n={n}, bucket={bucket}, max_bucket={self.max_bucket}"
            )
        bs = paged_cache.block_size

        # 1) host 侧先按 padding 配置铺满整桶，再覆盖前 n 行为真实值。
        toks = [0] * bucket
        poss = [0] * bucket
        slots = [scratch_block * bs] * bucket
        lens = [1] * bucket
        tables: list[list[int]] = []

        # 2) 采集真实行。**校验全部前置**——任一项不合法就直接 raise，此时 buffer
        #    一个字都还没被改（不会留下"前几行新数据 + 后面旧数据"的半个批状态）。
        for i, seq in enumerate(seqs):
            pos = seq.num_computed_tokens
            block_table = seq.block_table
            if len(block_table) > self.max_blocks:
                raise ValueError(
                    f"请求 #{seq.seq_id} 块表 {len(block_table)} 超过 max_blocks "
                    f"{self.max_blocks}（静态 buffer 容不下）"
                )
            # 位置式取法：稳态 decode 时 = output[-1]；也覆盖"位置落在 prompt 内"
            # 的单 token 追赶（抢占恢复），故不再要求"必须有 output token"。
            toks[i] = seq.input_token_ids(pos, 1)[0]
            poss[i] = pos
            slots[i] = block_table[pos // bs] * bs + pos % bs
            lens[i] = pos + 1
            tables.append(block_table)

        # 3) 4 个向量各一次 H2D（桶前缀，padding 行一并送达）
        self.input_ids[:bucket].copy_(torch.tensor(toks))
        self.positions[:bucket].copy_(torch.tensor(poss))
        self.slot_mapping[:bucket].copy_(torch.tensor(slots))
        self.seq_lens[:bucket].copy_(torch.tensor(lens, dtype=torch.int32))

        # 4) block_table: 桶前缀 [:, :max_len] 一次 H2D。列宽取真实行的最大块表长度，
        #    padding 行首列指 scratch 块、其余列为 0。
        max_len = max(len(t) for t in tables)
        rows = [t + [0] * (max_len - len(t)) for t in tables]
        rows += [[scratch_block] + [0] * (max_len - 1)] * (bucket - n)
        if max_len < self.max_blocks:
            # 尾部列清零：防止上一轮更长的块表留下脏列（内核只读到
            # cdiv(seq_len, bs) 列，这里保持与改造前一致的防御性）
            self.block_table[:bucket, max_len:].zero_()
        self.block_table[:bucket, :max_len].copy_(
            torch.tensor(rows, dtype=torch.int32)
        )
