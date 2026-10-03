"""P7 · Decode CUDA Graph：分桶捕获与回放（vLLM `FULL_DECODE_ONLY` 的简化版）。

对照 vLLM（本快照）:
  - 枚举: `CUDAGraphMode.FULL_DECODE_ONLY = (FULL, NONE)`（config/compilation.py:62）
    —— decode 批走全图，其余批（prefill / 超桶）走 eager。本模块只实现这两级，
    PIECEWISE 需编译栈，roadmap §2 已裁剪。
  - 分桶 + padding: `CudagraphDispatcher._create_padded_batch_descriptor` 与
    `_bs_to_padded_graph_size`（v1/cudagraph_dispatcher.py:71-154）——按 capture size
    向上取整。本实现同一思想：bucket = 最小的 >= n 的桶。
  - 键只对 uniform decode 建（FULL 要求 query_len 一致、num_reqs 精确）：
    本实现把「decode 前向（每请求恰 1 token）」当作 uniform decode，天然满足；
    非 decode（prefill）根本不进这条路径。
  - `build_for_cudagraph_capture` 捕获时把 seq_lens 填小值加速捕获
    （backends/triton_attn.py:185-192）：本实现更彻底——捕获/热身阶段所有行都指向
    预留 scratch 块。
  - P1 · logits 在**图外**算：vLLM 是 `hidden_states[logits_indices]` →
    `model.compute_logits(...)`（gpu_model_runner.py:4509-4510）；本实现同一结构
    （图输出 hidden_states，`EngineCore._decode_logits` 投影）。
  - P2 · 输入缓冲**一套按最大桶分配 + `[:bucket]` 前缀视图共享**：vLLM 的
    persistent buffers 按 `max_num_tokens` 分配、每步写 `[:num_tokens_padded]`
    （gpu_model_runner.py:812-825）。

与 vLLM 的差异（记录即可）:
  1. 不做**完整 dispatcher**：判据只有「纯 decode 且 n ≤ max bucket」，没有
     BatchDescriptor 五元键 / LoRA specialization / 多 routine。
  2. padding 行会真实写 KV，故永久预留 1 个 scratch 块（vLLM 用
     `slot_mapping=-1` 让写入核跳过 + dummy 请求）。
  3. 不做 `is_padding` 掩码：vLLM 里它只服务 **MoE 专家路由**跳过 padding token
     （`VLLM_MOE_SKIP_PADDING`，消费者全是 deepseek/kimi 系 MoE 模型），dense 模型
     没有消费者。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from nano_vllm.attention.metadata import AttentionMetadata
from nano_vllm.cudagraph.buffers import DecodeBuffers

DEFAULT_BUCKETS: tuple[int, ...] = (1, 2, 4, 8, 16, 32)


@dataclass
class CudaGraphStats:
    """图命中/回退埋点（P7 验证产出）。"""

    replays: int = 0                # 图回放次数（命中）
    fallback_eager: int = 0         # 回退 eager 次数（超桶 / 未启用）
    padded_seqs: int = 0            # 累计 padding 行数（浪费度量）
    captured_buckets: int = 0       # 已捕获桶数
    capture_ms: float = 0.0         # 捕获总耗时（含 warmup）
    bucket_hist: dict[int, int] = field(default_factory=dict)

    @property
    def hit_rate(self) -> float:
        total = self.replays + self.fallback_eager
        return self.replays / total if total else 0.0


class DecodeGraphRunner:
    """按桶管理 decode 全图：捕获一次，之后每步填充静态 buffer + replay。"""

    def __init__(
        self,
        model: torch.nn.Module,
        paged_cache,
        config,
        *,
        buckets: tuple[int, ...] = DEFAULT_BUCKETS,
        warmup_runs: int = 2,
        attn_impl: str = "triton",
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if attn_impl != "triton":
            raise ValueError(
                f"图捕获只支持 attn_impl='triton'（自研 Triton 核为张量寻址、图兼容），"
                f"got {attn_impl!r}"
            )
        self.model = model
        self.paged_cache = paged_cache
        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype
        self.buckets = sorted(b for b in buckets if b > 0)
        if not self.buckets:
            raise ValueError("buckets 不能为空")
        self.warmup_runs = warmup_runs
        self.attn_impl = attn_impl
        self.block_size = paged_cache.block_size
        # 单行块表容量上界：受 max_position_embeddings 与物理块数双重限制
        self.max_blocks = min(
            paged_cache.num_blocks,
            config.max_position_embeddings // self.block_size + 1,
        )
        # padding 行落点（永久预留，跨 reset 稳定）
        self.scratch_block = paged_cache.reserve_scratch_block()
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        # P2: **一套**按最大桶分配的输入缓冲，各桶用 `buf[:bucket]` 前缀视图共享
        self.buffers = DecodeBuffers(
            self.max_bucket, self.max_blocks, self.device, self.dtype
        )
        # P1: 图输出（hidden_states）是**捕获的产物**，每桶一块、必须保活——
        # 地址由捕获那一刻的池分配决定，无法像输入那样共享一条。
        self.outputs: dict[int, torch.Tensor] = {}
        self._pool = None
        self.captured = False
        self.stats = CudaGraphStats()

    # ---------------- 查询 ----------------

    @property
    def max_bucket(self) -> int:
        return self.buckets[-1]

    def bucket_for(self, n: int) -> int | None:
        """最小的 >= n 的桶；超过最大桶返回 None（调用方回退 eager）。"""
        for b in self.buckets:
            if n <= b:
                return b
        return None

    def can_use(self, n: int) -> bool:
        return n > 0 and self.bucket_for(n) is not None

    # ---------------- 捕获 ----------------

    def capture(self) -> None:
        """分桶捕获（含 warmup dummy run）。重复调用无副作用。"""
        if self.captured:
            return
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("CUDA Graph 捕获需要 CUDA 设备")
        t0 = time.perf_counter()
        written_before = self.paged_cache.written_tokens
        torch.cuda.synchronize()

        for bucket in self.buckets:
            # 热身：先按真实 eager 跑几次，触发 kernel 编译/autotune 与分配器预热，
            # 避免把首次 Triton 编译烧进捕获（对应 roadmap 的 warmup dummy run）。
            self.buffers.fill_padding_only(self.scratch_block, self.block_size, bucket)
            for _ in range(self.warmup_runs):
                self._forward(bucket)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            # 第一个桶建池，其余共享同一池（统一管理 + 顺序回放无别名；
            # **显存收益未观测到**，见 docs/stages/P7.md 补记 3）
            with torch.cuda.graph(graph, pool=self._pool):
                self.outputs[bucket] = self._forward(bucket)
            if self._pool is None:
                self._pool = graph.pool()
            torch.cuda.synchronize()
            self.graphs[bucket] = graph

        # 热身是 dummy 写入（只碰 scratch 块），不计入真实 KV 写入统计
        self.paged_cache.written_tokens = written_before
        self.captured = True
        self.stats.captured_buckets = len(self.buckets)
        self.stats.capture_ms = (time.perf_counter() - t0) * 1e3

    def _forward(self, bucket: int) -> torch.Tensor:
        """一次 decode 前向：输入取 buffer 的**桶前缀**，输出是 **hidden_states**。

        P1: 图只到 final norm 后的 hidden（`skip_lm_head=True`），logits 由调用方在
        图外算（`EngineCore._decode_logits` → `model.compute_logits`）。这样 padding
        行不再白算 lm_head——**图内躲不掉这件事**（静态形状决定行数）。
        P2: 入参全部是同一套 buffer 的**前缀视图**，`data_ptr()` 与整条相同、stride
        相同，kernel 只碰 `[0, bucket)` 行 → 桶间共享安全。
        """
        bufs = self.buffers
        metadata = AttentionMetadata(
            is_prefill=False,
            slot_mapping=bufs.slot_mapping[:bucket],
            block_table_tensor=bufs.block_table[:bucket],
            seq_lens_tensor=bufs.seq_lens[:bucket],
            attn_impl=self.attn_impl,
        )
        # 布局: b = 桶（每行一个请求）、s = 1（decode 单 token）
        hidden, _ = self.model(
            bufs.input_ids[:bucket].unsqueeze(1),
            paged_cache=self.paged_cache,
            metadata=metadata,
            position_ids=bufs.positions[:bucket].unsqueeze(1),
            skip_lm_head=True,
        )
        return hidden[:, 0]  # [bucket, hidden_size]，base 地址 = 池内静态输出

    # ---------------- 回放 ----------------

    @torch.no_grad()
    def replay(self, seqs) -> torch.Tensor:
        """填充静态 buffer 并**图回放**，返回 `[n, hidden_size]` 的有效行 hidden。

        P1 起图只输出 hidden_states（logits 由图外的 `compute_logits` 算）；
        调用方需自行完成最后一步投影。

        调用前请确保 `can_use(len(seqs))`（否则用 eager 路径）。
        """
        if not self.captured:
            self.capture()
        n = len(seqs)
        bucket = self.bucket_for(n)
        if bucket is None:
            raise ValueError(f"请求数 {n} 超过最大桶 {self.max_bucket}")
        self.buffers.fill(seqs, self.paged_cache, self.scratch_block, bucket)
        self.graphs[bucket].replay()
        self.stats.replays += 1
        self.stats.padded_seqs += bucket - n
        self.stats.bucket_hist[bucket] = self.stats.bucket_hist.get(bucket, 0) + 1
        # 图内 write() 不执行 host 计数 → 这里补记真实写入 token 数
        self.paged_cache.note_written(n)
        return self.outputs[bucket][:n]

    @torch.no_grad()
    def replay_static_eager(self, seqs) -> torch.Tensor:
        """P7 归因实验用：只走「静态 buffer + 张量寻址」的 **eager** 前向，不捕获图。

        目的: 把 P7 的收益拆成两部分——
          (a) 静态 buffer / 张量寻址带来的 host 开销与逐层张量构建的消除
          (b) 真正的图回放（launch 摊平）
        调用方（bench）可临时把 `runner.graph_runner.replay` 指到本方法，
        其余流程（调度/采样/统计）完全不变 → 单变量对照。
        """
        n = len(seqs)
        bucket = self.bucket_for(n)
        if bucket is None:
            raise ValueError(f"请求数 {n} 超过最大桶 {self.max_bucket}")
        self.buffers.fill(seqs, self.paged_cache, self.scratch_block, bucket)
        # eager 前向会**真实执行** `PagedKVCache.write()` → 已按 bucket 计过写入数（且含 padding 行）。
        # 故先快照、后恢复，再只按真实行 n 补记；否则会与下面的 note_written(n) 双计，
        # 使 `waste_rate` 失真（P7.md 补记「顺带发现」记录的即此）。
        written_before = self.paged_cache.written_tokens
        # 新张量，不覆盖 self.outputs[bucket]（图输出仍留在池内）
        hidden = self._forward(bucket)
        self.stats.replays += 1
        self.stats.padded_seqs += bucket - n
        self.paged_cache.written_tokens = written_before
        self.paged_cache.note_written(n)
        return hidden[:n]

    def note_fallback(self) -> None:
        """记录一次 eager 回退（超桶 / 未启用），供命中率埋点。"""
        self.stats.fallback_eager += 1
