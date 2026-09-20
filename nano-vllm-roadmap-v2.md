# nano-vLLM · 从零实现路线 v2

> **定位**：从零实现一个单卡 LLM 推理引擎，用于**学习与求职**，非生产系统。
> **参考**：`vllm-src/`（vLLM 开发版快照，只读对照），机制层面对齐 vLLM V1 架构，不追版本号。
> **环境**：WSL2 · RTX 4070 Ti SUPER 16GB（Windows 桌面占用后实际可用 ≈13GB）· Python + PyTorch + Triton。
> **实验模型**：架构 config 驱动、保持通用；benchmark 与验证只在 **Qwen2.5-1.5B-Instruct** 上做。
> **双环境**：自研代码跑 `~/venvs/dev`；vLLM 对照跑 `~/venvs/vllm`。dev 环境装包一律 `-c constraints.txt` 锁 torch。

---

## 0. 相对 v1 路线的变更（决策记录）

| 变更 | 理由 |
| --- | --- |
| 保留 P0–P4 主干（eager → KV → 分页 → 调度） | 这是推理引擎的核心心智模型，面试必问 |
| v0.3 拆分：分页与 prefix caching 分成两版 | 一版一主题，避免调度/显存/缓存逻辑互相干扰 |
| **新增** 最小 Serving（OpenAI 兼容 API） | 求职 demo 需要"能 curl 的东西"，工作量 1–2 天 |
| v0.7 torch.compile + Piecewise **后置为可选** | Dynamo/Inductor 链路踩坑极深（vLLM 编译栈仅 backends.py 就 1300+ 行），性价比低 |
| v0.8 CudagraphDispatcher **降为可选简化版** | 只保留 BatchDescriptor 三级分派思想，不复刻完整 dispatcher |
| v0.9 Breakable Graph + MRv2 **降级为调研笔记** | vLLM 工程团队内部演进，理解动机即可，不复刻 |
| 新增统一 Benchmark 框架（P0 建立，全阶段复用） | 每个数字都要有出处、可复现 |

---

## 1. 硬约束与显存预算

**Qwen2.5-1.5B-Instruct**：28 层 / hidden 1536 / 12 Q 头 + 2 KV 头（GQA）/ head_dim 128 / vocab 151936 / tie_word_embeddings=true / 32K 上下文。

| 项 | 值 |
| --- | --- |
| 权重（bf16） | **3.1 GB** |
| KV / token | **28 KB** = 2(K,V) × 28 层 × 2 头 × 128 × 2B |
| 实际可用显存 | **≈13 GB**（16 − Windows 桌面占用） |
| 剩余给 KV + 激活 + 图 | **≈8.5 GB**（13 − 3.1 权重 − ~1.4 激活与 CUDA Graph） |
| KV 总容量 | ≈30 万 token → 2K 上下文约 **150 并发**，8K 约 37，32K 约 9 |

**Roofline 参照**：显存带宽 672 GB/s → batch=1 的 TPOT 硬下界 ≈ 3.1GB ÷ 672GB/s ≈ **4.6ms（217 tok/s）**。decode 是 memory-bound：增大 batch 摊销权重读取，存在饱和 batch（估算 ~300，必须实测）。不要拿 4090/A100 的公开数字对标。

**两个实现级约束**（16GB 特有，别照抄服务器默认值）：
1. logits `[num_tokens, 151936]` 是大头：8192 token chunk 在 bf16 下 2.49GB，`max_num_batched_tokens` 需按显存扫上界。
2. CUDA Graph 桶数要克制：每桶一份静态 buffer，建议 1/2/4/8/16/32 六档。桶数 vs 显存的 tradeoff 本身是可量化分析题。

**通用性原则**（继承 v1，浓缩）：架构参数全部读 config.json，不写死 1.5B 数值；算子特化用 **shape 特化键**管理而非模型名；1.5B 上测得的**机制结论**（分页、连续批处理）可推广，**数值结论**（block_size 最优值、饱和 batch、命中率）标注"在 1.5B 上测得"，不外推。

---

## 2. 明确裁剪清单（防过度设计）

| 处置 | 内容 | 理由 |
| --- | --- | --- |
| ❌ 不做 | 张量并行 / 流水线并行 / 分布式（TP、PP、EP、ray） | 单卡无法验证收益，只需能讲清 Column/Row Parallel 原理 |
| ❌ 不做 | 生产级容错：进程隔离恢复、优雅重启、metrics 看板、请求级优先级/公平性调度 | 非生产系统；面试口头讲架构即可 |
| ❌ 不做 | 多模态、LoRA、结构化输出、speculative decoding 完整实现、sliding window / MLA / hybrid 模型 | 与主线无关；作为"为什么 vLLM 有这些"的调研题 |
| ⏸ 后置可选 | torch.compile + Piecewise CUDA Graph | 复杂度高；先做 P7 的手写图捕获，理解后再决定是否上 compile |
| ⏸ 后置可选 | FP8 KV cache / 权重量化 | 1.5B 显存不紧张，收益是带宽实验而非容量刚需 |
| 📖 降为调研 | Breakable CUDA Graph、Model Runner V2、CudagraphDispatcher 完整实现、异步调度 async_scheduling | 读源码写笔记：解决什么问题、代价是什么，面试能讲清即可 |
| 📖 降为调研 | prefix caching 跨请求的复杂淘汰策略（TTL、多级） | 只做 LRU + 引用计数 |

---

## 3. 统一 Benchmark 框架（P0 建立，之后每阶段复用）

**指标定义**（全程统一，避免各阶段口径漂移）：
- **TTFT**（p50/p99，ms）、**TPOT**（p50/p99，ms）、**吞吐**（output tok/s、total tok/s，含/不含 prefill）
- **显存**：权重 / KV 已分配 / 峰值（`torch.cuda.max_memory_allocated`）
- **KV 浪费率**：分配但未写入的 KV 字节占比
- **kernel launch 开销占比**：torch profiler 中 CPU launch 时间 / GPU busy 时间

**工具目录**（自研，参考 vLLM `benchmarks/` 思路但极简）：
```
bench/
  dataset.py     # 固定 prompt 集（短 128 / 中 1K / 长 8K / 32K 共享前缀四类，seed 固定）
  run_local.py   # 压测自研引擎：并发数、输入分布、输出长度可配，输出 JSON
  run_vllm.py    # 同数据集压测 ~/venvs/vllm（对照用，P3 起可选）
  compare.py     # 汇总多份 JSON → 表格 + 曲线图（matplotlib）
golden/
  dump_hf.py     # HF transformers 逐层中间量 + logits dump（正确性 golden）
  check_diff.py  # 自研 vs golden 对拍工具
```
**规则**：每个阶段完成后跑指定 benchmark，JSON + 图落盘到 `bench/results/P{n}_*`；没有基准数据的"优化"不算完成。

---

## 4. 阶段路线

> 主线必做：**P0–P4（跑对）→ P7–P8（跑快核心）**；P5/P6 推荐做；P9 按兴趣选。每阶段只依赖前面阶段，不回改架构。

### P0 · 基建与正确性基线
- **必须实现**：项目骨架（`nano_vllm/` 包结构）；`golden/dump_hf.py` 逐层 dump；`bench/` 四件套脚手架；dev 环境依赖清单（torch/Triton/transformers/safetensors 锁版本）
- **验证产出**：`dump_hf.py` 对 Qwen2.5-1.5B 产出逐层 hidden + logits `.npy`；`run_local.py` 空转跑通
- **完成标准**：对拍工具能报告两份张量的 max abs diff；golden 数据落盘且带 config 快照
- **Benchmark**：无（建立口径与数据集），顺手跑一次 HF 的单流 TPOT 作最初参照

### P1 · Eager 单流生成（正确性里程碑）
- **必须实现**：config 驱动的 Qwen2 类模型（embedding / RMSNorm / 带 bias QKV / RoPE(theta=1e6) / GQA / SwiGLU / tie_emb）；safetensors 加载与权重名映射；greedy + temperature/top-p/top-k 采样；无 KV cache 的自回归循环（每步重算全序列）
- **验证产出**：`check_diff.py` 逐层对比 HF；greedy 生成 128 token 与 HF 逐 token 比对脚本
- **完成标准**：逐层 max abs diff < 1e-2；greedy 128 token 与 HF **完全一致**；采样路径单元测试（固定 seed 可复现）
- **Benchmark**：batch=1 单流 TPOT（重算版，作为全项目最差基线）；与 HF generate 的 TPOT 对比

### P2 · KV Cache 与静态批（引入缓存机制）
- **必须实现**：连续预分配 KV cache（**故意用浪费写法**，为 P3 留动机）；cache 位置索引管理；prefill/decode 分离的两段式 attention；静态 batch（padding 拼批）
- **验证产出**：与 P1 的输出逐 token 一致性脚本；padding 浪费率统计脚本
- **完成标准**：greedy 输出与 P1 完全一致；decode 单步耗时相对 P1 显著下降并出数据；浪费率数字落盘（作为 P3 的动机证据）
- **Benchmark**：batch=1 TPOT（P1 vs P2 提升倍数）；静态批 batch=8/16 吞吐 vs 串行

### P3 · PagedAttention 与首个自研算子（核心里程碑）
- **必须实现**：BlockPool（free list）/ block_table / slot_mapping；`reshape_and_cache` 写入核；**decode 用自研 Triton paged attention kernel**（含间接寻址 gather）；prefill 先调 flash-attn 的 paged/varlen 接口（自研 prefill kernel 不阻塞本阶段）；block_size 常量进 config
- **参考源码**：`vllm/v1/core/block_pool.py`、`vllm/v1/attention/backends/triton_attn.py`、`vllm/attention/`（旧版 kernels 思路）
- **验证产出**：Triton kernel vs torch 朴素实现 vs SDPA 的数值对拍与延迟曲线脚本；不同 block_size 的扫描脚本
- **完成标准**：KV 浪费率从 P2 的 X% 降到 <2%；自研 kernel 数值对拍通过（相对误差 <1e-2）；长上下文（32K 单条）可跑通
- **Benchmark**：block_size ∈ {8,16,32,64} 扫描（TPOT + 显存）；三实现 decode 延迟 vs seq_len 曲线；P2 vs P3 同等 batch 的 KV 容量对比

### P4 · 连续批处理调度（吞吐里程碑）
- **必须实现**：Scheduler 状态机（waiting/running/finished）、Sequence 生命周期；token budget（`max_num_batched_tokens` 按显存扫上界）；chunked prefill；抢占与恢复（recompute 路线）；varlen 输入拼装（prefill 走 FA varlen）；EngineCore 主循环（schedule → execute → update）
- **参考源码**：`vllm/v1/core/sched/scheduler.py`（`schedule()` 主逻辑）、`vllm/v1/engine/core.py` 的 step 循环、`vllm/v1/engine/output_processor.py`
- **验证产出**：长短请求混合压测脚本；TPOT p99 监控输出
- **完成标准**：长 prompt prefill 期间已运行请求的 TPOT p99 劣化 <2×（chunked prefill 生效证据）；同 batch 吞吐 ≥ 静态批的 2×；抢占-恢复后输出仍与 P1 一致
- **Benchmark**：混合负载（prompt 长度 128/1K/8K 混排）：TTFT p50/p99、TPOT p99、吞吐；开启/关闭 chunked prefill 对照

### P5 · 最小 Serving（迷你版，推荐）
- **必须实现**：async 引擎循环；FastAPI 暴露 `/v1/completions` 与 `/v1/chat/completions`（含 streaming）；请求参数透传（temperature/top_p/max_tokens）
- **验证产出**：curl + openai SDK 双路 demo；压测脚本走 HTTP 的对照数据
- **完成标准**：streaming 输出与本地引擎一致；HTTP 吞吐损耗 <10% 并有数字
- **Benchmark**：`run_local.py` 直连 vs HTTP 两条路径的吞吐对比

### P6 · Prefix Caching 与 COW（推荐，面试高频）
- **必须实现**：块级滚动 hash（含 extra keys：salt/ LoRA 位预留概念）；hash→block 索引 + LRU 淘汰；命中块复用（跳过重算）；引用计数 + copy-on-write（多请求共享尾部块被追加时）；调度器接入 `get_computed_blocks` / `allocate_slots`
- **参考源码**：`vllm/v1/core/kv_cache_utils.py`（hash）、`vllm/v1/core/block_pool.py`（COW/evict）、`kv_cache_manager.py`
- **验证产出**：命中率统计埋点；多轮对话 + 共享前缀数据集的重放脚本
- **完成标准**：同前缀重放场景 TTFT 显著下降并出数据；COW 触发时输出仍逐 token 一致（正确性红线）
- **Benchmark**：共享前缀命中率 vs TTFT 关系曲线；多轮对话 trace 重放的吞吐对比

### P7 · Decode CUDA Graph（性能里程碑）
- **必须实现**：decode-only 全图捕获与回放；batch 分桶（1/2/4/8/16/32）+ padding；**输入 buffer 地址固定**（input_ids/positions/block_table/slot_mapping 全部预分配复用——P3 的 block_table 由此改为预分配 buffer）；warmup dummy run；多图按桶管理；mixed batch 回退 eager 的分派逻辑（简化版，不做完整 dispatcher）
- **参考源码**：`vllm/compilation/cuda_graph.py`、`vllm/v1/worker/gpu_model_runner.py` 的 cudagraph 接入点、`vllm/v1/cudagraph_dispatcher.py`（只读，理解 BatchDescriptor 思想）
- **验证产出**：图命中/回退统计埋点；torch profiler 的 launch 统计脚本
- **完成标准**：输出与 eager 逐 token 一致（含边界 batch size）；uniform decode 全部命中图；**kernel launch 开销占比从 10–30% 降到 ~1%**，TPOT 相对 P4 下降 15–25%（出实测数字）
- **Benchmark**：P4 vs P7 的 TPOT 全曲线；桶数 3 档 vs 6 档的（显存, TPOT）tradeoff 实验；命中/回退率统计

### P8 · 融合算子与权重预拼接
- **必须实现**：Triton 融合 residual+RMSNorm；RoPE 内联进 QKV 后处理；SwiGLU 融合；加载时按 config 预拼接 QKV（3 GEMM→1）与 gate/up（2→1）；attention 保留三实现（Triton 自研 / flash-attn / SDPA）做对照开关
- **参考源码**：`vllm/v1/worker/gpu_model_runner.py` 的 custom_ops 调用点、`vllm/_custom_ops.py`（只读）
- **验证产出**：逐 token 与 P7 一致性脚本（拼接前后、融合前后各一）；torch profiler 单层 kernel 计数脚本
- **完成标准**：输出逐 token 一致；单层 kernel launch 数下降 ≥30%（profiler 前后对比）；TPOT 再降 5–15%
- **Benchmark**：单算子微基准（融合 vs 不融合，µs 级）；端到端 P7 vs P8 的 TPOT/吞吐；launch 数与 GPU busy 占比对比

### P9 · 可选扩展（按性价比排序，做前先看第 2 节裁剪清单）

| 方向 | 要点 | 性价比判断 |
| --- | --- | --- |
| torch.compile + Piecewise（后置大项） | Dynamo trace → attention 处切图 → Inductor 编译子图 → 子图捕获；先只对 decode 路径开 | ⚠️ 复杂度最高的一项；做完 P7/P8 后再评估，目标是理解而非复刻 vLLM 编译栈 |
| 轻量动态分派 | BatchDescriptor（num_tokens/uniform）→ FULL/PIECEWISE(eager 混排)/NONE 三级选择 | 中；P7 分派逻辑的自然延伸，一天量级 |
| mega-kernel / shape 特化 | Triton autotune 按 shape 键搜 config；QKV+RoPE+写 cache 的参数化融合 kernel | 中；Triton 深度练习，出 kernel 级对比数据 |
| 整图捕获实验 | 按 (batch, seq) 分桶让 attention 静态化，尝试跳过 piecewise 直接整图 | 研究题；有 P7 数据后可做对照结论 |
| FP8 KV cache | sm_89 原生支持；1.5B 下收益是带宽实验 | 低优先级 |
| 投机解码 | draft-verify 对图捕获的影响；rejection sampling 思想 | 仅调研 + 笔记，不实现 |
| 📖 纯调研 | Breakable CUDA Graph、MRv2、异步调度 | 读 `vllm/compilation/breakable_cudagraph.py` 与 MRv2 相关代码，写"解决什么问题/代价"笔记 |

---

## 5. vLLM 参考地图（本快照实际结构）

| 主题 | 位置 | 备注 |
| --- | --- | --- |
| 引擎主循环 | `vllm/v1/engine/core.py`（EngineCore.step） | 请求经 `core_client.py` 分发 |
| 调度器 | `vllm/v1/core/sched/scheduler.py` | `schedule()` 产出 SchedulerOutput |
| KV cache 管理 | `vllm/v1/core/`：kv_cache_manager.py（门面）→ kv_cache_coordinator.py → block_pool.py（分配/COW/evict） | prefix hash 在 `kv_cache_utils.py` |
| Model Runner | `vllm/v1/worker/gpu_model_runner.py`（7600+ 行，只看 execute_model 主干） | 输入批 → attention metadata → forward → logits |
| 采样 | `vllm/v1/sample/sampler.py` | 与 execute_model 分离，两次有状态调用 |
| Attention backend | `vllm/v1/attention/`：registry.py（枚举注册）、selector.py（选择）、backends/（flash_attn / triton_attn / flashinfer） | 旧 `vllm/attention/` 已瘦身为平台层 |
| 编译与图 | `vllm/compilation/`：backends.py（VllmBackend 切图）、piecewise_backend.py、cuda_graph.py | splitting_ops 默认 attention 系 |
| 图分派 | `vllm/v1/cudagraph_dispatcher.py` | BatchDescriptor → FULL/PIECEWISE/NONE |
| 输出处理 | `vllm/v1/engine/output_processor.py` + detokenizer.py | 增量反解码 |
| 模型实现参考 | `vllm/models/` 已按家族重组（qwen2.py 不存在） | 对照实现建议读 HF `modeling_qwen2.py`，比 vLLM 的平台拆分版更易读 |

---

## 6. 反模式（红线）

1. ❌ 没有 P1 的 eager 基线就开优化——所有数字失去参照
2. ❌ 图捕获与调度同时改——P4 跑对调度才进 P7
3. ❌ 跳过 torch 朴素实现直接写 Triton——两遍实现法（先对、再快）
4. ❌ 一版塞多个主题——每个 P 编号只解决一个问题
5. ❌ 把 1.5B 数值结论当普适规律——机制可推广，数字标注实测条件
6. ❌ 追求与 vLLM 绝对性能对比——目标是**可解释的性能**：每个数字能说清来自哪个设计决策

## 7. 附录 · Qwen2.5-1.5B 实现坑点（P1 就会踩）

1. **QKV 带 bias**（Llama 没有）——照抄 Llama 不报错只会静默算错，必须逐层对拍
2. **tie_word_embeddings=true**——lm_head 复用 embed_tokens，加载时别找不存在的 key
3. **RoPE theta=1e6**（Llama-3 是 5e5），HF 风格 inv_freq/position_ids 约定——对不齐的头号原因
4. **GQA 12:2**——query 按 6 倍展开匹配 KV，repeat 维度顺序易错
5. **vocab 151936 + 32K 上下文**——logits 显存与长上下文压测正好验证 P3/P6