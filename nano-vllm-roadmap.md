# 从零实现 vLLM 推理引擎 · 版本演进路线

> **主线**：动态图执行（CUDA Graph → torch.compile → 分段图捕获与动态分派）+ 算子优化（Attention backend → 融合算子 → 单模型特化）
> **环境**：单卡 **RTX 4070 Ti SUPER 16GB**（AD103 / sm_89 / 672 GB/s）· Python + PyTorch + Triton · 业余时间长期推进
> **目标模型**：架构保持通用（config 驱动）；**实验与验证只在 Qwen2.5-1.5B-Instruct 上做**
> **对齐目标**：vLLM **v0.29.0**（2026-09-09，Model Runner V2 全模型默认）
> 下方版本号是**本项目的版本**，与 vLLM 版本号无关；「对齐」列指出该步对应的 vLLM 能力点

---

## 0. 范围与基线

- **模型架构**：保持通用。层数 / hidden / head_dim / KV 头数 / vocab / tie_emb 全部从 `config.json` 读取，**不做任何 1.5B 专属硬编码**
- **实验范围**：benchmark 与验证只在 Qwen2.5-1.5B-Instruct 上做。其他尺寸不测、不保证兼容，但也不因实现限制而排除
- **复用**：torch 张量运算、safetensors、tokenizers、FastAPI
- **自研**：执行引擎（图捕获与分派）、Attention backend、融合算子、调度与显存管理、Sampler
- **两个基准**：HF transformers 作正确性 golden；**v0.1 的 eager 模式作性能 baseline**（没有它，后面所有优化都无法量化）

### 0.1 实验对象 Qwen2.5-1.5B：参数与显存预算

| 项 | 值 |
| --- | --- |
| 层数 / hidden | 28 / 1536 |
| Q 头 / KV 头 | 12 / 2（GQA） |
| head_dim | 128 |
| intermediate | 8960 |
| vocab | 151936 |
| tie_word_embeddings | 是（lm_head 复用 embed_tokens） |
| 上下文 | 32K |
| **权重（bf16）** | **3.1 GB**（1.54B 参数） |
| **KV / token** | **28 KB** = 2(K,V) × 28 层 × 2 KV 头 × 128 × 2 字节 |
| **剩余给 KV** | **≈11 GB**（15.5 可用 − 3.1 权重 − ~1.4 激活与图） |

**并发容量上限**（11 GB ÷ 28 KB ≈ 41 万 token）：

| 平均上下文 | 理论最大并发 |
| --- | --- |
| 2K | ≈200 条 |
| 8K | ≈50 条 |
| 32K（打满） | ≈12 条 |

**两个仍然存在的硬约束：**

1. **logits 是大头。** `[num_tokens, 151936]` 在 bf16 下：2048 chunk = 622 MB、4096 = 1.24 GB、8192 = **2.49 GB**。`max_num_batched_tokens` 必须自己扫上界，不能照抄服务器默认值。
2. **CUDA Graph 桶数要克制。** 每个捕获尺寸占一份静态 buffer，16GB 下建议 1/2/4/8/16/32 六档而非十几档——桶数与显存的 tradeoff 本身就是一个可量化分析的题目。

### 0.2 「只做实验」≠「只做优化」（关键区分）

收窄的是**验证范围**，不是**架构通用性**。这点必须分清，否则项目会退化成只能跑一个模型的 demo：

| | 只优化 1.5B（❌ 错误理解） | 只在 1.5B 上实验（✅ 正确理解） |
| --- | --- | --- |
| 架构参数 | 常量写死 | 全部来自 config.json |
| head_dim | 只支持 128 | 任意（1.5B 恰好是 128） |
| 算子特化 | 为 1.5B 手调 | 以 **shape 特化键**管理，1.5B 只是当前被实测的那组键 |
| 结论表述 | 「我优化了 1.5B」 | 「我实现了通用机制，并在 1.5B 上量化验证」 |

**收窄带来的真实收益是「验证矩阵变小」**：不必为多尺寸的兼容性测试分心，省下的时间全部投入深度优化与测量。但架构不能因此退化——下面这些优化本来就是通用的，照做即可：

1. **权重预拼接是通用优化。** QKV 三投影合成一个大 GEMM、gate/up 同理，对任何模型成立，实现时按 config 算拼接维度。1.5B 上的数字（q/k/v 的 N 为 1536/256/256 → 拼接后 2048）只用来说明收益来源。
2. **算子融合是通用的。** residual+RMSNorm、RoPE 内联、SwiGLU 与模型尺寸无关。
3. **常量折叠是通用的。** RMSNorm 的 eps、RoPE 的 `inv_freq`、attention 的 `1/sqrt(d)` 都按 config 在初始化时预计算。
4. **shape 特化要用「特化键 + 通用兜底」，不能写死。** 这正是 vLLM 的思路：对常见 (batch, seq) 组合编译特化版本，另留一个 symbolic shape 兜底图。特化键是 **shape 而非模型名**，换尺寸依然有效。
5. **mega-kernel 要参数化生成。** 按 config 的 hidden / heads / head_dim 生成 Triton 的 constexpr，而不是为 1.5B 手写一个。
6. **全图捕获边界可以推得更远。** 按 (batch, seq) 分桶让 attention 静态化，尝试跳过 piecewise 直接整图捕获——这是能出结论的实验题，正好落在你的研究方向上。

> ⚠️ **实验结论的可推广性要说清楚**：1.5B 是 28 层 / **2 个 KV 头** / 32K 上下文；7B 是 28 层 / **4 个 KV 头** / 128K。分页显存与连续批处理这类**机制性结论**可以推广，但**数值结论不行**——`block_size` 最优值、饱和 batch、prefix 命中率都随 KV 头数与上下文长度变化。报告里标注「在 1.5B 上测得，换尺寸需重测」，不要外推。

---

## 1. 版本演进：每版新增什么

### v0.1 · Eager 基线
- **新增**：**config 驱动**的模型实现（层数 / hidden / head_dim / KV 头数 / vocab / tie_emb 全部从 config.json 读，不写死 1.5B）、embedding / RMSNorm / 带 bias 的 QKV / RoPE / GQA / SwiGLU / 共享或独立 lm_head、safetensors 权重加载与 name mapping、greedy + temperature/top-p 采样
- **执行模式**：eager，无图无编译
- **对齐**：vLLM `cudagraph_mode=NONE` / `-O0`
- **验收**：每层中间输出与 HF max abs diff < 1e-2；greedy 前 128 token 与 HF **完全一致**

### v0.2 · KV Cache
- **新增**：KV Cache 预分配（**故意先用连续预分配的浪费写法**）、prefill/decode 分离的自回归、静态批处理
- **执行模式**：eager
- **对齐**：vLLM 早期形态（未分页）
- **验收**：decode 单步耗时相对 v0.1 下降倍数；输出与 v0.1 逐 token 一致；测出 padding 浪费率（作为 v0.3 的动机数据）

### v0.3 · 分页显存 + 首个自研算子
- **新增**：BlockPool / KVCacheManager / block_table / slot_mapping；**Triton paged attention kernel** + `reshape_and_cache`；prefix caching（滚动 hash + LRU 双向链表）；copy-on-write 与 ref_cnt
- **执行模式**：eager
- **对齐**：PagedAttention + Attention backend 抽象
- **验收**：显存浪费率 X% → Y%；Triton vs torch 朴素版 vs FlashAttention 的延迟曲线；block_size 8/16/32/64 扫描
- **注意**：只需支持 head_dim=128（已锁定 1.5B）；KV cache tensor 按 28 KB/token 算，别照抄 Llama-3-8B 的 128 KB/token

### v0.4 · 连续批处理调度
- **新增**：Scheduler（waiting / running、token budget、chunked prefill、抢占）、Sequence 状态机、varlen 输入拼装
- **执行模式**：eager
- **对齐**：vLLM V1 scheduler（取消 prefill/decode 阶段划分，只维护 `num_computed_tokens`）
- **验收**：长 prompt 不阻塞 decode（用 TPOT p99 证明）；吞吐 vs HF baseline 提升倍数

> v0.1–v0.4 是「跑对」阶段。**从这里开始进入研究主线。**

### v0.5 · Full CUDA Graph（decode 全图捕获）
- **新增**：decode-only 全图捕获与回放；batch size 分桶（1/2/4/8/…）+ padding；**输入 buffer 地址固定**（input_ids / positions / block_table / slot_mapping 全部预分配，KV cache 指针不变）；warmup dummy run；多张图按桶管理
- **执行模式**：`FULL_DECODE_ONLY`
- **对齐**：vLLM `CUDAGraphMode.FULL_DECODE_ONLY`
- **验收**：**kernel launch 开销占比从 10–30% 降到 ~1%，端到端 TPOT 下降 15–25%**；输出与 eager 逐 token 一致
- **关键坑**：CUDA Graph 要求图内所有张量地址固定 —— 这是 v0.3 的 block_table 必须改成预分配 buffer 的直接原因

### v0.6 · 算子融合 + 权重预拼接
- **新增**：融合算子（residual+RMSNorm、RoPE 内联、SwiGLU、KV cache 写入融合）；**加载时按 config 把 QKV 与 gate/up 权重预拼接**（3 次 GEMM → 1 次、2 次 → 1 次；这是通用优化，不是 1.5B 特化）；Attention backend 保留三实现（Triton 自研 / FlashAttention / torch SDPA）用于对照
- **执行模式**：eager 与 v0.5 图共存
- **对齐**：vLLM `custom_ops` 融合算子栈
- **验收**：单层 kernel launch 数下降计数（Nsight / torch profiler 前后对比）；权重拼接后输出与拼接前逐 token 一致

### v0.7 · torch.compile 集成 + Piecewise CUDA Graph（核心里程碑）
- **新增**：Dynamo trace 出整张前向图 → 在 `splitting_ops`（attention）处切图 → 每个子图送 Inductor 编译 → **每个子图独立捕获 CUDA Graph**；`PiecewiseBackend` 按 batch size 分派：对 `compile_sizes`（1/2/4/8…）编译特化版本，另有一个 symbolic shape 兜底图；**Inductor 产物落盘缓存**加速重启
- **特化策略**：对常见 (batch, seq) 组合编译特化版本 + symbolic shape 兜底图。特化键是 shape 而非模型名；实验阶段只需为 1.5B 的 shape 预热，编译与重编译次数可控
- **执行模式**：`PIECEWISE`
- **对齐**：vLLM `CompilationMode.VLLM_COMPILE`（mode 3）+ `cudagraph_mode=PIECEWISE`
- **验收**：**mixed batch（prefill 与 decode 混排）也能吃到图加速**（这是 v0.5 做不到的）；首次启动 vs 缓存命中启动耗时对比

### v0.8 · CudagraphDispatcher 动态分派
- **新增**：`BatchDescriptor` 分派键（num_tokens / num_reqs / uniform）；**FULL > PIECEWISE > NONE 三级优先分派**；FULL 外层包装与 PIECEWISE 内层包装嵌套共存，运行时按模式透传
- **执行模式**：`FULL_AND_PIECEWISE`
- **对齐**：**vLLM 当前默认模式** `CUDAGraphMode.FULL_AND_PIECEWISE` 与 `CudagraphDispatcher`
- **验收**：uniform decode 命中 FULL、mixed batch 落 PIECEWISE、异常形状回退 eager；输出分派命中率统计

### v0.9 · Breakable CUDA Graph + MRv2 风格执行器
- **新增**：可中断 CUDA Graph（attention 断点处允许调度器介入，抢占 / 长序列不再导致整图失效）；流水线气泡消除；执行器与编译逻辑正交解耦
- **执行模式**：Model Runner V2 形态
- **对齐**：vLLM **Model Runner V2**（v0.25 起对 dense 模型默认，**v0.29 全模型默认**）+ breakable CG（#44050）
- **验收**：抢占 / 超长序列 / 变长 batch 场景下图的可用率；与 v0.8 的 TPOT 对比

---

## 2. 后续可选扩展（按性价比排序）

| 方向 | 新增内容 | 说明 |
| --- | --- | --- |
| **shape 特化与 mega-kernel（推荐）** | Triton `autotune` 按 shape 特化键搜最优 config 并缓存；mega-kernel（QKV-GEMM+RoPE+写 KV cache 融合）按 config 的 constexpr **参数化生成**；KV cache 布局可切换对照 | 特化键是 shape 不是模型名，换尺寸仍有效 |
| **跳过 piecewise 的整图捕获实验** | 按 seq 长度分桶让 attention 也静态化，尝试整图捕获并与 piecewise 对比 | 直接落在动态图执行研究方向上，可出对比结论 |
| FP8 KV cache | 1.5B 显存不紧张，收益主要是带宽而非容量 | 可选；sm_89 原生支持，可实测 |
| 投机解码与图的兼容 | draft-verify 多 token forward 对图捕获的影响 | 需额外小 draft 模型 |
| 张量并行算子 | Column/Row parallel Linear、all-reduce 融合 | ❌ 单卡只能验正确性，建议放弃 |

> **量化不再是必需项**：1.5B bf16 权重仅 3.1GB，16GB 卡留 11GB 给 KV，容量完全够。量化从「必需」降为「可选的带宽实验」。

---

## 3. 本机性能参照与 roofline

| 指标 | 值 |
| --- | --- |
| 显存带宽 | 672 GB/s（约 4090 的 2/3；**不要拿 4090/A100 的公开数据直接对标**） |
| **batch=1 的 TPOT 理论下界** | 读权重 3.1GB ÷ 672GB/s ≈ **4.6 ms**（≈217 tok/s）—— 这是硬下界，任何优化都突破不了，除非量化权重 |
| 每 token 计算量 | 2 × 1.31B（非 embedding 参数）≈ 2.62 GFLOP |
| 饱和 batch 估算 | ≈300（按机型标称 BF16 算力推算，**必须实测**） |

**饱和点实测方法**（这条数据是你 roofline 分析的落脚点）：固定 seq_len，把 batch 从 1 扫到 512，画 TPOT 与吞吐曲线。**饱和点之前，增大 batch 几乎不增加单步时间**（都在读同一份权重），吞吐近似线性增长——这正是连续批处理收益的物理来源；过了拐点转 compute-bound，TPOT 开始线性上升。

| 精度支持 | 状态 |
| --- | --- |
| bf16 | ✅ 原生，默认精度 |
| FP8 | ✅ sm_89 原生 tensor core，可实测 |
| FP4 / NVFP4 | ❌ 仅 Blackwell（sm_100+），只能调研 |
| 多卡 | ❌ 单卡，张量并行建议放弃 |

---

## 4. 研究主线：这两个方向必须能讲到底层

### 动态图执行
1. **为什么 attention 不能进全图** —— batch 内序列长度不同，shape 相关决策发生在运行时；block_table 是运行时数据不是常量。这是 piecewise 存在的根本原因
2. **FULL / PIECEWISE / NONE 的取舍** —— FULL 最快但只吃 uniform decode；PIECEWISE 通用但子图间有 eager 缝隙
3. **地址固定约束** —— 图内所有 buffer 地址不可变，反向决定了 v0.3 的 block_table 与 v0.5 的输入 buffer 设计
4. **分桶与 padding 的 tradeoff** —— 桶越少显存越省但 padding 浪费算力，桶越多显存占用越大
5. **Inductor 特化 vs symbolic 兜底** —— 特化快但要预编译多份，symbolic 慢一档但通用
6. **breakable graph 解决什么** —— 抢占与动态 shape 导致整图失效
7. **shape 特化下能不能跳过 piecewise**（本项目可做实验）—— 按 (batch, seq) 分桶让 attention 静态化，整图捕获 vs piecewise 的对比结论

### 算子优化
1. **paged attention 的间接寻址代价** —— 为什么必须把 gather 与 gemm 融合，否则多一次显存往返
2. **融合算子的边界** —— residual+RMSNorm 能融，attention 不能融；判断依据是「是否有运行时 shape 依赖」
3. **decode 为什么是 memory-bound** —— 每生成一个 token 要把权重读一遍，增大 batch 是在摊销权重 I/O，存在饱和 batch size 后转 compute-bound
4. **kernel launch 开销怎么量化** —— Nsight / torch profiler 对比 launch 数与 GPU 忙闲比，给出 10–30% → 1% 的实测数字
5. **权重预拼接的收益来源** —— 不只是省 launch，更是让 GEMM 的 N 维变大、提高 tensor core 利用率（1.5B 上 q/k/v 的 N 分别是 1536/256/256，拼接后 2048）

---

## 5. 反模式（六条）

1. ❌ 没有 v0.1 的 eager baseline 就开始优化 —— 后面所有数字都无法量化
2. ❌ 图捕获与调度同时改 —— 先 v0.4 跑对调度，再 v0.5 上图
3. ❌ 跳过 torch 朴素实现直接写 Triton —— 先跑对再换算子，两遍实现法
4. ❌ 只做 happy path —— 抢占、OOM、超长输入、空输入都要覆盖，这正是图失效的高发场景
5. ❌ 把 1.5B 上测出的**数值结论**当成普适规律 —— 机制可以推广，但 `block_size` 最优值、饱和 batch、命中率这些数字不行，报告里标注「在 1.5B 上测得」
6. ❌ 过早追求与 vLLM 的绝对性能对比 —— 目标是**可解释的性能**，每个数字都要能说清来自哪个设计

---

## 6. Qwen2.5-1.5B 实现坑点（v0.1 就会遇到）

1. **QKV 带 bias。** Qwen2 的 `q_proj / k_proj / v_proj` **有 bias**，Llama 没有。照抄 Llama 实现不会报错，只会静默算错——必须靠逐层对拍才能发现。
2. **tie_word_embeddings = true。** lm_head 复用 `embed_tokens` 权重，没有独立的 `lm_head.weight`；加载时别去找这个 key。
3. **RoPE theta = 1e6**（Llama-3 是 5e5），且用 HF 风格的 `inv_freq` 与 `position_ids` 约定。这是与 HF 对不齐的头号原因。
4. **vocab 151936，GQA 的 KV 头只有 2 个。** attention 里 query 要按 6 倍（12/2）展开去匹配 KV，repeat_interleave 的维度顺序容易搞错。
5. **上下文 32K。** 压测长上下文时正好用来验证分页显存与 prefix caching 的效果。
