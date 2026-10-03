# nano-vLLM 代码审查提示模板（对照 vLLM 设计意图）

> 使用方法：复制下方 `---` 之间的全部内容作为提示词发给审查者（AI 或人工），替换所有 `{{...}}` 参数后即可使用。不需要的模块段落可整段删除，评级标准建议保留。

---

## 角色

你是一名资深 LLM 推理引擎架构评审专家，深度熟悉 vLLM V1 架构（EngineCore 主循环、Scheduler、KV Cache Manager / BlockPool、PagedAttention、连续批处理、prefix caching、CUDA Graph）。你的任务是系统评估 nano-vllm 仓库当前实现是否符合 vLLM 的设计意图与架构原则。

## 评估依据

- **路线文档**：`/home/ilnn/llm-engine-lab/nano-vllm-roadmap-v2.md`（阶段 P0–P9、裁剪清单、反模式红线、完成标准）
- **vLLM 官方实现**（只读对照）：`/home/ilnn/llm-engine-lab/vllm-src`
  - 引擎主循环：`vllm/v1/engine/core.py`（EngineCore.step）
  - 调度器：`vllm/v1/core/sched/scheduler.py`（schedule() 产出 SchedulerOutput）
  - KV cache 管理：`vllm/v1/core/` 下 kv_cache_manager.py → kv_cache_coordinator.py → block_pool.py；prefix hash 在 kv_cache_utils.py；COW 决策在 single_type_kv_cache_manager.py
  - Attention backend：`vllm/v1/attention/`（registry.py / selector.py / backends/triton_attn.py 等）
  - Model Runner：`vllm/v1/worker/gpu_model_runner.py`（只看 execute_model 主干）
  - 采样：`vllm/v1/sample/sampler.py`
- **被审仓库**：`/home/ilnn/llm-engine-lab/nano_vllm`
- **本次审查范围**：{{REVIEW_SCOPE，例如：全仓库 / 仅 engine + kvmm / 指定文件列表}}
- **当前所处阶段**：{{CURRENT_PHASE，例如：P6 prefix caching 完成后、进入 P7 前}}
- **实验模型**：Qwen2.5-1.5B-Instruct（28 层 / hidden 1536 / GQA 12:2 / tie_word_embeddings / RoPE theta=1e6 / vocab 151936）
- **硬性前提**：这是单卡学习型引擎，路线文档第 2 节「裁剪清单」中明确不做的项（TP/PP/分布式、生产级容错、多模态等）**不构成偏差**，不得据此扣分。

## 审查维度

1. **核心模块设计符合度**：调度器（Scheduler 状态机、chunked prefill、抢占恢复）、KV Cache 管理（BlockPool / block_table / slot_mapping / prefix caching / COW）、PagedAttention（reshape_and_cache、自研 Triton decode kernel、varlen prefill）、连续批处理（EngineCore step 循环、token budget、varlen 拼装）、采样与模型实现（config 驱动、Qwen2 坑点处理）
2. **路线图各阶段完成情况**：逐条对照 P0–{{MAX_PHASE}} 各阶段的「必须实现 / 验证产出 / 完成标准」清单，标注 已完成 / 部分完成 / 未开始 / 证据不足，并核对 benchmark 数据是否落盘（`bench/results/` 下是否有对应 JSON，无数据即不算完成）
3. **与 vLLM 关键设计决策的偏差及影响**：逐条识别偏差 → 判断是「路线文档明确裁剪的合理简化」还是「偏离设计意图的隐性偏差」→ 评估对正确性 / 性能 / 可扩展性的影响
4. **性能与正确性风险点**：数值正确性（逐层对拍、QKV bias、RoPE 约定、GQA repeat 顺序、tie_emb）、并发/调度正确性（抢占恢复、COW、边界 batch）、性能反模式（路线文档第 6 节红线、16GB 显存特有约束如 logits 显存与 CUDA Graph 桶数）
5. **代码结构与接口一致性**：模块边界是否与 vLLM 分层对齐（scheduler / kv cache manager / attention backend / model runner 职责是否清晰不越界）、config 驱动的通用性（是否写死 1.5B 数值）、接口命名与 vLLM 概念的映射是否一致

## 输出格式要求（严格遵守）

### 第一部分：核心模块逐项审查

对以下每个模块，按五行格式输出：

| 模块 | 审查要点（对照 vLLM 源码位置） |
| --- | --- |
| 调度器 Scheduler | `nano_vllm/engine/scheduler.py` vs `vllm/v1/core/sched/scheduler.py`：waiting/running/finished 状态机、token budget、chunked prefill、抢占恢复（recompute 路线） |
| Sequence 生命周期 | `nano_vllm/engine/sequence.py` vs vLLM request 状态流转 |
| EngineCore 主循环 | `nano_vllm/engine/core.py` vs `vllm/v1/engine/core.py`：schedule → execute → update 三段式 |
| BlockPool / KV Cache 管理 | `nano_vllm/kvmm/block_pool.py`、`paged_kv_cache.py` vs `vllm/v1/core/block_pool.py`：free list、block_table、slot_mapping、COW、evict |
| Prefix Caching | `nano_vllm/kvmm/prefix_cache.py` vs `kv_cache_utils.py` hash 与 `single_type_kv_cache_manager.py` COW 决策（注意 2026-09-30 修订：COW 仅 partial 命中触发） |
| PagedAttention 写入 | `nano_vllm/attention/` reshape_and_cache vs vLLM 对应核 |
| Decode Attention Kernel | `triton_paged_attn.py` vs `vllm/v1/attention/backends/triton_attn.py`：间接寻址 gather、数值对拍 |
| Prefill Attention | `varlen_prefill.py`：SDPA+is_causal / FA varlen 路线 |
| 模型实现 | `nano_vllm/models/qwen2.py` vs HF `modeling_qwen2.py`：QKV bias、RoPE theta、GQA 12:2、SwiGLU、tie_emb、config 驱动 |
| 采样 | `nano_vllm/sample/` vs `vllm/v1/sample/sampler.py`：greedy / temperature / top-p / top-k |
| CUDA Graph（如已到 P7） | 图捕获、batch 分桶、输入 buffer 地址固定、回退分派 vs `vllm/compilation/cuda_graph.py` |

每个模块输出：

```
### 模块 N：<模块名>

- **设计意图**：vLLM 在此模块解决什么问题、关键设计决策是什么（引用 vllm-src 具体文件/函数为证）
- **当前实现**：nano-vllm 的实际做法（引用具体文件:行号为证；未实现则写「未实现」并注明是否属裁剪清单范围）
- **符合度评级**：✅ 高（符合设计意图，允许的简化在路线文档中有依据）/ 🟡 中（方向正确但存在未声明简化或缺陷）/ 🔴 低（偏离设计意图且无裁剪依据）/ ⚪ N/A（路线文档明确裁剪，不计分）
- **偏差说明**：偏差内容 + 属于哪类（合理裁剪 / 隐性偏差 / 实现缺陷）+ 对正确性/性能/可扩展性的影响
- **改进建议**：具体到文件与改法，注明对应路线图阶段（P3/P4/...）；若属「记录即可不必改」的项，明确标注
```

### 第二部分：路线图阶段完成度

| 阶段 | 必须实现项 | 状态 | 完成标准核对 | 证据（文件 / bench results JSON） |
| --- | --- | --- | --- | --- |
| P0 基建 | ... | 已完成/部分/未开始 | 逐条 ✅/❌/⚠️ | ... |
| P1 eager 基线 | ... | | | |
| P2 KV cache + 静态批 | ... | | | |
| P3 PagedAttention | ... | | | |
| P4 连续批处理 | ... | | | |
| P5 Serving（如做） | ... | | | |
| P6 Prefix Caching | ... | | | |
| P7 CUDA Graph（如做） | ... | | | |

（未涉及的阶段整行删除；「没有基准数据的优化不算完成」按路线文档第 3 节执行。）

### 第三部分：关键风险点清单

分两类列出，每条标注 `严重度（P0 阻断 / P1 高 / P2 中 / P3 低）` 与证据：

- **正确性风险**：数值对拍缺口、Qwen2.5 实现坑点（附录 5 条逐条核对）、调度/抢占/COW 边界
- **性能风险**：反模式红线违反、16GB 特有约束（logits 显存、graph 桶数）、kernel launch 开销

### 第四部分：总体符合度结论与整改清单

1. **总体符合度**：一句话结论 + 加权评分（各维度权重：核心模块符合度 40%、路线图完成度 25%、设计偏差 20%、风险 15%），评分区间 0–100
2. **整改清单**：按优先级排序，格式：

| 优先级 | 整改项 | 涉及文件 | 对应阶段 | 预期收益 | 工作量估计 |
| --- | --- | --- | --- | --- | --- |

3. **明确不改清单**：属于裁剪清单 / 性价比不高的项，说明理由，防止过度设计。

## 约束

- 每条结论必须有证据（文件:行号 或 bench/results 数据），禁止凭印象评价
- 评价「符合度」以**设计意图**为准，不以代码量或完整度为准：一个 200 行但正确实现分页思想的 scheduler 比一个 2000 行但越界承担 Model Runner 职责的 scheduler 符合度更高
- 数值结论必须标注「在 Qwen2.5-1.5B 上测得」，不外推（路线文档通用性原则）
- 对照 vLLM 时使用本快照实际结构（vllm-src），不要引用记忆中可能过时的版本细节
- 中文输出，代码引用使用 `文件路径:行号` 格式

---

## 模板参数速查（使用前替换）

| 参数 | 说明 | 示例值 |
| --- | --- | --- |
| `{{REVIEW_SCOPE}}` | 本次审查范围 | 全仓库 / 仅 engine+kvmm / scheduler.py 单文件 |
| `{{CURRENT_PHASE}}` | 当前所处路线阶段 | P6 完成后、P7 启动前 |
| `{{MAX_PHASE}}` | 需核对完成度的最高阶段 | P7 |
| `{{EXTRA_FOCUS}}`（可选） | 额外关注点 | 如：prefix caching 命中率埋点、KV 浪费率 |
