# nano-vllm 项目评估报告：架构设计 · 模块划分 · 开发路线

> 评估对象：`https://github.com/ilnnfover/nano-vllm`（HEAD = `e9327d1`，14 个提交，main 单分支）
> 评估方式：完整 clone 后通读全部 2704 行代码与文档，并把文档中的性能数字逐条与 `bench/results/*.json` 对账
> 评估时间：2026-09-22
> **修订说明（2026-09-22 二次复核）**：本报告已按逐条复核结果就地更正，所有更正处标注了 `复核更正 / 复核补充`。完整的复核意见（含原报告的 1 处事实错误、4 处逻辑或表述过强、2 处遗漏的明细）与**分优先级的 P3 开工前问题清单**见 `nano-vllm-报告复核与P3前问题清单.md`。

---

## 0. 总体结论

**这是一份质量明显高于同类"从零实现推理引擎"项目的前三阶段实现，开发方法论（基线优先、一版一主题、对拍驱动、数据留档）几乎无可指摘。真正的问题不在"做得对不对"，而在两件事：一是"抽象边界还没有出现"——唯一的编排层 `NanoRunner`（223 行）同时承担引擎主循环、掩码构造、静态批、指标统计、两条数值路径与对外 API；二是**一个被 roadmap 列为验收标准、却没有任何阶段负责解决的问题**——prefill 对整段序列过 `lm_head`，32K 单条的 logits 单独就是 9.98 GB，P3 的"32K 单条可跑通"在不改这里时不可达（§3.2、§3.2.1）。**

| 维度 | 评分 | 一句话判断 |
| --- | --- | --- |
| 开发方法论 / 路线规划 | ★★★★★ | P0-P9 主干、裁剪清单、反模式红线、统一 bench 口径——这是全项目最值钱的部分 |
| 数据可信度 / 可复现性 | ★★★★☆ | 文档数字与结果 JSON 逐条吻合（我全部核对过），但缺 pin 与 CI，换台机器跑不起来 |
| 正确性验证严谨度 | ★★★★☆ | 三层证据（位级 / 跨精度 / 逐 token）+ 踩坑记录质量极高；缺口是**回归防护与免模型单测** |
| 模块划分与代码组织 | ★★☆☆☆ | 目录名照抄 vLLM V1，实现却是 V0 的巨石执行器；3 个空壳包是"结构先行"的产物 |
| 抽象设计（接口层） | ★★☆☆☆ | 缺五件套 `Sequence` / `Scheduler` / `AttentionMetadata` / `AttentionBackend` / `KVCacheManager`，外加一个重要交付项（输出处理：detokenizer / `finish_reason`） |
| 工程化基线 | ★★☆☆☆ | 无 `pyproject.toml` / lock / ruff / CI / tags / LICENSE；README 还是环境文档 |
| 当前实现的具体缺陷 | ★★☆☆☆ | 有 3 个必须马上修的隐患（chunked prefill 前缀丢失、prefill 全序列 logits 9.98 GB、s² mask 显存）、若干性能与可维护性问题 |

**一句话路线建议**：**P3 开工前，先花 1–2 天补一层接口（`AttentionMetadata` + `AttentionBackend` + `Sequence` + 公开 API），然后把 P3 拆成 P3a（块池 + SDPA-paged，只要正确性）与 P3b（Triton kernel）**。这一步能把后面 P3/P4/P7 三次"穿透式重构"合并成一次。

---

## 1. 这个项目做得好的地方（不要改）

先明确哪些是要**保护**的资产，避免后面的重构把它们改掉：

1. **"没有基线就不优化"的执行力是真的**。P0 阶段一行引擎代码没写，先把 golden 对拍链、统一 bench 口径、环境快照做完了。`docs/stages/P0.md` 里那句"P0 不写引擎代码"是工程纪律的体现。
2. **数字全部可对账**。我逐条核对了文档与 JSON：
   - `batch_bench`：serial 42.2 tok/s → batch=8 279.2（6.61×）→ batch=16 392.3（9.29×）；pad 浪费 41.2%、KV 浪费 15% —— 与 `batch_bench_Qwen2.5-1.5B-Instruct_bf16.json` **完全一致**；
   - `p2_nano_cache_gpu`：TPOT p50 22.27ms / 44.85 tok/s；`p2_nano_eager_gpu` 44.66ms / 20.24 tok/s → 两处比值 2.005× 与 2.22×，与 P2 笔记的"整体 TPOT 2.00×、吞吐 2.22×"**完全一致**；
   - HF 基线 TPOT p50 26.81ms（38.84 tok/s），与 P0 笔记一致。
   **在我评估过的同类学习型项目里，文档数字能做到零漂移的是少数。**这条要守住。
3. **踩坑记录是本项目最有价值的"二次产出"**。P1 的 8 条（HF 5.x `output_hidden_states` 末元素语义、`repetition_penalty=1.1` 让 HF 的 greedy 不是纯 argmax、tie 权重下 `state_dict()` 不去重、bf16 diff 必须与 bf16 固有损失对比才有意义）和 P2 的 6 条（短请求 position_ids、SDPA mask 维度、预分配容量必须从数据推导而非拍默认值）都是硬信息，面试时是能直接展开讲的料。
4. **"故意写差一版"的教学设计**。P2 的 `KVCache` 明确声明"故意用浪费写法为 P3 留动机"，并把 15% / 41.2% 两个浪费率落盘成证据。这是很好的方法论。
5. **裁剪清单和反模式红线**（roadmap §2 / §6）。敢写"❌ 不做 TP/PP/分布式"、"❌ 跳过 torch 朴素实现直接写 Triton"的项目不多。
6. **提交粒度与信息质量**：`feat(P2): ...` / `docs(P2): ...` 的规范化前缀 + 阶段标签，14 个提交每个都有意义，没有 `update` / `fix`。

---

## 2. 架构与模块划分的问题

### 2.1 目录名抄了 vLLM V1，实现却是 V0 的巨石 —— 命名与职责错位

当前 `nano_vllm/` 实际布局：

```
nano_vllm/
├── __init__.py                     # 0 字节，无公开 API
├── config.py                       # 57 行  ✅ 职责清晰
├── kv_cache.py                     # 62 行  ⚠️ 核心子系统放在包根
├── attention/__init__.py           # 0 字节  空壳
├── core/__init__.py                # 0 字节  空壳
├── engine/__init__.py              # 0 字节  空壳
├── models/qwen2.py                 # 259 行 ⚠️ 模型 + 缓存 IO + 后端选择 混在一起
├── model_executor/runner.py        # 223 行 ⚠️ 唯一的"引擎"，什么都干
└── sample/sampler.py               # 44 行
```

三个问题的性质不同，但根源相同——**结构先行，职责未定**：

**问题 A：3 个空壳包是过早结构化的化石。**
`attention/`、`core/`、`engine/` 在 P0 就建了（`docs/stages/P0.md` 明确写"包骨架占位，P1 起填充"），但到 P2 结束仍然是 0 字节。站在评审者角度，这传递的信号是"目录是猜的"：`engine/` 与 `model_executor/` 的边界至今没有被定义，`core/` 到底是 V1 的 `core`（scheduler + kv_cache_manager）还是别的，也没人知道。**建议：立刻删掉没有内容的包，需要时再建。**空目录不是架构，是负债。

**问题 B：`model_executor` 是 V0 的命名，与 roadmap 声称的"对齐 vLLM V1"矛盾。**
roadmap 第 4 行写"机制层面对齐 vLLM V1 架构"，§5 参考地图也整张表指向 `vllm/v1/*`。但 V1 里模型执行的角色叫 **`worker/`**（`gpu_model_runner.py`），`model_executor/` 是 V0 的名字，而且 V0 的 `model_executor` 语义是"逐层权重加载 + 层算子分发"，跟这里 `runner.py` 干的活（prefill/decode 循环、掩码构造、批处理、统计）完全不是一回事。这份代码里的 `model_executor/runner.py` 实际上是 **`engine_core` + `model_runner` + 一点 scheduler** 的混合体。

**问题 C：`kv_cache.py` 放包根，而 `core/` 空着。**
KV 缓存是 P3–P7 的主战场（块池、引用计数、COW、前缀哈希、重分配预算）。它现在是一个 62 行的顶层模块，且**混了两个职责**：`capacity` / `waste_rate` 是"管理器"的职责，`k_cache` / `v_cache` 是"存储"的职责。P3 一进来这两者必然分家（vLLM 的做法是 `KVCacheManager` 管分配、`BlockPool` 管块、张量池只是被持有的内存）。被动等 P3 再拆，意味着 `runner.py` 里所有调用点都要改。

### 2.2 缺五个关键抽象（外加一个交付项），而路线图后三个阶段都要求它们存在

| 缺失抽象 | 谁会立刻需要它 | 现状 | 不补的代价 |
| --- | --- | --- | --- |
| `AttentionMetadata`（block_table / slot_mapping / seq_lens / query_start_loc / max_seq_len） | P3（分页）、P4（chunked prefill）、P7（图捕获） | 现在是 `is_prefill` + `cache_seq_len` + `attn_mask` 三个裸参数，从 `Qwen2ForCausalLM.forward` 一路手工透传到 `Attention.forward` | 每个阶段都要改一层签名，**穿 4 个文件**（`models/qwen2.py`、`runner.py`、`bench/*`、`tests/*`） |
| `AttentionBackend` 接口 + 注册表 | P3（roadmap 明确要求 Triton / flash-attn / SDPA **三实现可切换**，P8 要保留对照开关） | `qwen2.py:115` 一行 `if self.num_kv_groups > 1 and q.is_cuda and self.head_dim <= 256:` 现场决定走 GQA 融合还是 `repeat_kv` | 三实现无法共存；更要命的是**这个分支让数值路径随设备变化**，正是 P2 踩坑 6"batch≠serial"的嫌疑源之一（见 §3.4） |
| `Sequence` / `Request` 状态对象 | P4（waiting/running/finished 状态机、token budget、抢占恢复） | 全在 `generate_batch` 的局部变量里：`seq_lens`、`done`、`outs`、`valid_mask` | P4 会把 `generate_batch` 整个删掉重写，P2 在这上面花的所有打磨（pad mask、逐请求 position_ids）都无法复用 |
| `Scheduler` + `EngineCore.step()` | P4 | 不存在。`generate` 是"给定 prompt 阻塞式跑完" | 与 P4 完成标准直接冲突：chunked prefill、抢占恢复、token budget 都需要"每步调度一次"的主循环 |
| 输出处理（detokenizer / `finish_reason` / stop 条件） | P5（OpenAI 兼容 API 必须报 `finish_reason`） | `generate` 返回 `list[int]`，EOS 被 break 掉、**不记录停止原因**，`max_new_tokens` 到达也无信号 | P5 无法实现，且 streaming 增量反解码无处安放。roadmap 全文没有提到 detokenizer（只在 §5 参考地图里出现过一次） |
| `KVCacheManager`（分配 / 追加 / 释放 / 显存预算） | P3a（块池一落地就要有）、P4（token budget 与抢占） | `kv_cache.py` 一个 62 行类同时是"张量池"和"管理器"；`seq_len` 还在 `runner.py:67,75,163,208` 被外部手工赋值 | P3 换 BlockPool 时，这类隐式契约是最容易漏改的地方（§3.9） |

**这五个抽象不是"以后再说"的东西，而是 P3 第一个提交就要用到的。**当前 roadmap 声称"每阶段只依赖前面阶段，不回改架构"，但 P2 的笔记自己就记录了 3 个已有文件的改写（`models/qwen2.py`、`model_executor/runner.py`、`bench/run_local.py`——顺带一提，该笔记的小节标题写的是"修改代码（2 个）"，**计数本身就是错的**，见 §3.9）；按现在的设计，P3/P4/P7 会各再来一次。

### 2.3 模型层承担了不属于它的职责（`models/qwen2.py`）

`Attention.forward` 内部做了三件不该模型管的事：

1. **写 KV 缓存**（`qwen2.py:103` `kv_cache.write(layer_idx, k, v, cache_seq_len)`）——缓存写入策略（写哪、写多少、是否越界）是缓存管理器的事。P3 引入 `slot_mapping` 后，写入位置由块表决定，模型不该知道；P6 引入前缀复用后，"要不要写"也要由管理器决定（命中块跳过重算时某些位置根本不该写）。
2. **读写语义分支**（`qwen2.py:105-108`）——详见 §3.1，这里藏着一个真实的隐患。
3. **选择 attention 后端**（`qwen2.py:115`）——见上表。

**建议的接口形态**：模型只负责"给我 q/k/v 和一份 metadata，我给你 attention 输出"：

```
Attention.forward(hidden_states, position_embeddings, attn_metadata) -> Tensor
    q,k,v = qkv_proj(...) + rope(...)
    return self.backend.forward(q, k, v, attn_metadata, layer_idx)
```

写入缓存的动作挪到后端/模型执行器里（`reshape_and_cache` 本身就是 P3 的独立 kernel，它天然属于后端而非模型）。这样 P3b 换 Triton kernel、P7 加图捕获、P8 融合 RoPE 都不需要碰模型定义文件。

### 2.4 公开 API 缺失，且 bench 直接调私有方法

- `bench/run_local.py:74,81` 调用的是 `self.runner._prefill(...)` / `self.runner._decode(...)` —— **下划线私有方法**。这意味着每次引擎内部重构都会打破压测框架，而压测框架是"数字可复现"这一核心原则的载体。这是当前最紧的一处耦合。
- `SamplingParams` 定义在 `model_executor/runner.py:15-20`。它是**公开 API 类型**（bench、tests、未来 HTTP 层、以及用户代码都会用），却住在一个内部执行器文件里。应在 `nano_vllm/types.py`（或 `sample/params.py`），并由 `nano_vllm/__init__.py` 导出 `LLM` / `SamplingParams` —— 现在 `__init__.py` 是 0 字节。
- 没有 `LLM` 门面。vLLM 的用户视角是 `LLM(model).generate(prompts, SamplingParams)`；本项目目前唯一的入口是 `NanoRunner`，而它同时是"引擎"和"用户 API"，所以它没法变成引擎。

### 2.5 两条数值路径挤在一个方法里

`runner.py:79-111` 的 `generate(use_cache=...)` 用布尔开关在同一方法内维护两套自回归循环（eager 全量重算 / KV cache 增量）。P1 的 eager 基线是必须留的资产，但它属于**参考实现**，不属于引擎主类。建议：

- 把 eager 路径移到独立的参考执行器（例如 `nano_vllm/reference/eager_runner.py`）或直接留在 `tests/` 作为对拍工具；
- 引擎主类只保留一条路径。

否则 P4 之后 `NanoRunner` 里会长出 `use_cache` / `use_continuous_batching` / `use_cuda_graph` / `use_paged` 一串开关，变成配置地狱。

---

## 3. 当前实现的具体问题（按严重度）

### 3.1 🔴 chunked prefill / 前缀复用下会静默丢掉前缀（正确性隐患）

`models/qwen2.py:102-108`：

```
if kv_cache is not None:
    kv_cache.write(layer_idx, k, v, cache_seq_len)
    total_len = cache_seq_len + s
    if is_prefill:
        k_attn, v_attn = k, v          # ← 只用当前 chunk
    else:
        k_attn, v_attn = kv_cache.read(layer_idx, total_len)
```

`is_prefill=True` 时**无条件只用当前 chunk 的 K/V**，忽略了 `cache_seq_len > 0` 时缓存里已有的历史。P2 阶段这没问题，因为 prefill 永远是 `cache_seq_len=0`（`runner.py:66` 确实传 0）。但：

- **P4 的 chunked prefill** 正是"prefill 且 `cache_seq_len > 0`"，语义上必须 attention 到 `cache[0:cache_seq_len] + 当前 chunk`。按现在的代码，第二个 chunk 会看不到第一个 chunk。roadmap P4 的验收标准是"长 prompt prefill 期间已运行请求的 TPOT p99 劣化 <2×" + "抢占-恢复后输出仍与 P1 一致"——**如果抢占-恢复走的是分块重算，第二条会直接挂掉；即使它走整段重算（`cache_seq_len=0`，不受影响），chunked prefill 本身也已经错了**。失败形态是最难查的那种：不报错、不崩，只是输出变了。
  > ⚠️ 表述更正：原报告把"抢占-恢复会挂"写成必然，实际是条件性的——vLLM 的抢占-恢复若整段重算 prefill 就不触发。**必然触发本 bug 的是 P4 的 chunked prefill 与 P6 的前缀复用。**
- **P6 的 prefix caching 命中块复用**同理：命中意味着 prefill 时 `cache_seq_len > 0`。
- 代码里没有任何断言或测试守住这个语义。

**修法**：把 `is_prefill: bool` 换成能表达"本次调用是 chunk"的语义（例如 metadata 里的 `query_start_loc` / `num_cached_tokens`），attention 一律按 `[历史 KV] ++ [当前 chunk KV]` 构造。**并且在 P3 之前就加一个 CPU 上的 chunked-prefill 单测把这个语义钉死**——即使 P3 还没实现分页，用现在的连续缓存也能测（分两次调用 `model.forward`，第二次传 `cache_seq_len=len(chunk1)`，与一次全量 prefill 的 logits 对比）。

### 3.2 🔴 `s²` 稠密 mask：对正确性零贡献，但吃掉 2.15 GB（长上下文的第二障碍）

`runner.py:112-118` 的 `_build_prefill_mask` 构造 `[b, 1, s, s]` 的实数掩码（float16/bf16，不是 bool）：

| 场景 | mask 显存 |
| --- | --- |
| batch=1, s=1024（medium） | 2 MB |
| batch=1, s=8192（long） | 134 MB |
| **batch=1, s=32768（shared_prefix）** | **2.1 GB** |
| batch=8, s=1024 | 16 MB |

`bench/dataset.py` 里 `shared_prefix` 类刻意做了 32768 前缀（为 P6 准备），而 roadmap P3 的完成标准写着"长上下文（32K 单条）可跑通"。掩码本身的账单是：s=8192 → 128 MiB，s=32768 → **2.15 GB**（再叠加 `tril` / `full` 两个 bool 中间量，构造期峰值约 4 GB，而同一条序列的 KV 只有 0.94 GB）。**但要先说清楚：掩码不是 32K 单条的第一障碍，logits 才是——见下面 §3.2.1。**另外"`-inf` 掩码损失精度"这个担心不成立（`-inf` 在 fp16/bf16 中都精确可表示，`score + (-inf) = -inf` 是 IEEE 精确行为）；真正的代价只有两条：掩码必须被完整物化，以及传实数掩码会让 SDPA 不走 flash 因果 kernel。

**为什么现在还没爆炸**：`docs/stages/P2.md` 的 benchmark 只跑了 `short`（prompt 128）和 `medium`（prompt 1024），`long`(8192) 与 `shared_prefix`(32768) 这两类数据从 P0 生成之后**从未被任何 bench 或测试跑过**。所以这不是"还没暴露"，是"还没试过"。

### 3.2.1 🔴 32K 单条真正的显存大头是 logits，不是掩码（复核新增）

`models/qwen2.py:245` 对整段 `[b, s, hidden]` 过 `lm_head`，产出 `[b, s, vocab]`；而 `runner.py:165-166` 只取了 `last_idx` 那一行。于是长上下文的显存账单是：

| s | logits `[1,s,151936]` bf16 | mask `[1,1,s,s]` bf16 | KV（1 序列） |
| --- | --- | --- | --- |
| 1024 | 0.31 GB | 2.0 MiB | 0.03 GB |
| 8192 | 2.49 GB | 128 MiB | 0.23 GB |
| 32768 | **9.96 GB** | 2.15 GB | 0.94 GB |
| 32832（`shared_prefix` 实际长度） | **9.98 GB** | 2.16 GB | 0.94 GB |

**旁证**：`batch_bench` 记录的 `batch=16, max_prompt=1024` 峰值 8.18 GB ≈ 3.1（权重）+ 4.98（logits）——**这 8.18 GB 里约 6 成是 logits**。roadmap §1 自己就写了"logits `[num_tokens, 151936]` 是大头：8192 token chunk 在 bf16 下 2.49GB"，但**没有任何一个阶段负责把它修掉**。

因此：

- `long`(8192)：掩码 128 MiB 不构成威胁，2.49 GB 的 logits 可跑；
- **32K 单条：3.1 + 9.98 = 13.1 GB > 13 GB 预算，不改 logits 就一定 OOM。删掉掩码并不能让 P3 的"32K 单条可跑通"变成可达。**

**修法**：prefill / decode 都只对 `last_idx` gather 后再过 `lm_head`（`hidden[arange(b), last_idx]` → `lm_head`），显存降三个数量级；golden 对比用的全量 logits 路径保留开关（`golden/check_diff.py` 需要逐位置 logits）。**注意这会改变 GPU 上 `lm_head` 的 GEMM shape，属数值路径变更，需同步重跑 P1/P2 的 bench 与对拍。**

**顺带一个新发现**：`bench/dataset.py:24` 的 `shared_prefix` 实际 `prompt_len = 32768 + 64 = 32832`，**已经超过 Qwen2.5-1.5B 的 `max_position_embeddings = 32768`**。拿它当"32K 单条可跑通"的验收样本，等于用越界样本验收，且 RoPE 位置超出训练范围、输出质量不可解释。建议前缀改 32704，或显式标注为越界样本单独讨论。

**修法（已在本机实测验证，脚本见 `mask_fix_proof.py`）**

先给结论，这条比原报告第 2 条更省事：**这个稠密掩码对正确性的贡献是零，可以整块删掉。**

原因：`runner.py:137-141` 把 prompt 放在 `input_ids[i, :len_i]`，**padding 在右**。对任一有效 query 列 `i < len_i`，因果条件 `j ≤ i` 蕴含 `j < len_i`，所以 padding 位置的 key 本来就永远不会被有效行看到——`causal ∧ padding` 与单纯 `causal` 在有效行上**完全等价**。掩码唯一改变的是 padding 行自身的输出，而那些行会被 `last_idx = len_i - 1` 丢弃。

实测（`prompt_lens=[3,9,17]`，12 Q 头 / 2 KV 头的 GQA 形态，4 组 dtype × 实现组合）：

| dtype | KV 展开方式 | 有效行 max\|Δ\| | padding 行 max\|Δ\| |
| --- | --- | --- | --- |
| float32 | repeat_kv | **0.000e+00** | 2.799 |
| float32 | enable_gqa | **0.000e+00** | 2.231 |
| bfloat16 | repeat_kv | **0.000e+00** | 2.586 |
| bfloat16 | enable_gqa | **0.000e+00** | 3.672 |

四组全为**逐位一致**。所以修法是：

1. **立刻可落地（改 2 行，零架构改动）**：`generate_batch` 里 prefill 传 `attn_mask=None`，删掉 `_build_prefill_mask` 整个函数。`Attention.forward` 里 `is_causal = is_prefill and not use_mask` 会自动变成 `True`，走 SDPA 的原生因果路径。收益有三层：
   - 掩码显存 **2.15 GB → 0**（对 32K 是必要但不充分：还需要 §3.2.1 的 logits 改造）；
   - 少一次 `torch.tril(32768²)` + `masked_fill_` 的全量访存；
   - 附带效果：prefill 侧少一个"有无掩码导致 SDPA 换 kernel"的数值分叉因子。**但注意这并没有把分叉因子减到 1 个**——decode 侧仍必须传掩码（`generate_batch` 里未写入的 cache 位置是 0 填充，不掩蔽会稀释输出），而串行路径的 decode 没有掩码，两条路径在 decode 阶段依旧选了不同后端。§3.4 的对照实验必须把这一项也作为变量。
   - 代价：padding 行仍然参与 forward 计算（41.2% 的 padding 计算浪费是既有的，属于 P4 的账，这次不动）。
2. **P3b / P4 阶段（消除 padding 本身）**：用 varlen 打包（`cu_seqlens`）把 pad token 从批里彻底去掉。这需要 flash-attn varlen 或自研 kernel，属于分页 attention 的同一批工作。
3. **P4 chunked prefill（长上下文的通用解）**：用 `max_num_batched_tokens` 限制单次 forward 的 `s`，长 prompt 切成若干块。这既把 `s²` 从源头掐死，又正好暴露并强制修掉 §3.1 那个"prefill 丢前缀"的 bug——**两件事是同一个改动**。
4. **不要用稠密 bool 掩码当折中**：只省 2×，32K 仍是 1 GB，且照样禁用 flash 因果 kernel。

**⚠️ 更正原报告的一处建议**：我原先写的第 2 条推荐用 `flex_attention` + `create_block_mask` 替代 `s×s`，**这条是错的**，实测后撤回。看 torch 源码 `torch/nn/attention/flex_attention.py:1213 → 1009`：`create_block_mask()` 内部先调 `create_mask(mask_mod, B, H, Q_LEN, KV_LEN)` **物化出完整的 `[B, H, Q, KV]` 稠密 bool 张量**，再把它归约成块掩码。也就是说它省的是"回放期"显存，不是"构造期"：

| s | 现行 `[b,1,s,s]`（head 维靠广播） | flex 构造期 `[B,H,s,s]`（H=12） | 倍数 | 构造后的 BlockMask 存储 |
| --- | --- | --- | --- | --- |
| 8192 | 128 MB | 768 MB | 6× | 4 KB |
| 32768 | 2048 MB | 12288 MB | 6× | 64 KB |

本机实测：`s=4096` 构造成功（临时密张量 192 MB）产出 1.5 KB 的 BlockMask；**`s=32768` 直接 OOM，进程尝试分配 64 GiB**。结论：BlockMask 只在"同一 mask 结构能被反复复用"时才划算（固定长度桶、跨请求共享的模式），对"单请求、长度每次不同、只调一次"的 prefill 反而是负优化。你现在的 `[b,1,s,s]` 之所以还没那么贵，纯粹是因为 head 维靠广播——这是它唯一的优点。

**防护措施（避免以后重新引入）**：
- 在 `Attention.forward` 或 `AttentionMetadata` 构造处加断言，禁止出现 `shape[-2] == shape[-1] and ndim == 4` 的实数 mask；只允许传 varlen metadata 或 `None`。
- 把 `long`(8192) 与 `shared_prefix`(32768) 两类数据纳入**每个阶段**的 smoke bench（现在 `bench/dataset.py` 生成了却没人跑）。
- 加一条常量级单测：prefill 路径的 SDPA 调用必须满足 `(attn_mask is None) or (attn_mask.dtype == torch.bool)`。


### 3.3 🟠 decode 循环的写法与 P7 CUDA Graph 天然不兼容

`runner.py:174-212` 的 decode 循环里，每一步都在重新分配张量：

- `runner.py:193` `torch.tensor([tokens], ...).T` —— 每步新建输入张量（**地址不固定**）
- `runner.py:194` `torch.tensor([seq_lens], ...).T` —— 同上
- `runner.py:195-198` 用 Python 列表推导构造 `new_valid` 张量
- `runner.py:199` `valid_mask = torch.cat([valid_mask, new_valid], dim=1)` —— **每步让掩码张量长大一格**

CUDA Graph 捕获的硬性前提是"输入输出 buffer 地址固定、形状固定、无动态分配"。P7 的必备项（roadmap 自己写了"输入 buffer 地址固定：input_ids/positions/block_table/slot_mapping 全部预分配复用"）意味着这套循环要**整体重写**。这不是说现在写错了——静态批阶段这么写最直观——但要在 roadmap 里显式标注"P7 = 重写 decode 循环 + 引入 InputBatch"，不要以为只是"加个 capture"。

**建议**：在 P3 就把 `InputBatch`（持有预分配的 `input_ids` / `positions` / `slot_mapping` / `block_table` buffer，按 `max_num_seqs` 和 `max_num_batched_tokens` 一次分配）引入，后续所有阶段只往里写值。这样 P7 的工作量从"重写"降到"包一层 capture"。

### 3.4 🟠 "batch 输出 ≠ 串行输出"的根因没有真正定位（可能不是你以为的那个原因）

P2 笔记踩坑 6 的结论是：GPU 上 SDPA 依据"有无 attn_mask"选了不同 kernel backend，归约顺序不同 × bf16 → argmax 翻转，**定性为数值噪声非 bug**，并用"CPU fp32 同路径 11/11 一致"锁定逻辑正确性。

推理链本身是合理的，但**排查没有隔离变量**：代码里同时存在两个会造成跨路径差异的因子——

1. mask 有无 → SDPA kernel 选择（笔记已归因）；
2. `qwen2.py:115` 的 `if self.num_kv_groups > 1 and q.is_cuda` —— **CPU 走 `repeat_kv` 显式展开，GPU 走 `enable_gqa=True` 融合路径**，两者归约顺序同样不同。

笔记在 CPU 上验证了"mask 路径 vs `is_causal=True` 路径一致"，但**没有验证 GPU 上 `repeat_kv` vs `enable_gqa` 是否一致**。

> ⚠️ **复核更正：原报告此处逻辑不成立。** 原文把"GPU 上两个因子叠加才翻转"当作"CPU 一致、GPU 不一致"的最短解释，但 `enable_gqa` 由 `q.is_cuda` 决定（`qwen2.py:115`），**在 GPU 的串行路径与批处理路径上同时存在**，因此它无法解释"同一条 GPU 上两条路径之间的差异"。它能解释的是另一件事：CPU 的"11/11 一致"证据**不能外推到 GPU**。
>
> 同时被忽略的更可疑因子有两个：
> - **decode 侧的掩码差异**——串行走 `_decode`（`runner.py:71-76`，无 `attn_mask`），批处理走 `generate_batch`（`runner.py:202-207`，传 `[b,1,1,total]` 实数掩码），两条路径在 decode 阶段就已经选了不同 SDPA 后端；
> - **batch shape**——b=16 与 b=1 本身就会改变 decode kernel 的切分。
>
> 所以正确的因子集合是 {prefill 掩码有无, decode 掩码有无, batch size, gqa/repeat}，**原报告设计的四组对照不足以分离变量**。

**为什么值得花半天搞清楚**：因为 P4/P6/P7 的验收标准里写着"抢占-恢复后输出仍与 P1 一致"、"COW 触发时输出仍逐 token 一致（正确性红线）"。**如果"跨路径逐 token 一致"在数值上根本不成立，那这些红线就是不可达的验收标准**——P4 做到最后会发现自己在追一个追不上的目标。见 §4.2。

**建议动作**：给 attention 加一个显式开关（`attn_impl: "gqa" | "repeat"`，正好也是 §2.2 里 `AttentionBackend` 的一部分），然后在**固定 shape** 的前提下按 {prefill 掩码有无} × {decode 掩码有无} × {b=1 / b=16} 做对照，CPU 与 GPU 各跑一遍；`gqa` / `repeat` 单独做一组（只在 GPU 上有意义）以判断"CPU 证据能否外推"。半天的工作量，收益是**后面所有阶段的一致性判据都能站得住**。

### 3.5 🟠 每调用一次 `generate_batch` 就整块重新分配 KV 显存

`runner.py:143-152`：每次 `generate_batch` 都 `KVCache(...)` + `torch.zeros`，尺寸是 `[28, b, max_seq_len, 2, 128]`。

- b=16, max_seq_len=1280 → **293 MB × 2（K/V）= 587 MB**，每次调用都重新分配 + 清零；
- 加上 `NanoRunner.__init__`（`runner.py:41-49`）已经常驻一个 batch=1 的 cache，**峰值是两份**；
- `KV 预分配浪费 15%` 这个已落盘的数字，正是"容量从数据推导"的副产品，但**分配时机**（每批一次）没有被当作优化点讨论过。

**修法**：`max_num_seqs` 维度上一次性预分配，`reset()` 复用（P3 之后由 `BlockPool` 统一管）。同时 `SamplingParams` 里的 `max_num_seqs` 应当是显式配置项，而不是"传进来多少就是多少"（原文误写作 `SequencingParams`，已更正）。

### 3.6 🟠 采样器：批内共享一个 generator + Python 循环逐请求采样

`sampler.py:12-18` 每个 `Sampler` 只有一个 `torch.Generator`；`runner.py:176-183` 在 decode 循环里**逐个请求**调用 `sampler.sample(last_logits[i], ...)`。后果有三：

1. **顺序耦合**：同一批里请求 i 采样消耗掉的随机数会影响请求 i+1，于是"批量结果"和"逐条结果"在 temperature>0 时**逻辑上就不可能一致**（P2 的一致性判据只能建立在 greedy 上，正是因为这一点）；
2. **不可复现的 per-request 随机性**：无法做到"给请求 X 固定 seed → 结果确定"，而 vLLM 的 `SamplingParams.seed` 是标准能力；
3. **性能**：每步每请求一次 Python 调用 + 一次 `multinomial`，共 b 次 kernel launch。P7 的目标是"launch 开销占比降到 ~1%"，而这一步在 P4/P7 必须变成**一次 `multinomial` 处理 `[b, vocab]`**。

**修法**：采样接口改成 batch 化（`sample(logits[b,vocab], params) -> [b]`），generator 按 `(sampler, seed)` 允许每请求持有。注意 greedy 与 sampling 混合批次（一个 batch 里部分请求 greedy、部分采样）的处理——vLLM 的做法是分别处理再 scatter 回原位，这点值得提前想。

### 3.7 🟡 `Sampler.sample` 原地修改入参

`sampler.py:33-34` `logits[logits < kth] = -inf`、`sampler.py:42` `logits[sorted_idx[mask]] = -inf`。当前不会出 bug（temperature>0 时 `logits = logits.float() / temperature` 已经产生副本，temperature≤0 时提前返回），但这是**指针级别的隐性契约**：只要将来有人在 temperature 缩放之前复用这份 logits（例如 P7 的图捕获要复用静态输出 buffer、或做 speculative decoding 的 draft-verify），就会踩雷。建议改成非原地写法（`masked_fill` 到新张量）。

### 3.8 🟡 统计口径把两种浪费混成一个数

`runner.py:214` 的 `waste = 1.0 - total_tokens / (b * batch_cache.seq_len)` 同时包含了：
- prefill 阶段短请求 pad 到最长（"padding 浪费"）；
- decode 阶段已 done 请求随批空转（"空转浪费"）。

P2 笔记用"padding 41.2% / KV 15%"作为 P4 的动机证据——方向没错，但**这个字段名掩盖了它其实混了两种浪费**：prefill 阶段短请求 pad 到最长，以及 decode 阶段已 done 请求随批空转。
> 复核补充：这次实测里空转≈0。batch=16 的 16 条请求都跑满 64 个 decode 步，`total_tokens` ≈ 9216（prompt）+ 1019（decode）= 10235，`1 − 10235/17408 = 0.412` ✓；纯 pad 的理论值是 `(16×1024 − 9216)/(16×1024) = 43.75%`，被真实 decode token 稀释到 41.2%。**所以 41.2% 基本就是 padding，不是"已被污染"。** 拆指标仍建议做，但理由是"P4 的两项工作需要可分别归因"。

建议拆成 `prefill_pad_waste` 与 `decode_idle_waste` 两个指标分别落盘（这两个数正好分别对应 P4 的 chunked prefill 与 continuous batching 两项工作，验收时能一一对应）。

### 3.9 🟡 其它可维护性细节

| 位置 | 问题 |
| --- | --- |
| `golden/check_diff.py:70` | `limit = args.tol if fname.startswith("hidden") else args.tol` —— 三元两边完全相同，死代码（应该是想给 logits 单独一个阈值） |
| `bench/run_local.py:17,26`、`bench/batch_bench.py:21,28`、`golden/dump_nano.py:10,16` | `from pathlib import Path` 重复 import 两次（行号已按实际更正） |
| `runner.py:50` | `self.pad_id = 0` 硬编码，没从 tokenizer/config 取 |
| `tests/test_p2_correctness.py:75` | `eos = {151645, 151643}` 硬编码重复了 `config.eos_ids` 的能力 |
| `tests/test_p2_correctness.py:88` | `assertRaises(Exception)` 过宽，应精确到 `ValueError`（`kv_cache.py:52` 抛的就是 ValueError） |
| `bench/compare.py:77` | 图标题硬编码 `"P0 benchmark compare (HF baseline)"`，工具已被复用到 P1/P2 |
| `bench/results/` | 同时存在 `smoke_hf_cpu_0.5b.json` 与 `smoke_hf_cpu_0.5b_v2.json`，命名无法自解释哪份有效；建议每份结果写入 `git_sha` + 命令行，或加 `bench/results/README.md` 索引 |
| `nano-vllm-roadmap-v2.md` | 放在仓库根目录，应移入 `docs/` |

**复核新增的 4 条（原报告遗漏）**：

| 位置 | 问题 |
| --- | --- |
| `docs/stages/P2.md:48-57` | 小节标题写"修改代码（2 个）"，下面实际列了 3 个文件（`qwen2.py`、`runner.py`、`bench/run_local.py`）；"新增代码（1 个）"之外又有一个"补充新增"节加了 2 个文件。笔记是"数字唯一真相源"，计数必须更正 |
| `bench/run_local.py:103-113` | bench 用自己的 `argmax` 解码，**绕过了 `NanoRunner.sampler`**；HF backend 也走 HF 自己的 greedy。于是"bench 口径"与"引擎输出"不是同一条路：引擎的 EOS / `max_new_tokens` / stop 逻辑一旦变化，bench 数字不会变也不会报警 |
| `kv_cache.py:33,36` + `runner.py:67,75,163,208` | `KVCache.seq_len` 定义在缓存里，赋值却散落在 runner 的 4 处。这是"缓存状态由外部猜"的反模式，P3 换 `KVCacheManager` 时最容易漏改 |
| `sampler.py:34`（而非 33-34） | 原地改写语句在第 34 行，第 33 行是 `topk`；`sampler.py:42` 无误 |

---

## 4. 开发流程与路线图的合理性评估

### 4.1 路线总体判断：主干是对的，但 P3 的切分方式违背了自己的原则

**合理的部分**（不建议动）：

- **P0 → P1 → P2 → P3 → P4 递进**（基线 → eager → KV cache → 分页 → 连续批处理）完全符合"机制依赖顺序"，且每个阶段都有可量化的验收标准。
- **P7（CUDA Graph）不早于 P4**、**P9（torch.compile）后置为可选**——两个判断都很成熟，避免了在调度还没跑对时去碰图捕获。
- **裁剪清单**把分布式/多模态/LoRA 明确排除，理由写得很具体（"单卡无法验证收益"），这是防止学习项目失控的关键。
- **§3 的指标定义**（TTFT/TPOT p50/p99、显存、KV 浪费率、kernel launch 占比）方向非常正确。

**需要调整的部分**：

**（1）P3 应该拆成 P3a / P3b——把自己在 §6 红线里写的"两遍实现法"用在阶段内部。**

roadmap §6 反模式第 3 条："❌ 跳过 torch 朴素实现直接写 Triton —— 两遍实现法（先对、再快）"。原则很好，但**P3 一个阶段里同时塞进了**：BlockPool + block_table + slot_mapping + `reshape_and_cache` 写入核 + **自研 Triton paged attention kernel** + flash-attn prefill 接入 + block_size 扫描。

问题在于三点：

1. **红线自己的原则没被用在阶段内部**——"两遍实现法（先对、再快）"的意思正是块管理与 kernel 不该同期交付；
2. **两个独立风险源被绑在一期**：块池/`slot_mapping` 的正确性，与 Triton kernel 的正确性互不相干，但绑在一期后任一失败都会阻塞整个 P3；
3. **P3 的完成标准本身过载**：同一期要"浪费率 15%→<2%"+"32K 单条可跑通"+"kernel 相对误差 <1e-2"，而"32K 可跑通"实际还依赖 P4 的 chunked prefill（§3.2.1），把它算进 P3 只是延长了这条链路。

> ⚠️ **复核更正**：原报告这里用了"Triton kernel 到 P4 大概率要重设计，等于写两遍"作论据，**这条论据不成立**。P3 要写的是 **decode** paged attention，其天然入参（q、paged K/V、`block_table`、`seq_lens` / `context_lens`）本身就是 batch-agnostic 的：连续批处理与抢占只改变每步的 batch 组成，不改变该 kernel 的接口形态；真正对 varlen 敏感的是 prefill，而 roadmap 已把 prefill 交给 flash-attn（现已改为 SDPA）。所以"kernel 要重写"不能作为拆分理由——但上面的三条理由足以支撑拆分。

建议：

| 子阶段 | 内容 | 验收 |
| --- | --- | --- |
| **P3a** | `BlockPool`（free list）+ `block_table` + `slot_mapping` + `reshape_and_cache`（torch 实现）+ **SDPA 版 paged attention**（块 gather 后调 SDPA）。**只求正确与省显存，不求快** | KV 浪费率 15% → <2%；输出与 P2 逐 token 一致（同路径 CPU fp32）；32K 单条可跑通 |
| **P3b** | Triton paged attention kernel 替换 P3a 的 SDPA 实现，三实现（Triton / SDPA / flash-attn）对拍 + block_size ∈ {8,16,32,64} 扫描 | 相对误差 <1e-2；decode 延迟曲线；与 P3a 输出逐 token 一致 |

**额外收益**：P3a 做完之后，P4 可以先在 SDPA 实现上把调度跑对，再考虑 kernel 性能——**解耦"调度正确性"与"kernel 正确性"两个风险源**。

**（2）P3 之前先补一次接口层重构（1–2 天），把后面三次重构合并成一次。**

按 §2.2 的清单，引入 `AttentionMetadata` / `AttentionBackend` / `Sequence` / `InputBatch` / 公开 API。判断依据很简单：**这五件套是 P3 第一个提交的刚需，而不是 P5 以后的事**。晚做一次，就要在 P3、P4、P7 各付一次"穿 4 个文件"的改造费。

**（3）roadmap 缺一批"必须交付的工具"，而验收标准已经把它们当既成事实。**

| 缺口 | 证据 | 影响 |
| --- | --- | --- |
| `bench/profile.py`（launch 开销占比 / kernel 计数） | §3 定义了"**kernel launch 开销占比**"这个指标；P7 验收写"从 10–30% 降到 ~1%"、P8 写"单层 kernel launch 数下降 ≥30%"，但**全仓库没有任何代码产出这个数**（grep 只命中 roadmap 和 README 的文字） | P7/P8 的验收无法执行 |
| `bench/run_vllm.py` | §3 工具目录里列了它（"P3 起可选"），P5 验收又要求"HTTP 吞吐损耗 <10% 并有数字" | 与 vLLM 的横向对照是求职 demo 的核心卖点，但工具不存在 |
| `bench/regress.py` + `baseline.json` | 10 个阶段每个都声称 TPOT 改善；无任何自动回归门禁 | 某次重构把 P2 的 22.3ms 悄悄变回 30ms 也不会有人知道 |
| 每阶段一键复现（`make verify-p2`） | P0/P1/P2 的复现命令写在 md 里，靠人手敲；且文档现在已经是"数字唯一真相" | "每个数字都有出处、可复现"的原则目前依赖人工纪律 |
| detokenizer / `finish_reason` | roadmap 全文未作为交付项出现（只在参考地图里）。**复核补充**：`generate` 遇 EOS 直接 break（`runner.py:96-99`、`184-188`），既不记录停止原因也不区分 `max_new_tokens` 到达 | P5 无法实现；且 **P4 的 Scheduler 已经需要 `finish_reason` 来标记 finished**，建议提前到 P4 补 |

**（4）验收判据需要统一——现在的判据在不同阶段自相矛盾，而 P2 已经证明了它不可达。**

对照 roadmap 各阶段的措辞：

| 阶段 | 完成标准原文 | 问题 |
| --- | --- | --- |
| P1 | greedy 128 token 与 HF **完全一致** | 可达（同路径 CPU fp32） |
| P3 | kernel 数值对拍 **相对误差 <1e-2** | 可达 |
| P4 | 抢占-恢复后输出**仍与 P1 一致** | ⚠️ 跨路径逐 token 一致，P2 已证不可达 |
| P6 | COW 触发时输出**仍逐 token 一致（正确性红线）** | ⚠️ 同上 |
| P7 | 输出与 eager **逐 token 一致**（含边界 batch size） | ⚠️ batch 变化本身就破坏一致性 |

P2 笔记踩坑 6 的结论是"**跨数值路径不做逐 token 一致性承诺**"，但 roadmap 的 P4/P6/P7 验收标准仍在要求跨路径逐 token 一致。**这不是措辞问题——它会让 P4 之后的每一个阶段都卡在一个追不上的目标上，或者被迫用"这不算 bug"来结束阶段，从而让"正确性红线"这个概念贬值。**

建议替换为一套分层的判据定义（写进 roadmap §3）：

| 判据 | 适用 | 定义 |
| --- | --- | --- |
| **J1 位级一致** | 同设备、同 dtype、同路径 | golden 对拍 diff = 0（P1 已验证可达） |
| **J2 同路径逐 token 一致** | 路径重构类改动（P2 cache、P3 分页、P6 COW） | 同设备同 dtype 同 attention 后端下，greedy 输出逐 token 相等 |
| **J3 跨路径数值等价** | 跨后端 / 跨 batch shape / 跨精度（P3b、P4、P7、P8） | 逐步 logits 的 `max abs diff ≤ ε`（按 dtype 定），**且** 首个 token 分歧处 top-1 与 top-2 的 margin < ε（用 margin 解释翻转，而不是要求不翻转） |
| **J4 结构性一致** | 调度类改动（P4 抢占恢复） | 恢复后的**完成输出**与参考路径一致；若不一致，必须归因到 J3 允许的数值翻转且给出 margin 证据 |

这样"正确性红线"才有可执行的定义，而不是一句表态。

**（5）缺失的阶段/交付项**

- **序列化输出与停止条件**（detokenizer、`finish_reason`、`stop` 字符串、`ignore_eos`）——建议明确并入 P5，或在 P4 就补 `finish_reason`（Scheduler 需要它来标记 finished）。
- **显存预算管理**（等价于 vLLM 的 `gpu_memory_utilization`）：roadmap §1 手算了"剩余 8.5 GB 给 KV + 激活"，P4 提到"`max_num_batched_tokens` 按显存扫上界"，但**没有任何一个阶段负责把这个手算变成代码里的自动预算**。建议 P3a 交付 `KVCacheManager` 时一并实现（按可用显存反推 `num_blocks`），否则 P4/P7 的"扫上界"会变成人工试参数。
- **「32K 单条可跑通」这条验收标准本身需要重定**（原报告遗漏，见 §3.2.1）：(a) 真正的显存大头是 prefill 的全序列 logits（9.98 GB），不是掩码；(b) 验收数据 `shared_prefix` 本身是 32832 token，超过模型 `max_position_embeddings = 32768`；(c) 即使做完 (a)，32K 单次 prefill 仍需要 P4 的 chunked prefill 才能稳妥落地。**建议：要么把这条从 P3 移到 P4，并把 P3 改成"8K 单条可跑通 + 分块 prefill 语义可跑通"；要么在 P3 前补上"只算 last-token logits"（P0-03）。**
- **P3 prefill 的依赖风险**：P0 遗留问题明确写了"flash_attn 未装，P3 需补装或先用 SDPA"；`golden/env_report.txt` 显示 flashinfer 已装、flash_attn 未装。而 flash-attn 在 torch 2.14 + cu130 + sm_89 上属于"编译可能失败"的依赖。**修正后的建议（已在本机实测收窄）：P3a 的 prefill 就用现在的 SDPA + `is_causal=True`（右对齐 padding 下正确性已实例验证，见 §3.2），先不要引入任何新 attention 后端**；varlen 打包（`cu_seqlens`）推迟到 flash-attn 可用或自研 kernel 落地时。**`flex_attention` 建议从中期方案里彻底移除，只作调研笔记**——实测它**没有 `is_causal` 参数**（`flex_attention.py:1396`），因果性只能靠 BlockMask 表达，而 `create_block_mask()` 内部先调 `create_mask()` 物化完整 `[B,H,Q,KV]` 稠密张量再归约（`flex_attention.py:1213 → 1009`），对一次性 prefill 是负优化：s=8192 构造期 768 MB（是现行 `[b,1,s,s]` 的 6 倍），s=32768 本机直接 OOM，进程尝试分配 64 GiB（§3.2 更正小节）。
- **roadmap 文本需要同步改写**（否则旧结论下阶段会重新生效）：`roadmap:107` 的"prefill 先调 flash-attn 的 paged/varlen 接口"、`roadmap:124` 的"P3 需要补装 flash_attn（prefill varlen）"两处，与上面"先用 SDPA"的结论冲突。

**（6）工程化基建整体缺失（影响"求职 demo"这个定位）**

| 缺口 | 现状 | 建议 |
| --- | --- | --- |
| 依赖声明 | 只有 `constraints.txt`，内容是 `torch==2.14.0+cu130` / `torchvision==0.29.0+cu130`（**本地版本号 tag，换台机器/换索引未必装得上**）。transformers / triton / numpy / safetensors / xxhash / flashinfer 全部未声明。README 提到用 `uv`，但没有 `uv.lock` | `pyproject.toml`（含 optional extras：`flash-attn` / `vllm`）+ `uv.lock`。transformers 5.x 是有破坏性变更的大版本（P0 踩坑 3 已经踩过），必须 pin |
| 测试入口 | `tests/test_p2_correctness.py:20` 用 `MODEL = "models/Qwen2.5-0.5B-Instruct"` **硬编码相对路径**，而 `.gitignore` 忽略了 `/models/`；每个测试文件自己 `sys.path.insert` | **该文件的 4 项测试在干净 clone 上跑不起来。**（⚠️ 复核更正：`tests/test_sampler.py` 的 **7 项只依赖 torch 与 `nano_vllm.sample.sampler`，干净 clone 上本来就能跑**——原报告"所有测试都跑不起来"是错的。11 项里 **7 项可跑、4 项不可跑**。这条更正带来一个"免费"结论：**CPU-only CI 今天就能加，先只跑 sampler 即可，不必等免模型单测写完。**） |
| CI | 无 | 不必等免模型单测——先用 `tests/test_sampler.py` 起一个 CPU-only GitHub Actions（lint + 单测），成本极低，是求职信号；免模型单测就位后再把 `test_p2_correctness` 纳进来 |
| Lint / 类型 | 无 ruff / mypy 配置（代码本身用了 `from __future__ import annotations` + 完整类型标注，基础很好） | 加 ruff + pyright 配置即可，几乎零成本 |
| README | **仍是 P0 的环境搭建文档**：讲 WSL/驱动/显存/学习路径，提到 "myengine"，还引用了一个**已过期的绝对路径** `C:\Users\25184\WorkBuddy\2026-09-20-10-47-54\llm-engine-env`。没有架构图、没有进度表、没有实测数据、没说怎么跑 bench/test | 见下方 "README 重写" |
| 阶段留痕 | 无 git tag、无 GitHub Release、无 milestone | 给每个完成阶段打 tag（`v0.2-p2`）+ 用 Release 承载该阶段的 bench 表 —— 让"进度"在仓库层面可见 |
| LICENSE | 无 | 补一个（MIT 即可） |

---

## 5. 改进方向与落地路径

### 5.1 批次 A：P3 开工前的接口层重构（约 1–2 天，最高优先级）

目标：**把 P3/P4/P7 的三次穿透式改造合并成这一次**，并保证 bench 数字不回退。

| # | 动作 | 涉及文件 | 验收 |
| --- | --- | --- | --- |
| A1 | 引入 `AttentionMetadata`（`block_table` / `slot_mapping` / `seq_lens` / `num_cached_tokens` / `max_seq_len`），替换裸透传的 `is_prefill` + `cache_seq_len` + `attn_mask` | `models/qwen2.py`、`model_executor/runner.py` | 现有 11 项测试全绿；bench TPOT 不回退（p2_cache ≈22.3ms） |
| A2 | 引入 `AttentionBackend` 抽象 + registry；先落一个 SDPA backend（含 `attn_impl` 显式开关，分离 §3.4 的两个因子） | 新建 `attention/`（**这次是真的填充**）、`models/qwen2.py` | GPU 四组对照实验（mask×gqa / mask×repeat / causal×gqa / causal×repeat）出数据，写入 P2 笔记勘误 |
| A3 | 抽出 `Sequence`（`input_ids` / `output_ids` / `seq_len` / `status` / `finish_reason` / `sampling_params`）；`SamplingParams` 迁到 `nano_vllm/types.py` 并从 `__init__.py` 导出 | 新建 `core/sequence.py`、`types.py` | `from nano_vllm import SamplingParams` 可用 |
| A4 | `KVCache` → 拆成"存储"与"管理"：`worker/cache_engine.py`（张量池，纯存储）+ 预留 `core/kv_cache_manager.py` 空接口；**按 `max_num_seqs` 一次预分配 + `reset()` 复用**；并把 `seq_len` 从"外部手工赋值"（现在散在 `runner.py:67,75,163,208`）收进管理器 | `kv_cache.py`（删除）、`runner.py` | 重复调用 `generate_batch` 不再触发整块重分配（`torch.cuda.max_memory_allocated` 前后一致）；`runner` 中不再直接赋值 cache 的内部状态 |
| A5 | 公开 API：`NanoRunner` → 拆出 `LLM`（离线门面，`LLM.generate()`）+ `ModelRunner`（批执行）；`bench/run_local.py` 不再调 `_prefill/_decode` | `model_executor/runner.py`、`bench/run_local.py` | bench 只依赖公开接口；`NanoRunner` 的 `use_cache` 开关移除（eager 路径移入 `reference/` 或 tests） |
| A6 | 清理：删除 `engine/` 空包（`attention/`、`core/` 在本批次中直接填充，**不要删**——原报告写"删掉 3 个空包"与 A2/A3 的"新建"冲突，按此更正执行次序）、重复 import、`check_diff.py:70` 死代码、`pad_id` 从 config 取、测试里的 eos 硬编码、`compare.py` 图标题 | 多处 | ruff 零告警 |
| A7 | **免模型单测**：用 config dict 构造 2 层 tiny 模型（随机权重，不下载、不需要 GPU），覆盖 `KVCache` 边界 / 掩码形状与数值 / position_ids / config 解析 / 新增的 chunked-prefill 语义 | 新建 `tests/fixtures.py` + 若干 test | **干净 clone 无需任何模型即可跑通**；可进 CI |
| A8 | **删掉 `_build_prefill_mask`**（prefill 一律 `attn_mask=None` 走 `is_causal=True`），并加"禁止稠密 `[*,*,s,s]` 掩码"的断言 | `model_executor/runner.py` | 32K 单条 prefill 不再分配 2.15 GB 掩码；`long`/`shared_prefix` 两类数据纳入 smoke bench 跑通；复现脚本 `mask_fix_proof.py` 作为依据 |
| A9 | **prefill / decode 只对 `last_idx` 取 logits 再过 `lm_head`**（golden 的全量 logits 路径保留开关）——这是 32K 验收的真正前提，见 §3.2.1 | `models/qwen2.py:245`、`model_executor/runner.py:159-166` | 32K 单条 prefill 峰值 < 6 GB（logits 9.98 GB → 约 10 MB）；`long`(8192) 可跑通 |
| A10 | **修 `bench/dataset.py` 的越界样本**：`shared_prefix` 实际 32832 token > `max_position_embeddings` 32768 | `bench/dataset.py:24-26` | 数据 spec 与模型上限一致 |
| A11 | **加 chunked-prefill 语义单测**（用现在的连续缓存即可，不需要 P3 的分页）：分 2/3 次 `forward`，第二次传 `cache_seq_len=len(chunk1)`，与一次全量 prefill 的 last-token logits 对比 | 新建 `tests/test_chunked_prefill.py`；依据 `models/qwen2.py:102-108` | 1/2/3 段切分全绿；修好 §3.1 的前缀丢失 |

> A7 是这一批里"性价比最高、最容易被跳过"的一项。**4 项**测试（`test_p2_correctness.py`）依赖 2.9 GB 本地模型 + 相对路径，等于**在换机器的自己面前拿不出任何可验证的证据**；另外 7 项（`test_sampler.py`）本就不依赖模型，可以先接进 CI，不必等 A7 完成。

### 5.2 批次 B：P3 重排（P3a / P3b）

见 §4.1 的表格。补充两条：

- **P3a 完成后立即做 P4 的 Scheduler，P3b（Triton kernel）与 P4 可以并行推进**——因为 Scheduler 的正确性与 attention 后端的快慢正交，两者共用同一份 `AttentionMetadata` 接口。这是"先补接口"带来的直接红利。
- P3a 交付 `KVCacheManager` 时顺手做**显存预算自动推导**（可用显存 → `num_blocks`），让 P4 的"`max_num_batched_tokens` 按显存扫上界"从人工试参变成配置计算。

### 5.3 批次 C：工程化与可见性（可与其他批次并行，按 30 分钟粒度推进）

1. `pyproject.toml` + `uv.lock`（pin transformers / triton / numpy 等全部直接依赖）+ ruff/pytest 配置；
2. 把 `nano-vllm-roadmap-v2.md` 移入 `docs/`，README 指向它；
3. **README 重写**（对"求职 demo"定位而言这是 ROI 最高的一项）：

   ```
   # nano-vLLM：从零实现单卡 LLM 推理引擎
   ## 这是什么 / 不是什么（明确 16GB 单卡、学习与求职定位）
   ## 当前进度表
   | 阶段 | 机制 | 关键实测 | 相比上一阶段 |
   | P0 | golden 对拍 + bench 口径 | HF TPOT p50 26.81ms | 基线 |
   | P1 | eager 单流（无 cache） | TPOT 43.81ms | 全项目最差基线 |
   | P2 | KV cache + 静态批 | TPOT 22.27ms / batch16 392.3 tok/s | 2.00× / 9.29× |
   | P3 | PagedAttention | 进行中 | — |
   ## 架构图（模块分层）
   ## 快速开始（含"不装模型也能跑的 CPU 单测"）
   ## 每个数字怎么复现（给出命令与结果文件路径）
   ## 与 vLLM 的对照说明（机制对齐 V1，规模不追）
   ```
4. 阶段 tag + Release（Release 正文贴该阶段 bench 表）；
5. LICENSE；
6. `bench/profile.py`（launch 占比 / kernel 计数，补齐 §3 的指标缺口）+ `bench/regress.py` + `bench/baseline.json`；
7. `Makefile` 或 `scripts/` 包装：`make test` / `make bench-p2` / `make verify-p3`。

### 5.4 建议的目标目录结构

```
nano_vllm/
├── __init__.py                 # 导出 LLM / SamplingParams
├── types.py                    # SamplingParams / RequestOutput / FinishReason
├── config.py                   # ModelConfig / CacheConfig / SchedulerConfig
├── core/
│   ├── sequence.py             # Sequence 状态机 + block_table + 生命周期
│   ├── scheduler.py            # P4: schedule() -> SchedulerOutput
│   ├── kv_cache_manager.py     # 分配 / 追加 / 释放 + 显存预算 + 统计
│   ├── block_pool.py           # P3: free list / ref_cnt / COW(P6)
│   └── kv_cache_utils.py       # P6: 滚动 hash
├── worker/
│   ├── input_batch.py          # 预分配固定地址 buffer（P3 定接口，P7 直接用）
│   ├── model_runner.py         # execute_model(InputBatch) -> logits
│   └── cache_engine.py         # KV 张量池（纯存储）
├── attention/
│   ├── metadata.py             # AttentionMetadata
│   ├── registry.py             # 后端注册/选择
│   ├── backends/{sdpa.py, triton_paged.py, flash_attn.py}
│   └── ops/reshape_and_cache.py # P3a
├── models/
│   ├── qwen2.py                # 纯模型定义（backend 无关）
│   └── loader.py               # safetensors 加载 + 名称映射 + 预拼接(P8)
├── sample/
│   ├── sampler.py              # 批量化 + 每请求 generator
│   └── params.py
└── engine/
    ├── engine_core.py          # step: schedule -> execute -> update
    ├── processor.py            # 输出聚合 / detokenize / finish_reason
    └── llm.py                  # LLM 门面
```
> 命名为 `worker/` 而非 `model_executor/`：既然声明对齐 vLLM V1，就不要留 V0 的名字（V1 里 `vllm/v1/worker/gpu_model_runner.py` 才是模型执行角色）。

---

## 6. 一页速查清单

> 完整的分优先级清单（含定位、修法、验收）见 **`nano-vllm-报告复核与P3前问题清单.md`**；本节只是速览。

**必须马上做（P3 之前）**
- [ ] 加 chunked-prefill / 前缀复用的语义单测，修 `qwen2.py:102-108` 的前缀丢失隐患
- [ ] **prefill / decode 只算 `last_idx` 的 logits 再过 `lm_head`**——32K 单条 logits 9.98 GB，这才是"32K 可跑通"的第一障碍 → §3.2.1
- [ ] **删掉 `_build_prefill_mask`**——实测有效行逐位一致，掩码零贡献（32K 单条白吃 2.15 GB）→ §3.2
- [ ] 修 `bench/dataset.py:24` 的越界样本（32832 > 32768）
- [ ] 引入 `AttentionMetadata` + `AttentionBackend`，收敛跨层参数
- [ ] 抽出 `Sequence` + `KVCacheManager`（存储/管理分离）+ 公开 `LLM` / `SamplingParams`，bench 停用私有方法
- [ ] KV 预分配改一次性 + 复用（消除 560 MiB/批的重复分配），`seq_len` 收进管理器
- [ ] 加免模型 CPU 单测；**并先把 `tests/test_sampler.py`（7 项，无模型依赖）接进 CPU CI**

**应该做（P3–P4 期间）**
- [ ] 拆分 P3 为 P3a（块池 + SDPA-paged）/ P3b（Triton kernel）
- [ ] 隔离"batch≠serial"的根因——**因子有 4 个**（prefill 掩码 / decode 掩码 / batch shape / `enable_gqa` vs `repeat_kv`，`qwen2.py:115`），四组对照不够
- [ ] 统一验收判据（J1–J4），替换 roadmap 里跨路径逐 token 一致的表述
- [ ] 重定"32K 单条"验收的归属（P3 还是 P4）
- [ ] 补齐 `bench/profile.py`（launch 占比）与 `bench/run_vllm.py`
- [ ] 补 detokenizer / `finish_reason`（P4 的 Scheduler 已需要它）
- [ ] 显存预算自动化（可用显存 → num_blocks / max_num_batched_tokens）
- [ ] 同步改写 roadmap `:107` / `:124` 的 flash-attn 表述

**工程化（可并行）**
- [ ] `pyproject.toml` + `uv.lock`，pin 全部直接依赖
- [ ] README 重写（进度表 + 架构图 + 复现命令）
- [ ] CI（CPU-only lint + 单测）、ruff、阶段 tag/Release、LICENSE
- [ ] `bench/regress.py` + `baseline.json` 回归门禁

---

## 7. 需要你确认的取舍（会影响上面的取舍）

1. **这个仓库的读者是谁？** 如果是求职作品集，README 需要重写甚至双语化，`docs/stages/*.md` 的中文"踩坑记录"反而是加分项（证明真实调试能力），但 roadmap §2 的"❌ 不做分布式"这类裁剪清单要注意表达方式（面试官容易理解成"不会"，建议改写成"单卡范围内主动排除，原理已梳理"）。
2. **P3 的 prefill 后端选哪条路？**（已用本机实测收窄）`flex_attention` 已排除——它没有 `is_causal`，因果性必须靠 BlockMask，而 BlockMask 构造要物化 `[B,H,Q,KV]`（32K 实测 OOM）。剩下：**A. 保持 SDPA + `is_causal=True`**（今天就可用、零掩码，代价是 padding 行仍参与计算，属 P4 的账）；**B. 装 flash-attn 走真正 varlen 打包**（消除 padding，但有 torch 2.14+cu130+sm_89 编译风险）。建议先 A 把 P3a 跑通，再评估 B。这条不再是"需要你拍板"，而是"建议按 A→B 顺序推进"。
3. **要不要做 vLLM 横向对照？** roadmap §3 列了 `run_vllm.py`、P5 验收要求"HTTP 吞吐损耗 <10%"，这些都需要 `~/venvs/vllm` 环境可用。如果这条不打算走通，建议现在就从路线图里删掉，而不是留着一个永远不会被满足的验收标准。
4. **【复核新增】"32K 单条"算 P3 还是 P4 的验收？** 若算 P3，则 A9（只算 last-token logits）必须在本轮做完；若移到 P4，P3 的完成标准要改成"8K 单条可跑通 + 分块 prefill 语义可跑通"。
5. **【复核新增】是否接受重跑 P2 的 bench 数字？** A8 / A9 都会改变 GPU 上 prefill 的数值路径与耗时，`bench/results/p2_nano_*.json` 与 P2 笔记的 22.27 ms / 9.29× / `match_serial=false` 需要重跑确认，并留下一份"改动前后对照"。
