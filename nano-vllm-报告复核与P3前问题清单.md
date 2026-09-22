# nano-vllm 评估报告 · 复核意见与 P3 开工前问题清单

> 复核对象：`nano-vllm-架构与开发路线评估报告.md`（2026-09-22 版）
> 复核方式：把报告中每一条论断回溯到仓库 HEAD `e9327d1` 的代码行 / `nano-vllm-roadmap-v2.md` 原文 / `bench/results/*.json` / `docs/stages/*.md`，关键显存数字重新计算，可疑逻辑重新推导。
> 复核结论已于同日就地写回原报告（第 2 节各条的"更正"部分）。

---

## 0. 复核结论

**主干判断全部立得住**：抽象层缺失、P3 该拆、验收判据自相矛盾、工程化基线缺失——这四条经逐条核对后证据充分。

但有 **1 处事实错误、4 处逻辑或表述过强、2 处遗漏**，其中 1 处遗漏直接决定 P3 的验收标准能否成立。

| 类别 | 条数 | 最高严重度 |
| --- | --- | --- |
| 事实错误 | 1 | 高（"测试全不可跑"的结论是错的） |
| 逻辑不成立 / 表述过强 | 4 | 中 |
| 遗漏 | 2 | **高：P3 的"32K 单条可跑通"在不改 logits 时不可达** |
| 行号 / 计数偏差 | 3 | 低 |

---

## 1. 已核实无误的论断

| # | 报告中的论断 | 核对证据 |
| --- | --- | --- |
| 1 | `qwen2.py:105-108` prefill 无条件只用当前 chunk 的 K/V | ✅ 原文一致；调用点 `runner.py:66`、`runner.py:161` 均传 `cache_seq_len=0` |
| 2 | `runner.py:112-118` 构造 `[b,1,s,s]` 实数掩码（非 bool） | ✅ 行号精确；`dtype=self.dtype`（bf16） |
| 3 | 32K 单条掩码 2.1 GB | ✅ 精确值 2.15 GB（2048 MiB） |
| 4 | `attention/` `core/` `engine/` 三个 0 字节空壳包 | ✅ 三者的 `__init__.py` 均 0 字节；`nano_vllm/__init__.py` 亦为 0 |
| 5 | `qwen2.py:115` 由 `q.is_cuda` 决定 GQA 融合 vs `repeat_kv` | ✅ 行号与条件精确 |
| 6 | `SamplingParams` 定义在 `runner.py:15-20` | ✅ 精确 |
| 7 | `bench/run_local.py:74,81` 调用私有 `_prefill` / `_decode` | ✅ 精确 |
| 8 | KV 每批重分配 293 MB × 2 = 587 MB（b=16, max_seq_len=1280） | ✅ 280 MiB × 2 = 560 MiB = 587 MB（十进制）；`batch_bench.py:110-111` 推出 1280 属实 |
| 9 | `runner.py:214` 浪费率公式混口径、字段名却叫 `padding_waste_rate` | ✅ 精确 |
| 10 | P2 踩坑 6 的结论是"跨数值路径不做逐 token 一致性承诺" | ✅ `P2.md:83` 原句一致 |
| 11 | roadmap 的 P4 / P6 / P7 验收仍要求跨路径逐 token 一致 | ✅ `roadmap:117` / `130` / `137` 原文一致 |
| 12 | roadmap §3 定义了"kernel launch 开销占比"，但全仓库无代码产出 | ✅ `roadmap:67`；仅 `scripts/verify_env.py:178` 用过 profiler，与本指标无关 |
| 13 | `bench/run_vllm.py` 列在工具目录里但不存在 | ✅ 不存在（但 `scripts/test_vllm.py` 是已有的 vLLM 冒烟脚本，可作起点） |
| 14 | 文档数字与 `bench/results/*.json` 零漂移 | ✅ serial 42.24 / b8 279.2(6.61×) / b16 392.3(9.29×) / pad 0.412 / kv 0.15；TPOT 22.270 vs 44.664（2.005×）、吞吐 44.846 vs 20.245（2.215×）；HF 26.811 ms / 38.837 tok/s |
| 15 | P1 有 8 条踩坑、P2 有 6 条，报告引用的 4 个 P1 例子准确 | ✅ `P1.md:76-83`，共 8 条 |
| 16 | `golden/check_diff.py:70` 三元两边完全相同 | ✅ 原句 `limit = args.tol if fname.startswith("hidden") else args.tol` |
| 17 | `bench/results/` 同时存在 `smoke_hf_cpu_0.5b.json` 与 `_v2.json` | ✅ 两份都在（n=2 与 n=1） |
| 18 | 645 行核心代码 / 2704 行代码与文档 / 14 个提交 / main 单分支 / 无 tag | ✅ 645 = 57+62+259+223+44；2704 = py 2088 + md 616 |

---

## 2. 需要更正的论断

### 2.1 【事实错误 · 高】"干净 clone 上没有任何测试跑得起来"

**原文**：§4.1(6) 测试入口行与 §5.1 A7 注称测试硬编码模型路径 + `.gitignore` 忽略 `/models/`，因此"干净 clone 上所有测试都跑不起来"、"没有任何人能验证你的工作"。

**核对**：`tests/test_sampler.py` 共 7 项，只 import `torch` 与 `nano_vllm.sample.sampler`，**不依赖任何模型文件**，干净 clone 上直接可跑。跑不起来的只有 `tests/test_p2_correctness.py` 的 4 项。

**更正**：11 项测试里 **7 项可跑、4 项不可跑**。这条更正还带来一个"免费"结论：**CPU-only CI 今天就能加**（先只跑 sampler），不必等免模型单测写完——原报告把 CI 排在 A7 之后，是没有必要的串行。

### 2.2 【逻辑不成立 · 中】§3.4 对"batch ≠ serial"的根因解释

**原文**：CPU 上只有 1 号因子（mask 有无），"GPU 上两个因子叠加才翻转"，并把 `enable_gqa` vs `repeat_kv` 称为"最短解释"。

**问题**：`enable_gqa` 由 `q.is_cuda` 决定（`qwen2.py:115`），**在 GPU 的串行路径与批处理路径上同时存在**，所以它无法解释"同一条 GPU 上两条路径之间的差异"。它能解释的是另一件事——CPU 的"11/11 一致"证据不能外推到 GPU。

同时被忽略的更可疑因子有两个：

- **decode 侧的掩码差异**：串行走 `_decode`（`runner.py:71-76`，无 `attn_mask`），批处理走 `generate_batch`（`runner.py:202-207`，传 `[b,1,1,total]` 实数掩码）——两条路径在 decode 阶段就已经选了不同 SDPA 后端；
- **batch shape**：b=16 与 b=1 本身会改变 decode kernel 的切分。

**更正**：假说集合应为 {prefill 掩码有无, decode 掩码有无, batch size, gqa/repeat_kv}。原报告设计的"四组对照"不足以分离变量，需要 2×2×2 固定 shape 的对照（见清单 P1-01）。

### 2.3 【表述过强 · 中】§3.2 "顺带消掉一个数值分叉因子…只剩 enable_gqa 一个"

删掉 prefill 掩码后，**decode 侧仍然必须传掩码**（`generate_batch` 里未写入的 cache 位置是 0 填充，不掩蔽会稀释输出），而串行路径的 decode 没有掩码——分叉因子并没有降到 1 个。

**更正**：删 prefill 掩码的主要收益是"2.15 GB → 0 + 少一次 `tril` / `masked_fill_` 全量访存"，"少一个数值因子"是附带的、不是"只剩一个"。补丁注释已按此改写（`prefill-mask-fix.patch` 第 33-35 行）。

### 2.4 【表述过强 · 高】§3.2 "s² mask 会吃光显存 → 32K 撞墙"

掩码 2.15 GB 属实，但**它不是 32K 单条的第一障碍**。详见 §3.1：prefill 把整段 s 过 `lm_head`，32K 单条的 logits 单独就是 9.98 GB。**即使掩码归零，32K 单条依然跑不通。** 原报告把"32K 撞墙"完全归因于掩码，是定位错误。

### 2.5 【表述不当 · 中】§0 / §1 "645 行核心代码把模型、缓存、调度、批处理、采样全压在一个 NanoRunner 里"

模型在 `models/qwen2.py`、缓存在 `kv_cache.py`、采样在 `sampler.py`，**已经分文件了**。准确的表述是两句：

- **编排层过载**：唯一的编排层 `NanoRunner`（223 行）同时承担 prefill/decode 主循环、掩码构造、静态批、指标统计、两条数值路径（eager / KV）、以及对外 API；
- **缺接口契约**：模型 / 缓存 / 采样虽已分文件，但彼此靠裸参数穿透，没有可替换的抽象。

前者是"该拆的类没拆"，后者是"该有的抽象没有"——原报告把两件事揉成了一句，导致 §2.1（目录与职责）和 §2.2（抽象缺失）的边界模糊。

### 2.6 【内部不一致 · 低】

- §0 评分表说缺"`Sequence` / `Scheduler` / `AttentionMetadata` / `AttentionBackend` / **KV 池管理**"五件套，而 §2.2 表列的第五项是"**输出处理**（detokenizer / `finish_reason`）"，`KVCacheManager` 只出现在 §2.1 与 §5.1。两处口径需统一。
- §5.1 批次 A 内部次序冲突：A2/A3 要求"新建 `attention/`、`core/`（这次是真的填充）"，A6 又要求"删除 3 个空包"。按编号顺序执行会**删掉刚填充好的包**（见清单 P1-04）。

### 2.7 【表述过重 · 低】§3.8 "41.2% 里混了空转"

口径确实混了（这是真问题），但**这次实测里空转≈0**：batch=16 的 16 条请求都跑满 64 个 decode 步，`total_tokens` ≈ 9216（prompt）+ 1019（decode）= 10235，`1 − 10235 / 17408 = 0.412` ✓；纯 pad 的理论值是 `(16×1024 − 9216) / (16×1024) = 43.75%`，被真实 decode token 稀释到了 41.2%。所以这个数**基本就是 padding**。

**更正**：拆指标仍建议做，但理由是"P4 的 chunked prefill 与 continuous batching 需要可分别归因"，不是"这个数现在已经被污染"。

### 2.8 【论证偏弱 · 低】§4.1(1) "Triton kernel 到 P4 大概率要重设计"

P3 要写的是 **decode** paged attention，其天然入参（q、paged K/V、`block_table`、`seq_lens` / `context_lens`）本身就是 batch-agnostic 的：连续批处理与抢占只改变每步的 batch 组成，不改变该 kernel 的接口形态。真正对 varlen 敏感的是 **prefill**，而 roadmap 已把 prefill 交给 flash-attn。所以"kernel 要重写"这条论据不足。

**更正**：拆 P3a / P3b 的理由应换成三条更硬的——

1. roadmap §6 红线自己的"两遍实现法（先对再快）"；
2. 块管理（P3a）与 kernel 正确性（P3b）是两个独立风险源，绑在一期会让任一失败阻塞整期；
3. P3 的完成标准同时要求"浪费率 < 2% + 32K 可跑通 + kernel 相对误差 < 1e-2"，其中"32K 可跑通"实际还依赖 P4 的 chunked prefill（见 3.1 / 3.2）。把 kernel 绑在这一期只会连带阻塞。

### 2.9 【行号偏差 · 低】

| 报告写的 | 实际位置 |
| --- | --- |
| `bench/batch_bench.py:20,28` | 21、28 |
| `golden/dump_nano.py:13,18` | 10、16 |
| `sampler.py:33-34`（原地改写） | 原地语句在第 34 行（33 行是 `topk`） |

（`bench/run_local.py:17,26` 无误。）

### 2.10 【不支持 · 低】§3.2 "bf16 的 `-inf` 掩码加法本身也损失精度"

这句不成立：`-inf` 在 fp16 / bf16 中都能精确表示，加法掩码路径 `score + (-inf) = -inf` 是 IEEE 精确行为，不会"污染接近 0 的 logit"。真正的问题只有两条——(a) 传实数掩码会让 SDPA 不走 flash 因果 kernel，改变数值路径；(b) 掩码必须被完整物化，吃显存。已从报告中删除该句。

---

## 3. 遗漏项

### 3.1 🔴 P3 验收的真正障碍是 logits，不是掩码

`models/qwen2.py:245` 对整段 `[b, s, hidden]` 过 `lm_head`，产出 `[b, s, vocab]`；而 `runner.py:165-166` 只取了 `last_idx` 那一行。

| s | logits `[1,s,151936]` bf16 | mask `[1,1,s,s]` bf16 | KV（1 序列） |
| --- | --- | --- | --- |
| 1024 | 0.31 GB | 2.0 MiB | 0.03 GB |
| 8192 | 2.49 GB | 128 MiB | 0.23 GB |
| 32768 | **9.96 GB** | 2.15 GB | 0.94 GB |
| 32832（`shared_prefix` 实际长度） | **9.98 GB** | 2.16 GB | 0.94 GB |

**旁证**：P2 的 `batch=16, max_prompt=1024` 峰值记录 8.18 GB ≈ 3.1（权重）+ 4.98（logits）——**这 8.18 GB 里约 6 成是 logits**。roadmap §1 自己也写了"logits `[num_tokens, 151936]` 是大头"，但**没有任何阶段负责把它修掉**。

**结论**：

- `long`(8192)：掩码 128 MiB 不构成威胁，logits 2.49 GB 可跑，但仍应统一优化；
- **32K 单条：不改 logits 就一定 OOM**（3.1 + 9.98 = 13.1 GB > 13 GB 预算）。
- 因此 P3 完成标准里的"长上下文（32K 单条）可跑通"必须二选一：改成"32K **分块** prefill 可跑通"（等于移入 P4），或现在就做"prefill 只算 last-token logits"。

### 3.2 🟠 "32K 单条"的验收数据本身越界

`bench/dataset.py:24` 的 `shared_prefix` 是 `prompt_len = 32768 + 64 = 32832`，已超过 Qwen2.5-1.5B 的 `max_position_embeddings = 32768`。用它作为"32K 单条可跑通"的验收样本，等于拿越界样本验收；同时 RoPE 位置超出训练范围，输出质量不可解释。

### 3.3 🟠 golden 数据不在仓库里，正确性证据链无法在干净 clone 上复现

`.gitignore` 忽略 `*.npy`，仓库里只有 `golden/*/meta.json`，而 `check_diff.py` 需要两份 golden 目录才能对拍。于是"每个数字都有出处、可复现"目前**只能在有 2.9 GB 权重 + GPU 的机器上重现**；新人 clone 后拿不到任何对拍证据。建议：文档化再生步骤，并提交一份 0.5B CPU fp32 的小体量 golden（几十 MB 量级）供 CI 自检。

### 3.4 🟡 把 chunked prefill 的 bug 直接绑到"P4 抢占-恢复会挂"是条件性的

抢占-恢复走 recompute 时，vLLM 的做法是**整段重算 prefill**（`cache_seq_len = 0`），此时旧代码不丢前缀。必然触发该 bug 的是 **(a) P4 的 chunked prefill** 和 **(b) P6 的命中块复用**。

**更正**：表述收窄，避免把优先级建立在错误因果上（该 bug 仍必须在 P3 前修）。

### 3.5 🟡 `P2.md` 的文件变更计数自相矛盾

`docs/stages/P2.md:48` 写"修改代码（2 个）"，下面实际列了 3 个（`qwen2.py`、`runner.py`、`bench/run_local.py`）；"新增代码（1 个）"之外又有一个"补充新增（验证补齐轮）"节加了 2 个文件。笔记是"数字唯一真相源"，计数应更正。

### 3.6 🟡 `bench/run_local.py` 用 `argmax` 自己解码，绕过了 `Sampler`

`bench/run_local.py:103-113` 直接 `last.argmax()`，没走 `NanoRunner.sampler`；HF backend 也走 HF 自己的 greedy。于是"bench 口径"与"引擎输出"不是同一条路：引擎的 EOS / `max_new_tokens` / stop 逻辑一旦改变，bench 数字不受影响也不会报警。

### 3.7 🟡 `KVCache.seq_len` 由调用方手工维护

`kv_cache.py:33,36` 定义 `seq_len`，但赋值散落在 `runner.py:67,75,163,208`。这是"缓存的状态由外部猜"的反模式；P3 换成 `KVCacheManager` 时最容易漏改的就是这类隐式契约，建议与 KV 拆分（P0-09）一起做。

### 3.8 🟡 roadmap 的 P3 条目与 P0 遗留需要同步改写

`roadmap:107` 仍写"prefill 先调 flash-attn 的 paged/varlen 接口"，`roadmap:124` 的 P0 遗留仍写"P3 需要补装 flash_attn（prefill varlen）"。这两处与"先用 SDPA + `is_causal=True`"的结论冲突；不同步修改，旧结论下阶段会重新生效。

### 3.9 🟡 补丁自身的两处小瑕疵（已修）

- 注释原先只说了 prefill 侧因子，已补充"decode 侧分叉因子仍然存在"的提示；
- 保留了上游 `runner.py` 中"方法之间无空行"的既有风格问题（`return out` 直接跟 `def _build_decode_mask`），ruff 会报 E301，A6 统一清理时不要漏。

---

## 4. P3 开工前问题清单

### 4.1 P0 · 阻塞 P3 验收或静默失败（必须先做）

| ID | 类别 | 问题 | 定位 | 为什么必须在 P3 前 | 修法 | 验收 |
| --- | --- | --- | --- | --- | --- | --- |
| **P0-01** | 正确性 | `is_prefill=True` 时无条件只用当前 chunk 的 K/V，忽略 `cache_seq_len > 0` 的历史 → prefill 丢前缀 | `nano_vllm/models/qwen2.py:102-108`；调用点 `runner.py:161` | P4 chunked prefill 与 P6 命中块复用都会构造"`is_prefill=True` 且 `cache_seq_len>0`"，失败形态是**静默输出改变** | 用 metadata 表达"本次是 chunk"；attention 一律按 `[历史 KV] ++ [当前 chunk KV]` 拼 K/V | 分 2 / 3 次 `forward` 与一次全量 prefill 的 last-token logits 位级一致 |
| **P0-02** | 正确性 | 上述语义没有任何断言或测试守着 | `tests/`（现有 4 + 7 项均不覆盖） | 不固化 = 修了也会回退 | 新增 `tests/test_chunked_prefill.py` | 1 / 2 / 3 段切分全绿 |
| **P0-03** | 显存 | prefill 对整段 s 过 `lm_head`：32K 单条 logits 9.98 GB，叠加 3.1 GB 权重已超 13 GB 预算 | `models/qwen2.py:245`；`runner.py:159-166` 只取了 `last_idx` | P3 完成标准"32K 单条可跑通"在不改这里时**不可达**（掩码只是第二障碍） | 先 gather `last_idx` 再过 `lm_head`；golden 的全量 logits 路径保留开关 | 32K 单条 prefill 峰值 < 6 GB；`long`(8192) 可跑通 |
| **P0-04** | 显存 | `_build_prefill_mask` 构造 `[b,1,s,s]` 实数掩码（32K → 2.15 GB，叠加 bool 中间量峰值约 4 GB），而该掩码对有效行贡献为零 | `runner.py:112-118`、`158-162` | 同 P0-03；且删掉后少一次 `tril` / `masked_fill_` 全量访存 | 应用 `prefill-mask-fix.patch`（prefill 一律 `attn_mask=None`） | 应用后 `tests/test_p2_correctness.py` 4 项仍绿（CPU fp32）；`mask_fix_proof.py` 为实测依据 |
| **P0-05** | 数据 | `shared_prefix` 实际 prompt 长度 32832 > `max_position_embeddings` 32768 | `bench/dataset.py:24-26` | 验收样本越界，结论不可解释 | 前缀改 32704，或显式标注为越界样本并单独讨论 | spec 与模型上限一致 |
| **P0-06** | 接口 | `is_prefill` / `cache_seq_len` / `attn_mask` 三个裸参数穿透 4 层 | `models/qwen2.py:90-92,157-159,191-193,234-236` → `runner.py` → `bench/*` → `tests/*` | P3 起每次签名变更都要穿一整套 | 引入 `AttentionMetadata`（`block_table` / `slot_mapping` / `seq_lens` / `num_cached_tokens` / `max_seq_len`） | 调用点只剩一个 metadata 入参 |
| **P0-07** | 接口 | attention 后端选择硬编码在模型里（`q.is_cuda` 决定 GQA 融合 vs `repeat_kv`） | `models/qwen2.py:115-126` | P3 要求 Triton / SDPA / flash-attn 可切换；P8 要保留对照开关 | `AttentionBackend` 接口 + registry + 显式 `attn_impl` 开关 | 同一进程内切换后端不改模型文件 |
| **P0-08** | 接口 | 无公开 API：`SamplingParams` 住在内部执行器、`__init__.py` 0 字节、bench 调私有方法 | `runner.py:15-20`、`nano_vllm/__init__.py`、`bench/run_local.py:74,81` | 压测框架是最紧耦合点，P3 重构必然打破它 | `types.py` + `LLM` 门面 + 公开批接口；`__init__` 导出 | bench 中不再出现 `_` 前缀调用 |
| **P0-09** | 接口 | KV 每批整块重分配（560 MiB / 批）；`seq_len` 由调用方手工赋值 | `runner.py:41-49,143-152`、`kv_cache.py:33-36` | P3 要换成 BlockPool，越晚改调用点越多 | 按 `max_num_seqs` 一次性预分配 + `reset()` 复用；状态收进管理器 | 连续两次 `generate_batch` 不再新增分配 |

### 4.2 P1 · P3 同期（1–2 天，与 P0 同批次完成最省）

| ID | 类别 | 问题 | 定位 | 为什么必须做 | 修法 | 验收 |
| --- | --- | --- | --- | --- | --- | --- |
| **P1-01** | 诊断 | "batch ≠ serial" 的根因未隔离，且"GPU 两因子叠加"的解释不成立 | `models/qwen2.py:115`、`runner.py:71-76` vs `202-207`、`P2.md:83` | P4 / P6 / P7 的一致性红线建立在这个结论上 | 固定 shape，按 {prefill 掩码} × {decode 掩码} × {b=1 / b=16} 做对照，GPU / CPU 各一遍 | 出数据 + 写回 P2 笔记勘误 |
| **P1-02** | 防护 | 无断言阻止重新引入稠密实数掩码 | `runner.py:112`（删除处） | 防回归 | 入口断言 `attn_mask is None or attn_mask.dtype == bool`，禁止 `ndim==4 且 shape[-2]==shape[-1]` 的实数掩码 | 单测覆盖 |
| **P1-03** | 接口 | decode 循环每步新建张量（`runner.py:193-194`）、`valid_mask` 逐步 `cat`（`:199`）、每步重建 decode 掩码（`:202`）——地址与形状都不固定 | `runner.py:193-202` | P7 CUDA Graph 的硬前提；P3 定接口成本最低 | 引入 `InputBatch`：预分配 `input_ids` / `positions` / `slot_mapping` / `block_table` buffer | decode 循环内零新建张量 |
| **P1-04** | 接口 | 空包删除与新建的次序冲突（原报告 A2/A3 先建、A6 后删会误删） | `nano_vllm/{attention,core,engine}/__init__.py` | 顺序错就直接破坏工作 | 次序固定为：先删本轮不用的 `engine/`；`attention/`、`core/` 直接填充不删 | 目录状态明确、无残留空包 |
| **P1-05** | 显存 | 显存预算全靠手工（roadmap §1 手算 8.5 GB） | `roadmap:34`；P4 要求"按显存扫上界" | 否则 P4 的扫上界退化为人工试参 | `KVCacheManager` 由可用显存反推 `num_blocks` | 预算配置化 |
| **P1-06** | 可验证 | 测试硬编码 `MODEL = "models/Qwen2.5-0.5B-Instruct"` 且该目录被 gitignore → **4 项**测试在干净 clone 不可跑（**7 项 sampler 测试可跑**） | `tests/test_p2_correctness.py:20` | 现在就能上 CPU CI | tiny 2 层随机权重 fixture + `--model` 参数化 | `python tests/*.py` 全绿且不需下载模型 |
| **P1-07** | 工程化 | 无 `pyproject.toml` / lock；`constraints.txt` 只有 torch / torchvision（带 `+cu130` 本地 tag）；transformers 5.x 的破坏性变更未 pin | `constraints.txt`、仓库根 | 换机器不可复现 | `pyproject.toml` + `uv.lock` + optional extras | `uv sync` 可复现 |
| **P1-08** | 口径 | `padding_waste_rate` 把 pad 与空转混成一个数 | `runner.py:214-222` | P4 的 chunked prefill 与 continuous batching 需要可分别归因 | 拆成 `prefill_pad_waste` / `decode_idle_waste` | 两个数分别落盘 |
| **P1-09** | 文档 | `P2.md` 变更计数错误（"修改代码（2 个）"实列 3 个） | `docs/stages/P2.md:48-57` | 笔记是数字唯一真相源 | 更正为 3 个 | — |
| **P1-10** | 判据 | 验收判据自相矛盾：P4 / P6 / P7 要求跨路径逐 token 一致，而 P2 已证不可达 | `roadmap:117,130,137`、`P2.md:83` | 不修则 P4 起每期都卡在一个追不上的目标上 | 换 J1–J4 分层判据（位级 / 同路径逐 token / 跨路径数值等价含 margin 归因 / 结构性一致），写进 roadmap §3 | roadmap §3 有可执行判据 |

### 4.3 P2 · 可并行或延后（不阻塞 P3）

| ID | 类别 | 问题 | 定位 | 说明 |
| --- | --- | --- | --- | --- |
| **P2-01** | 采样 | 单个 `Generator` + 逐请求 Python 采样（b 次 kernel launch）；无法按请求固定 seed | `sampler.py:12-18`、`runner.py:176-183` | P4 / P7 必须 batch 化（P7 目标 launch 占比 ~1%）；需提前想 greedy 与 sampling 混批的 scatter |
| **P2-02** | 采样 | `Sampler.sample` 原地修改入参 | `sampler.py:34,42` | 当前不出 bug，但 P7 静态 buffer 复用时会踩；改非原地 |
| **P2-03** | 工具 | 缺 `bench/profile.py`（launch 占比）、`bench/run_vllm.py`、`bench/regress.py` + `baseline.json` | `roadmap:67,74`；可用 `scripts/test_vllm.py` 起步 | P7 / P8 的验收直接引用这些数字 |
| **P2-04** | 可验证 | golden `*.npy` 被 ignore，仓库只留 meta → 对拍链无法在干净 clone 复现 | `.gitignore`、`golden/*/meta.json` | 文档化再生步骤 + 提交 0.5B CPU 小 golden 供 CI |
| **P2-05** | 工程化 | README 仍是 P0 环境文档（含过期绝对路径 `2026-09-20-10-47-54` 与 `myengine`） | `README.md:41,53` | 对"求职作品集"定位而言 ROI 最高 |
| **P2-06** | 工程化 | 无 ruff / CI / tag / Release / LICENSE | 仓库根 | 求职信号 + 回归门禁；CI 可先只跑 sampler（见 P1-06） |
| **P2-07** | 清理 | `check_diff.py:70` 死代码；重复 import（`run_local.py:17,26`、`batch_bench.py:21,28`、`dump_nano.py:10,16`）；`pad_id` 硬编码 `runner.py:50`；eos 硬编码 `tests/test_p2_correctness.py:75`；`assertRaises(Exception)` `:88`；图标题 `bench/compare.py:77`；`bench/results/` 无索引；roadmap 在根目录 | 见左 | 一次做完，做到 ruff 零告警 |
| **P2-08** | 交付项 | 缺 detokenizer / `finish_reason`：`generate` 遇 EOS 直接 break，不记录停止原因 | `runner.py:96-99,184-188`；`roadmap:121` | P5 无法实现；P4 的 Scheduler 已需要它标记 finished，建议 P4 就补 |
| **P2-09** | 文档 | roadmap P3"必须实现"仍要求 flash-attn prefill；P0 遗留也仍写"需补装 flash_attn" | `roadmap:107,124` | 与"先用 SDPA + `is_causal`"的结论冲突，需同步改写 |
| **P2-10** | 工具 | bench 用 `argmax` 自己解码，绕过 `Sampler` | `bench/run_local.py:103-113` | bench 口径与引擎输出不同一条路；显式声明或收敛到引擎解码 |

---

## 5. 建议执行顺序（含依赖）

| 阶段 | 内容 | 前置 | 完成后必须做的事 |
| --- | --- | --- | --- |
| **0. 半天，无需 GPU** | 只删 `engine/` 空包（P1-04）；修 `P2.md` 计数（P1-09）；加 sampler-only CI（P1-06 前半）；`pyproject` 骨架（P1-07） | — | CI 变绿 |
| **1. 一天，纯正确性与显存** | P0-04 掩码 → P0-03 logits → P0-05 数据 → P0-01 / P0-02 chunked prefill 语义与单测 | — | **重跑 `bench/run_local.py` 与 `bench/batch_bench.py`**：P0-03 / P0-04 会改变 GPU 上 SDPA 的 kernel 选择，P2 记录的 22.27 ms / 9.29× / `match_serial=false` 需要重新确认（TPOT 不应回退超过噪声） |
| **2. 1–2 天，接口层一次到位** | P0-06 → P0-07 → P0-08 → P0-09 → P1-03 → P1-02 | 阶段 1 | 每步跑 `tests/` + `batch_bench --batches 1 8 16`；确认 TPOT ≈ 22.3 ms 不回退 |
| **3. 并行，半天** | P1-01 对照实验（GPU 与 CPU）→ 写回 `P2.md` 勘误；P1-10 把 J1–J4 写进 roadmap §3；P1-05 预算推导 | 阶段 2（需要 `attn_impl` 开关） | 一致性判据从此可执行 |
| **4. P3 开工** | P3a（块池 + SDPA-paged，只求正确与省显存）/ P3b（Triton） | 阶段 1–3 | **P3 的"32K 单条"验收需重新定义**（见 P0-03 / P0-05） |

---

## 6. 与原报告 §5.1 / §6 清单的差异

| 变化 | 内容 |
| --- | --- |
| **新增为 P0** | P0-03（logits，原报告完全遗漏）、P0-05（32K 数据越界，遗漏）、P0-01 / P0-02（原报告只在速查清单提了"加语义单测"，未给定位与验收） |
| **新增为 P1** | P1-01 重新设计对照实验（原"四组"不足以分离变量）、P1-02 掩码防护断言、P1-04 次序冲突、P1-05 预算推导、P1-08 口径拆分、P1-09 笔记更正 |
| **优先级下调** | 原 §4.1(3) 的 `bench/profile.py` / `run_vllm.py` / `regress.py` → P2-03（P7 / P8 才引用）；原 §4.1(6) 的 README / CI / tag → P2-05 / P2-06（可并行） |
| **优先级上调** | 原"应该做（P3–P4 期间）"的统一验收判据 → P1-10（P3 自己的"逐 token 一致"验收也要用同一套判据） |
| **修正** | 原"立刻删掉 3 个空包" → 只删 `engine/`，`attention/` 与 `core/` 直接填充（P1-04） |
| **修正** | 原建议保留 `flex_attention` 作为"可选对照后端" → 建议从中期方案里彻底移除，仅作调研笔记（实测：无 `is_causal`，构造 BlockMask 要物化 `[B,H,Q,KV]`，32K 直接 OOM） |
| **修正** | 原"测试完全不可跑 → CI 排在 A7 之后" → sampler 7 项本就可跑，CI 今天可加（P1-06） |

---

## 7. 仍需你确认的取舍（已收窄）

1. **仓库读者定位**：是否作为求职作品集？（决定 README 是否重写 / 双语，以及裁剪清单的表达方式）
2. **P3 的 prefill 后端**：已收窄为 **A. SDPA + `is_causal=True`（先做，零新依赖）→ B. flash-attn varlen（消除 padding，但有 sm_89 + cu130 编译风险）**。
3. **是否真做 vLLM 横向对照**：P5 验收要求"HTTP 吞吐损耗 < 10% 且有数字"，`bench/run_vllm.py` 不存在。要现在补进计划，还是把该验收降级。
4. **【新增】"32K 单条"算 P3 还是 P4 的验收**：若算 P3，则 P0-03（只算 last-token logits）必须在本轮做完；若移到 P4，P3 的完成标准要改成"8K 单条可跑通 + 分块 prefill 可跑通"。
5. **【新增】是否接受重跑 P2 的 bench 数字**：P0-03 / P0-04 会改变 GPU 上 prefill 的数值路径与耗时，`bench/results/p2_nano_*.json` 与 P2 笔记的数字会失效，需要重跑并留下一份"改动前后对照"。
