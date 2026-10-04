# 整改 · P8 前置四项（2026-10-03）

> 背景：进入 P8（算子融合 + 权重预拼接）前，把「阻塞 P8 验收的前置」和一处历史命名冲突清掉。
> 来源：`docs/stages/P7.md`「未完成与残留」/「后续」、`nano-vllm-P6前优化审计.md` P2-03、
> `nano-vllm-架构与开发路线评估报告.md` §4.1(3)、以及 P7.md 补记的「顺带发现」。
> 明确不做：C 类工程化老账（README 重写 / CI / LICENSE / golden 可复现），不在本轮。

---

## 0. 先清命名：P8 这个编号此前指向两件事

| 口径 | P8 指什么 | 出处 |
| --- | --- | --- |
| 旧的文件名与 commit | **抢占恢复对齐** | `bench/p8_preempt_bench.py`、`bench/results/p8_preempt_align.json`、commit `1c24a8d` |
| roadmap / 阶段笔记 | **融合算子 + 权重预拼接** | `nano-vllm-roadmap-v2.md:144`、`docs/stages/P7.md:176` |

抢占恢复对齐在 `docs/notes/preempt-recompute-align.md:1-5` 自述是**「整改」**，
其待办来源是 pre-P7 清单（`pre-p7-fixes.md`）——它不是阶段。若不清掉，`docs/stages/P8.md`
一建立，代码/bench/commit 里的 "P8" 就会同时指向两件事。

处置：`git mv` 改名（保留历史），引用同步。

| 原 | 现 |
| --- | --- |
| `bench/p8_preempt_bench.py` | `bench/preempt_align_bench.py` |
| `bench/results/p8_preempt_align.json` | `bench/results/preempt_align.json` |
| 脚本内 `--tag` 默认值 `p8_preempt_align` | `preempt_align` |

引用点共 5 处，已全部更新（`preempt-recompute-align.md:65,83` + 脚本内 3 处）。

---

## 1. attention 第二实现：flash-attn → SDPA

### 1.1 为什么放弃真装 flash-attn

原写法（roadmap:145）要求「attention 保留三实现（Triton 自研 / **flash-attn** / SDPA）做对照开关」。
对 `flash-attn==2.8.3.post1` 的 sdist 做了兼容性静态审查：

| 检查点 | 结论 |
| --- | --- |
| 版本门 | `setup.py:173` 只有下界 `>= CUDA 11.7`，**无上界** → CUDA 13.3 通过；README 口径「CUDA 12.0 and above」 |
| 分发形式 | PyPI **只有 sdist**（`--only-binary=:all:` 无匹配）→ 必须**源码编译** |
| 架构分支 | `setup.py:179-191` 只有 `80` / `90` / `100` / `120`，**没有 `89`**。设 `FLASH_ATTN_CUDA_ARCHS=89` 会得到**空的 `-gencode`**（静默失效）；本机 sm_89 只能用 `80`（Ada 可运行 sm_80 cubin，NVIDIA 同 major 向后二进制兼容） |
| 规模 | 73 个 TU（`cutlass/test` 那 2500+ 不参与），本机 16 核 / 15GB，估 20–40 分钟 |
| 真正的不确定 | `torch 2.14.0+cu130` 比 flash-attn 的测试矩阵新一大截；FA2 内联 PTX + bundled cutlass 对 CUDA 13 的兼容**静态无法排除**（`csrc/flash_attn/` 自身代码未用任何已移除的 torch C++ API，但这只排除了一个风险源） |

**判定**：无硬阻断，但「20–40 分钟编译 + 中等失败概率」换来的唯一收益是「多一条对照实现」，
而该收益用已存在的 SDPA 就能拿到 → **不装**。

### 1.2 实现：只补注册，不改 oracle

`paged_attention_sdpa`（`nano_vllm/attention/paged_attn.py:65`）**P3 时就已存在**
（gather 成连续 KV 后走 `F.scaled_dot_product_attention`，`tests/test_p3_correctness.py:122` 在用），
只是**没注册进注册表**。所以改动只有注册本身：

```python
PAGED_ATTN_BACKENDS = {
    "torch":  paged_attention_torch,   # einsum + softmax 朴素 oracle
    "sdpa":   paged_attention_sdpa,    # F.scaled_dot_product_attention
    "triton": paged_attention_triton,  # 自研 kernel（P7 图路径唯一实现）
}
```

**刻意没有**把 `paged_attention_torch` 改成 SDPA：它是独立 oracle（einsum + 手写 softmax），
与 triton、sdpa 一起构成三条**不共享代码**的路径；把 oracle 换掉反而少一条对照。

`nano_vllm/models/qwen2.py` **无需改动**：批量 decode 里 `attn_impl == "triton"` 走 batch kernel
（`qwen2.py:219`），否则逐条调 `attn_fn` → SDPA 自然落在那条分支。图路径仍只支持 triton
（`decode_graph.py:77`），`enable_cudagraph` 缺省按此自动决定（`runner.py:50-54`）。

### 1.3 验证（Qwen2.5-1.5B / bf16 / sm_89）

| 判据 | 方法 | 结果 |
| --- | --- | --- |
| **J3 函数级数值等价** | 同一 q/KV/block_table，相对 `torch` oracle 的最大绝对差；seq_len ∈ {16, 48, 63, 64} | sdpa 与 triton **均 9.77e-04**（输出自身 abs max = 0.223，即相对 ~0.4%），三条路径等价 |
| **J2 同路径确定性** | `attn_impl="sdpa"` 跑两次同一负载 | 逐 token 一致 ✅ |
| 跨后端逐 token | sdpa vs triton | **不一致**——属预期：按 `nano-vllm-报告复核与P3前问题清单.md:215`（P1-10）的分层判据，**跨路径只要求 J3 数值等价**，不要求逐 token 相同 |

> 这条顺带修正了 roadmap 里 P8 的验收口径：原文写「输出逐 token 一致」，
> 与 P1-10 已确立的 J1–J4 判据冲突（P8 的融合/预拼接本身就在改数值路径）。已在 roadmap:148 改写为 J3。

---

## 2. 单层 kernel 计数（新工具 `bench/layer_kernels.py`）

roadmap P8 的完成标准是「**单层** kernel launch 数下降 ≥30%（profiler 前后对比）」，
但 `bench/p7_profiler.py` 只有**整步**粒度——融合带来的「每层少几个 kernel」会被
28 层 + 采样 + 调度摊平。

### 2.1 方法：减法归属（而不是 profiler 父链）

`KinetoEvent.cpu_parent` 本是干这个的，但**实测在 torch 2.14 下未被填充**
（链上只有事件自身），父链法不可用。故改用减法：

```
单层 kernel 数 = (整模型一步 − 28 层全部置空后一步) / 28
```

置空 = 把 `DecoderLayer.forward` 换成 identity（`lambda hidden, *a, **k: hidden`）。
层外部分（embedding / rotary / final norm / lm_head / 采样 / 调度）不受影响，差值即 28 层的贡献。
图模式下必须在**捕获之前**打补丁（图一旦捕获就固化了 kernel 序列），所以每次都建新 engine。

**同构性交叉校验**（本工具最容易被质疑的前提）：再测一次「只置空第 1 层」，
`整模型 − 只置空一层` 应 ≈ 单层均值。

### 2.2 基线（batch=8 / prompt_len=64，`bench/results/layer_kernels*.json`）

| 配置 | 单层 kernel | 整步 | 层外 | 同构性偏差 |
| --- | --- | --- | --- | --- |
| triton + 图（P7 生产路径） | **42** | 1198 | 22 | **0.0%** |
| triton eager | **43.0** | 1273 | 69 | **0.0%** |

### 2.3 给 P8 的直接线索

单层 42 个 kernel 的构成（名称分解，全模型口径出现次数）：

| 次数 | kernel | 归属 |
| --- | --- | --- |
| 197 | `cutlass::Kernel2` | ~7 个/层：QKV / o_proj / gate / up / down 的 GEMM |
| 169 + 112 + 59×2 + 57×4 | `elementwise_kernel` 系 | RMSNorm、SiLU、residual 相加 |
| 56 | `at::native::(anonymous namespace)::CatArrayBatchedCopy` | **2 个/层 = `rotate_half`(RoPE) 里的 `torch.cat`**（q 与 k 各一次，`qwen2.py`）——归 RoPE 项，**不是**权重预拼接（2026-10-04 更正） |
| 56 | `index_elementwise_kernel` | KV cache 写入 / slot 相关 |
| 28 | `_paged_attn_decode_batch_kernel` | 1 个/层 |

即 P8 的「单层 kernel 数下降 ≥30%」目标 ≈ 42 → ≤ 29。

> **2026-10-04 更正**：原写「消掉 2 个 `CatArrayBatchedCopy` 靠权重预拼接」**是错的**。做完①后实测
> `CatArrayBatchedCopy` 前后恒为 56（Δ=0）：它来自 `rotate_half` 的 `torch.cat`，属 **RoPE 项的收益**，
> 与权重预拼接无关。①的真实收益在 **GEMM 数**上：每层 7 个 Linear → 4 个（−3 个 `cutlass::Kernel2`）、
> 合并后不再触发 split-k（−2 个 `cublasLt::splitKreduce_kernel`）、净增 1 个 elementwise
> ⇒ **−4/层**（42 → 38）。详见 `docs/notes/p8-prejoin.md`。
>
> **后续进展**：② 「融合 RMSNorm」再 −14/层，当前默认 **24.0 个/层**（P8 起点 42.0，累计 −42.9%）；
> 见 `docs/notes/p8-fused-norm.md`。上面这个「42」的基线文件已刷新为 28.0（与当前默认只差一个开关）。

---

## 3. 回归门禁（新工具 `bench/regress.py` + `bench/results/baseline.json`）

审计文档 P2-03 / 评估报告 §4.1(3) 点名的缺口：「10 个阶段每个都声称 TPOT 改善；
无任何自动回归门禁。某次重构把 P2 的 22.3ms 悄悄变回 30ms 也不会有人知道。」
这个缺口在 P0–P7 只是记账，到 **P8 变成真风险**——P8 动的正是数值路径（权重布局与 GEMM 分段），
预拼接写错不报错，只静默漂移。

### 3.1 两档设计（以及为什么性能档不能做成硬门）

| 档 | 内容 | 判据 |
| --- | --- | --- |
| **正确性（硬门）** | `pytest tests/ -q` 子进程 | 退出码 ≠ 0 → FAIL |
| **性能（比值门）** | **同进程 A/B**：同一 shape 下 eager 与 graph 各测一遍 decode TPOT | `graph_speedup = eager_ms / graph_ms` 低于 `基线 × (1 − tol)` → FAIL |
| 性能（绝对值） | `eager_ms` / `graph_ms` | **只 WARN，不门禁** |

**为什么绝对值不能做门**：本机（WSL + 单卡）**跨 run 漂移实测 21%**——同一份 profiler 脚本
两次运行，未被改动的 eager 路径 TPOT 从 45.28 → 35.59 ms（`P7.md` 踩坑记录 7）。
绝对值做门会随机红。而 A/B **比值**在同一次运行里两臂一起漂，受影响小得多。
代价是「整体同比例变快/变慢」检测不到（那需要独占机器与固定频率）。

### 3.2 基线（2026-10-03）

| 项 | 值 |
| --- | --- |
| 正确性 | 130 passed（138.2s） |
| eager TPOT | 34.94 ms/step |
| graph TPOT | 8.13 ms/step |
| **比值** | **4.30×**（门禁门槛 = 3.01×，即基线 −30%） |

复跑验证：比值 4.67×、绝对漂移 5.5% / 2.9% → PASS。
（`--update-baseline` 重录；换机器或换 torch 后需重录。）

---

## 4. `replay_static_eager` 写入计数双计（P7.md 补记的「顺带发现」）

**问题**：归因实验路径 `replay_static_eager` 里 `_forward` 是 eager，会**真实执行**
`PagedKVCache.write()` → `_written_tokens += bucket`（还含 padding 行）；紧接着又调
`note_written(n)` → 该路径写入计数**双计**，使 `waste_rate` 失真。

**修复**（`nano_vllm/cudagraph/decode_graph.py::replay_static_eager`）：前向前后对
`written_tokens` 做快照/恢复，再只按真实行 `n` 补记。

**影响面**：仅 `waste_rate` 诊断指标的绝对值；生产路径（图回放的 `note_written(n)`）本就准确。

---

## 5. 改动文件

| 类型 | 文件 |
| --- | --- |
| 改名 | `bench/p8_preempt_bench.py` → `bench/preempt_align_bench.py`；`bench/results/p8_preempt_align.json` → `bench/results/preempt_align.json` |
| 修改 | `nano_vllm/attention/backend.py`（注册 sdpa + 口径注释）、`nano_vllm/model_executor/runner.py`（docstring）、`nano_vllm/cudagraph/decode_graph.py`（B1） |
| 修改 | `nano-vllm-roadmap-v2.md`（P8 的 attention 对照开关 + 验收判据改 J3）、`docs/notes/preempt-recompute-align.md`（引用改名） |
| 新增 | `bench/layer_kernels.py`、`bench/regress.py`、`bench/results/baseline.json`、`bench/results/layer_kernels.json`、`bench/results/layer_kernels_eager.json`、本文档 |

## 6. 验证汇总

| 项 | 结果 |
| --- | --- |
| 全量测试 | **130 passed** |
| `bench/regress.py` 基线录制 + 比对 | PASS（比值 4.67× vs 基线 4.30×，退出码 0） |
| SDPA 后端 J3（函数级） | sdpa / triton 相对 oracle 均 9.77e-04 |
| SDPA 后端 J2（同路径确定性） | 逐 token 一致 |
| `layer_kernels.py` 同构性校验 | 偏差 **0.0%**（图档与 eager 档均是） |
