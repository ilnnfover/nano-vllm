# 整改 · 抢占恢复对齐 vLLM（位置式 token 流 + 保留 output）

> 日期：2026-10-03
> 来源：架构评审后的待办「抢占恢复对齐 vLLM」（评审记录于
> `docs/notes/pre-p7-fixes.md` 与 `docs/stages/P7.md` 的「后续」）。
> 参考源码：`vllm/v1/core/sched/scheduler.py::_preempt_request`、
> `vllm/v1/core/kv_cache_manager.py::get_computed_blocks`。
> 结论：**重算量 352 → 114 token（3.09×）、端到端墙钟 3213 → 2107 ms（1.53×）、
> 抢占次数 18 → 3，且两种口径输出逐 token 一致**（1.5B/bf16 实测）。

---

## 1. 问题：nano 的抢占恢复为什么比 vLLM 贵一个量级

| 维度 | nano（改动前） | vLLM |
| --- | --- | --- |
| 抢占时对已生成 token | `reset_for_preemption` **清空 `output_token_ids`** | 只把 `num_computed_tokens = 0`，**保留 output**（`scheduler.py:1544-1568`） |
| 恢复时的命中范围 | 只在 **prompt** 上查命中，钳制 `num_prompt_tokens - 1` | 全流 `num_tokens - 1`（`kv_cache_manager.py:295`），`num_tokens = prompt + output` |
| 恢复代价 | 重算 prompt 尾部 **+ 全部已生成 token 重新生成** | 重算 ≤ 1 个 block，末段 logits 产出**下一个** token |
| 流式副作用 | 已推送的 token 会被重复产出（序列回退而客户端不撤回） | 无（不会重新产出已发 token） |

根因不是"钳制公式"，而是**执行模型假设「prompt = prefill、output = decode、output 只增不减」**：
`is_prefill` 决定给 chunk 还是给 1 token，输入 token 的取法也按阶段二分
（prefill 取 `prompt_token_ids[p]`、decode 取 `output_token_ids[-1]`）——于是
"重算 output 区间"这个动作**表达不出来**，只能退化成"清空 output 从 prompt 重跑"。

---

## 2. 做法：三条公式，四步落地

### 2.1 统一规则（核心）

设 `T = prompt + output`（`Sequence.all_token_ids`），`n = len(T)`，
`c = num_computed_tokens`（KV 覆盖 `[0, c)`，稳态恒有 `c = n - 1`）：

| 规则 | 表达式 | 含义 |
| --- | --- | --- |
| 输入 token | `input(p) = T[min(p, n-1)]` | 位置 p 的输入；`p == n` 时重喂最后一个 token（decode） |
| 本步算多少 | `num_new = min(n - c, budget)` | 稳态 decode 天然 = 1；追赶段 = 整段 |
| 何时采样 | `c + num_new >= n` | chunk 终点到达已知流末尾才采样 |

三条公式同时覆盖：新请求 prefill、稳态 decode、**抢占恢复的 output 区间追赶**
（中间段只重算 KV 不采样 → 不重复产出；末段 logits 产出下一个 token）。

### 2.2 路由（保留原有 kernel 选择，避免数值路径漂移）

```python
def _use_prefill_route(seq, num_new) -> bool:
    return num_new > 1 or seq.num_computed_tokens < seq.num_prompt_tokens
```

| 情形 | 路由 |
| --- | --- |
| 多 token chunk（新请求 prefill / 恢复追赶段） | varlen prefill |
| 单 token 且仍在 prompt 区（末尾 prompt token） | varlen prefill（**与改动前完全一致**） |
| 单 token 且在 output 区（稳态 decode / 尾部追赶） | decode 路径（**可进 P7 图**） |

### 2.3 四步实施（每步独立验证，符合"一版一主题"）

| 步骤 | 内容 | 验证 |
| --- | --- | --- |
| **S1** | `Sequence` 新增 `all_token_ids` / `input_token_ids(start,k)`（位置式取法） | 纯新增，11 项单测；全量 126 绿 |
| **S2** | 执行路径统一取 token：`_run_prefill*` / `_run_decode*` / `DecodeBuffers.fill` 全部改走 `input_token_ids` | 等价重构，全量 126 绿 |
| **S3** | 行为变化：`reset_for_preemption` 保留 output；查找传全流 + 钳制 `num_tokens - 1`；调度器统一记账；采样判据统一；`ScheduledSeq.num_tokens` → `num_scheduled_tokens` | 新增 4 用例；全量 **130 绿** |
| **S4** | 对照基准 `bench/preempt_align_bench.py`（monkeypatch 复刻旧口径，单变量） | 落盘 `bench/results/preempt_align.json` |

`S3` 的关键改动点：

```python
# scheduler.get_computed_blocks —— 全流查找 + num_tokens-1
seq.all_token_ids, max_tokens=seq.num_tokens - 1
# scheduler.schedule —— 统一 chunk 大小（不再按阶段二分）
num_new = min(seq.num_new_tokens, token_budget)
# ScheduledSeq.samples_this_step —— 采样判据
seq.num_computed_tokens + num_scheduled_tokens >= seq.num_tokens
# sequence.reset_for_preemption —— 保留 output（只归零 num_computed_tokens）
```

---

## 3. 实测（Qwen2.5-1.5B-Instruct / bf16 / RTX 4070 Ti SUPER，不外推）

`bench/results/preempt_align.json`；负载：6 条 48-token prompt、max_new=32、
**紧池 20 块**（故意触发抢占）、3 次重复取中位数；两种口径同池同负载同种子。

| 指标 | legacy（改动前） | aligned（改动后） | 变化 |
| --- | --- | --- | --- |
| **恢复重算 token 数** | **352** | **114** | **3.09× ↓** |
| 抢占次数 | 18 | 3 | 6× ↓ |
| prefill 路径 token 数 | 640 | 402 | 1.59× ↓ |
| 前向 token 总数 | 842 | 586 | 1.44× ↓ |
| 端到端墙钟 | 3213 ms | 2107 ms | 1.53× ↓ |
| 输出 token 数 | 192 | 192 | — |
| **输出逐 token 一致** | — | — | ✅ `outputs_identical=true` |

**两个额外结论**：

1. **抢占次数也下降（18→3）**：legacy 下每次抢占都把请求打回 prompt 起点，反复重置 →
   反复竞争；aligned 下恢复后**有净进展**（已生成 token 保留），不再来回弹。
   这比"单次恢复更省"更重要——它减少了抢占的发生本身。
2. **"≤1 block" 是理想上界**：命中链（滚动 hash 逐块）在紧池下会被挤掉中间块
   （自己释放的块被别的请求分配 → hash 被 evict → 链条提前断），此时重算是
   "尾块 + 断链处之后的部分"。114/3 ≈ 38 token/次即由此而来。
   vLLM 同样受此影响（它的命中也是链式的）。

---

## 4. 验证与回归网

```
~/venvs/dev/bin/python -m pytest tests/ -q
============================= 130 passed in 131.00s =============================
```

| 用例 | 覆盖 |
| --- | --- |
| `tests/test_preempt_align.py::TestUnifiedTokenStream`（7） | 位置式取法：prompt 区 / output 区 / 跨边界 / 越界重喂 / 空流 / 返回副本 |
| `...::TestNumNewTokensUnified`（3） | `num_new_tokens`：稳态=1、抢占后=整段、无 output 时=prompt |
| `...::TestResumeReuseKV`（3） | ① 抢占保留 output；② **重算量 = `n - floor((n-1)/bs)*bs ≤ bs`** 且命中超过 prompt 长度（证明复用了已生成 token 的块）；③ 恢复后 output 既有前缀不变（不重复产出） |
| `...::TestResumeMatchesEagerNano`（1） | 端到端：抢占×3 请求，恢复后输出与 P1 eager 逐 token 一致 |
| `test_p4_correctness.py::test_preemption` | 既有对拍（小池抢占 + P1 eager）保持绿 |
| `test_p6_correctness.py::test_preemption_with_prefix_cache` | 抢占×前缀缓存×P1 一致保持绿 |

---

## 5. 已知前提与风险（诚实边界）

1. **不宣称"逐位一致"**：恢复时重算的 KV 与原先（不同 batch 形状下的前向）不是逐位相同，
   两种口径的输出一致性是"greedy 逐 token 一致"（`outputs_identical=true` 在本次负载下成立），
   不是位级保证。这与 P6 既有的"开/关缓存一致"是同一类前提。
2. **budget 饥饿**：`max_num_batched_tokens` 很小时，追赶段可能被切成多段；
   中间段**不采样**（由 `samples_this_step` 保证），不会串味。
3. **图覆盖**：P7 的图只覆盖 decode 路由的 chunk（稳态 decode + output 区单 token 追赶）。
   prompt 区的单 token chunk 仍走 varlen prefill（刻意为保持 kernel 选择不变）。
4. **R1 边界的"接管块"优化仍不做**：见 `docs/notes/pre-p7-fixes.md` §1 与
   P7 讨论——收益是每请求一次性 ≤1 块，成本是块所有权状态机，优先级低于本次整改。

---

## 6. 与 vLLM 的对照（本快照源码）

| 机制 | vLLM 位置 | nano 对应 |
| --- | --- | --- |
| 抢占只归零计算进度、保留 output | `scheduler.py:1544-1568` | `Sequence.reset_for_preemption` |
| 全流查找 + `num_tokens - 1` 钳制 | `kv_cache_manager.py:292-298` | `Scheduler.get_computed_blocks` |
| 无 prefill/decode 阶段，按 `num_computed_tokens` 追赶 | `scheduler.py:572-581` | `ScheduledSeq` + `num_new_tokens` 公式 |
| 参考 token 流 = prompt + output | `request.num_tokens` | `Sequence.all_token_ids` |

**保留的差异**：nano 仍以"块粒度"做命中（不做 fine-grained/partial），
且恢复时因块未满而不注册 output 的尾块（与 vLLM 的 `max_cache_hit_length` 行为一致）。
