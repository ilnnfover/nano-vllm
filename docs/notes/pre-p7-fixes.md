# 整改 · P7 前置修复记录（R1 / R2 / evict 守卫 / 双前向量化）

> 日期：2026-09-30
> 来源：`nano-vllm` 全仓库架构评审（P6 完成后、进入 P7 前）识别出的两个 P1 级
> 正确性风险（R1 全 prompt 命中、R2 调度 livelock）与一个 P1 级性能待量化项
> （Q1 step 内双前向）。
> 结论先行：**三项全部完成**，全仓测试 105/105 通过；双前向开销量化为
> **混合负载下融合收益保守上界 ≈15% 墙钟**（1.5B/bf16 实测），支持 P7 先做
> decode-only 全图、混批留作 P9 对照的既定路线。

---

## 1. R1 · 全 prompt 前缀命中 → 0 新 token（正确性，P1 级）

### 1.1 问题

同一 prompt 二次提交（重放 / 多轮重 roll）且 `prompt_len % block_size == 0` 时，
`find_longest_hit` 的命中可覆盖**整个 prompt** → `Sequence.num_new_tokens == 0`
（`nano_vllm/engine/sequence.py`）→ 仍被调度进 prefill 批（`scheduler.schedule`
waiting 分支不校验 `num_new > 0`）→ 空前向（0 token 拼 batch）→ 采样取
`logits[i]` 越界或请求永不产出首 token。

### 1.2 vLLM 证据

vLLM 在 `KVCacheManager.get_computed_blocks` 显式以 `num_tokens - 1` 作查找上界
（`vllm-src/vllm/v1/core/kv_cache_manager.py:264-321`，注释明确：保证至少重算
最后一个 token 以产出 logits）。这是 vLLM 的防线，nano 复刻 hash 机制时漏掉了
这一行边界。

### 1.3 修复

- `nano_vllm/kvmm/prefix_cache.py` `find_longest_hit` 新增 `max_tokens` 参数：
  逐块循环中块覆盖 token 数 `end > max_tokens` 即停（按块覆盖边界判断，
  兼容将来 partial 块语义）。
- `nano_vllm/engine/scheduler.py` `get_computed_blocks` 调用处传
  `max_tokens=seq.num_prompt_tokens - 1`，与 vLLM 逐字对齐。

### 1.4 验证

- `tests/test_pre_p7_fixes.py::TestR1FullPromptHitClamp`：32-token prompt（16 整除）
  两块全命中 → 命中钳到 16、`num_new_tokens=16`；33-token prompt 行为不变
  （钳到 32，与修复前一致——非整除场景无多余损失）。
- `tests/test_pre_p7_fixes.py::TestR1SamePromptReplayE2E`：0.5B/CPU 同 prompt 连续
  两次 `engine.generate`，第二次全 prompt 命中，输出与第一次逐 token 一致
  （修复前此用例在 prefill 空批上崩溃/挂起）。
- 既有断言更新：`test_p6_prefix_cache_unit.py::test_extra_keys_gate_sharing`
  的 prompt=8 恰为 block_size=4 整数倍，本身就是 R1 场景——命中断言按钳制语义
  从 8 更新为 4（并注明原因）。这是**语义变化**而非测试迁就：多命中一个块
  就会重现 0 新 token。

### 1.5 命中率影响

钳制只影响「prompt 长度恰为 block_size 整数倍」的请求，代价 = 多重算至多
1 个 block（16 token）。P6 bench 的 replay/hit-rate 数据不受影响（负载里
prompt = prefix + suffix，suffix 存在时非整除；prefix_len=4096 场景 suffix=64）。

---

## 2. R2 · 调度 livelock 两个形态（正确性，P1 级）

### 2.1 形态 A：超容量请求（独占全池也无法完成）

`prompt + max_new_tokens` 所需块数 > 全池总量时，该请求即使抢光所有块也永远
跑不完。原实现的 waiting 分支会为此抢占全部 running，然后自己仍调度失败 →
下一步重试 → 无限空转；`EngineCore.generate` 的 `while has_requests()` 永不退出。

**修复**（`scheduler.add_request`）：准入时校验
`blocks_needed(prompt + max_new_tokens) <= pool.num_blocks`，超出直接 `ValueError`
拒绝入队（错误信息含所需块数与建议）。语义与 vLLM 一致：请求无法完成时以
失败告终，而不是无限空转。

注意：判据用 `max_new_tokens`（而非「prompt 能放下就行」）是**完成性保证**的
强判据——decode 增长超池的请求同样会 livelock（见形态 B 的退化情形）。
副作用：早停（EOS）本可完成的请求若 `max_new_tokens` 填得过大会被拒，
属于「把不可能完成的契约提前失败」，文档化即可。既有 bench 负载
（13489 块池、最大 8K prompt）均不受影响。

### 2.2 形态 B：waiting 抢占 running → ping-pong

原 waiting 分支块不足时 LIFO 抢占 `running[-1]` 并 `continue` 重试。组合需求
超过池容量时（每条请求单独都可行）：

```
step k:   A(decode) 抢掉 B → A prefills
step k+1: B(waiting) 抢掉 A → B prefills/decodes
step k+2: A 抢掉 B → …
```

每次抢占 `reset_for_preemption` 清空 `output_token_ids` → 双方进度反复归零，
**谁也到不了 max_new_tokens** → livelock。

vLLM 证据：`vllm-src/vllm/v1/core/sched/scheduler.py` 中抢占只发生在 **RUNNING**
调度段（allocate_slots 失败时抢占 `running[-1]` 或自身，L737-805）；waiting 段
块不足直接停止准入，且本 step 一旦发生过抢占就不再调度 waiting（L863-864）。
waiting 永远不抢占 running——新请求靠 running 完成后自然释放的块入场，
饥饿有界（running 必在 max_new_tokens 内完成）。

**修复**（`scheduler.schedule` waiting 分支）：删去抢占逻辑，块不足直接
`break`。RUNNING 段的自身抢占（原 Branch 2）保持不变，成为唯一抢占路径。

### 2.3 语义变化与代价

| 项 | 修复前 | 修复后 |
| --- | --- | --- |
| waiting 块不足 | 抢占 running 最新者后重试（可级联） | 停止准入，等自然释放 |
| 新请求 TTFT（池紧张时） | 可能立即入场（踩着别人） | 有界饥饿：≤ 当前 running 的剩余 max_new_tokens |
| 抢占触发点 | waiting / running 均可 | 仅 running（增长失败者自身 recompute） |
| livelock | 形态 A/B 均可发生 | 消除（每请求单独可行 + 有界完成时间 ⇒ 必然终止） |

评审中「自我抢占 vs 抢 `running[-1]`」维持原样（自我抢占）：影响仅是抢占者
丢失自身进度 vs 丢失更新者的进度，两者都能终止，改动收益低，记录即可。

### 2.4 验证

- `TestR2CapacityGuard`：不可行请求 `ValueError`；恰占满全池的边界请求放行。
- `TestR2WaitingDoesNotPreempt`：块不足时 waiting 留队、零抢占；
  running 自身抢占路径不受影响。
- `TestR2WaitingDoesNotPreempt::test_pingpong_terminates`：**修复前的确切
  livelock 场景**（两条 48-token prompt + max_new 8，池 6 块，组合 8 块 > 6），
  驱动 300 步上限——修复后 ~20 步内双双完成；修复前互相抢占、进度反复归零，
  该用例会在步数上限失败。
- 既有 `test_p4_correctness::test_preemption`、`test_p6_correctness::
  test_preemption_with_prefix_cache` 均通过（抢占语义改为 running 自身抢占后，
  两条用例的触发路径变为「decode 增长竞争」，仍产生抢占且输出与 P1 一致）。
- 既有单测适配：`test_p6_prefix_cache_unit.py` 的 4 个 Scheduler 级用例
  原以默认 `SamplingParams(max_new_tokens=128)` 配 16 块小池，被新准入守卫
  正确拒绝——补显式 `max_new_tokens=8`（守卫行为正确，测试夹具适配）。

---

## 3. 附带修复 · `_evict_hashes` hash 别名误删（性能，P2 级）

R1 场景的连带排查发现：同内容块被重算后，`register_hash` 会把
`_hash_index[h]` 指向**新块**，而旧块仍残留同名 `block_hash`。旧块之后被
重新分配时，原 `_evict_hashes` 无条件 `pop(_hash_index[h])` → **误删新块的
注册**（命中丢失，只损性能不算错——hash 相同即内容相同）。

修复（`nano_vllm/kvmm/block_pool.py` `_evict_hashes`）：仅当索引项确实指向
本块时才 pop。验证：`TestEvictHashAliasGuard`。

遗留观察（记录不改）：`register_hash` 重复注册时旧块的陈旧 hash 字段仍会
保留到其被重新分配为止，由本守卫兜底，命中率影响可忽略；vLLM 用
`cache_full_blocks` 的 promotion 断言从源头避免重复注册，将来做 fine-grained
hash 时可一并收紧。

---

## 4. Q1 · 「step 内双前向」开销量化（性能，P7 设计输入）

### 4.1 方法

`bench/p7_pre_bench.py`（不改引擎源码，脚本内包装 `_execute` 与三个前向入口，
GPU 前后 `torch.cuda.synchronize` 逐 step 计时）。负载：4×2048 长 prompt +
4×128 短 prompt 同时提交，budget=512（长 prompt 必然多 chunk），max_new=32，
`enable_prefix_cache=False`（隔离变量），attn=triton，1.5B/bf16，3 次重复。

### 4.2 结果（在 Qwen2.5-1.5B-Instruct / bf16 / RTX 4070 Ti SUPER 上测得，不外推）

`bench/results/p7_pre_dual_forward.json`，3 次 run 稳定：

| 指标 | 中位数 |
| --- | --- |
| 总 step 数 | 49 |
| 混合 step（同 step 既有 prefill 又有 decode） | 14（28.6%） |
| 混合 step 平均：prefill 前向 / decode 前向 / step | 83.2 / 32.8 / 116.4 ms |
| 双前向多出的前向次数（vs 混批） | 14 |
| **融合收益保守上界**（Σ min(prefill, decode)） | **461 ms，占墙钟 15.1%** |
| 单次 run 波动 | 14.1% / 15.1% / 15.8% |

口径说明：若混批，两次前向合并为一次，耗时近似 ≥ max(两者)，故可省 ≤
min(两者)——这是**保守上界**；实际融合收益还包含省掉的 host 侧第二次
张量拼装与全部 kernel launch（该部分由 P7 的 torch profiler 数据补充）。
decode-only step（31 步）无此开销，纯 decode 负载下双前向结构零损失。

### 4.3 对 P7 的决策含义

1. **支持既定路线**：P7 做 decode-only 全图捕获——decode 前向独立反而使图捕获
   简单（无 prefill varlen 混入）；混合负载 ~15% 的结构性损失留作 P9
   「混批 vs 双前向」对照实验的量化对象。
2. **P7 验收基准选纯 decode 负载**测 TPOT（图命中率与收益都干净），混合负载
   数字单独归档，避免把 15% 的结构损失误读为 P7 图收益不达标。
3. P7 的输入 buffer 固定（input_ids/positions/block_table/slot_mapping 预分配）
   对混合 step 中的 decode 前向同样生效，故混合负载也会部分受益——但收益来源
   应分账记录（图回放 vs 混批），这是面试可讲的「可解释的性能」。

---

## 5. 全量验证记录

```
~/venvs/dev/bin/python -m pytest tests/ -v
======================= 105 passed in 144.30s (0:02:24) =======================
```

| 测试文件 | 结果 | 说明 |
| --- | --- | --- |
| `tests/test_pre_p7_fixes.py` | 11 passed（新增） | R1 钳制 / evict 守卫 / R2 守卫与不抢占 / ping-pong 终止 / 重放 E2E |
| `test_p4_correctness.py` | passed | 抢占恢复输出仍与 P1 eager 一致 |
| `test_p6_correctness.py` | passed | 开关缓存逐 token 一致、抢占 × 前缀缓存 |
| `test_p6_prefix_cache_unit.py` | passed（5 处断言/夹具按新语义更新） | 见 §1.4、§2.4 |
| `test_p6_throttle.py` / `test_p6_guards.py` / `test_sampler.py` / `test_chunked_prefill.py` / `test_p2/p3/p5` | passed | 无回归 |

## 6. 本次不改清单（防过度设计）

| 项 | 理由 |
| --- | --- |
| prefill/decode 混批实现 | 属 P9 对照实验；P7 前改动会让「图收益」与「混批收益」两组变量混在一起（违反一版一主题） |
| 抢占 victim 改为 vLLM 的 priority/arrival 语义 | 请求级优先级属 roadmap §2 裁剪项；自我抢占与抢 `running[-1]` 均能终止，收益低 |
| `register_hash` 重复注册源头去重（复刻 vLLM promotion 断言） | 已由 `_evict_hashes` 守卫兜底，仅在 fine-grained hash 落地时才有实际价值 |
| R1 钳制放宽（如仅对「真·重复请求」钳制） | 需要请求级去重语义，复杂度不成比例；统一钳 1 token 的代价可忽略 |

---

## 7. 后续修订（2026-10-03）

本文档记录的是当时（P7 前）的口径。此后「抢占恢复对齐 vLLM」整改改动了其中两点，
以本仓当前实现为准：

| 项 | 本文档当时口径 | 当前实现 | 变更原因 |
| --- | --- | --- | --- |
| R1 命中钳制 | `num_prompt_tokens - 1` | **`num_tokens - 1`**（`num_tokens = prompt + output`） | 抢占恢复要对齐 vLLM：新请求时两者等价；恢复时能复用已生成 token 的块 |
| 抢占语义 | `reset_for_preemption` 清空 `output_token_ids` | **保留 output**，只归零 `num_computed_tokens` | 同上（vLLM `_preempt_request` 亦如此） |

数据与实施细节：`docs/notes/preempt-recompute-align.md`（重算 352→114 token，3.09×）。
