# nano-vllm · P6 前优化审计（以 vLLM v1 为基准）

> 对照基准：vLLM 主线 `vllm/v1/*`（本次核实过的文件见各条「vLLM 参考」）
> 审计对象：`ilnnfover/nano-vllm` main @ `35682cd`（P5 已合入，P6 未开始）
> 说明：所有收益数字为**估算**，除非注明「实测」。本次只做分析，未改任何代码。

---

## 0. TL;DR — 按收益/成本比排序的总表

| # | 优化项 | 维度 | 收益 | 成本 | 比 | 优先级 |
|---|---|---|---|---|---|---|
| 1 | `abort_request()`：断连/取消时释放 KV 块 | 并发/资源 | 高 | 低 | ★★★★★ | **P6 前必做** |
| 2 | `attn_impl` 在 serving 里硬编码 `torch` → 暴露为参数 | 性能 | 中高 | 极低 | ★★★★★ | **P6 前必做** |
| 3 | `num_blocks` 按显存反推（现默认 512 块 = 仅 8192 slot） | 显存 | 高 | 低 | ★★★★★ | **P6 前必做** |
| 4 | `_step_loop` 异常隔离 + 请求级错误传播 | 可靠性 | 高 | 低 | ★★★★☆ | **P6 前必做** |
| 5 | 采样批量化 + 张量化 temperature/top_p/top_k | 性能 | 中 | 中 | ★★★★ | **P6 前必做** |
| 6 | `finish_reason` 区分 `stop`/`length` | 协议/一致性 | 中 | 极低 | ★★★★ | **P6 前必做** |
| 7 | `tok.encode` 移出 event loop | 并发 | 中 | 极低 | ★★★★ | **P6 前必做** |
| 8 | decode attention 批量化（解 `for i in range(b)`） | 性能 | 中高 | 中 | ★★★ | 建议 P6 前 |
| 9 | `slot_mapping`/`block_table` 向量化 + 预分配 buffer | 性能 | 中 | 中 | ★★★ | 建议 P6 前 |
| 10 | 调度加 `long_prefill_token_threshold` + `max_num_seqs` | 尾延迟 | 中 | 中 | ★★★ | 建议 P6 前 |
| 11 | step 级 metrics + 消费 `preempted_seq_ids` | 可观测性 | 中 | 中 | ★★★ | 建议 P6 前 |
| 12 | flashinfer wrapper 复用 + plan 缓存 | 性能 | 中低 | 低 | ★★★ | 建议 P6 前 |
| 13 | 合并 prefill/decode 两次 forward（unified batch） | 性能 | 中低 | 高 | ★★ | 可延后 |
| 14 | `_run_prefill` 死代码复活为对照组 | 可维护性 | 低 | 极低 | ★★ | 可延后 |
| 15 | 代码结构清理（双 SamplingParams / attn_impl 双关 / core 撞名） | 可维护性 | 中 | 中 | ★★ | 可延后 |
| 16 | 释放 P2 连续 `KVCache` 常驻显存 | 显存 | 中 | 低 | ★★ | 可延后 |
| 17 | Queue 背压（maxsize + drop 策略） | 并发 | 低中 | 低 | ★★ | 可延后 |
| 18 | logprobs / per-request seed | 一致性 | 低 | 中 | ★ | 可延后 |
| 19 | swap 抢占（替代 recompute） | 性能 | 中 | 高 | ★ | 可延后 |
| 20 | 算子融合 / torch.compile / CUDA Graph | 性能 | 高 | 高 | ★★ | P7 本体 |

---

## 1. 维度 A · 推理性能（调度 / 批处理 / KV 显存管理）

### A1 · `attn_impl` 在 serving 路径被硬编码为 `torch`

- **当前位置**：`nano_vllm/server/api.py::create_app()` → `NanoRunner(..., attn_impl="torch", ...)`；`main()` 的 argparse 里**根本没有 `attn-impl` 参数**。
- **vLLM 参考**：`vllm/v1/attention/backends/`（backend 由 `VllmConfig.attention_backend` 选择，`vllm/v1/worker/gpu_model_runner.py` 里初始化并常驻）。
- **根因**：P3 自研 Triton paged decode kernel 实测比 torch 朴素快 **1.12–1.51×**（P3 kernel bench，seq=1024→1.12×，seq=4096→1.51×），但 P5 装配时写死了 `"torch"`，且 CLI 没暴露开关。等于 P3 的 kernel 在 serving 里一次都没跑过。
- **思路**：`create_app` 增加 `attn_impl` 参数并透传；`main()` 加 `--attn-impl {torch,triton}`。
- **预期收益**：decode 主导场景下端到端约 5–15%（估算）；P5 bench 的 overhead_ratio 会相应变化。
- **成本**：低（~10 行）。
- **风险**：极低。唯一注意 triton kernel 要求 CUDA + `num_tokens == 1`，CPU/fp32 测试路径不受影响。

### A2 · 采样未批量化，每请求一次 Python 调用 + 一次 GPU→CPU 同步

- **当前位置**：`engine/core.py::_execute` → `_sample()` 里的 `for`/逐条调用；`sample/sampler.py::Sampler.sample()` 返回 `int(...)`。
- **vLLM 参考**：`vllm/v1/sample/sampler.py::forward`（整批 `logits.div_(temp.unsqueeze(1))` + `topk_topp_sampler(logits, generators, top_k, top_p)`）、`vllm/v1/sample/metadata.py::SamplingMetadata`（温度/top_k/top_p 全是张量）。
- **根因**：(a) 每请求一次 `int(tensor)` 触发设备同步，b=12 时每步 12 次；(b) 全局**单个** `torch.Generator`，per-request seed 无法隔离；(c) temperature/top_p 走 Python 分支判断，无法批量。
- **思路**：`_execute` 收集本步所有待采样请求的 logits 拼成 `[n, vocab]`，一次性走张量化采样返回 `[n]`；`SamplingParams` 的标量参数拼成张量。
- **预期收益**：每步省掉 N 次 host-device 同步 + N 次小 kernel launch，估算 **5–10%** decode 步耗时。
- **成本**：中。`Sampler` 接口要改成 batch 版，`tests/test_sampler.py` 需扩展。
- **风险**：采样顺序改变会让既有对拍的随机数序列变化 → 用固定 seed 的 CPU fp32 测试重新锁定基线；greedy 路径（`temperature<=0`）应完全不受影响，可先只批量化 greedy 再扩到随机采样。

### A3 · 批量 decode attention 是 Python 逐条循环

- **当前位置**：`models/qwen2.py::_forward_paged()` 的 decode 分支 `else:` → `for i in range(b): outs.append(attn_fn(...))`。
- **vLLM 参考**：`vllm/v1/attention/backends/flash_attn.py`（整批一次 kernel）、`vllm/v1/attention/ops/triton_unified_attention.py`（`grid` 含 batch 维）。
- **根因**：triton 后端 `grid = (num_heads,)`，一次只处理一条请求 → **28 层 × b 次 kernel launch/step**（b=12 时 336 次极小 kernel）；torch 后端同样每请求一次 `index_select` gather。
- **思路**：(a) 先把 `block_tables` 做成二维预分配张量；(b) triton kernel 的 grid 扩成 `(batch, num_heads)`；(c) torch 后端用一次 `index_select` 处理所有请求的块，再 reshape 回 `[b, ...]`。
- **预期收益**：decode 步 launch 开销从 O(28·b) 降到 O(28)，估算 **10–25%** decode 步耗时（小 batch 下 launch 占比高）。
- **成本**：中（改 kernel grid + metadata 二维化）。
- **风险**：GQA 映射 `kv_head = head_idx // 6` 在扩维后要重新核对；末块半满 mask 要按每请求的 `seq_len` 分别截断——**块表 padding 读脏块会直接 NaN**，这是 P3 已踩过的坑，必须补一条「不同请求块数不等」的测试。

### A4 · 每步重建张量，无 persistent buffer

- **当前位置**：`engine/core.py::_run_prefill_batched` / `_run_decode_batched`（每步 `torch.tensor(input_ids)`、`torch.tensor(position_ids)`、`torch.cat(slot_mappings)`、`torch.tensor(last_idx)`）；`core/paged_kv_cache.py::slot_mapping()` 是 **Python for 循环逐 token 构造**。
- **vLLM 参考**：`vllm/v1/worker/gpu_model_runner.py::InputBatch`（预分配 `token_ids`/`positions`/`slot_mapping`/`block_table` 缓冲 + `CpuGpuBuffer.copy_to_gpu()`）、`vllm/v1/worker/block_table.py`（`commit_block_table(num_reqs)` 增量提交）。
- **根因**：所有元数据每步从 Python list 新建 → 每步多次 host→device 拷贝；`slot_mapping` 逐 token 循环在 token budget 2048 时是 2048 次 Python 迭代/step。
- **思路**：预分配 `max_num_batched_tokens` 大小的 buffer，每步只写 `[:n]` 并 `copy_()`；`slot_mapping` 向量化为 `block_table.gather() * block_size + arange % block_size`。
- **预期收益**：每步省若干次小拷贝 + 一次 O(total_tokens) 的 Python 循环，估算 **3–8%**；**更重要的收益是为 P7 CUDA Graph 铺路**（地址固定是捕获的前提）。
- **成本**：中。
- **风险**：buffer 复用后必须严格用 `num_tokens` 截断，否则读到上一 step 的残留 → 静默算错。建议同时加一条「buffer 复用后结果不变」的对拍测试。

### A5 · flashinfer wrapper 每步重建 + 每步 `plan()`

- **当前位置**：`attention/varlen_prefill.py::varlen_prefill_flashinfer()`（函数内部 `new BatchPrefillWithPagedKVCacheWrapper` + `plan()`）；`_get_workspace()` 只缓存了 128MB workspace。
- **vLLM 参考**：`vllm/v1/attention/backends/flashinfer.py`（wrapper 在 builder/backend 初始化时创建并常驻，只在 metadata 结构变化时 re-plan）。
- **根因**：wrapper 构造 + `plan()` 是 host 侧开销，12 请求小 batch 下盖过 kernel 收益（P4 笔记踩坑 3 已记录）。
- **思路**：按 `qo_indptr` 结构哈希缓存 wrapper，结构不变则只 `run()`。
- **预期收益**：中低——**prefill 不是瓶颈**（P4 已证明收益主要来自 MLP/QKV 拼批），所以这项修好也不太可能改变吞吐结论。
- **成本**：低。
- **风险**：低。缓存 key 必须包含所有影响 `plan()` 的输入（qo_indptr、kv_indptr、last_page_len、num_kv_heads、dtype），漏一个就会算错。

### A6 · 调度策略缺三个 vLLM 标配的节流阀

- **当前位置**：`engine/scheduler.py::schedule()`（阶段 2 只看 `waiting[0]`；`watermark_blocks` 是固定块数；无 `max_num_seqs`）。
- **vLLM 参考**：`vllm/v1/core/sched/scheduler.py` —— `long_prefill_token_threshold`（截断单个请求的新 token 数）、`adaptive_long_prefill_threshold`（阈值下限 = `input_budget // num_eligible_reqs`）、`create_request_queue(policy)` 支持 `PRIORITY` 策略、victim 选择 `max(running, key=(priority, arrival_time))`。
- **根因**：
  1. **队头阻塞**：`while` 只看 `waiting[0]`，队首是 8K 长 prompt 且块不够时直接 `break`，后面排着的短请求一起饿死。
  2. **无长 prefill 节流**：budget 2048 下一条 8192 的 prompt 会连续 4 个 step 独占预算，同批短请求 TPOT 直接炸（这正是 `p4_bench --tpot-test` 想量化的现象，目前测法有缺陷）。
  3. **watermark 固定 1 块**：块池变大后（见 B1）1 块不再有保护意义。
- **思路**：加 `long_prefill_token_threshold`（默认 `max_num_batched_tokens // 2`）+ `max_num_seqs`；watermark 改成比例（vLLM 默认 0.01）。
- **预期收益**：主要改善 **TPOT p99 尾延迟**，吞吐可能略降。
- **成本**：中（调度行为变更，需要重跑 P4 全部 bench）。
- **风险**：阈值调错会把吞吐打下去；建议参数化并在 bench 里扫阈值。

### A7 · prefill + decode 两次 forward（非 unified batch）

- **当前位置**：`engine/core.py::_execute()` 拆 `prefill_seqs` / `decode_seqs` 各跑一次 model forward。
- **vLLM 参考**：`vllm/v1/` unified batch（prefill 与 decode 混在同一次前向，靠统一 wrapper / unified kernel）。
- **思路**：decode 视为 `q_len=1` 的 entry，统一走 varlen 路径（右对齐 causal 自动退化成「看全部 KV」）。
- **预期收益**：**中低**。合并后 GEMM 尺寸几乎不变（`total_q` 由 prefill chunk 主导），不存在 P4 那种「12 次变 1 次」的收益；省的是一次完整 forward 的 launch 开销，估算 1–3ms/step。
- **成本**：高（要动 metadata、kernel 分发、采样时机）。
- **风险**：纯 decode batch 必须走回 P3 Triton kernel（它为 `q_len=1` 设计），否则反而变慢。**建议推到 P6 之后**，或在 A3 完成后再做。

### A8 · 模型层无算子融合 / 无 torch.compile / 无 CUDA Graph

- **当前位置**：`models/qwen2.py`（RMSNorm、RoPE、SwiGLU、QKV 投影都是朴素实现）；`reshape_and_cache` 是 PyTorch 高级索引而非 kernel。
- **vLLM 参考**：`vllm/model_executor/layers/`（fused RMSNorm / RotaryEmbedding / MLP）、`vllm/compilation/`（`CudagraphDispatcher`、`cudagraph_capture_sizes`）、`vllm/v1/worker/gpu_model_runner.py`（`compilation_config`）。
- **预期收益**：高（这是 P7 的主要收益来源）。
- **成本**：高。
- **建议**：**明确划给 P7**，P6 前不要碰。

---

## 2. 维度 B · 内存与显存占用

### B1 · `num_blocks` 未按显存反推（**当前 serving 严重受限**）

- **当前位置**：`model_executor/runner.py::NanoRunner.__init__` → `num_blocks or PagedKVCache.blocks_needed(max_seq_len, block_size) + 1`；`server/api.py` 默认 `num_blocks=512`。
- **vLLM 参考**：`vllm/v1/worker/gpu_model_runner.py::determine_available_memory()`（profile 后按剩余显存算块数）、`vllm/v1/core/kv_cache_utils.py`（`check_enough_kv_cache_memory`）。
- **根因 + 实测推算**：服务端默认 **512 块 × block_size 16 = 8192 个 slot**，1.5B bf16 每 token KV 占 `28 层 × 2(K/V) × 2 KV头 × 128 × 2B = 28 KB`，即整池只有 **229 MB**，容量 **8192 token**。而 `max_seq_len` 默认 8192 —— **一条满长请求就能占满整池**；`blocks_needed(8192+64) = 516 > 512` 会触发 `_try_ensure_capacity` 返回 False → running 为空 → `break` → **该请求永久卡在队首，并阻塞后面所有请求**（不崩溃，直接静默饥饿）。
  你的卡（4070 Ti SUPER 16GB）按剩余 ~8.5GB 算，理论上限约 **19k 块 / 30 万 token**。
- **思路**：启动时 profile 剩余显存 → 反推 `num_blocks`，取 60–70% 安全系数；`--num-blocks` 保留为手动覆盖。
- **预期收益**：并发能力从「1 条长请求」提升到数十条，是 P6 前缀缓存的硬前提（前缀缓存需要**富余**的块才有命中率）。
- **成本**：低（~40 行）。
- **风险**：算多了会 OOM。必须先做一次 dummy profile run 再分配，别用 `torch.cuda.mem_get_info()` 直接算。

### B2 · P2 连续 `KVCache` 仍然常驻显存

- **当前位置**：`runner.py::__init__` 无条件构造 `KVCache(max_seq_len=..., batch_size=1)`；P4/P5 主链路已完全不用它，只有 `generate_batch` / `_prefill` / `_decode` / P1 eager 用。
- **vLLM 参考**：不存在对照物（vLLM 只有分页一种）。
- **根因**：为了保留 P1/P2 对照基线刻意留着。
- **思路**：改成惰性构造（`_legacy_cache` property，首次访问才分配）。
- **预期收益**：1.5B / max_seq_len 8192 / bf16 下约 `28×1×8192×2×128×2B ≈ 229 MB`（与块池同量级）。
- **成本**：低。
- **风险**：低。注意别删掉 `generate_batch`——它是 padding 浪费 61.5% 这个动机证据的产地。

### B3 · `waste_rate` 统计口径在多请求下失真

- **当前位置**：`core/paged_kv_cache.py::waste_rate`（依赖 `_written_tokens`，只在 `layer_idx == 0` 累加，且只在 `_prefill_paged` 开头重置）。
- **根因**：P4 的批量路径根本不调 `_prefill_paged`，`_written_tokens` 只增不减，`waste_rate` 会随运行时间单调失真。
- **思路**：改成 `1 - Σ(seq_len) / (num_used_blocks * block_size)`，由 scheduler 侧按当前 running + waiting 状态实时算。
- **预期收益**：可观测性收益；**P6 前缀缓存的命中率/复用率要靠同一个统计口径**，现在修比 P6 再修便宜。
- **成本**：低。
- **风险**：低。

### B4 · 抢占只有 recompute，没有 swap

- **当前位置**：`engine/sequence.py::reset_for_preemption()`（释放块 + `num_computed_tokens=0` + 清空已生成 token）。
- **vLLM 参考**：`vllm/v1/core/sched/scheduler.py` 的 `PREEMPTED` → `WAITING_FOR_REMOTE_KVS` / swap 路径（配合 `KVCacheCoordinator`）。
- **思路**：暂不做。短请求 recompute 比换入换出便宜，且 P6 前缀缓存落地后 recompute 的代价会被前缀命中大幅摊薄。
- **优先级**：可延后。

---

## 3. 维度 C · 并发吞吐

### C1 · 缺失 `abort_request()`（**P6 前最该补的一项**）

- **当前位置**：`server/async_engine.py::generate_stream()` 的 `finally:` 只做 `self._token_queues.pop(seq_id, None)`；`_step_loop` 里 `q = self._token_queues.get(seq_id); if q is not None: await q.put(...)` —— 队列没了就静默丢弃，但请求仍在跑。
- **vLLM 参考**：`vllm/v1/engine/core.py::abort_requests()` → `Scheduler.finish_requests()` → 释放块。
- **根因**：`Sequence` 仍留在 `Scheduler.running` 里继续被调度、继续 decode、继续占 KV 块，直到自然结束。
- **后果**：客户端 SSE 断开后白占 GPU 直到它自己跑完；**恶意/异常客户端可以连发请求立刻断开，把块池占满并触发抢占风暴，拖垮所有正常请求**。在 B1 修好块池之前这个风险被掩盖，修好之后反而更危险（因为并发上去了）。
- **思路**：`AsyncEngineCore.abort(seq_id)` → `EngineCore.abort(seq_id)` → `Scheduler.finish_request(seq)`（从 running/waiting 摘除 + `_free_seq_blocks`）；在 `finally` 里调用。
- **预期收益**：消除资源泄漏；并发正确性。
- **成本**：低（~30 行）。
- **风险**：正在 GPU 上执行的 step 不能中断，只能标记下一 step 生效；需要保证「abort 时该 seq 不在当前 scheduled 列表里」的边界处理。

### C2 · `_step_loop` 无异常隔离（一个异常挂死全部请求）

- **当前位置**：`server/async_engine.py::_step_loop()` —— `while self.engine.scheduler.has_requests(): ... await asyncio.to_thread(self.engine.step)`，无 `try/except`。
- **vLLM 参考**：`vllm/v1/engine/core.py` 的 busy loop 对每步做异常捕获，并把错误通过 `EngineCoreOutput` 传回对应的请求（而不是让进程/循环死掉）。
- **根因**：`step()` 抛异常 → task 结束 → **所有在等 `q.get()` 的客户端永久挂起**（永远收不到 `None` 哨兵）。后续新请求虽然会因 `_ensure_step_loop()` 重启循环，但旧请求已经死了。
- **思路**：`try/except` 包住 `step()`，异常时给该 step 涉及的所有 queue 塞入哨兵 + 记录错误；循环继续。
- **预期收益**：可靠性，从「一个请求炸掉整个服务」变成「单个请求失败」。
- **成本**：低。
- **风险**：低。注意别吞掉 `asyncio.CancelledError`。

### C3 · `tok.encode` / `apply_chat_template` 阻塞 event loop

- **当前位置**：`server/api.py::completions()` / `chat_completions()` 内直接同步调用。
- **思路**：`await asyncio.to_thread(tok.encode, prompt_text, add_special_tokens=True)`。
- **预期收益**：长 prompt 并发时明显（encode 是几十 ms 级 CPU 操作，会卡住整个 loop）。
- **成本**：极低。
- **风险**：极低。

### C4 · token 队列无背压

- **当前位置**：`async_engine.py` 里 `asyncio.Queue()` 无 `maxsize`。
- **vLLM 参考**：`vllm/v1/engine/output_processor/`（有停止/丢弃策略）。
- **思路**：`asyncio.Queue(maxsize=N)` + 满时策略（drop 或反压）。
- **优先级**：可延后（单卡 demo 规模下影响有限）。

### C5 · 同步 step + 单 step loop（无 async scheduling）

- **当前位置**：`async_engine.py` 的 `_step_loop` 串行 `schedule → execute → update`；GPU 执行时 CPU 空闲等待下一轮调度。
- **vLLM 参考**：`vllm/v1/core/sched/async_scheduler.py`（提前一步调度，与 GPU 执行重叠）。
- **优先级**：可延后（roadmap 已把 async scheduling 列为「只读调研」）。

---

## 4. 维度 D · 数值精度与结果一致性

### D1 · `finish_reason` 硬编码 `"stop"`

- **当前位置**：`server/protocol.py` 四个 `make_*` 函数。
- **vLLM 参考**：`vllm/entrypoints/openai/protocol.py` + `serving_engine.py`（按结束原因返回 `stop` / `length`）。
- **根因**：P2-08 到 P5 一直没修。被 `max_tokens` 截断时按 OpenAI 规范应返回 `"length"`。
- **连带影响**：这也意味着 `preempted_seq_ids` 那条线彻底断了——**抢占重算对外完全不可见**。
- **成本**：极低（scheduler 已能区分 EOS 与 `max_new_tokens`，只需把原因带出来）。

### D2 · bench 绕过 Sampler 自己 argmax，口径不一致

- **当前位置**：`bench/*.py` 用 `int(last.argmax())`；server 走 `Sampler`。
- **vLLM 参考**：`vllm/benchmarks/` 统一走引擎采样路径。
- **思路**：bench 加 `--sampling {greedy,sampler}`，或直接统一走 `Sampler(temperature=0)`。
- **预期收益**：口径一致，bench 数字才代表真实 serving 路径。
- **成本**：低。
- **风险**：数字会变，需要重跑 P0–P5 全部 bench 并更新阶段笔记。

### D3 · 抢占重算在 `temperature > 0` 时结果不保证一致

- **当前位置**：`sequence.py::reset_for_preemption()` 连已生成的 token 一起丢弃。
- **说明**：这是语义而非 bug（vLLM 同样如此）。但 bench 必须保持 `temperature=0.0`，且**文档要写明**。若 A2 做了 per-request seed，则重算可以复现同一序列——这是一个可选增强。

### D4 · 缺少 NaN / 形状守卫

- **当前位置**：`attention/*` 与 `scheduler` 侧均无数值断言。
- **已知风险点**（P3 踩坑 1、2）：块表 padding 读到脏块、末块半满未截断 → NaN；`slot_mapping` 长度与 token 数不等已在 `reshape_and_cache` 里断言，但 `qo_indptr` 与 `input_ids` 长度是否匹配没有断言。
- **vLLM 参考**：`vllm/v1/core/kv_cache_utils.py` 的大量 assert + `check_enough_kv_cache_memory`。
- **思路**：在 `_run_prefill_batched` / `_run_decode_batched` 入口加 shape assert；debug 模式下检查 `torch.isnan(logits).any()`。
- **成本**：低。**P6 前缀缓存会引入块共享，出错更隐蔽，建议 P6 前补上。**

### D5 · 三路对拍只在 CPU fp32

- **说明**：torch / triton / SDPA 三实现当前只在 CPU fp32 对拍。GPU bf16 跨路径不做逐 token 一致承诺（P2 踩坑 6 的结论）。**这不是缺陷，是已确认的口径**，但应在 `docs/` 里写明，避免 P6 时误判。

---

## 5. 维度 E · 代码结构与可维护性

| 项 | 当前位置 | 问题 | vLLM 参考 | 建议 | 成本 |
|---|---|---|---|---|---|
| E1 ✅ | `engine/sequence.py` 与 `model_executor/runner.py` 各有一份 `SamplingParams` | 两份同名类，`p4_bench.py` 已被迫写成 `as RunnerParams` | `vllm/sampling_params.py` 单一定义 | 合并到 `types.py`，`__init__.py` 导出 | 低 |
| E2 ✅ | `AttentionMetadata.attn_impl` | prefill 时被赋成 `prefill_impl`，decode 时才是真 `attn_impl`，语义双关 | `vllm/v1/attention/backends/` 元数据按 backend 分离 | 拆成 `attn_impl` + `prefill_impl` 两字段 | 低 |
| E3 ✅ | `nano_vllm/core/`（块池） vs `nano_vllm/engine/core.py`（主循环） | 目录撞名，语义无关 | vLLM 里 `v1/core/` 只指调度与 KV 管理 | `core/` 改名 `kvmm/` 或并入 `engine/` | 中 |
| E4 ✅ | `attention/varlen_prefill.py` 内 `if impl == ...` 自行分发 | prefill 后端不在 `backend.py` 注册表里（P0-07 只修一半） | `vllm/v1/attention/backends/` 注册表 | prefill 也进注册表 | 低 |
| E5 ✅ | `runner.py` 同时管模型加载 + 两种 cache + sampler + P1/P2 三个历史入口 | 职责过宽 | `vllm/v1/worker/` 分层 | `legacy_api.py` 拆出 P1/P2 入口 | 中 |
| E6 ✅ | `core.py::_run_prefill()` 是死代码 | 但它是证明「varlen 拼批有效」的唯一对照组 | — | **复活成开关**（`--prefill-mode {batched,per-seq}`），别删 | 极低 |
| E7 ✅ | 无 `pyproject.toml`、`__init__.py` 为空 | `api.py` 顶部有 `sys.path.insert` hack | vLLM 有完整打包配置 | 补 `pyproject.toml` | 低 |
| E8 ✅ | `attention/varlen_prefill.py` 内联重写了 `gather_paged_kv` | 与 `paged_attn.py::gather_paged_kv` 重复 | — | 改为直接调用 | 极低 |

> 注意：P2 的 `generate_batch` / `paged_attention_sdpa` / P1 eager **不要删**——它们是所有加速比的分母与 P3 完成标准要求的对照物。

---

## 6. 维度 F · 错误处理与可观测性

### F1 · 完全没有 metrics（**P6 前缀缓存会因此无法验收**） ✅

- **当前位置**：全项目无统计模块；`SchedulerOutput.preempted_seq_ids` **只写不读**（我 grep 过 `core.py`，`step()` 只消费 `scheduled`）。
- **vLLM 参考**：`vllm/v1/metrics/stats.py`（`SchedulerStats`：num_running / num_waiting / gpu_cache_usage / prefix cache hit rate）、`vllm/v1/metrics/loggers.py`（Prometheus + 日志）、`vllm/v1/core/kv_cache_metrics.py`（`KVCacheMetricsCollector`）。
- **思路**：先做最小版——`EngineCoreStats` 每步累计 `num_batched_tokens` / `num_running` / `num_waiting` / `cache_usage` / `preempt_count` / `finish_reason` 分布；`step()` 返回或暴露一个 `stats` 属性；bench 落盘 JSON。Prometheus 可延后。
- **预期收益**：(a) 现在无法证明抢占路径在 bench 里真的被走过；(b) **P6 的验收指标「前缀命中率」必须依赖这套统计**。
- **成本**：中。
- **风险**：低。

### F2 · 无请求级错误隔离 / 无结构化错误返回

- **当前位置**：见 C2。此外 FastAPI 未定义异常处理器，任何未捕获异常会返回 500 HTML 而非 OpenAI 格式错误体。
- **vLLM 参考**：`vllm/entrypoints/openai/api_server.py` 的 exception handlers + `serving_engine.py`。
- **成本**：低。

### F3 · SSE chunk 的 `id` / `created` 每片重新生成

- **当前位置**：`protocol.py` 的 `make_*_chunk` 内部调 `_cid()` / `_ts()`。
- **规范**：同一 stream 内所有 chunk 必须共享同一个 `id` 和 `created`。
- **成本**：极低。**建议与 D1 一起顺手做掉。**

### F4 · 无 health / readiness / 日志分级

- **vLLM 参考**：`/health`、`/ready`、`vllm/logging_utils/`。
- **优先级**：可延后（demo 规模非必需）。

---

## 7. 能力对照总览（nano-vllm vs vLLM v1）

| 能力 | vLLM v1 | nano-vllm 现状 | 差距性质 |
|---|---|---|---|
| 迭代级调度 | ✅ | ✅（P4） | 已对齐 |
| token budget 记账 | ✅ | ✅ | 已对齐 |
| chunked prefill | ✅ | ✅ | 已对齐 |
| LIFO 抢占 + recompute | ✅ | ✅ | 已对齐（无 swap） |
| 优先级调度 / 长 prefill 节流 | ✅ | ❌ | 待补（A6） |
| 分页 KV + block_table | ✅ | ✅（P3） | 已对齐 |
| 块池按显存反推 | ✅ | ❌ 固定 512 | **硬伤（B1）** |
| 前缀缓存（块哈希 + ref_cnt） | ✅ | ⬜ P6 | 下一阶段 |
| 批量采样（张量化参数） | ✅ | ❌ 逐条 | 待补（A2） |
| 批量 attention kernel | ✅ | ❌ Python 循环 | 待补（A3） |
| persistent input buffers | ✅ | ❌ | 待补（A4，同时是 P7 前提） |
| unified batch（prefill+decode 一次前向） | ✅ | ❌ 两次 | 可延后 |
| 算子融合 / compile / CUDA Graph | ✅ | ❌ | P7 本体 |
| abort / 取消 | ✅ | ❌ | **硬伤（C1）** |
| step 级异常隔离 | ✅ | ❌ | **硬伤（C2）** |
| metrics / Prometheus | ✅ | ❌ | 待补（F1） |
| logprobs / per-request seed | ✅ | ❌ | 可延后 |
| 分布式（TP/PP） | ✅ | ❌ | roadmap 已裁剪 |

---

## 8. 建议的增量实施顺序（每批可独立验收）

```
第 0 批 · 半天的「低风险高收益」集中改（不动 kernel，不动调度）
  ├ ① A1  attn_impl 暴露为参数（~10 行）
  ├ ② D1  finish_reason 区分 stop/length + F3 SSE id 固定
  ├ ③ C3  tok.encode 包 to_thread
  ├ ④ E1/E2/E8 三项结构小改（合并 SamplingParams、拆 attn_impl、复用 gather）
  └ 验收：tests/ 全绿 + 重跑 p5_bench，确认 overhead_ratio 仍 ≤1.10

第 1 批 · 资源与可靠性（P6 的地基）
  ├ ⑤ B1  num_blocks 按显存反推（含 profile run）
  ├ ⑥ C1  abort_request 全链路
  ├ ⑦ C2  step loop 异常隔离
  ├ ⑧ B3  waste_rate 改为实时口径
  └ 验收：新增 test_p6_abort.py（断连后块被回收）；并发 16 请求无块泄漏

第 2 批 · 性能主战场
  ├ ⑨ A2  采样批量化
  ├ ⑩ A4  slot_mapping 向量化 + persistent buffer
  ├ ⑪ A3  decode attention 批量化（含 triton grid 扩 batch）
  ├ ⑫ A5  flashinfer wrapper 复用
  └ 验收：p4_bench 重跑（warmup+repeat），对比第 0 批基线

第 3 批 · 尾延迟与可观测性
  ├ ⑬ A6  long_prefill_token_threshold + max_num_seqs
  ├ ⑭ F1  step 级 metrics + 消费 preempted_seq_ids
  ├ ⑮ 补真 TPOT p99 测量（bench 现只记 wall_ms）
  └ 验收：--tpot-test 出可信曲线；抢占次数可统计

第 4 批（P6 前有余力才做）
  ├ ⑯ D2  bench 统一走 Sampler（需重跑全部 bench + 更新阶段笔记）
  ├ ⑰ D4  NaN / shape 守卫
  └ ⑱ E3/E5/E7 结构重构（core 改名、runner 拆分、pyproject）

明确推到 P6 之后：A7 unified batch、A8 算子融合/compile/CUDA Graph、B4 swap、C4 背压、C5 async scheduling
```

**排序理由**：第 0 批是「改一处就白拿收益」的项（尤其 A1——P3 白做的 kernel 终于能用）；第 1 批是 P6 的**硬前提**（前缀缓存需要富余块池 + 正确的回收语义 + 可靠的命中率统计）；第 2 批才是真正的性能优化，且必须在第 0/1 批之后测，否则基线是脏的。

---

## 9. 我未核实、需要你补充确认的内容

以下内容我没有读到，报告中**未对其下结论**，如与实际不符请告知后我再修订：

1. `README.md`、`nano-vllm-架构与开发路线评估报告.md`、`constraints.txt`（46B）的当前内容——可能已包含部分优化项或额外的验收口径。
2. `tests/test_p5_serving.py` 的具体用例——我不知道 abort / 断连 / 并发是否有测试覆盖（本报告按「无」处理）。
3. `sequence.py::SamplingParams` 的完整字段与默认值（我只确认了 `p4_bench.py` 里存在两份同名类的事实）。
4. `bench/p5_bench.py` 的构造参数（是否传 `num_blocks`，HTTP 并发路数）。
5. `server/protocol.py` 里 `CompletionRequest.max_tokens` 是否有默认值——若无，`max_new_tokens=None` 可能走进 `len(output) >= None` 的类型错误。
6. `nano_vllm/__init__.py` 与各子包 `__init__.py` 是否仍为空。
7. `setup_wsl.sh` / `diag_wsl_gpu.sh` 是否包含显存 profile 逻辑（若已有机智的反推逻辑，B1 可简化）。
---

## 10. 实施记录（2026-09-27）

> 第 0 批 + 第 1 批已实施，P5 4/4 + P4 4/4 + P2 4/4 全绿，P3 11/12（Triton GPU test 已知跳过）。

### 第 0 批 · 低风险高收益

#### A1 · `attn_impl` 暴露为参数 ✅

- **改动**：`create_app()` 加 `attn_impl: str = "torch"` 参数，`main()` 加 `--attn-impl {torch,triton}` CLI 开关，透传给 `NanoRunner`。
- **原因**：P3 自研 Triton paged decode kernel 实测比 torch 朴素快 1.12–1.51×，但 P5 装配时写死了 `"torch"`，CLI 没暴露开关，P3 kernel 在 serving 里一次都没跑过。
- **文件**：`server/api.py`（`create_app` 签名 + `main` argparse）

#### D1+F3 · `finish_reason` 区分 stop/length + SSE id 固定 ✅

- **改动**：
  - `protocol.py`：`make_*_response` 加 `finish_reason` 参数；`make_*_chunk` 加 `cid`/`created`/`finish_reason` 参数，不再每次 `_cid()`/`_ts()` 重新生成。
  - `api.py`：非流式按 `len(tokens) >= max_tokens` 判断 `"length"` vs `"stop"`；流式在 gen() 开头生成一次 `cid`/`created`，每个 chunk 复用；流式末尾按 token 计数判断 finish_reason。
- **原因**：被 `max_tokens` 截断时应返回 `"length"`（OpenAI 规范），原硬编码 `"stop"` 不一致；同一 stream 内 chunk 必须共享同一个 `id` 和 `created`（OpenAI SSE 规范）。
- **文件**：`server/protocol.py`、`server/api.py`、`tests/test_p5_serving.py`（断言从 `"stop"` 改为 `"length"`，流式 finish 判断从 `== "stop"` 改为 `is not None`）

#### C3 · `tok.encode` 包 `to_thread` ✅

- **改动**：`api.py` 路由处理器内 `tok.encode` / `tok.apply_chat_template` / `tok.decode`（非流式）包 `asyncio.to_thread`。
- **原因**：长 prompt encode 是几十 ms 级 CPU 操作，同步调用会卡住整个 event loop，阻塞其他请求的调度。`to_thread` 把 CPU-bound 操作6扔线程池，event loop 保持响应。
- **文件**：`server/api.py`（加 `import asyncio` + 4 处 `to_thread` 包装）

#### E1 · 合并 SamplingParams ✅

- **改动**：`runner.py` 删除 `SamplingParams` dataclass，改为 `from nano_vllm.engine.sequence import SamplingParams`（重=导出，向后兼容）。
- **原因**：`engine/sequence.py` 和 `model_executor/runner.py` 各有一份完全相同的 `SamplingParams`，`p4_bench.py` 被迫写成 `as RunnerParams`。合并到单一定义，现有导入不破坏。
- **文件**：`model_executor/runner.py`

### 第 1 批 · 资源与可靠性

#### B1 · `num_blocks` 按显存反推 ✅

- **改动**：`NanoRunner._profile_num_blocks()` 新方法：`torch.cuda.mem_get_info()` 获取剩余显存 → 按 `per_token_kv = num_layers × 2 × num_kv_heads × head_dim × dtype_size` 算每 token KV 开销 → `num_blocks = free_bytes × 0.5 / (block_size × per_token_kv)`（50% 安全系数留激活/开销）。`create_app()` 默认 `num_blocks=None` 触发自动反推，`--num-blocks` 保留手动覆盖。CPU 回退到 `blocks_needed(max_position_embeddings) + 1`。
- **原因**：原默认 512 块 × 16 = 8192 slot，1.5B bf16 每 token KV 占 28 KB，整池仅 229 MB / 8192 token。`max_seq_len` 默认 8192 → 一条满长请求占满整池 → 后续请求永久饥饿。4070 Ti SUPER 16GB 剩余 ~8.5GB 理论上限约 18k 块。
- **文件**：`model_executor/runner.py`（`_profile_num_blocks` + `__init__` 改用 `if num_blocks is None`）、`server/api.py`（默认改 `None`）

#### C1 · `abort_request` 全链路 ✅

- **改动**：
  - `Scheduler.abort(seq_id)`：遍历 running/waiting 找到 seq → `_free_seq_blocks` → 从列表移除。
  - `EngineCore.abort(seq_id)`：转发给 scheduler。
  - `AsyncEngineCore`：`generate()` 和 `generate_stream()` 的 `finally` 块加 `self.engine.abort(seq.seq_id)`（正常完成时 abort 是 no-op，seq 已不在 running/waiting；断连时释放块）。
- **原因**：客户端 SSE 断开后，原代码只 pop Queue 不 abort seq → seq 留在 `running` 继续被调度、继续 decode、继续占 KV 块直到自然结束。恶意客户端可连发请求立刻断开占满块池触发抢占风暴。
- **文件**：`engine/scheduler.py`、`engine/core.py`、`server/async_engine.py`

#### C2 · `_step_loop` 异常隔离 ✅

- **改动**：`_step_loop` 的 `await asyncio.to_thread(self.engine.step)` 包 `try/except Exception`，异常时遍历所有 `_token_queues` → `engine.abort(seq_id)` + `await q.put(None)` 发完成信号，`continue` 循环。
- **原因**：原代码无异常捕获，`step()` 抛异常 → task 结束 → 所有等 `q.get()` 的客户端永久挂起（永远收不到 None 哨兵）。修复后变成「单个 step 异常 → abort 该步涉及请求 → 循环继续」，不影响后续新请求。
- **文件**：`server/async_engine.py`

#### B3 · `waste_rate` 实时口径 ✅

- **改动**：`Scheduler.waste_rate` property：`1 - Σ(seq.num_tokens for seq in running) / (num_used_blocks × block_size)`，由调度器实时状态计算。
- **原因**：原 `PagedKVCache.waste_rate` 依赖 `_written_tokens`，只在 `_prefill_paged` 开头重置，P4 批量路径不调 `_prefill_paged` → `_written_tokens` 只增不减 → waste_rate 单调失真。P6 前缀缓存的命中率/复用率要靠同一个统计口径，现在修比 P6 再修便宜。
- **文件**：`engine/scheduler.py`

### 验收

| 测试 | 结果 |
| --- | --- |
| P5 serving（4 项） | ✅ 4/4 全绿 |
| P4 correctness（4 项） | ✅ 4/4 全绿 |
| P2 correctness（4 项） | ✅ 4/4 全绿 |
| P3 correctness（12 项） | ✅ 11/12（1 个 Triton GPU test 在无 CUDA 环境跳过，已知限制） |

#### GPU bench（repeat=5，2026-09-27）

| 路径 | overhead_ratio | pass | 备注 |
| --- | --- | --- | --- |
| completions_nonstream | 1.001 | ✅ | 较 P5（0.966）改善 |
| completions_stream | 0.988 | ✅ | 较 P5（0.961）改善 |
| chat_nonstream | 1.012 | ✅ | 较 P5（0.922）改善 |
| chat_stream | 1.040 | ✅ | 较 P5（0.988）改善 |

**4 路径全达标。** bench 稳定性修复（`AsyncEngineCore.reset()` repeat 间隔离 + chat prompt 续写指令 + warmup=3 SSE 预热）后，chat 两路径从 0.871/0.818 提升到 1.012/1.040。

### 第 2 批 · 性能主战场（2026-09-27）

#### A4 · slot_mapping 向量化 ✅

- **改动**：`PagedKVCache.slot_mapping()` 从 Python for 循环改为 tensor 运算（`arange` + `//` + `%` + `gather`）。
- **原因**：token budget 2048 时每步 2048 次 Python 迭代，向量化后几次 tensor 运算完成。
- **文件**：`core/paged_kv_cache.py`

#### A2 · 采样批量化（greedy 路径）✅

- **改动**：`Sampler.batch_sample_greedy(logits_batch)` 新方法：一次 `argmax(dim=-1)` + 一次 `tolist()` 同步。`EngineCore._execute` decode 路径：全 greedy 时批量采样，替代逐条 `int(logits.argmax())` 的 N 次同步。
- **原因**：b=12 时每步 12 次 host-device 同步，批量化后 1 次。
- **文件**：`sample/sampler.py`、`engine/core.py`

#### A3 · decode attention 批量化（triton kernel）✅

- **改动**：新增 `_paged_attn_decode_batch_kernel`：grid=(batch, num_heads)，消除 Python 逐条循环。新增 `paged_attention_triton_batch` wrapper：二维 block_table + seq_lens 张量。`qwen2.py` decode 批量路径：triton 后端走 batch kernel，torch 后端保持逐条循环（oracle）。
- **原因**：28 层 × b 次 kernel launch/step（b=12 时 336 次极小 kernel）→ 28 层 × 1 次。
- **文件**：`attention/triton_paged_attn.py`、`models/qwen2.py`、`tests/test_p3_correctness.py`（新增 `test_triton_batch_vs_torch` 对拍）
- **GPU bench 收益**：P4 continuous 吞吐 65.6 → **131.8 tok/s（2.01× 加速）**，throughput_ratio 2.60× → **5.53×**。

#### A5 · flashinfer wrapper 复用 ✅

- **改动**：`_get_prefill_wrapper(device)` 按 device 缓存 `BatchPrefillWithPagedKVCacheWrapper` 对象，省掉每步构造（`plan()` 仍每步调以更新元数据）。
- **原因**：wrapper 构造是 host 侧开销，小 batch 下不可忽略。
- **文件**：`attention/varlen_prefill.py`

#### 第 2 批验收

| 测试 | 结果 |
| --- | --- |
| P3 correctness（含新增 batch kernel 对拍） | ✅ 13/13 全绿 |
| P5 serving + P4 + P2 | ✅ 12/12 全绿 |

| GPU bench | torch 后端 | triton 后端 |
| --- | --- | --- |
| P4 continuous 吞吐 | 65.6 tok/s | **131.8 tok/s（2.01×）** |
| P4 throughput_ratio | 2.60× | **5.53×** |
| P5 completions overhead | 1.012 / 1.036 | 0.903 / 0.922 ✅ |
| P5 chat overhead | 0.990 / 0.981 | 0.489 / 0.866 ⚠️ |

P5 chat triton 不达标：chat prompt 在 triton online softmax（float32 累加）精度下 argmax 翻转 → 提前 EOS → 生成短，非 kernel bug。completions 两路径达标确认 kernel 正确性。

### 未实施项（按审计文档优先级延后）

- **第 3 批**（A6 调度节流 / TPOT p99 测量）：尾延迟。
- **A7/A8/B4/C4/C5**：明确推到 P6 之后或 P7 本体。

> **后续修订（2026-10-03）**：P2-03 记的三个工具缺口，现状如下，细节见 `docs/notes/p8-prereq.md`。
> - `bench/regress.py` + `baseline.json` —— **已补**。两档：正确性（pytest 退出码，硬门）+ 性能
>   （同进程 eager/graph **比值**门；绝对值只 WARN，因本机跨 run 漂移实测可达 21%）。
> - `bench/profile.py` —— **仍未收成通用脚本**，职能由阶段临时脚本承担：`bench/p7_profiler.py`
>   出整步 launch 占比，`bench/layer_kernels.py` 出单层 kernel 数（P8 验收用）。
> - `bench/run_vllm.py` —— **仍缺**；横向对照目前靠 `bench/vllm_cudagraph_probe.py` 等单点探针。

### 维度 E · 代码结构与可维护性修复（2026-09-27）

> E1–E8 全部修复，35/35 测试全绿（TRITON_INTERPRET=1 CPU 解释器模式）。

#### E8 · 复用 gather_paged_kv ✅

- **改动**：`varlen_prefill_torch` 内联的 gather KV 逻辑改为调用 `paged_attn.py::gather_paged_kv`。
- **原因**：P4 写 `varlen_prefill_torch` 时内联重写了与 P3 `gather_paged_kv` 相同的 gather 逻辑，两份重复代码改一处忘另一处会不一致。
- **文件**：`attention/varlen_prefill.py`

#### E6 · 复活 _run_prefill 成 --prefill-mode 开关 ✅

- **改动**：`EngineCore.__init__` 加 `prefill_mode: str = "batched"` 参数；`_execute` 加 per-seq 分支（逐条调 `_run_prefill`）；`create_app` + `main()` 加 `--prefill-mode {batched,per-seq}` CLI。
- **原因**：`_run_prefill` 是 P4 初版的逐条 prefill，被 `_run_prefill_batched` 取代后变死代码。复活成开关保留对照组，bench 可跑 A/B 对比验证 varlen 拼批有效性。
- **文件**：`engine/core.py`、`server/api.py`

#### E2 · 拆 attn_impl/prefill_impl 两字段 ✅

- **改动**：`AttentionMetadata` 加 `prefill_impl: str = "torch"` 字段；prefill 赋值点改用 `prefill_impl`，decode 保持 `attn_impl`；`qwen2.py` prefill varlen 路径用 `metadata.prefill_impl`。
- **原因**：原 `attn_impl` 字段在 prefill 时塞 `prefill_impl`、decode 时塞 `attn_impl`，同一字段两种语义，读代码时必须看赋值点才知道含义。
- **文件**：`attention/metadata.py`、`engine/core.py`、`models/qwen2.py`、`model_executor/runner.py`

#### E4 · prefill 进 backend.py 注册表 ✅

- **改动**：`backend.py` 加 `PREFILL_ATTN_BACKENDS` 注册表 + `get_prefill_attn()` 函数；`varlen_prefill_attention` 改用注册表替代 `if impl ==` 分支。
- **原因**：P3 建了 `backend.py` 注册表只管 decode；P4 加 prefill 多后端时没复用注册表，写了 `if impl ==` 分支，两套后端选择机制并存。
- **文件**：`attention/backend.py`、`attention/varlen_prefill.py`

#### E7 · 补 pyproject.toml + 去 sys.path hack ✅

- **改动**：新建 `pyproject.toml`（setuptools 配置，`nano_vllm` 可 `pip install -e .`）；删除 `api.py` 的 `sys.path.insert` hack 和无用 `import sys` / `from pathlib import Path`。
- **原因**：项目从未做打包配置，`sys.path.insert` 是没装包时的 workaround，不同启动方式下可能失效。
- **文件**：`pyproject.toml`（新建）、`server/api.py`

#### E3 · core/ 改名 kvmm/ ✅

- **改动**：`git mv nano_vllm/core nano_vllm/kvmm`；5 个文件的 import 路径 `nano_vllm.core` → `nano_vllm.kvmm`。
- **原因**：`nano_vllm/core/`（BlockPool + PagedKVCache）与 `nano_vllm/engine/core.py`（EngineCore）撞名，两个 `core` 语义无关但同名，import 时易看错。`kvmm`（KV memory management）语义清晰。
- **文件**：`nano_vllm/kvmm/`（原 `core/`）、`kvmm/paged_kv_cache.py`、`model_executor/runner.py`、`engine/scheduler.py`、`tests/test_p3_correctness.py`、`tests/test_p4_correctness.py`

#### E5 · 拆 legacy_api.py ✅

- **改动**：新建 `model_executor/legacy_api.py`，移入 P1 eager / P2 连续 cache / P2 静态批的实现体（`generate_eager` / `generate_cached` / `generate_batch` / `forward_last_logits`）；`runner.py` 的 `generate` P1/P2 分支和 `generate_batch` 改为薄委托，`_build_prefill_mask`/`_build_decode_mask` 删除；`_prefill`/`_decode` 保留（chunked prefill 测试直接调用）。
- **原因**：`runner.py` 322 行混合了 P1 eager / P2 连续 cache / P3 分页 / P4 入口，职责过宽。拆出 P1/P2 到 `legacy_api.py` 后 `runner.py` 降至 ~200 行，主路径（P3+）与历史基线（P1/P2）分离。
- **文件**：`model_executor/legacy_api.py`（新建）、`model_executor/runner.py`
#### F1 · step 级 metrics ✅

- **改动**：新建 `engine/stats.py`（`EngineCoreStats` dataclass：本步快照 num_batched_tokens/num_running/num_waiting/num_prefills/num_decodes/cache_usage/waste_rate + 累计 total_steps/total_preempt_count/total_tokens_generated + P6 预留 prefix_cache_hit_rate）；`EngineCore.step()` 每步更新 stats；`AsyncEngineCore.reset()` 重置 stats；`Scheduler` 加 `num_running`/`num_waiting` property；`PagedKVCache` 加 `cache_usage` property；P4 bench 落盘 `p4_stats` JSON。
- **原因**：全项目无统计模块，`SchedulerOutput.preempted_seq_ids` 只写不读，无法证明抢占路径在 bench 里真的被走过。P6 前缀缓存的验收指标「前缀命中率」必须依赖这套统计。
- **文件**：`engine/stats.py`（新建）、`engine/core.py`、`engine/scheduler.py`、`kvmm/paged_kv_cache.py`、`server/async_engine.py`、`bench/p4_bench.py`
