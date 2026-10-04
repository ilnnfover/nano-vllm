# P8 ① · 权重预拼接（QKV 3→1、gate/up 2→1）

> 日期：2026-10-04　对应 roadmap `nano-vllm-roadmap-v2.md:145` 的第一条
> 「加载时按 config 预拼接 QKV（3 GEMM→1）与 gate/up（2→1）」。
> 目的：把每层 7 次独立 GEMM 收敛到 4 次，减少 kernel launch 数（P8 的硬指标）。

---

## 1. 结论（先给数）

| 指标 | 拼接前 | 拼接后 | 变化 |
| --- | --- | --- | --- |
| **单层 kernel 数**（triton + 图） | **42.0** | **38.0** | **−4.0（−9.5%）** |
| 单层 kernel 数（triton eager） | 43.0 | 39.0 | −4.0 |
| 整步 kernel 数（batch=8） | 1198 | 1086 | −112 |
| 层外 kernel（embedding/rotary/final norm/lm_head/采样） | 22 | 22 | 不变 |
| 参数量 | 1,543,714,304 | 1,543,714,304 | 0（无冗余副本） |

没有达到 roadmap 的「≥30%↓」目标——那是四项之和，本项只贡献 −4/层。

> **2026-10-04 更新（②「融合 RMSNorm」落地后重测）**：上表的绝对值是在 ② 落地**之前**测的。
> ②（含 ②-b）落地后，用同一脚本重测的单变量对照为 **26.0 → 22.0（仍为 −4.0）**；
> `bench/results/layer_kernels_noprejoin.json` 已刷新为前者，以保证「消融文件 vs 当前默认」
> **只差一个开关**（而不是差两个）。当前默认（① + ②-b）为 **22.0 个/层**，
> P8 起点 42.0 ⇒ **累计 −47.6%**。见 `docs/notes/p8-fused-norm.md`（②-a）与 `p8-triton-norm.md`（②-b）。
> （上表 42 → 38 是 ① 刚落地时的快照；`noprejoin`/`layer_kernels` 两个文件名固定、
> 内容随默认配置演进而刷新，所以当前读数与该快照不同，但**差值恒为 −4.0**。）

---

## 2. 改了什么

### 2.1 模型侧 `nano_vllm/models/qwen2.py`

| 位置 | 改动 |
| --- | --- |
| `_prejoin_state_dict()`（新增，模块级） | 加载期把 HF 的 `q/k/v_proj`、`gate/up_proj` 权重按 `dim=0` 拼成 `qkv_proj` / `gate_up_proj`，并**消费掉原始键** |
| `Attention.__init__(config, prejoin=True)` | `prejoin=True` 时只建一个 `qkv_proj`（`out = (heads + 2·kv_heads)·head_dim`，**bias=True**）；否则保持原三投影 |
| `Attention._project_qkv()`（新增） | 两条路径统一产出 `[b, heads, s, head_dim]` 的 q/k/v |
| `MLP.__init__(config, prejoin=True)` | `prejoin=True` 时只建一个 `gate_up_proj`（`out = 2·intermediate`，bias=False） |
| `DecoderLayer` / `Qwen2Model` / `Qwen2ForCausalLM` | 透传 `prejoin` |
| `Qwen2ForCausalLM.load_weights` | 在 `load_state_dict` **之前**做键重映射（`self.prejoin` 为真时） |

### 2.2 打通开关（用于「拼接前后」对照）

| 文件 | 改动 |
| --- | --- |
| `nano_vllm/model_executor/runner.py` | `NanoRunner(..., prejoin: bool = True)`，传给模型 |
| `nano_vllm/server/api.py` | `create_app(..., prejoin)` + CLI `--prejoin/--no-prejoin`；顺带把上次注册的 `sdpa` 补进 `--attn-impl` 的 `choices`（否则 CLI 用不了第三个实现） |
| `bench/layer_kernels.py` | 新增 `--prejoin/--no-prejoin`，使 kernel 对照可复现 |

### 2.3 测试（新增，免模型免 CUDA）

`tests/test_p8_prejoin.py`：8 项，0.9 s 跑完。

| 用例 | 锁住什么 |
| --- | --- |
| `test_join_order_and_bias` | 顺序必须是 `[q,k,v]`/`[gate,up]`，且 **weight 与 bias 都拼** |
| `test_original_keys_consumed` | 原始键必须被消费（否则加载报「未知键」） |
| `test_unrelated_keys_passthrough` | `embed_tokens`/`o_proj`/`down_proj` 原样透传 |
| `test_missing_bias_raises` / `test_missing_gate_raises` | 缺件显式 `KeyError`，**不静默跳过** |
| `test_attention_projection_matches_three_linears` | 与三投影数值等价（CPU fp32） |
| `test_attention_projection_needs_no_copy` | q/k/v 是**切片视图**而非拷贝（用 stride 判定） |
| `test_mlp_matches_two_linears` | 与 gate/up 两段投影数值等价 |

---

## 3. 五个设计决策与依据

### 3.1 合并顺序三处必须一致，且要显式校验

`_prejoin_state_dict` 的 `cat([q,k,v], 0)` → 模型侧 `split([q,k,v], -1)` → `unflatten`。三处顺序一旦不一致：

- **不会崩**，只会静默算错（q 拿到 k 的权重）；
- 而 HF 权重里 `q_proj` 与 `k_proj` 的形状通常不同（本模型 `[1536,1536]` vs `[256,1536]`），所以这类错误**有时侥幸不报错**。

因此做了两条显式校验，**缺件即 raise**，而不是静默跳过（`p8` 的红线：把静默算错转成显式失败，同 D4 思路）。

### 3.2 QKV 的 bias 必须一起拼

Qwen2 的 QKV **带 bias**（`qwen2.py` 里三个 `nn.Linear(bias=True)`），Llama 没有（roadmap 附录坑点 1）。
合并后 bias 变成 `[2048]`，同样按 `[q,k,v]` 排。**weight 拼对了但 bias 漏拼/拼错顺序不会报错**，所以单测专门覆盖。

### 3.3 加载期重映射放在 `load_state_dict` 之前

`load_weights` 里有：

```python
missing, unexpected = self.load_state_dict(weights, strict=False)
if unexpected: raise KeyError(f"权重文件中存在未知键: {unexpected[:5]}")
```

原始的 `q/k/v_proj`（28 层 ×(weight+bias)×3）若不先被消费，会立刻被判定为「未知键」→ **加载直接失败**（这是好事：失败响，不是静默错）。所以重映射必须在 `load_state_dict` **之前**完成。

### 3.4 模型侧用 `unflatten`，不用 `view`；也**不能用 reshape**

`split` 出来的切片**不是连续张量**：合并缓冲区的行 stride 是总宽（2048），而 q 切片只有 1536 列。

| 写法 | 结果 |
| --- | --- |
| `.view(b, s, heads, hd)` | **直接报错**（非连续） |
| `.reshape(...)` | 能用，但会**触发一次拷贝 kernel**——于是「3 GEMM 合 1」省下的开销被拷贝吃回去 |
| **`.unflatten(-1, (heads, hd))`** | 作用在最后一维（该维 stride 恒为 1）→ **零拷贝视图** ✅ |

这条有单测兜底（`test_attention_projection_needs_no_copy` 用 stride 判定；若哪天有人改成 `reshape`/`contiguous`，s 维 stride 会从 `total` 退化成 `head_dim`，测试立刻红）。

### 3.5 做成开关（默认开），不是无条件替换

roadmap 明确要求产出「**拼接前后**各一份对照」，所以必须有 `prejoin=False` 路径；同时它也是数值等价验证的参照臂。
默认 `True`（新路径即生产路径），`--no-prejoin` 用于对照与排查。

**没有**动 `o_proj` / `down_proj`：它们本来就是单个 GEMM，无可合并。

---

## 4. kernel 数变化逐项归因

同进程、同配置（batch=8 / prompt_len=64 / attn_impl=triton / 图开），**只切 `prejoin` 一个变量**：

| kernel | 拼接前 | 拼接后 | Δ | Δ/层 | 机制 |
| --- | --- | --- | --- | --- | --- |
| `cutlass::Kernel2` | 197 | 113 | −84 | **−3.0** | 每层 Linear 数 **7 → 4**（qkv、o、gate_up、down） |
| `cublasLt::splitKreduce_kernel` | 56 | 0 | −56 | **−2.0** | 合并后的 GEMM 不再触发 split-k 规约（附带收益） |
| `elementwise_kernel` 等 4 项 | — | — | +28 | **+1.0** | 净增 1（bias 广播不再被 GEMM epilogue 吸收） |
| `CatArrayBatchedCopy` | 56 | 56 | 0 | 0 | **与预拼接无关**（见 §5） |
| `_paged_attn_decode_batch_kernel` | 28 | 28 | 0 | 0 | attention 未改动 |
| **合计** | 1198 | 1086 | **−112** | **−4.0** | |

数据：`bench/results/layer_kernels_noprejoin.json`（前）/ `layer_kernels.json`（后，当前默认）。

---

## 5. 更正上一份文档的一处错误归因

`docs/notes/p8-prereq.md` 原写：

> `CatArrayBatchedCopy` 2 个/层 = QKV 与 gate/up 的运行时 `cat` ← 权重预拼接要消掉的就是这两个

**这是错的。** 做完本项后实测该 kernel 前后**恒为 56（Δ=0）**；它来自 `qwen2.py` 里
`rotate_half` 的 `torch.cat`（RoPE，每层对 q 与 k 各调用一次），属 **RoPE 项**的收益。
已在 `p8-prejoin.md` 与本笔记中更正。

> 教训：把「运行时 `cat`」和「加载期 `cat`」混为一谈。权重预拼接是**加载期**的（不进 runtime kernel），
> runtime 里的 `aten::cat` 只可能来自模型 forward 本身——grep `torch.cat` 即可定位，当时没做这一步。

---

## 6. 数值验证（四层证据）

### 6.1 结构性：拼出来的权重逐位正确

| 检查 | 结果 |
| --- | --- |
| `qkv_proj.weight` vs `cat([q,k,v])` | **`torch.equal` = True**（形状 `[2048,1536]`） |
| `qkv_proj.bias` vs `cat([q,k,v])` | **True** |
| `gate_up_proj.weight` vs `cat([gate,up])` | **True**（形状 `[17920,1536]`） |
| `o_proj` / `down_proj` | 未参与合并，逐位相同 |
| 参数量 | 1,543,714,304 → 同（无副本、无冗余） |

### 6.2 隔离级：单层投影的数值等价（随机权重、只比 q/k/v）

| dtype | q | k | v |
| --- | --- | --- | --- |
| **fp32** | Δ = **0.0**（逐位相同） | Δ = 4.8e-7（相对 4.2e-7） | Δ = 3.7e-7（相对 3.2e-7） |
| bf16 | Δ = 0.0 | Δ = 7.8e-3（相对 6.9e-3） | Δ = 3.9e-3（相对 3.4e-3） |

fp32 下 q 逐位相同、k/v 差 ~4e-7 ⇒ **纯粹是 GEMM 形状不同带来的舍入**，不是逻辑错误。
（k/v 是 256 宽、q 是 1536 宽，cuBLAS 选到的 tiling 不同。）
MLP 隔离：fp32 与 bf16 **双双 Δ = 0.0**（gate/up 等宽，切分对称）。

### 6.3 整模型级：与 bf16 噪声底对比（关键判据）

对照臂取「同一模型 bf16 相对 fp32」的固有误差（即仓库 `golden/verify_bf16_noise.py` 的判据 A）：

| 层 | 预拼接相对误差 | bf16 噪声底 | 判定 |
| --- | --- | --- | --- |
| 1 | 1.01e-02 | 1.94e-02 | 更小 |
| 2 | 5.54e-03 | 9.08e-03 | 更小 |
| 4 / 8 / 16 | ~4.9e-03 | ~3.5–4.3e-03 | 同量级 |
| 27 | 3.38e-01 | 4.02e-01 | 更小 |
| 28（final norm 后） | 9.11e-02 | 1.15e-01 | 更小 |
| **logits** | **1.15e-01** | **1.69e-01** | **更小** |

**结论：预拼接引入的扰动 ≤ bf16 量化本身的扰动** ⇒ `verify_bf16_noise.py` 的判据「C ≈ A → 正常累积，不是 bug」成立。

> 踩坑提示：最初只看到「层 27 的 max|Δ| = 64」会误判为 bug。实情是**层 27 的幅值本身有 189**，
> bf16 在该层的固有相对误差就有 40%（与预拼接无关）。**绝对差必须除以该层幅值才有意义**。

### 6.4 引擎级：逐 token 与 margin 归因（J2 / J3）

| 判据 | 结果 |
| --- | --- |
| **J2 同路径确定性**（`prejoin=True` 跑两次） | 4/4 请求逐 token 一致 ✅ |
| 跨路径（`prejoin` on vs off） | 4 条里 **1 条完全一致**，3 条在 token 16 / 2 / 2 处翻转 |

按 roadmap P8 修订后的判据（J3：须用 top1/top2 margin 解释翻转），实测首个分歧处：

| 请求 | 分歧@ | ε = max\|Δlogits\| | 本路 margin | margin/ε | 结论 |
| --- | --- | --- | --- | --- | --- |
| 1 | 16 | 9.52e+00 | 1.375 | **0.14** | 翻转在数值差以内 ✅（prejoin 侧 margin = 0，两个候选**完全并列**） |
| 2 | 2 | 3.35e+00 | 1.250 | **0.37** | ✅ |
| 3 | 2 | 6.09e-01 | 0.3125 | **0.51** | ✅ |

三次翻转的 top1/top2 间距都**小于**该步两条路径 logits 的最大差 ⇒ 翻转由数值扰动解释，**非逻辑错误**。

---

## 7. 与 P7 CUDA Graph 的交互

融合改的是模型前向 → **图里烧的 kernel 序列随之变化，旧图必须重捕**。
nano 的图是**懒捕获**（首次 decode 时 capture，见 `DecodeGraphRunner.capture`），
所以换 `prejoin` 只要新建 engine 就会自动重捕，无需额外处理。

实测印证：图档与 eager 档的**层外 kernel 数（22 / 69）在两版之间完全不变**，只有层内核数变了 —— 说明
变的是层内 kernel 序列，不是图的分派结构。

---

## 8. 验证汇总

| 项 | 结果 |
| --- | --- |
| 全量测试 | **138 passed**（原 130 + 新增 8） |
| 新增单测 | `tests/test_p8_prejoin.py` 8/8，0.9 s，免模型免 CUDA |
| `bench/layer_kernels.py` 同构性校验 | 前后两版均 **0.0%** 偏差 |
| `bench/regress.py` 性能门 | PASS（本轮比值 4.62×，门槛 3.80×） |
| lint | 改动的 4 个文件 0 诊断 |

---

## 9. 遗留与已暴露的问题

1. **收益只到 1/3**：−4/层 vs 目标 −13/层。剩余靠 ②RNSorm+residual 融合（约 −4）、
   ③RoPE（约 −3~5）、④SwiGLU（约 −2）。
2. **净增 1 个 elementwise**：合并 GEMM 的 bias 广播没被 epilogue 吸收。理论上可写成
   `addmm` 显式融合，但优先级低（1 个 kernel/层）。
3. **`bench/regress.py` 的比值本身也有波动**：同一提交内两次测量 eager 绝对值漂 17.5%，
   比值 5.43× → 4.62×。当前 30% 容差能吸收，但若要更稳，应改为「多次重复取中位数」
   （与 `p7_fill_bench.py` 已采用的口径一致）。
4. **`attn_impl` 三实现对拍尚未纳入常规回归**：`sdpa` 已注册并有函数级验证（见 `p8-prereq.md`），
   但还没有测试把「三后端 J3 等价」锁住。
