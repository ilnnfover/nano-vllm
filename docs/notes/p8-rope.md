# P8 ③ · 自研 Triton RoPE 融合核

> 日期：2026-10-04　对应 roadmap `nano-vllm-roadmap-v2.md:145` 的「RoPE 内联进 QKV 后处理」。
> 本项把 RoPE 从 ~10 个 kernel/层压到 **2 个**（q、k 各一次 launch），
> **单层 kernel 22.0 → 14.0（−8.0）**，是本项目单次收益最大的一项。

---

## 1. 结论

| 指标 | `rope_impl="torch"` | **`rope_impl="triton"`** | 变化 |
| --- | --- | --- | --- |
| **单层 kernel 数**（图开 / batch=8） | **22.0** | **14.0** | **−8.0（−36.4%）** |
| 整步 kernel 数 | 631 | **407** | −224 |
| TPOT（图开，ms/step） | 6.975 | **6.460** | **1.080×** |
| TPOT（eager） | 31.21 | 32.46 | 区间重叠，**不定论** |
| **P8 累计**（①+②+③） | — | — | **42.0 → 14.0（−66.7%）** |

数据：`bench/results/layer_kernels_ropetorch.json`（前）/ `layer_kernels.json`（后，当前默认）。

---

## 2. 消融先行：先知道上限，再写核

老规矩（② 就是靠这个发现"纯 norm 反而比库慢"）。消融 = 把 `apply_rotary_pos_emb` 换成恒等：

| kernel | 基线 | 无 RoPE | Δ/层 |
| --- | --- | --- | --- |
| `elementwise_kernel`（乘/加） | 140 | 28 | **−4.00** |
| `CatArrayBatchedCopy`（`rotate_half`） | 56 | 0 | **−2.00** |
| `vectorized_elementwise_kernel` | 56 | 0 | **−2.00** |
| `elementwise_kernel` | 56 | 0 | **−2.00** |
| （名字桶重分配） | 28 | 84 | +2.00 |
| **合计** | 631 | **407** | **−8.00/层** |

上限 **−8.0/层**（22.0 → 14.0），比原先估的 −3~5 大得多。
**实测落地也是 −8.0，整步 407 与消融的 407 一字不差 ⇒ 100% 命中了上限。**

---

## 3. 为什么 torch 层省不下来（只能写核）

torch 参考每层对 q、k 各做一遍：

```python
rotate_half(x) = torch.cat((-x2, x1), dim=-1)      # 求负 1 个 kernel + cat 拷 1 个
out = q * cos + rotate_half(q) * sin               # 乘、乘、加 3 个
```

**试过但没用的替法**：把 `cat` 换成「预计算索引 + 符号」：

```python
rot = x.index_select(-1, idx) * sign               # gather 1 个 + 乘 1 个 —— 还是 2 个
```

个数没变（消融里的 B 组实验本想验证这点，因在图捕获里建了 CPU 张量而崩，未跑完，但
从算式上可判定）。要真正降到 1 个，只能把整段算式写成一个核。

---

## 4. 核的推导：把 `rotate_half` 从「拼接」改写成「两半分别算」

`rotate_half(x) = concat([-x2, x1])`，代进 `out = x*cos + rotate_half(x)*sin` 展开：

```
前半  out1 = x1*cos1 - x2*sin1
后半  out2 = x2*cos2 + x1*sin2
```

（`x1/x2` 是 x 的前/后半，各 D/2 个；`cos1/cos2`、`sin1/sin2` 同理。）

这样**既不需要拼接、也不需要负号**：一个 program 只要分别 load 两个半段、算 4 个乘 2 个加减、
再分别 store —— 全程在寄存器里。`-x2*sin1` 与 `x*cos + (-x2)*sin` 在 IEEE 下等价
（取负、乘以 0、`a + (-b) = a - b` 都是精确的）。

grid = `(b × heads × s,)`，一个 program 处理「一条 (batch, head, token) 的 D 个元素」。

---

## 5. 接进模型时发现的真问题：**必须支持 stride**

这是本项最值得记的一条。预拼接（①）之后，q/k 是**合并缓冲区上的非连续切片视图**：

```
qkv 输出 [b, s, 2048]  →  切出 q  [b, s, 1536]  →  unflatten + transpose  →  [b, 12, s, 128]
                                                                              strides = (2048, 128, 1536, 1)
```

`is_contiguous() == False`。如果按"要求输入连续"处理，就要先 `.contiguous()` ——
**那会多一个拷贝 kernel，把融合省下的开销又吃回去**（正是 ① 里 `view` vs `unflatten` 同一个坑的另一处化身）。

所以核改成**按显式 stride 寻址**（`stride_xb/stride_xh/stride_xs`），输出则显式分配成**连续**张量
（与 torch 参考一致，且下游 attention 读连续布局更友好）。实测 strided 视图下误差仍是 0.71 ulp。

> 记录：这也说明"① 的收益"和"③ 的可行性"是耦合的 —— 如果 ① 当初把 q/k 做成连续副本，
> ③ 的核就只能吃连续输入，那 ① 省下的 GEMM 会被拷贝抵消。两个决策要一起看。

---

## 6. 数值

### 6.1 核级（vs torch 参考，多种形状/dtype/布局）

| 用例 | max\|Δ\|（≈ulp） |
| --- | --- |
| 连续布局 `(8,12,1,128)` | 0.44 |
| **预拼接式 strided 视图**（strides `(2048,128,1536,1)`） | **0.71** |
| prefill 形状 `(1,12,200,128)` | 0.60 |
| 小 head_dim `(2,4,3,8)` | 0.30 |

**全部 < 1 ulp。** 差异来源：参考实现每步都在 bf16 里往返舍入，本核在 fp32 里做中间运算、
写回时统一舍入一次（方向同 `fused_norm.py`，更准一侧）。

退化不变量也已断言：`cos=1, sin=0` 时**逐位等于输入**；`cos` 只在前半为 1 时，后半输出**恰好为 0**
（后者专门用来抓"两半写反/错位"，这类错不会崩、只会静默算错）。

### 6.2 全模型 / 引擎级

| 判据 | 结果 |
| --- | --- |
| logits 相对误差（triton vs torch） | **4.28e-02** |
| 逐层相对误差（层 2/4/8/16） | 2.2e-05 ~ 5.7e-05 |
| J2 同路径确定性（两次运行） | 逐 token 一致 ✅ |
| J3 跨路径 | 4 条里 2 条完全一致；2 条在 token 20 / 7 处翻转，`margin/ε = 0.67 / 0.04` ⇒ **翻转在数值差以内** ✅ |

**本项是三项改动里数值扰动最小的**：logits 相对 4.28e-02 < bf16 噪声底 1.16e-01
（① 预拼接 1.15e-01、② 融合 norm 8.26e-02）。合理：RoPE 是纯逐元素算子，没有归约、
也不改 GEMM 的 tiling。

---

## 7. 性能

### 7.1 单算子微基准（`bench/ops_micro_bench.py`，µs）

| (batch, seq) | torch 参考 | **自研核** | 加速 |
| --- | --- | --- | --- |
| 8 × 1（decode） | 103.18 | **48.47** | **2.13×** |
| 32 × 1 | 92.47 | 47.40 | 1.95× |
| 1 × 200（prefill） | 96.29 | 48.62 | 1.98× |
| 4 × 64 | 92.77 | 48.20 | 1.92× |

**约 2× 全面领先**（对照 ②-b 的 norm 核：纯 norm 反而比 `F.rms_norm` 慢 70%）。
`num_warps = 1` 与 `4` 相当，`2` 略差；默认取 1。

### 7.2 端到端 TPOT（同进程、只切 `rope_impl`、7 轮中位数）

| 模式 | torch | **triton** | 变化 |
| --- | --- | --- | --- |
| **图开（生产路径）** | 6.975 `[6.831, 7.292]` | **6.460** `[6.433, 7.003]` | **+0.515 ms（1.080×）** |
| eager | 31.21 `[30.80, 35.79]` | 32.46 `[25.96, 34.03]` | 区间重叠 → **不定论** |

图开档中位数分离明显（triton 的 min-max 区间与 torch 只轻微重叠），**+8% 可信**。
eager 档这轮噪声太大（triton 的 min 25.96 比它的中位数低 20%），不下结论
——这与 ②-b 那次"eager 明确变慢"不同，那次两轮复现一致，这次不一致。

---

## 8. 验证汇总

| 项 | 结果 |
| --- | --- |
| 全量测试 | **161 passed**（150 + 新增 `tests/test_p8_rope.py` 11 项，2.82 s，免模型） |
| 消融上限 vs 实测 | 上限 −8.00/层，实测 **−8.00/层**（整步 407 完全一致） |
| `layer_kernels` 同构性 | 前后两档均 **0.0%** 偏差 |
| regress 门禁 | PASS（`graph_speedup` 4.47×，门槛 3.11×） |
| lint | 0 error |

---

## 9. 剩余 14.0 个/层的构成与下一步

| 每层 | kernel | 说明 |
| --- | --- | --- |
| 4.04 | `cutlass::Kernel2` | 4 个 GEMM（已被 ① 最小化：7 → 4） |
| 2.00 | `_rope_kernel` | 本项（q、k 各一次 launch） |
| **2.00** | `index_elementwise_kernel` | **来源未定，待查**（③ 之前就在，与 RoPE 无关） |
| 2.00 | `_fused_add_rms_norm_kernel` | ②-b |
| 1.00 ×3 | `elementwise_kernel` | 待归因 |
| 1.00 | `_paged_attn_decode_batch_kernel` | 自研 paged attention |

可继续做的：

1. **把 RoPE 内联进 QKV 后处理**（roadmap 的字面说法）：现在 q、k 各一次 launch，
   合起来能到 1 个/层（再 −1）。代价是核要同时处理合并输出上的 q/k/v 三段的错位布局。
2. **`index_elementwise_kernel` 2/层 的来源**：查清后可能再省 2。
3. **cos/sin 缓存**：`RotaryEmbedding.forward` 每步现算（`cat` + `cos` + `sin`，属层外），
   预计算成表可按 `position_ids` 索引 —— 只影响步级，属小项。
4. ②-b 留下的：拆行 + 两阶段归约让 norm 核超过库函数。
