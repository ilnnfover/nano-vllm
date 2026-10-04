# P8 ④ · 自研 Triton SwiGLU 融合核

> 日期：2026-10-04　对应 roadmap `nano-vllm-roadmap-v2.md:145` 的「SwiGLU 融合」。
> 把 `self.act_fn(gate) * up` 从 **2 个** elementwise kernel 合成 **1 个**：单层 14.0 → 13.0。
> 同时本项**证实了 ②-b 笔记里那条"未验证假说"**（见 §5），这是它最大的产出。

---

## 1. 结论

| 指标 | `mlp_impl="torch"` | **`mlp_impl="triton"`** | 变化 |
| --- | --- | --- | --- |
| **单层 kernel 数**（图开 / batch=8） | **14.0** | **13.0** | **−1.0（−7.1%）** |
| 整步 kernel 数 | 407 | **379** | −28 |
| TPOT（图开，ms/step） | 6.449 | 6.460 | **−0.010 ⇒ 无可测差异** |
| TPOT（eager） | 32.17 | 30.74 | 区间重叠，**不定论** |

数据：`bench/results/layer_kernels_mlptorch.json`（前）/ `layer_kernels.json`（后，当前默认）。

**这是四项里唯一"只赢 kernel 数、端到端测不出收益"的一项** —— §5 解释了为什么，以及它顺带证明了什么。

---

## 2. 消融先行：上限是 −2，本项拿回 1

把 `MLP.forward` 分别改成不调 silu / 不调乘法：

| 消融 | 单层 | 可省 |
| --- | --- | --- |
| 基线（`silu(gate) * up`） | 14.00 | — |
| **A**：`down_proj(gate)`（silu 与乘法都不做） | **12.00** | **−2.00/层** |
| **B**：`down_proj(silu(gate))`（只去掉乘法） | **13.00** | **−1.00/层** |

⇒ `aten::silu` 与 `aten::mul` **各占 1.0 个/层**，上限 −2.0。

融合核 `out = silu(gate) * up` 一次算完 ⇒ 拿回 **1.0/层**（14 → 13）。剩下的 1 个要吃满，
必须把 SwiGLU 写进 **`gate_up_proj` 的 GEMM epilogue**（gate/up 全在寄存器里就完成 silu 与相乘，
不物化中间结果）——那要自己写 matmul，代价远超 1 个 kernel 的收益，故**不做**，记录在 §7。

---

## 3. 核

`nano_vllm/ops/swiglu.py`：

```python
g = load(gate ...)        # fp32
u = load(up   ...)
y = g * tl.sigmoid(g) * u      # silu(g)*u，一次算完
store(out ...)
```

grid = `(rows, cdiv(H, 1024))`。`silu(x) = x·sigmoid(x)`，`x` 大负时 `exp(-x)` 溢出为 `inf`
→ `sigmoid → 0` → 输出 0，与 torch 同路径（都在 fp32 里算 sigmoid）。

### 第三次遇到同一个坑：非连续行视图

预拼接之后：

```
gate_up_proj(x) → [b, s, 17920]（连续）
.split(8960, -1) → gate = [b, s, 8960]  strides = (s·17920, 17920, 1)   ← 非连续
                   up   = 同一缓冲区的后半，同样非连续
```

所以核按**显式 `stride_row`** 寻址，并要求「前导维拍平后行 stride 一致」
（`stride(d) == shape(d+1)·stride(d+1)`，本布局满足）；输出写成连续张量。
若在这里 `.contiguous()`，会多**两个**拷贝 kernel，把融合收益全吃掉。

> **这是同一个坑的第三次出现**（① 的 `view` vs `unflatten`、③ 的 stride 支持、④ 的这里）。
> 值得当成一条规律记下来：**预拼接把"权重拼接"变成了"到处是非连续视图"**，
> 任何新算子只要碰到 q/k/v 或 gate/up，就默认要支持 stride。

`_row_layout()` 会**校验**行 stride 一致，不一致直接 raise —— 因为按错的行距读**不会崩，只会静默算错**。

---

## 4. 数值

| 判据 | 结果 |
| --- | --- |
| 核级 vs `F.silu(gate)*up`：连续布局 | 0.40 ulp（bf16）/ 0.80（fp32） |
| 核级：**预拼接式 strided 切片** | **0.80 ulp** |
| 核级：prefill 形状 `[1, 500, 8960]` | < 1 ulp |
| **silu 饱和/溢出边界**（−100、±30、100） | **逐位一致**（两半没写反才会如此精确） |
| 全模型 logits 相对误差 | **6.23e-02**（< bf16 噪声底 1.16e-01） |
| J2 同路径确定性 | 逐 token 一致 ✅ |
| J3 跨路径 | 4 条里 2 条完全一致；2 条翻转，`margin/ε = 0.29 / 0.66` ⇒ 在数值差以内 ✅ |

---

## 5. 性能：kernel 数赢了，端到端没测出收益（以及它证明了什么）

### 5.1 单算子微基准：自研核**更慢**（`bench/ops_micro_bench.py`）

| (batch, seq) | `F.silu(gate)*up`（2 kernel） | 自研核（1 kernel） | 结论 |
| --- | --- | --- | --- |
| **8 × 1**（decode） | **19.36 µs** | 29.00 µs | **慢 1.5×** |
| 32 × 1 | 17.52 | 24.33 | 慢 |
| 1 × 200（prefill） | 18.38 | 25.48 | 慢 |
| 4 × 64 | 25.52 | 24.96 | 1.02× |

一个 kernel 反而比两个 kernel 慢 —— 这只能有一个解释：**Triton 的 Python 侧 launch 开销
远高于 ATen**（自研核每次 ~25–30 µs，ATen elementwise 每次 ~8–10 µs）。

### 5.2 这**证实了 ②-b 笔记 §6.3 那条"未验证假说"**

三个自研核的微基准数据放在一起，规律非常一致：

| 核 | 自研（每次 launch） | 对照 | 差 |
| --- | --- | --- | --- |
| `rms_norm` | ~19 µs | `F.rms_norm` ~11 µs | +8 |
| `swiglu` | ~25–29 µs | `aten::silu`+`mul` ~17–19 µs（两次） | +10 |
| `apply_rope` | ~24 µs（2 次 48） | torch 参考 ~95 µs（~4 次） | **−45** |

⇒ **单看"每次 launch"，自研核比 ATen 贵约 10–20 µs；只有在"少掉的 launch 数 × ATen 单价"
能覆盖这个差价时才赢**。RoPE 省了 8 个 kernel/层所以大幅赢；SwiGLU 只省 1 个，差价为负。

而 **CUDA Graph 会把这条 launch 路径整个绕开**（只重放 GPU 侧），所以图开档的排序与微基准**不同**
—— 与 P7 的结论（"eager 的瓶颈在 CPU 侧 launch，图把它打掉"）完全一致。
这条规律解释了本阶段所有"eager 慢 / 图快"的观测，②-b 里它只是假说，现在有数据了。

### 5.3 端到端 TPOT（同进程、只切 `mlp_impl`、7 轮中位数）

| 模式 | torch | triton | 结论 |
| --- | --- | --- | --- |
| **图开（生产路径）** | 6.449 `[6.421, 6.712]` | 6.460 `[6.356, 6.707]` | **−0.010 ms（0.998×）⇒ 无差异** |
| eager | 32.168 `[26.94, 34.97]` | 30.735 `[28.43, 33.64]` | 区间重叠 ⇒ 不定论 |

**为什么图开也测不出**：图开一步约 6.5 ms，而省下的这 1 个 kernel 是一次 71680 元素的逐元素运算
（几十 µs 量级、且只占 379 个 kernel 里的 1 个）⇒ 期望收益 ~0.1%，**在噪声以下**。

> **结论性认识**：`kernel 数` 与 `TPOT` 这两个指标在**边际项上会脱钩**。
> P8 前几项（① 少 4 个 GEMM/层、② 少 14 个、③ 少 8 个）都在 TPOT 上可测；
> 到第 ④ 项只剩 1 个，kernel 数仍降 7.1%，但 TPOT 已进噪声。
> 所以 roadmap 的「kernel ↓≥30%」与「TPOT ↓5–15%」是两个独立判据，都要报。

---

## 6. 验证汇总

| 项 | 结果 |
| --- | --- |
| 全量测试 | **171 passed**（161 + 新增 `tests/test_p8_swiglu.py` 10 项，1.94 s，免模型） |
| 消融上限 vs 实测 | 上限 −2.00/层（不做 silu+乘法）；本项拿回 **−1.00/层** |
| `layer_kernels` 同构性 | 前后两档均 **0.0%** 偏差 |
| regress 门禁 | PASS（`graph_speedup` 4.53×，门槛 3.47×；graph TPOT 基线 6.53 ms） |
| lint | 0 error |

**顺带修掉一个测试**：`tests/test_p8_prejoin.py::test_mlp_matches_two_linears` 在 CPU 上直接构造
`MLP`，而新默认 `mlp_impl="triton"` 会拒绝 CPU。该测试测的是**预拼接的权重布局**，
与 SwiGLU 无关，故显式传 `mlp_impl="torch"`。（`NanoRunner` 侧不受影响——它已按设备显式降级。）

---

## 7. 剩余 13.0 个/层的构成，与 ④ 的"进一步"

| 每层 | kernel | 说明 |
| --- | --- | --- |
| 4.04 | `cutlass::Kernel2` | 4 个 GEMM（① 已把 7 减到 4） |
| 2.00 | `_rope_kernel` | ③ |
| **2.00** | `index_elementwise_kernel` | **来源仍未定**（③④ 之前就在，与本阶段改动无关） |
| 2.00 | `_fused_add_rms_norm_kernel` | ②-b |
| 1.00 | `elementwise_kernel` | 待归因 |
| 1.00 | `_paged_attn_decode_batch_kernel` | 自研 paged attention |
| 1.00 | `_swiglu_kernel` | ④ |

④ 想吃满消融上限（再 −1），只能做 **`gate_up_proj` 的 GEMM epilogue 融合**：
在同一个 Triton matmul 里读完 gate/up 就做 silu 与相乘。收益 = 1 个 kernel/层 + 少一次
`[b,s,17920]` 的往返写读（这个可能比 kernel 数更值钱）。代价 = 要自己写一块 matmul kernel。

---

## 8. P8 四项总账

| 项 | 单层 kernel | 该项收益 | 图开 TPOT（该项） |
| --- | --- | --- | --- |
| 起点（进入 P8） | **42.0** | — | ~7.64 ms |
| ① 权重预拼接 | 38.0 | −4.0 | 未单独测 |
| ②-a 融合 RMSNorm（lib） | 24.0 | −14.0 | 1.055× |
| ②-b 自研 norm 核 + 延迟残差 | 22.0 | −2.0 | 1.035× |
| ③ 自研 RoPE 融合核 | 14.0 | −8.0 | 1.080× |
| ④ 自研 SwiGLU 融合核 | **13.0** | −1.0 | ~1.000×（噪声内） |
| **合计** | **42.0 → 13.0** | **−29.0（−69.0%）** | **1.18×（可测段）** |

roadmap 的两条 P8 判据：

| 判据 | 目标 | 实测 |
| --- | --- | --- |
| 单层 kernel launch 数下降 | ≥30% | **−69.0%** ✅ |
| TPOT 再降 | 5–15% | **−15.4%（1.18×）** ✅（④ 贡献 ~0，由 ②③ 达成） |

**遗留**（按价值排序）：

1. **`index_elementwise_kernel` 2/层 的来源**（可能是唯一"不花钱"的收益，尚未查清）。
2. ④ 的 GEMM epilogue 融合（−1 kernel/层 + 省一次中间张量往返）。
3. ③ 的进一步：把 RoPE 内联进 QKV 后处理，q/k 合成 1 次 launch（再 −1/层）。
4. ②-b 的拆行 + 两阶段归约（让 norm 核超过库函数）。
5. **开关已到 5 个**（`attn_impl` / `prejoin` / `norm_impl` / `rope_impl` / `mlp_impl`），
   每个都穿透 4 层构造器。P8 已收尾、不再新增，但**下阶段开工前建议先收成一个 options dataclass**
   （审计报告早就点过"裸参数穿透"这个问题）。
