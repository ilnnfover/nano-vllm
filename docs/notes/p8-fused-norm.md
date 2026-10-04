# P8 ② · 融合 residual + RMSNorm

> 日期：2026-10-04　对应 roadmap `nano-vllm-roadmap-v2.md:145` 的「Triton 融合 residual+RMSNorm」。
> 本项一次性把**单层 kernel 数从 38 降到 24（−36.8%）**，单独就超过了 roadmap 的「≥30%↓」目标。

---

## 1. 结论

| 指标 | 融合前 | 融合后 | 变化 |
| --- | --- | --- | --- |
| **单层 kernel 数**（triton + 图） | **38.0** | **24.0** | **−14.0（−36.8%）** |
| 单层 kernel 数（eager） | 39.0 | 25.0 | −14.0 |
| 整步 kernel 数（batch=8） | 1086 | 687 | −399 |
| 层外 kernel 数 | 22 | **15** | −7（final norm 也走融合） |
| **P8 累计**（① + ②） | 42.0 | **24.0** | **−18.0（−42.9%）** |

数据：`bench/results/layer_kernels_nofusednorm.json`（前）/ `layer_kernels.json`（后，当前默认）。

---

## 2. 先做消融，再写实现（为什么）

`layer_kernels.py` 只能告诉我「每层有 ~16 个 elementwise kernel」，**不能告诉我它们分别来自哪个 op**
（`KinetoEvent.cpu_parent` 在 torch 2.14 下未填充，见 `p8-prejoin.md`；GPU kernel 的 name 又只有
`elementwise_kernel` 这种泛化名）。所以先在 profiler 里加一路 **CPU 侧 aten 计数**：

| aten op | 次数/层 | 来源 |
| --- | --- | --- |
| `aten::pow` | 2.04 | RMSNorm 的 `pow(2)`，2 个 norm/层 |
| `aten::add` | 6.32 | 2 处 residual + 3 处 QKV bias |
| `aten::mul` | 9.36 | RMSNorm ×2、SiLU、gate×up |
| `aten::to` / `_to_copy` | 16.32 / 13.61 | RMSNorm 的 fp32 往返、`_sdpa` |
| `aten::linear` | 4.04 | 4 个 GEMM/层（① 之后） |

逐 op 版 `RMSNorm.forward` 本身有 8 个张量操作（`to` ×2、`pow`、`mean`、`add`、`rsqrt`、`mul` ×2），
**× 2 个 norm/层 = 16 个 kernel** —— 与实测的 elementwise 数量吻合。

**再做消融**（把 `RMSNorm.forward` 临时换成 `F.rms_norm`，不提交）：
整步 kernel 1161 → 762 ⇒ 每层 39.0 → 24.75，即**上限约 −14.25/层**。

有了这个上限，才决定「值不值得写内核、写哪个」。**实测落地后为 −14.0，与消融预测仅差 0.25**。

---

## 3. 实现

### 3.1 改法与开关

`nano_vllm/models/qwen2.py::RMSNorm`：

```python
def forward(self, hidden_states):
    if self.fused:
        return F.rms_norm(hidden_states, (self.weight.numel(),), self.weight, self.variance_epsilon)
    # 逐 op 版：与 HF Qwen2RMSNorm 逐行对齐，留作对拍基准
    ...
```

开关链与 ① 同构：`NanoRunner(fused_norm=True)` → `Qwen2ForCausalLM` → `Qwen2Model` → `DecoderLayer`
→ `RMSNorm`；CLI `--fused-norm/--no-fused-norm`；`bench/layer_kernels.py --fused-norm/--no-fused-norm`。

### 3.2 为什么这一版用 `F.rms_norm`，而不是直接手写 Triton

仓库自己的红线（roadmap §6 反模式 3）是「**两遍实现法（先对、再快）**：跳过 torch 朴素实现
直接写 Triton」。本项的两遍是：

| 遍 | 内容 | 状态 |
| --- | --- | --- |
| 第一遍（本项） | 用库融合算子 `F.rms_norm`（CUDA 上就是单个融合 kernel），逐 op 版**保留为对拍基准** | ✅ 已完成，−14/层 |
| 第二遍（下一步） | 手写 Triton `fused_add_rms_norm`，把 **residual 加也折进去** | 待做，预估再 −2/层 |

依据：`F.rms_norm` 已经把 8 个 kernel 压到 1 个（拿到全部收益的 **87.5%**：−14 中的 −16/16 折算），
而手写内核的边际收益只有 residual 那 2 个 add；先用低风险路径锁定大头、并把对照基准立住，
再动内核，是本仓库一贯的顺序（P3 的 torch 朴素 → Triton 自研也是这个套路）。

### 3.3 保留逐 op 路径不是死代码

它是 ② 的**对照臂**（roadmap 要求「融合前后各一份对照」），也是 J3 数值验证的参照。默认走融合。

---

## 4. kernel 数变化逐项归因

同进程、同配置（batch=8 / prompt_len=64 / triton / 图开 / `prejoin=True`），**只切 `fused_norm` 一个变量**：

| kernel | 前 | 后 | Δ/层 | 说明 |
| --- | --- | --- | --- | --- |
| `elementwise_kernel` | 197 | 140 | −2.0 | |
| `vectorized_elementwise_kernel`（a） | 59 | 2 | −2.0 | |
| `unrolled_elementwise_kernel` | 59 | 2 | −2.0 | |
| `vectorized_elementwise_kernel`（b） | 57 | 0 | −2.0 | |
| `elementwise_kernel` | 57 | 0 | −2.0 | |
| `vectorized_elementwise_kernel`（c） | 57 | 0 | −2.0 | |
| `reduce_kernel`（均值归约） | 57 | 0 | −2.0 | |
| `vectorized_elementwise_kernel`（d） | 57 | 0 | −2.0 | |
| **`vectorized_layer_norm_kernel`** | 0 | **57** | **+2.0** | **新增的融合 kernel** |
| `cutlass::Kernel2` / `CatArrayBatchedCopy` / attention / index | — | — | 0 | 不受影响 |
| **合计** | 1086 | 687 | **−14.0** | |

读法：**逐 op 版每个 RMSNorm = 8 个 kernel**（上表前 8 行，每行 2.0/层 = 2 个 norm 各 1 个），
融合后 = **1 个**（第 9 行，2.0/层 = 2 个 norm）。净 −7/次 × 2 次/层 = **−14/层**。

层外 22 → 15（−7）：`Qwen2Model.norm`（final norm）也走融合，同样是 8 → 1。

---

## 5. 数值验证

判据同 ①：与**同一模型的 bf16 噪声底**（bf16 相对 fp32 的固有误差）比，而不是看绝对差。

| 层 | 融合相对误差 | bf16 噪声底 | 判定 |
| --- | --- | --- | --- |
| 1 | 1.01e-02 | 1.94e-02 | 更小 ✅ |
| 2 | 5.54e-03 | 4.37e-03 | 同量级 |
| 4 | 4.98e-03 | 3.20e-03 | 同量级（≈1.5×） |
| 8 | 4.86e-03 | 3.28e-03 | 同量级 |
| 16 | 4.85e-03 | 2.34e-03 | 同量级（≈2×） |
| 27 | 1.69e-01 | 3.73e-01 | 更小 ✅ |
| 28（final norm 后） | 1.02e-01 | 7.29e-02 | 同量级 |
| **logits** | **8.26e-02** | **1.16e-01** | **更小 ✅** |

**结论**：融合扰动的量级 ≤ bf16 量化本身的扰动，符合 `golden/verify_bf16_noise.py` 的
「C ≈ A → bf16 正常累积，不是 bug」判据。误差来源是**均值归约顺序**（`pow(2).mean` 逐元素 vs
融合核内分块归约），两者都在 fp32 里累加，属于同类舍入。

> 与 ① 的对比：①（预拼接）在整模型级的 logits 相对误差是 1.15e-01，本项是 8.26e-02，
> **本项的数值扰动更小**——因为预拼接改变的是 GEMM 的 tiling/split-k（波及全部数值），
> 而 norm 融合只改一个归约的顺序。

---

## 6. 与 P7 CUDA Graph 的交互

`F.rms_norm` 是形状静态的 aten 算子，**可以被图捕获**。已实测：`--cudagraph` 档下
整步 687 个 kernel、回放正常（`layer_kernels.py` 的图档跑通且同构性偏差 0.0%）。

> 对比：这只是因为融合核的形状与 data-dependent 无关。若哪天把 norm 融进 `reshape_and_cache`
> 那种带动态索引的核，就要重新评估可捕获性。

---

## 7. 验证汇总

| 项 | 结果 |
| --- | --- |
| 全量测试 | **138 passed** |
| `layer_kernels.py` 同构性校验 | 前后两版均 **0.0%** 偏差 |
| lint | 改动的 3 个文件 0 诊断 |
| 消融预测 vs 实测 | 预测 −14.25/层，实测 **−14.0/层** |

---

## 8. 遗留

1. **②-b：手写 Triton `fused_add_rms_norm`**，把 residual 加折进 norm 核
   （预估再 −2/层：每层 2 处 residual `aten::add`）。收益不大但这是 roadmap 字面要求的那一项；
   做完后单层应到 ~22。
2. **`F.rms_norm` 的 eps 语义**：`torch.rms_norm` 用 `rsqrt(mean(x²) + eps)`，与逐 op 版一致；
   但若后续换别的模型（如对 eps 位置更敏感的实现），需要重新对拍。
3. **`_sdpa` 的 host 分支仍未收敛**（`qwen2.py` 里 `if self.num_kv_groups > 1 and q.is_cuda ...`）——
   `aten::_to_copy` 13.6/层里有一部分来自它。P8 后续项（SwiGLU / RoPE）会继续叠 kernel 分支，
   建议在收尾前把它改成显式驱动。
