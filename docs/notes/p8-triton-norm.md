# P8 ②-b · 自研 Triton RMSNorm 融合核 + 延迟残差

> 日期：2026-10-04　对应 roadmap `nano-vllm-roadmap-v2.md:145` 的「**Triton** 融合 residual+RMSNorm」。
> 上一版（②-a）先用库算子 `F.rms_norm` 把收益锁定（见 `p8-fused-norm.md`），本版是
> 「两遍实现法（先对、再快）」的第二遍：**自己写核**，并把残差加一起融掉。

---

## 1. 结论

| 指标 | `torch`（逐 op oracle） | `lib`（`F.rms_norm`，②-a） | **`triton`（本版）** |
| --- | --- | --- | --- |
| 单层 kernel 数（图开 / batch=8） | 38.0 | 24.0 | **22.0** |
| 单层 kernel 数（eager） | 39.0 | 25.0 | **23.0** |
| 整步 kernel 数 | 1086（②-a 前） | 687 | **631** |
| TPOT（图开，ms/step） | 7.64 | 7.24 | **6.99** |
| TPOT（eager） | 38.76 | 31.90 | 33.29 |

**P8 累计：单层 42.0 → 22.0（−47.6%）**；图开 TPOT 从 7.64 → 6.99（1.09×），
其中 ②-a 贡献 1.055×、②-b 再贡献 1.035×。

但请看 §6：**本版的性能账并不全是正面的**，有一处明确的退步，值得记住。

---

## 2. 两个核

`nano_vllm/ops/fused_norm.py`：

| 核 | 用途 | 调用次数/前向 |
| --- | --- | --- |
| `_rms_norm_kernel` | 纯 RMSNorm | **1**（只有第 0 层的 `input_layernorm`） |
| `_fused_add_rms_norm_kernel` | `(norm(x+residual), x+residual)` | **56**（28 层 × 2） |

行-块映射：`grid = (num_rows,)`，一个 program 处理一整行 hidden（1536 个数），
行内归约**不需要跨 program 通信**（因此没有 atomic、没有二阶段归约）。代价见 §6。

### 三个容易写错的地方

| 坑 | 现象 | 处理 |
| --- | --- | --- |
| **`H` 不是 2 的幂** | `BLOCK_H = next_power_of_2(1536) = 2048`，25% 的 lane 被 mask 掉 | 均值除的是 **`H` 而不是 `BLOCK_H`** —— mask 掉的 lane 在 `tl.sum` 里贡献 0，若除以 2048 会把方差算小 33%，输出整体偏大约 15% |
| **bf16 直接平方求和** | 1536 个数在 bf16 里累加会明显丢精度 | 先 `.to(tl.float32)` 再平方、再 `tl.sum` |
| **一个核写两个输出** | `s` 既要在寄存器里参与归约，又要落盘给下一段残差用 | 先 `tl.store` 再算 `var`，两者共用寄存器里的 `s`，**不读回** |

### 舍入顺序刻意跟 oracle 对齐

oracle（HF `Qwen2RMSNorm`）是 `(x * rsqrt).to(dtype) * w` —— **先转回 bf16 再乘 weight**。
本核照抄这个顺序，而不是"全程 fp32 最后再转"。这一步不是随手写的，实测有回报（见 §5）。

---

## 3. 精度：为什么与 oracle 逐位相同，与逐 op 版差 1 ulp

这是本项最值得记的一条。三组对拍（bf16 / 同权重 / 同输入）：

| 对拍 | 结果 |
| --- | --- |
| `_rms_norm_kernel` vs 逐 op oracle | **逐位相同（max\|Δ\| = 0.0）**，H = 1024 / 1536 / 2048 / 8960 全覆盖 |
| `fused_add_rms_norm` 的 `s = x + residual` | **逐位相同**（与 `torch` 的 bf16 加法一致） |
| `fused_add_rms_norm` 的 norm vs **高精度 oracle**¹ | **逐位相同** |
| `fused_add_rms_norm` 的 norm vs **逐 op oracle** | 差 **1 ulp**（6.25e-2 / 幅值 10 = 0.6%） |

¹ 高精度 oracle = 在 fp32 里做加法，**并用未舍入的 fp32 和**算方差。

**所以那 1 ulp 不是我们的核的误差，是逐 op 基准丢了精度**：

```
逐 op 版:  s_bf16 = (x + residual).to(bf16)      ← 先舍入一次
           var    = mean(s_bf16.to(fp32)²)       ← 再读回来算方差
融合核  :  s_fp32 = x.to(fp32) + residual.to(fp32)
           var    = mean(s_fp32²)                ← 用**未舍入**的值算方差
           s_bf16 = s_fp32.to(bf16)              ← 只在这里舍入一次
```

融合把「加法」与「归约」放在同一份 fp32 中间值上，**少一次往返舍入**。
这不是"误差更大"，是**更准**。这条已固化为单测（`test_bitwise_matches_high_precision_oracle`），
把"1 ulp"钉成**已知且可解释**的性质，而不是含糊的"有噪声"。

---

## 4. 延迟残差（deferred residual）：本版真正的结构性改动

教科书式 decoder layer 每层自己做两次残差加：

```python
residual = x;  h = norm(x);   h = attn(h);  x = residual + h     # 独立 aten::add
residual = x;  h = norm(x);   h = mlp(h);   x = residual + h     # 独立 aten::add
```

vLLM 的做法是把**最后一次加法推迟**给下一层的 norm：

```python
# DecoderLayer.forward(..., residual=None) -> (hidden_states, residual)
if residual is None:                       # 只有第 0 层走到这里
    residual = hidden_states;  hidden = input_layernorm(hidden)
else:
    hidden, residual = input_layernorm.forward_with_residual(hidden, residual)   # 加+归一并成一个核
hidden = self_attn(hidden)
hidden, residual = post_attention_layernorm.forward_with_residual(hidden, residual)
hidden = self.mlp(hidden)
return hidden, residual                    # 「y = residual + mlp_out」不在这里做
```

`Qwen2Model.forward` 用 `residual` 变量逐层传递，末端由 `self.norm.forward_with_residual(...)` 收口。
**每层因此少两次独立 `aten::add`**（以及它们各自的显存往返）。

### 回溯验证：这个重构对数值是无操作

`norm_impl="torch"` 时，操作与顺序和重构前**完全一致**，所以 fp32 下应当逐位复现。
用重构前记录的值做回溯（同一 `torch.manual_seed(0)`、同一 200-token 输入）：

| | 层 1 | 层 2 | 层 4 | 层 8 | 层 16 | 层 27 | 层 28 | logits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 重构前记录 | 6.20 | 2890.25 | 6427.34 | 6589.70 | 6600.11 | 189.59 | 147.37 | 18.913 |
| 重构后实测 | 6.20 | 2890.25 | 6427.34 | 6589.70 | 6600.11 | 189.59 | 147.37 | 18.913 |

**全部一致 ⇒ 重构是数值无操作。** 对应的 kernel 数也印证：`torch` 档前后都是 **38.0**。

---

## 5. 三实现对拍

| 对拍（bf16 全模型前向） | logits 相对误差 |
| --- | --- |
| **`triton` vs `torch`（oracle）** | **7.66e-02** |
| `lib` vs `torch` | 8.28e-02 |
| `lib` vs `triton` | ≈ 上者的 2 倍（逐层也是 2 倍） |

**自研核比库算子更贴近 oracle**。原因正是 §2 那条舍入顺序：`F.rms_norm` 在 fp32 里
连乘 weight 再一次性转回，而 oracle 与我们的核都是"先转回 bf16、再乘 weight"。
「照着 oracle 的舍入顺序写」不是形式主义，是能测出来的。

---

## 6. 诚实的性能账（本版有一个明确的退步）

### 6.1 单算子微基准（`bench/norm_micro_bench.py`，µs，H=1536）

| rows | 纯 RMSNorm：逐 op | 纯 RMSNorm：`F.rms_norm` | 纯 RMSNorm：**我们的核** |
| --- | --- | --- | --- |
| 8 | 81.4 | **11.3** | 19.3（**比库慢 70%**） |
| 512 | 87.7 | **11.5** | 18.8 |

| rows | 残差加：`aten::add` + `F.rms_norm` | **我们的融合核** |
| --- | --- | --- |
| 8 | 30.9 | **23.0（快 1.34×）** |
| 512 | 22.7 | 24.7（略慢） |

**读法**：省掉一次 launch 的那一半赢了（1.34×），但**纯 norm 那一半明显输给库**（1.7×）。

### 6.2 端到端 TPOT A/B（同进程、只切 `norm_impl`、5–7 轮取中位数）

| 模式 | `torch` | `lib` | `triton` | lib → triton |
| --- | --- | --- | --- | --- |
| **图开（生产路径）** | 7.637 | 7.240 | **6.995** | **−0.245 ms（快 3.4%）** |
| eager | 38.76 | **31.90** | 33.29 | **+1.39 ms（慢 4.4%）** |

图开档的结论较稳：`triton` 的中位数（6.995）**低于** `lib` 的最小值（7.003）。
eager 档的退步也复现了两次（第一轮 −2.0 ms、第二轮 −1.4 ms），**不是噪声**。

### 6.3 机制假说（**未验证**，留作下一步）

两个已定位的弱点：

1. **SM 利用率**：一行一个 program，`rows=8`（decode batch=8）只占满 **8 / 84** 个 SM。
   库函数的切分方式不同，所以在小 row 数下更快 —— 这也解释了为什么 rows 从 8 增到 512 时
   我们的核耗时几乎不变（本来就没占满）。
2. **Triton 的 Python 侧 launch 开销**：本版每个前向要发 57 次 Triton launch（56 融合 + 1 纯）。
   图回放会**完全绕开**这条路径（只重放 GPU 侧），所以图开档赢、eager 档输 ——
   与 P7 的结论（"eager 的瓶颈在 CPU 侧 launch，图把它打掉"）方向一致。

**假说**：eager 档的退步来自 Triton launch 的 CPU 开销 > 省下的 `aten::add`。
未验证（没有单独量过两者的 host 侧时间），下一步可用 `bench/layer_kernels.py` 的 profiler
加一路 CPU 侧计时来确认。

### 6.4 由此得到的默认值选择

默认仍取 `norm_impl="triton"`：**生产路径（图开）赢 3.4%**，且这是 roadmap 字面要求的那一项；
eager 档的 4.4% 退步记录在此，供"无图部署"场景参考（那种场景应改用 `--norm-impl lib`）。

---

## 7. 已知弱点与下一步优化（学习清单）

1. **拆行 + 两阶段归约**：把一行切给多个 program，用 atomic 或第二个小核做跨 program 归约，
   解决小 row 数下的 SM 欠占。这是最值得做的一项。
2. **向量化 load/store**：`tl.load` 加 `eviction_policy` / 更宽的访问粒度。
3. **`num_warps` 自动选择**：扫描显示 rows≥8 时 `w=4/8` 差别不大，`w=16` 明显变差（回归压力）。
4. **`F.rms_norm` 与自研核混合**：纯 norm 只调用 1 次/前向（第 0 层），用库替掉它几乎没有代价 ——
   但这会让"三实现对照"变得不干净，故**没有**这么做，只记录为可选项。

---

## 8. 验证汇总

| 项 | 结果 |
| --- | --- |
| 全量测试 | **150 passed**（138 + 新增 `tests/test_p8_triton_norm.py` 12 项，1.98 s，免模型） |
| 核级对拍 | 纯 norm bf16 **逐位相同**；融合核的 `s` **逐位相同**；融合核的 norm 与高精度 oracle **逐位相同** |
| 延迟残差重构 | fp32 下 7 层 + logits **逐位复现重构前**；对应 kernel 数 38.0 前后一致 |
| 三实现对拍 | triton 比 lib **更贴近** oracle（7.66e-02 vs 8.28e-02） |
| `layer_kernels` 同构性 | 三档均 **0.0%** 偏差 |
| regress 门禁 | PASS（`graph_speedup` 4.64×，门槛 3.47×；graph TPOT 基线 7.30 → **6.82 ms**） |

---

## 9. 遗留

- §6.3 的假说未验证（Triton host 侧 launch 开销）。
- §7 的四项优化未做。
- **三实现（torch/lib/triton）的对拍目前只有单测与手动脚本**，尚未纳入 `bench/regress.py` 的常规回归；
  建议加一条"三实现 J3 等价"的自动检查（与 `p8-prereq.md` 里 `attn_impl` 三实现的欠账同源）。
