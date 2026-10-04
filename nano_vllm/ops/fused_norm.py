"""P8 ②-b · 自研 Triton RMSNorm 融合核（对照实现见 `docs/notes/p8-triton-norm.md`）。

两个核：

| 核 | 输入 | 输出 |
| --- | --- | --- |
| `rms_norm` | `x` | `norm(x)` |
| `fused_add_rms_norm` | `x`, `residual` | `norm(x + residual)`、`x + residual` |

`fused_add_rms_norm` 是「延迟残差」设计的基础：模型层不再单独做 `y = residual + attn_out`，
而是把这个加法折进下一层（或本层）的 norm 里 —— **一个核同时干「加」和「归一」两件事**。

## 语义必须对齐的对象

对齐 HF `Qwen2RMSNorm`（也就是 `qwen2.py` 里的逐 op 版，本模块的 oracle）：

    s   = x + residual                        # 融合核顺手做掉
    var = mean(s²)                            # **fp32 累加**
    y   = (s * rsqrt(var + eps)).to(dtype) * w

最后那个 `.to(dtype)` 的位置是刻意的：HF 是「归一化留在 fp32，**转回 bf16 之后再乘 weight**」。
把乘 weight 提到转换之前（全程 fp32 再一次性转换）数值上会差一点点，对拍时能看出来。

## 三个容易写错的地方（都在这段代码里显式处理）

1. **`H` 不是 2 的幂**（1536）：`BLOCK_H = next_power_of_2(H) = 2048`，越界列用 mask 屏蔽。
   关键：均值除的是 **H 而不是 BLOCK_H** —— mask 掉的列参与 `tl.sum` 时贡献 0（`other=0.0`），
   若除以 BLOCK_H 会把方差算小 33%。
2. **必须 `to(tl.float32)`**：bf16 直接平方求和，1536 个数累加会明显损失精度（roadmap 附录坑点 3
   的同类问题）。累加在 fp32 里做，这是与 `torch` oracle 对得上的前提。
3. **一个核写两个输出**：`s` 既要存回显存（作为下一段残差的输入），又要在寄存器里参与归约 —
   所以先 `tl.store` 再算 `var`，两者共用同一份寄存器里的 `s`，不需要读回。

## 行-块映射

`grid = (num_rows,)`，一个 program 处理**一整行** hidden（1536 个数）：
行内归约只在本 program 内完成，不需要跨 program 通信（也就不用 atomic / 二阶段归约）。
代价是行数少时并行度不足（decode batch=8 只有 8 个 program）——这是本核的已知取舍，
`docs/notes/p8-triton-norm.md` 里有 num_warps / BLOCK_H 的扫描数据。
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    eps,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    mask = cols < H
    offs = row * H + cols

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # 归约分母用 H（真实宽度），不是 BLOCK_H —— 否则被 mask 的 0 会把方差摊薄
    var = tl.sum(x * x, axis=0) / H
    rr = tl.math.rsqrt(var + eps)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    y = (x * rr).to(out_ptr.dtype.element_ty) * w
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def _fused_add_rms_norm_kernel(
    x_ptr,
    res_ptr,
    w_ptr,
    out_ptr,
    res_out_ptr,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    eps,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    mask = cols < H
    offs = row * H + cols

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(res_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s = x + r  # 残差加：原本是一个独立的 elementwise kernel

    # s 既要落盘（下一段残差要用），又要留在寄存器里参与归约 —— 先存，再算，不读回
    tl.store(res_out_ptr + offs, s.to(res_out_ptr.dtype.element_ty), mask=mask)

    var = tl.sum(s * s, axis=0) / H
    rr = tl.math.rsqrt(var + eps)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    y = (s * rr).to(out_ptr.dtype.element_ty) * w
    tl.store(out_ptr + offs, y, mask=mask)


# ------------------------------------------------------------------ 封装

# 与 `triton_paged_attn.py` 一样：显式传 constexpr，不用 autotune（扫描数据见笔记）
_DEFAULT_NUM_WARPS = 4


def _check(x: torch.Tensor, name: str) -> None:
    """守卫：本核按「行内连续」寻址（`row * H + cols`），非连续输入会静默算错。"""
    if x.device.type != "cuda":
        raise RuntimeError(f"Triton RMSNorm 仅支持 CUDA，{name} 在 {x.device} 上")
    if not x.is_contiguous():
        raise ValueError(f"Triton RMSNorm 要求 {name} 连续（按行内连续寻址），实际 strides={x.stride()}")


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    num_warps: int = _DEFAULT_NUM_WARPS,
) -> torch.Tensor:
    """`[*, H]` → `norm(x)`，返回与 `x` 同形状同 dtype 的新张量。"""
    _check(x, "x")
    h = weight.numel()
    rows = x.numel() // h
    out = torch.empty_like(x)
    _rms_norm_kernel[(rows,)](
        x, weight, out,
        H=h, BLOCK_H=triton.next_power_of_2(h), eps=eps,
        num_warps=num_warps,
    )
    return out


def fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    num_warps: int = _DEFAULT_NUM_WARPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(norm(x + residual), x + residual)` —— 一次 launch 出两个结果。

    刻意**不做原地更新**（vLLM 把 `residual` 原地改掉以省一次分配）。理由：原地写会让
    `residual` 与调用方的其它引用别名，在 CUDA Graph + 缓存分配器下这种隐式别名是难查的
    bug 来源；而显存往返次数与原地版**完全相同**（`s` 无论如何都要落盘），
    差别只有一次分配（缓存分配器下是廉价的）。见笔记的「为什么没做原地更新」。
    """
    _check(x, "x")
    _check(residual, "residual")
    if x.shape != residual.shape:
        raise ValueError(f"x 与 residual 形状必须一致：{tuple(x.shape)} vs {tuple(residual.shape)}")
    if x.dtype != residual.dtype:
        raise ValueError(f"x 与 residual dtype 必须一致：{x.dtype} vs {residual.dtype}")
    h = weight.numel()
    rows = x.numel() // h
    out = torch.empty_like(x)
    res_out = torch.empty_like(x)
    _fused_add_rms_norm_kernel[(rows,)](
        x, residual, weight, out, res_out,
        H=h, BLOCK_H=triton.next_power_of_2(h), eps=eps,
        num_warps=num_warps,
    )
    return out, res_out
