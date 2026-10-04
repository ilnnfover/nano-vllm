"""P8 ④ · 自研 Triton SwiGLU 融合核（对照记录见 `docs/notes/p8-swiglu.md`）。

`MLP.forward` 里 `self.act_fn(gate) * up` 在 torch 下是**两个** elementwise kernel
（`aten::silu` + `aten::mul`），本核合成一个：

    out = silu(gate) * up = gate * sigmoid(gate) * up

消融实测这两个 kernel 各占 **1.0 个/层**（合计 2.0/层）；融合后 1 个 ⇒ 净省 **1.0/层**。

## 为什么只省 1 个（上限是 2）

消融的上限是 −2.0/层（silu 与乘法全消失），但要吃满需要把 SwiGLU 写进
**`gate_up_proj` 的 GEMM epilogue**（一次读完 gate/up 直接进 reduce，不物化中间结果）
—— 那要自己写 matmul，代价远大于 1 个 kernel 的收益。故本项只做逐元素融合，
GEMM epilogue 方案记在笔记的「进一步」里。

## 布局：gate/up 是合并缓冲区上的**切片视图**

预拼接（①）之后：

    gate_up_proj(x) → [b, s, 17920]（连续）
    .split(8960, -1) → gate = [b, s, 8960]  strides = (s·17920, 17920, 1)
                       up   = [b, s, 8960]  strides = (s·17920, 17920, 1)   ← 同一缓冲区的后半

所以两个输入都**非连续**。核按显式 `stride_row` 寻址（要求最后一维 stride = 1，
且把前导维拍平后行 stride 仍然一致 —— 本布局满足：`stride(0) == shape(1) · stride(1)`），
输出写成连续张量。若在这里 `.contiguous()`，会多两个拷贝 kernel，把融合的收益全吃掉
（同 ① 的 `view` vs `unflatten`、③ 的 stride 支持，同一个坑的第三次出现）。
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    H,
    stride_g,
    stride_u,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < H

    g = tl.load(gate_ptr + row * stride_g + col, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(up_ptr + row * stride_u + col, mask=mask, other=0.0).to(tl.float32)

    # silu(g) = g * sigmoid(g)。x 很负时 exp(-x) 溢出为 inf → sigmoid→0 → 输出 0，
    # 与 torch 的 silu 同路径（都在 fp32 里算 sigmoid）。
    y = g * tl.sigmoid(g) * u
    tl.store(out_ptr + row * H + col, y.to(out_ptr.dtype.element_ty), mask=mask)


_DEFAULT_NUM_WARPS = 4
_DEFAULT_BLOCK = 1024


def _row_layout(x: torch.Tensor, name: str) -> tuple[int, int, int]:
    """把 `[..., H]` 折成 `(rows, H, stride_row)`，并**校验行 stride 确实一致**。

    前导维拍平后行 stride 一致 ⇔ 对每个前导维 d：`stride(d) == shape(d+1) · stride(d+1)`。
    不满足时**显式报错**（而不是让核按错的行距读，静默算错）。
    """
    if x.device.type != "cuda":
        raise RuntimeError(f"Triton SwiGLU 仅支持 CUDA，{name} 在 {x.device} 上")
    if x.stride(-1) != 1:
        raise ValueError(f"{name} 最后一维必须连续（stride=1），实际 strides={x.stride()}")
    h = x.shape[-1]
    if x.dim() == 1:
        return 1, h, h
    rs = x.stride(-2)
    exp = rs
    for d in range(x.dim() - 3, -1, -1):
        exp = exp * x.shape[d + 1]
        if x.stride(d) != exp:
            raise ValueError(
                f"{name} 前导维拍平后行 stride 不一致：dim{d} stride={x.stride(d)} 期望 {exp}"
            )
    return x.numel() // h, h, rs


def swiglu(gate: torch.Tensor, up: torch.Tensor, num_warps: int = _DEFAULT_NUM_WARPS) -> torch.Tensor:
    """`silu(gate) * up`，一次融合 kernel；返回连续的新张量（`[...]` 与输入同形）。

    两个输入都允许是**非连续的行视图**（预拼接后的 gate/up 就是）。
    """
    if gate.shape != up.shape:
        raise ValueError(f"gate 与 up 形状必须一致：{tuple(gate.shape)} vs {tuple(up.shape)}")
    if gate.dtype != up.dtype:
        raise ValueError(f"gate 与 up dtype 必须一致：{gate.dtype} vs {up.dtype}")
    rows, h, sg = _row_layout(gate, "gate")
    rows_u, h_u, su = _row_layout(up, "up")
    if (rows, h) != (rows_u, h_u):
        raise ValueError(f"gate/up 行布局不一致：{(rows, h)} vs {(rows_u, h_u)}")

    out = torch.empty(gate.shape, dtype=gate.dtype, device=gate.device)
    _swiglu_kernel[(rows, triton.cdiv(h, _DEFAULT_BLOCK))](
        gate, up, out,
        h, sg, su,
        BLOCK=_DEFAULT_BLOCK,
        # `num_warps` 是 Triton 的**启动元参数**，不在核签名里；静态检查器不建模它，故误报。
        # 实测扫描见 docs/notes/p8-swiglu.md §5（默认值 4 与最优值差异在噪声内）。
        num_warps=num_warps,  # pyright: ignore[reportCallIssue]  # pyrefly: ignore
    )
    return out
