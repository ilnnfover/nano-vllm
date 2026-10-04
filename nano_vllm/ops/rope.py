"""P8 ③ · 自研 Triton RoPE 融合核（对照记录见 `docs/notes/p8-rope.md`）。

## 为什么要融合

`qwen2.py::apply_rotary_pos_emb` 的 torch 写法每层要 4 个左右 kernel（按 q、k 各算一遍）：

    rotate_half(x) = cat([-x2, x1])       ← 求负 1 个 + cat 拷 1 个
    out = q * cos + rotate_half(q) * sin  ← 乘、乘、加 3 个

**torch 层没法再省**：把 `cat` 换成「预计算索引 gather + 符号乘」仍是 2 个 kernel
（这也试过）；要真正降到 1 个，只能把整段算式写成一个核。实测 RoPE 段占 **8.0 个 kernel/层**
（22.0 → 14.0），是本项目剩余项里最大的一块。

## 关键：把 `rotate_half` 从「拼接」改写成「两半分别算」

`rotate_half(x) = concat([-x2, x1])`，代进 `out = x*cos + rotate_half(x)*sin` 展开：

    前半 out1 = x1*cos1 - x2*sin1
    后半 out2 = x2*cos2 + x1*sin2

其中 `x1/x2` 是 x 的前/后半（各 D/2 个）。这样**不需要拼接、也不需要负号**：
一个 program 只要分别 load 两个半段、算 4 个乘 2 个加减、再分别 store —— 全程在寄存器里。

（`-x2*sin1` 与 `x*cos + (-x2)*sin` 在 IEEE 下等价：取负与 0 相乘、以及 `a + (-b) = a - b`
都是精确的。差异只来自我们**在 fp32 里算、最后统一舍入一次**，见下。）

## 与 torch 参考的数值差异

参考实现每一步都在 bf16 里往返舍入（`q*cos` 出 bf16，再 `+` 出 bf16），
本核在 fp32 里做中间运算、只在写回时舍入一次 ⇒ 与参考差 **1–2 ulp**（更准，方向同
`fused_norm.py`）。对拍数据见笔记。

## 布局约定

    q/k:      [b, heads, s, D]      （prefill 的 varlen 路径也是 b=1、s=total_q）
    cos/sin:  [b, s, D]             （按 head 广播）
    grid:     (b * heads * s,)      一个 program 处理「一条 (batch, head, token) 的 D 个元素」

decode 时 `s=1`、heads=12 ⇒ 8×12=96 个 program（k 只有 2 个 head ⇒ 16 个）——
**SM 利用率偏低**，与 `fused_norm.py` 同一类取舍；prefill 时 `s` 大，并行度充足。
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    num_heads,
    seq_len,
    stride_xb,
    stride_xh,
    stride_xs,
    D: tl.constexpr,
    HALF: tl.constexpr,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0)
    head = r % num_heads
    pos = (r // num_heads) % seq_len
    batch = r // (num_heads * seq_len)

    # 输入按**显式 stride** 寻址：预拼接后 q/k 是合并缓冲区上的非连续视图
    # （unflatten+transpose → strides 形如 (s·total, head_dim, total, 1)），
    # 若在这里做 `.contiguous()` 就多一个拷贝 kernel，把省下的开销又吃回去。
    # 输出写成**连续**布局：d 维 stride 恒为 1，其余由 h/s/d 推出。
    xbase = batch * stride_xb + head * stride_xh + pos * stride_xs
    obase = ((batch * num_heads + head) * seq_len + pos) * D
    cbase = (batch * seq_len + pos) * D

    offs = tl.arange(0, BLOCK)
    mask = offs < HALF

    x1 = tl.load(x_ptr + xbase + offs, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + xbase + HALF + offs, mask=mask, other=0.0).to(tl.float32)
    c1 = tl.load(cos_ptr + cbase + offs, mask=mask, other=0.0).to(tl.float32)
    c2 = tl.load(cos_ptr + cbase + HALF + offs, mask=mask, other=0.0).to(tl.float32)
    s1 = tl.load(sin_ptr + cbase + offs, mask=mask, other=0.0).to(tl.float32)
    s2 = tl.load(sin_ptr + cbase + HALF + offs, mask=mask, other=0.0).to(tl.float32)

    o1 = x1 * c1 - x2 * s1
    o2 = x2 * c2 + x1 * s2

    ety = out_ptr.dtype.element_ty
    tl.store(out_ptr + obase + offs, o1.to(ety), mask=mask)
    tl.store(out_ptr + obase + HALF + offs, o2.to(ety), mask=mask)


_DEFAULT_NUM_WARPS = 1


def _check(x: torch.Tensor, name: str, allow_strided: bool) -> None:
    if x.device.type != "cuda":
        raise RuntimeError(f"Triton RoPE 仅支持 CUDA，{name} 在 {x.device} 上")
    if x.stride(-1) != 1:
        raise ValueError(
            f"Triton RoPE 要求 {name} 最后一维连续（stride=1），实际 strides={x.stride()}"
        )
    if not allow_strided and not x.is_contiguous():
        raise ValueError(f"Triton RoPE 要求 {name} 连续，实际 strides={x.stride()}")


def _launch(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
            out: torch.Tensor, num_heads: int, num_warps: int) -> None:
    b, _, s, d = x.shape
    _rope_kernel[(b * num_heads * s,)](
        x, cos, sin, out,
        num_heads, s,
        x.stride(0), x.stride(1), x.stride(2),
        D=d, HALF=d // 2, BLOCK=triton.next_power_of_2(d // 2),
        num_warps=num_warps,
    )


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_warps: int = _DEFAULT_NUM_WARPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(q, k)` 各做一次 RoPE，返回新张量（与 torch 参考同为非原地）。

    形状：`q=[b, heads, s, D]`、`k=[b, kv_heads, s, D]`、`cos/sin=[b, s, D]`。
    """
    _check(q, "q", allow_strided=True)
    _check(k, "k", allow_strided=True)
    _check(cos, "cos", allow_strided=False)
    _check(sin, "sin", allow_strided=False)
    if q.shape[-1] != k.shape[-1] or cos.shape[-1] != q.shape[-1]:
        raise ValueError(
            f"head_dim 不一致: q={q.shape[-1]} k={k.shape[-1]} cos={cos.shape[-1]}"
        )
    if (q.shape[-1] % 2) != 0:
        raise ValueError(f"RoPE 要求 head_dim 为偶数，got {q.shape[-1]}")
    if cos.shape[:2] != q.shape[:1] + q.shape[2:3]:
        raise ValueError(
            f"cos/sin 应为 [b, s, D]，与 q 的 (b, s) 对应：cos={tuple(cos.shape)} q={tuple(q.shape)}"
        )
    if q.dtype != k.dtype or q.dtype != cos.dtype:
        raise ValueError(f"dtype 必须一致: q={q.dtype} k={k.dtype} cos={cos.dtype}")

    # 输出显式分配成**连续**张量：与 torch 参考（`q*cos` 也会新建连续张量）一致，
    # 且下游 attention kernel 读连续布局更友好
    q_out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    k_out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    _launch(q, cos, sin, q_out, q.shape[1], num_warps)
    _launch(k, cos, sin, k_out, k.shape[1], num_warps)
    return q_out, k_out
