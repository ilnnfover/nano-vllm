"""P8 ③ · 自研 Triton RoPE 融合核的单测（**免模型**：只构造张量）。

与 `test_p8_triton_norm.py` 同一套思路：正确性对拍 + 把"差多少"钉成**已知的 ulp 级性质**
（而不是含糊的"有噪声"）+ 守卫显式报错。

本核**必须支持非连续输入**：预拼接之后 q/k 是合并缓冲区上的切片视图
（`unflatten` + `transpose` → strides 形如 `(s·total, head_dim, total, 1)`）。
如果这里改成先 `.contiguous()`，会多一个拷贝 kernel，把融合省下的开销吃回去 ——
所以专门有一条针对 strided 视图的用例。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from nano_vllm.models.qwen2 import apply_rotary_pos_emb as rope_ref
from nano_vllm.ops.rope import apply_rope

CUDA = torch.cuda.is_available()
# (batch, heads, seq, head_dim) —— 覆盖 decode（s=1）、prefill（s 大）、以及小 head_dim
SHAPES = [(8, 12, 1, 128), (1, 12, 200, 128), (4, 12, 7, 128), (2, 4, 3, 8), (3, 2, 5, 64)]
ULP_TOL = 2.0  # 实测 0.3–0.8 ulp


def _ulp_rel(got: torch.Tensor, ref: torch.Tensor) -> float:
    """以 ulp 为单位的最大相对偏差（按 ref 的幅值归一）。"""
    eps = torch.finfo(ref.dtype).eps
    scale = ref.float().abs().max().item()
    return (got.float() - ref.float()).abs().max().item() / scale / eps


def _inputs(shape, dtype, seed=0):
    b, h, s, d = shape
    torch.manual_seed(seed)
    q = (torch.randn(b, h, s, d, device="cuda") * 0.5).to(dtype)
    k = (torch.randn(b, max(1, h // 4), s, d, device="cuda") * 0.5).to(dtype)
    cos = torch.randn(b, s, d, device="cuda").to(dtype)
    sin = torch.randn(b, s, d, device="cuda").to(dtype)
    return q, k, cos, sin


@unittest.skipUnless(CUDA, "Triton 核需要 CUDA")
class TestRopeKernel(unittest.TestCase):
    def test_matches_reference_all_shapes(self) -> None:
        for shape in SHAPES:
            for dt in (torch.bfloat16, torch.float32):
                q, k, cos, sin = _inputs(shape, dt)
                rq, rk = rope_ref(q, k, cos, sin)
                gq, gk = apply_rope(q, k, cos, sin)
                u = max(_ulp_rel(gq, rq), _ulp_rel(gk, rk))
                self.assertLessEqual(u, ULP_TOL, f"{shape} {dt}: {u:.2f} ulp")

    def test_strided_view_matches_reference(self) -> None:
        """预拼接式视图：q/k 是合并缓冲区上的非连续切片，核必须按 stride 寻址。"""
        b, h, hk, s, d = 8, 12, 2, 1, 128
        for dt in (torch.bfloat16, torch.float32):
            torch.manual_seed(0)
            total = (h + 2 * hk) * d
            buf = (torch.randn(b, s, total, device="cuda") * 0.5).to(dt)
            qv, kv, _ = buf.split((h * d, hk * d, hk * d), dim=-1)
            q_s = qv.unflatten(-1, (h, d)).transpose(1, 2)
            k_s = kv.unflatten(-1, (hk, d)).transpose(1, 2)
            self.assertFalse(q_s.is_contiguous())
            cos = torch.randn(b, s, d, device="cuda").to(dt)
            sin = torch.randn(b, s, d, device="cuda").to(dt)
            rq, rk = rope_ref(q_s, k_s, cos, sin)
            gq, gk = apply_rope(q_s, k_s, cos, sin)
            self.assertTrue(gq.is_contiguous())
            self.assertLessEqual(max(_ulp_rel(gq, rq), _ulp_rel(gk, rk)), ULP_TOL)

    def test_identity_when_cos_one_sin_zero(self) -> None:
        """退化不变量：`cos=1, sin=0` 时 RoPE 是恒等（且该情形应逐位精确）。"""
        q, k, _, _ = _inputs((2, 4, 3, 8), torch.bfloat16)
        one = torch.ones(2, 3, 8, device="cuda", dtype=torch.bfloat16)
        zero = torch.zeros(2, 3, 8, device="cuda", dtype=torch.bfloat16)
        gq, gk = apply_rope(q, k, one, zero)
        self.assertTrue(torch.equal(gq, q))
        self.assertTrue(torch.equal(gk, k))

    def test_zero_half_when_cos_masks_half(self) -> None:
        """只看前半的 cos：后半输出应恰好为 0（验证两半的错位没有写反）。"""
        q, k, _, _ = _inputs((2, 4, 3, 8), torch.bfloat16)
        half = torch.zeros(2, 3, 8, device="cuda", dtype=torch.bfloat16)
        half[..., :4] = 1
        zero = torch.zeros(2, 3, 8, device="cuda", dtype=torch.bfloat16)
        gq, _ = apply_rope(q, k, half, zero)
        self.assertTrue(bool((gq[..., 4:] == 0).all()))

    def test_output_is_contiguous(self) -> None:
        q, k, cos, sin = _inputs((8, 12, 1, 128), torch.bfloat16)
        gq, gk = apply_rope(q, k, cos, sin)
        self.assertTrue(gq.is_contiguous() and gk.is_contiguous())


@unittest.skipUnless(CUDA, "Triton 核需要 CUDA")
class TestRopeGuards(unittest.TestCase):
    def test_bad_last_stride_rejected(self) -> None:
        q = torch.randn(2, 2, 4, 8, device="cuda").transpose(0, 1)  # 最后一维 stride 仍为 1?
        if q.stride(-1) != 1:  # pragma: no cover - 仅在该构造下成立
            with self.assertRaises(ValueError):
                apply_rope(q, torch.randn_like(q), *[torch.randn(2, 4, 8, device="cuda")] * 2)
        else:
            # 构造一个最后一维不连续的输入
            bad = torch.randn(2, 4, 8, 2, device="cuda")[..., 0]
            self.assertNotEqual(bad.stride(-1), 1)
            with self.assertRaises(ValueError):
                apply_rope(bad, bad, torch.randn(2, 4, 8, device="cuda"),
                           torch.randn(2, 4, 8, device="cuda"))

    def test_odd_head_dim_rejected(self) -> None:
        with self.assertRaises(ValueError):
            apply_rope(torch.randn(2, 2, 2, 7, device="cuda"), torch.randn(2, 2, 2, 7, device="cuda"),
                       torch.randn(2, 2, 7, device="cuda"), torch.randn(2, 2, 7, device="cuda"))

    def test_cos_shape_mismatch_rejected(self) -> None:
        with self.assertRaises(ValueError):
            apply_rope(torch.randn(2, 2, 2, 8, device="cuda"), torch.randn(2, 2, 2, 8, device="cuda"),
                       torch.randn(2, 3, 8, device="cuda"), torch.randn(2, 3, 8, device="cuda"))

    def test_dtype_mismatch_rejected(self) -> None:
        with self.assertRaises(ValueError):
            apply_rope(torch.randn(2, 2, 2, 8, device="cuda", dtype=torch.bfloat16),
                       torch.randn(2, 2, 2, 8, device="cuda", dtype=torch.bfloat16),
                       torch.randn(2, 2, 8, device="cuda", dtype=torch.float32),
                       torch.randn(2, 2, 8, device="cuda", dtype=torch.float32))

    def test_head_dim_mismatch_between_q_and_k_rejected(self) -> None:
        with self.assertRaises(ValueError):
            apply_rope(torch.randn(2, 2, 2, 8, device="cuda"), torch.randn(2, 2, 2, 16, device="cuda"),
                       torch.randn(2, 2, 8, device="cuda"), torch.randn(2, 2, 8, device="cuda"))


class TestRopeGuardsCPU(unittest.TestCase):
    def test_cpu_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            apply_rope(torch.randn(2, 2, 2, 8), torch.randn(2, 2, 2, 8),
                       torch.randn(2, 2, 8), torch.randn(2, 2, 8))


if __name__ == "__main__":
    unittest.main()
