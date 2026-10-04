"""P8 ④ · 自研 Triton SwiGLU 融合核的单测（**免模型**：只构造张量）。

与前两个核同一套思路：正确性对拍 + 把偏差钉成 ulp 级性质 + 守卫显式报错。

本核**必须支持非连续的行视图**：预拼接后 `gate`/`up` 是 `gate_up_proj` 输出（连续
`[b, s, 17920]`）上的两半切片 → strides `(s·17920, 17920, 1)`。
故专门覆盖：
  * strided 切片视图；
  * **行 stride 不一致**的输入必须被拒绝（否则核会按错的行距读，静默算错）；
  * silu 的饱和/溢出边界（`x` 大负时 `exp(-x)` 溢出为 inf，须与 torch 同结果）。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from nano_vllm.ops.swiglu import swiglu

CUDA = torch.cuda.is_available()
ULP_TOL = 2.0  # 实测 0.4–0.8 ulp


def _ulp_rel(got: torch.Tensor, ref: torch.Tensor) -> float:
    eps = torch.finfo(ref.dtype).eps
    scale = ref.float().abs().max().item()
    if scale == 0.0:
        return 0.0
    return (got.float() - ref.float()).abs().max().item() / scale / eps


@unittest.skipUnless(CUDA, "Triton 核需要 CUDA")
class TestSwigluKernel(unittest.TestCase):
    def test_matches_reference(self) -> None:
        for shape in ((8, 1, 8960), (1, 200, 8960), (4, 7, 128), (2, 3, 64)):
            for dt in (torch.bfloat16, torch.float32):
                torch.manual_seed(0)
                gate = torch.randn(*shape, device="cuda").to(dt)
                up = torch.randn(*shape, device="cuda").to(dt)
                ref, got = F.silu(gate) * up, swiglu(gate, up)
                self.assertEqual(got.shape, ref.shape)
                self.assertLessEqual(_ulp_rel(got, ref), ULP_TOL, f"{shape} {dt}")

    def test_strided_slices_from_merged_buffer(self) -> None:
        """预拼接式：gate/up 是同一缓冲区上的两半切片，必须按 stride 读对。"""
        b, s, inter = 8, 1, 8960
        for dt in (torch.bfloat16, torch.float32):
            torch.manual_seed(0)
            merged = torch.randn(b, s, 2 * inter, device="cuda").to(dt)
            gate, up = merged.split(inter, dim=-1)
            self.assertFalse(gate.is_contiguous())
            ref, got = F.silu(gate) * up, swiglu(gate, up)
            self.assertTrue(got.is_contiguous())
            self.assertLessEqual(_ulp_rel(got, ref), ULP_TOL)

    def test_prefill_shape_row_stride(self) -> None:
        """prefill 时行数 = total_q（可能上千），验证跨行 stride 寻址没串行。"""
        torch.manual_seed(0)
        merged = torch.randn(1, 500, 2 * 8960, device="cuda", dtype=torch.bfloat16)
        gate, up = merged.split(8960, dim=-1)
        self.assertLessEqual(_ulp_rel(swiglu(gate, up), F.silu(gate) * up), ULP_TOL)

    def test_silu_saturation_bitwise(self) -> None:
        """silu 的饱和/溢出边界应与 torch **逐位一致**（两半都没写反才会如此精确）。"""
        g = torch.tensor([[[-100.0, -30.0, -1.0, 0.0, 1.0, 30.0, 100.0]]],
                         device="cuda", dtype=torch.bfloat16)
        u = torch.ones_like(g)
        self.assertTrue(torch.equal(F.silu(g) * u, swiglu(g, u)))

    def test_output_contiguous(self) -> None:
        merged = torch.randn(4, 1, 2 * 64, device="cuda", dtype=torch.bfloat16)
        gate, up = merged.split(64, dim=-1)
        self.assertTrue(swiglu(gate, up).is_contiguous())


@unittest.skipUnless(CUDA, "Triton 核需要 CUDA")
class TestSwigluGuards(unittest.TestCase):
    def test_shape_mismatch_rejected(self) -> None:
        with self.assertRaises(ValueError):
            swiglu(torch.randn(4, 8, device="cuda"), torch.randn(4, 9, device="cuda"))

    def test_dtype_mismatch_rejected(self) -> None:
        with self.assertRaises(ValueError):
            swiglu(torch.randn(4, 8, device="cuda", dtype=torch.bfloat16),
                   torch.randn(4, 8, device="cuda", dtype=torch.float32))

    def test_bad_last_stride_rejected(self) -> None:
        bad = torch.randn(4, 8, 2, device="cuda")[..., 0]
        self.assertNotEqual(bad.stride(-1), 1)
        with self.assertRaises(ValueError):
            swiglu(bad, torch.randn(4, 8, device="cuda"))

    def test_non_uniform_row_stride_rejected(self) -> None:
        """前导维拍平后行 stride 不一致 → 必须报错（否则按错行距读，静默算错）。"""
        # [b, s, h] 但 b/s 维置换 → stride(0) != shape(1)*stride(1)
        x = torch.randn(8, 4, 16, device="cuda").transpose(0, 1)
        self.assertEqual(x.stride(-1), 1)
        with self.assertRaises(ValueError) as cm:
            swiglu(x, x)
        self.assertIn("行 stride 不一致", str(cm.exception))


class TestSwigluGuardsCPU(unittest.TestCase):
    def test_cpu_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            swiglu(torch.randn(4, 8), torch.randn(4, 8))


if __name__ == "__main__":
    unittest.main()
