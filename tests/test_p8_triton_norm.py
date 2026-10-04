"""P8 ②-b · 自研 Triton RMSNorm 核的单测（**免模型**：只构造张量，不加载权重）。

三类断言：

1. **正确性**——与逐 op oracle（HF 语义）逐元素对拍；
2. **精度方向**——融合核在 fp32 里算 `x+residual` 并用**未舍入**的值算方差，因此它应当
   与「高精度 oracle」**逐位相同**，而与「先落盘成 bf16 再读回」的逐 op 版差 1 ulp。
   这条把"1 ulp 差异"钉成**已知且可解释**的性质，而不是含糊的"有噪声"。
3. **守卫**——非连续 / 形状不符 / dtype 不符 / 非 CUDA 全部显式报错，不静默算错。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from nano_vllm.ops.fused_norm import fused_add_rms_norm, rms_norm

CUDA = torch.cuda.is_available()
EPS = 1e-6
# 覆盖：2 的幂（1024/2048）、非 2 的幂（1536 = Qwen2.5-1.5B 的 hidden）、大宽度（8960 = intermediate）
WIDTHS = [1024, 1536, 2048, 8960]


def op_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """逐 op oracle：与 HF `Qwen2RMSNorm` 逐行对齐。"""
    dt = x.dtype
    h = x.to(torch.float32)
    var = h.pow(2).mean(-1, keepdim=True)
    return w * (h * torch.rsqrt(var + eps)).to(dt)


def hi_prec_fused_norm(x, r, w, eps):
    """高精度 oracle：在 fp32 里加、且用**未舍入**的 fp32 和算方差（融合核的语义）。"""
    s = x.float() + r.float()
    return w * (s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


@unittest.skipUnless(CUDA, "Triton 核需要 CUDA")
class TestRMSNormKernel(unittest.TestCase):
    def test_matches_oracle(self) -> None:
        for h in WIDTHS:
            for dt in (torch.bfloat16, torch.float32):
                torch.manual_seed(0)
                x = (torch.randn(37, h, device="cuda") * 0.5).to(dt)
                w = torch.randn(h, device="cuda").to(dt)
                ref, got = op_norm(x, w, EPS).float(), rms_norm(x, w, EPS).float()
                rel = (ref - got).abs().max().item() / ref.abs().max().item()
                # bf16 实测逐位相同；fp32 只差归约顺序（~1e-7）。给一点余量以免换卡即红。
                tol = 1e-6 if dt == torch.float32 else 1e-3
                self.assertLessEqual(rel, tol, f"H={h} {dt}: 相对误差 {rel:.3e} > {tol}")

    def test_bf16_is_bitwise_identical_to_oracle(self) -> None:
        """哨兵断言：bf16 下**逐位相同**（实测性质）。

        为什么敢断言逐位：本核与 oracle 的操作顺序完全一致（同一套乘法次序、都在 fp32 累加），
        唯一区别是归约的分块方式。若哪天这条挂了，说明核或 oracle 的语义被改动过 —— 值得立刻查。
        """
        for h in WIDTHS:
            torch.manual_seed(0)
            x = (torch.randn(37, h, device="cuda") * 0.5).to(torch.bfloat16)
            w = torch.randn(h, device="cuda").to(torch.bfloat16)
            self.assertTrue(
                torch.equal(op_norm(x, w, EPS), rms_norm(x, w, EPS)), f"H={h} 不再逐位相同"
            )

    def test_odd_rows(self) -> None:
        """行数不是 2 的幂 / 只有 1 行也要对（decode batch 会取到这些值）。"""
        for rows in (1, 3, 17):
            torch.manual_seed(1)
            x = torch.randn(rows, 1536, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(1536, device="cuda", dtype=torch.bfloat16)
            self.assertTrue(torch.equal(op_norm(x, w, EPS), rms_norm(x, w, EPS)))


@unittest.skipUnless(CUDA, "Triton 核需要 CUDA")
class TestFusedAddRMSNormKernel(unittest.TestCase):
    def test_residual_sum_is_bitwise(self) -> None:
        """`x + residual` 必须与 torch 的 bf16 加法逐位相同（否则下游残差链会被改数值）。"""
        torch.manual_seed(0)
        x = (torch.randn(13, 1536, device="cuda") * 0.5).to(torch.bfloat16)
        r = (torch.randn(13, 1536, device="cuda") * 0.5).to(torch.bfloat16)
        w = torch.randn(1536, device="cuda", dtype=torch.bfloat16)
        _, s_got = fused_add_rms_norm(x, r, w, EPS)
        self.assertTrue(torch.equal(x + r, s_got))

    def test_bitwise_matches_high_precision_oracle(self) -> None:
        """1 ulp 差异的归因：与**高精度** oracle 逐位相同，与逐 op oracle 差 1 ulp。

        这说明差异来自**逐 op 版先丢了一次精度**（`s` 落成 bf16 后再读回来算方差），
        而不是我们的核有误差 —— 融合核把加法与归约放在同一份 fp32 中间值上。
        """
        for h in (1024, 1536):
            torch.manual_seed(0)
            x = (torch.randn(37, h, device="cuda") * 0.5).to(torch.bfloat16)
            r = (torch.randn(37, h, device="cuda") * 0.5).to(torch.bfloat16)
            w = torch.randn(h, device="cuda").to(torch.bfloat16)
            n_got, _ = fused_add_rms_norm(x, r, w, EPS)
            n_hi = hi_prec_fused_norm(x, r, w, EPS)
            n_low = op_norm(r + x, w, EPS)
            self.assertTrue(torch.equal(n_hi, n_got), f"H={h}: 与高精度 oracle 不再逐位相同")
            # 与逐 op 版的差异应当很小（实测 ~1 ulp），但**不为 0**
            rel = (n_low.float() - n_got.float()).abs().max().item() / n_low.float().abs().max().item()
            self.assertLessEqual(rel, 2 * 2**-8, f"H={h}: 与逐 op 版差异 {rel:.3e} 超出 2 ulp")

    def test_three_impls_agree(self) -> None:
        """三种 impl（实测口径）在同一输入上互相一致：torch / lib / 自研核。"""
        import torch.nn.functional as F

        torch.manual_seed(0)
        x = (torch.randn(37, 1536, device="cuda") * 0.5).to(torch.bfloat16)
        r = (torch.randn(37, 1536, device="cuda") * 0.5).to(torch.bfloat16)
        w = torch.randn(1536, device="cuda").to(torch.bfloat16)

        s = r + x
        outs = {
            "torch": op_norm(s, w, EPS),
            "lib": F.rms_norm(s, (1536,), w, EPS),
            "triton": fused_add_rms_norm(x, r, w, EPS)[0],
        }
        scale = outs["torch"].float().abs().max().item()
        for a, b in (("torch", "triton"), ("lib", "triton"), ("torch", "lib")):
            rel = (outs[a].float() - outs[b].float()).abs().max().item() / scale
            self.assertLessEqual(rel, 5e-2, f"{a} vs {b}: 相对差 {rel:.3e} 过大")


@unittest.skipUnless(CUDA, "Triton 核需要 CUDA")
class TestGuards(unittest.TestCase):
    """静默算错 → 显式失败（D4 思路）。"""

    def test_non_contiguous_rejected(self) -> None:
        x = torch.randn(4, 16, device="cuda").t()  # 非连续
        self.assertFalse(x.is_contiguous())
        with self.assertRaises(ValueError):
            rms_norm(x, torch.randn(4, device="cuda"), EPS)

    def test_shape_mismatch_rejected(self) -> None:
        with self.assertRaises(ValueError):
            fused_add_rms_norm(
                torch.randn(4, 8, device="cuda"), torch.randn(4, 9, device="cuda"),
                torch.randn(8, device="cuda"), EPS,
            )

    def test_dtype_mismatch_rejected(self) -> None:
        with self.assertRaises(ValueError):
            fused_add_rms_norm(
                torch.randn(4, 8, device="cuda", dtype=torch.bfloat16),
                torch.randn(4, 8, device="cuda", dtype=torch.float32),
                torch.randn(8, device="cuda", dtype=torch.bfloat16), EPS,
            )


class TestGuardsCPU(unittest.TestCase):
    def test_cpu_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            rms_norm(torch.randn(4, 8), torch.randn(8), EPS)


class TestRMSNormModule(unittest.TestCase):
    """`RMSNorm.forward_with_residual` 的契约（CPU 上测 torch / lib 两条路径）。"""

    def _layer(self, impl):
        from nano_vllm.models.qwen2 import RMSNorm

        torch.manual_seed(0)
        n = RMSNorm(32, EPS, impl)
        return n

    def test_returns_norm_and_sum(self) -> None:
        for impl in ("torch", "lib"):
            n = self._layer(impl)
            x = torch.randn(2, 5, 32)
            r = torch.randn(2, 5, 32)
            with torch.no_grad():
                normed, s = n.forward_with_residual(x, r)
            self.assertTrue(torch.equal(s, x + r), impl)
            self.assertTrue(torch.equal(normed, n._op_norm(x + r)), impl)

    def test_plain_forward_matches_op_norm(self) -> None:
        for impl in ("torch", "lib"):
            n = self._layer(impl)
            x = torch.randn(2, 5, 32)
            with torch.no_grad():
                self.assertTrue(torch.equal(n(x), n._op_norm(x)), impl)


if __name__ == "__main__":
    unittest.main()
