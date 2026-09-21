"""验证 bf16 hidden diff 是舍入正常累积还是 bug。

跑三组对比（同模型同输入同 device）：
  A) HF fp32 vs HF bf16   → bf16 相对 fp32 的固有精度损失（参考量级）
  B) nano fp32 vs nano bf16 → 同上，自研侧
  C) nano bf16 vs HF bf16   → 实现差异（应 << A，否则有 bug）

判据：C 的 max diff 若与 A 同量级 → bf16 正常累积；若 C >> A → 实现有 bug。
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import torch
import transformers

from nano_vllm.model_executor.runner import NanoRunner

MODEL = "models/Qwen2.5-0.5B-Instruct"
PROMPT = "The meaning of life is"
DEVICE = "cpu"


def hf_hidden(dtype: torch.dtype) -> list[np.ndarray]:
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    ids = tok(PROMPT, return_tensors="pt").input_ids.to(DEVICE)
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype).to(DEVICE).eval()
    with torch.no_grad():
        out = model(ids, output_hidden_states=True, use_cache=False)
    return [h.float().cpu().numpy() for h in out.hidden_states]


def nano_hidden(dtype: torch.dtype) -> list[np.ndarray]:
    r = NanoRunner(MODEL, device=DEVICE, dtype=dtype)
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    ids = tok(PROMPT, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        _, hs = r.model(ids, output_hidden_states=True)
    return [h.float().cpu().numpy() for h in hs]


def max_diff(a: list[np.ndarray], b: list[np.ndarray]) -> float:
    return max(float(np.abs(x - y).max()) for x, y in zip(a, b))


def main() -> None:
    print(f"[verify] model={MODEL} device={DEVICE}")
    print("[verify] loading HF fp32 ...")
    hf32 = hf_hidden(torch.float32)
    print("[verify] loading HF bf16 ...")
    hf16 = hf_hidden(torch.bfloat16)
    print("[verify] loading nano fp32 ...")
    n32 = nano_hidden(torch.float32)
    print("[verify] loading nano bf16 ...")
    n16 = nano_hidden(torch.bfloat16)

    a = max_diff(hf32, hf16)
    b = max_diff(n32, n16)
    c = max_diff(n16, hf16)
    d = max_diff(n32, hf32)

    print()
    print("=== 0.5B CPU hidden max-diff 汇总 ===")
    print(f"A) HF fp32 vs HF bf16     : {a:.4f}   (bf16 固有精度损失)")
    print(f"B) nano fp32 vs nano bf16 : {b:.4f}   (自研侧 bf16 损失)")
    print(f"C) nano bf16 vs HF bf16   : {c:.4f}   (实现差异，应 << A)")
    print(f"D) nano fp32 vs HF fp32   : {d:.4f}   (fp32 逻辑差异，应 ~0)")
    print()
    if c < a * 2:
        print(f"结论: C({c:.4f}) 与 A({a:.4f}) 同量级 → bf16 diff 是舍入正常累积，非 bug")
    else:
        print(f"结论: C({c:.4f}) >> A({a:.4f}) → 实现存在 bf16 路径 bug，需排查")


if __name__ == "__main__":
    main()