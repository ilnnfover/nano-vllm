"""P1 · 采样器：greedy / temperature / top-k / top-p。

处理顺序对齐 vLLM v1 sampler（temperature 缩放 → top-k → top-p → softmax → multinomial）。
temperature<=0 视为 greedy（argmax），保证 greedy 主路径零随机性。
"""
from __future__ import annotations

import torch


class Sampler:
    def __init__(self, device: str | torch.device) -> None:
        dev = torch.device(device)
        gen_device = "cuda" if dev.type == "cuda" else "cpu"
        self.generator = torch.Generator(gen_device)

    def seed(self, seed: int) -> None:
        self.generator.manual_seed(seed)

    @torch.no_grad()
    def sample(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> int:
        if temperature <= 0.0:
            return int(logits.argmax())
        logits = logits.float() / temperature

        if top_k > 0:
            kth = torch.topk(logits, min(top_k, logits.shape[-1]))[0][-1]
            logits[logits < kth] = float("-inf")

        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cum = torch.cumsum(probs, dim=-1)
            mask = (cum - probs) > top_p
            mask[0] = False
            logits[sorted_idx[mask]] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1, generator=self.generator))