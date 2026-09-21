"""P1 · eager 执行器：无 KV cache，每步将全部序列重算一遍（全项目最差性能基线）。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from nano_vllm.config import Qwen2Config
from nano_vllm.models.qwen2 import Qwen2ForCausalLM
from nano_vllm.sample.sampler import Sampler


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    max_new_tokens: int = 128


class NanoRunner:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        seed: int | None = None,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.config = Qwen2Config.from_json(Path(model_path) / "config.json")
        self.model = Qwen2ForCausalLM(self.config).to(device=device, dtype=dtype).eval()
        self.model.load_weights(model_path)
        self.sampler = Sampler(device)
        if seed is not None:
            self.sampler.seed(seed)

    @property
    def eos_ids(self) -> set[int]:
        return self.config.eos_ids

    @torch.no_grad()
    def forward_last_logits(self, ids: list[int]) -> torch.Tensor:
        t = torch.tensor([ids], dtype=torch.long, device=self.device)
        logits, _ = self.model(t)
        return logits[0, -1]

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: list[int],
        params: SamplingParams | None = None,
    ) -> list[int]:
        params = params or SamplingParams()
        ids = list(prompt_ids)
        out: list[int] = []
        for _ in range(params.max_new_tokens):
            logits = self.forward_last_logits(ids)
            tok = self.sampler.sample(
                logits, temperature=params.temperature, top_k=params.top_k, top_p=params.top_p
            )
            if tok in self.eos_ids:
                break
            out.append(tok)
            ids.append(tok)
        return out
