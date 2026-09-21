"""P1 · config 驱动的模型架构参数，全部来自 config.json，不写死任何模型数值。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Qwen2Config:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    hidden_act: str
    tie_word_embeddings: bool
    max_position_embeddings: int
    head_dim: int
    eos_token_id: int | list[int]
    bos_token_id: int | None

    @property
    def num_kv_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def eos_ids(self) -> set[int]:
        e = self.eos_token_id
        return set(e) if isinstance(e, list) else {e}

    @classmethod
    def from_json(cls, path: str | Path) -> "Qwen2Config":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        head_dim = raw.get("head_dim") or raw["hidden_size"] // raw["num_attention_heads"]
        if raw["num_attention_heads"] % raw["num_key_value_heads"] != 0:
            raise ValueError("num_attention_heads 必须能整除 num_key_value_heads")
        if raw["hidden_act"] != "silu":
            raise ValueError(f"P1 仅支持 silu, got: {raw['hidden_act']}")
        return cls(
            hidden_size=raw["hidden_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=raw["num_attention_heads"],
            num_key_value_heads=raw["num_key_value_heads"],
            intermediate_size=raw["intermediate_size"],
            vocab_size=raw["vocab_size"],
            rms_norm_eps=raw["rms_norm_eps"],
            rope_theta=raw["rope_theta"],
            hidden_act=raw["hidden_act"],
            tie_word_embeddings=raw["tie_word_embeddings"],
            max_position_embeddings=raw["max_position_embeddings"],
            head_dim=head_dim,
            eos_token_id=raw["eos_token_id"],
            bos_token_id=raw.get("bos_token_id"),
        )