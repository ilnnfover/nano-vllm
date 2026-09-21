#!/usr/bin/env python3
"""P0 · benchmark 数据集生成：固定 seed、确定性、四类负载。

类别:
  short         8 条 × prompt 128 + out 64
  medium        8 条 × prompt 1024 + out 128
  long          1 条 × prompt 8192 + out 256
  shared_prefix 4 条共享 32768 前缀 + 各自 64 独立 + out 128  (P6 prefix caching 用)

产物: bench/data/datasets.json （prompt_ids 为 token id 序列，确定性可复现）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SPEC = {
    "short": {"count": 8, "prompt_len": 128, "output_len": 64},
    "medium": {"count": 8, "prompt_len": 1024, "output_len": 128},
    "long": {"count": 1, "prompt_len": 8192, "output_len": 256},
    "shared_prefix": {"count": 4, "prompt_len": 32768 + 64, "output_len": 128},
}
SHARED_PREFIX_LEN = 32768
VOCAB = 151936
ID_LO, ID_HI = 100, VOCAB - 400  # 避开 Qwen2 特殊 token 区 (≥151643)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="bench/data/datasets.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vocab", type=int, default=VOCAB)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    lo, hi = ID_LO, min(ID_HI, args.vocab - 400)
    shared_prefix = rng.integers(lo, hi, size=SHARED_PREFIX_LEN).tolist()

    requests, rid = [], 0
    for cat, spec in SPEC.items():
        for i in range(spec["count"]):
            if cat == "shared_prefix":
                uniq = rng.integers(lo, hi, size=spec["prompt_len"] - SHARED_PREFIX_LEN)
                prompt_ids = shared_prefix + uniq.tolist()
            else:
                prompt_ids = rng.integers(lo, hi, size=spec["prompt_len"]).tolist()
            requests.append(
                {
                    "id": rid,
                    "category": cat,
                    "prompt_len": len(prompt_ids),
                    "output_len": spec["output_len"],
                    "prompt_ids": prompt_ids,
                }
            )
            rid += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": args.seed,
        "vocab": args.vocab,
        "id_range": [lo, hi],
        "shared_prefix_len": SHARED_PREFIX_LEN,
        "spec": SPEC,
        "requests": requests,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"[dataset] {len(requests)} 条 -> {out}")
    for cat, spec in SPEC.items():
        print(f"  {cat:<14} {spec['count']} 条 × prompt {spec['prompt_len']} + out {spec['output_len']}")


if __name__ == "__main__":
    main()