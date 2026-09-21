#!/usr/bin/env python3
"""P0 · 正确性 golden 生成：对 HF transformers 逐层 dump 中间量。

产物（fp32 落盘，供 P1 起所有版本对拍）:
  hidden_{i:02d}.npy   # i=0 为 embedding 输出, 1..N 为各 decoder layer 输出
  logits.npy           # 最后一步全部位置 logits
  greedy_ids.npy       # greedy 续写结果（含 prompt）
  meta.json            # 输入 ids / config / 库版本 / 全部复现参数

方法论: 输入用固定 token id 序列（非文本+tokenizer，避免 tokenizer 版本漂移）。
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import torch
import transformers


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    p.add_argument("--device", default=None, help="默认 cuda，不可用回退 cpu")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--gen-tokens", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None, help="默认 golden/<模型名>_<dtype>")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    torch.manual_seed(args.seed)

    print(f"[dump] model={args.model} device={device} dtype={args.dtype} seed={args.seed}")
    from transformers import AutoModelForCausalLM

    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, attn_implementation="sdpa"
        )
        .to(device)
        .eval()
    )
    cfg = model.config

    # 固定 token id 序列，避开词表尾部特殊 token 区（Qwen2 special ids ≥ 151643）
    rng = np.random.default_rng(args.seed)
    ids = rng.integers(100, cfg.vocab_size - 400, size=args.seq_len)
    input_ids = torch.tensor(np.asarray([ids]), dtype=torch.long, device=device)

    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)

    out_dir = Path(args.out or f"golden/{Path(args.model).name}_{args.dtype}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, h in enumerate(out.hidden_states):
        np.save(out_dir / f"hidden_{i:02d}.npy", h.float().cpu().numpy())
    np.save(out_dir / "logits.npy", out.logits.float().cpu().numpy())

    with torch.no_grad():
        gen = model.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=args.gen_tokens,
            pad_token_id=cfg.eos_token_id,
        )
    np.save(out_dir / "greedy_ids.npy", gen.cpu().numpy())

    meta = {
        "model_path": str(args.model),
        "device": device,
        "dtype": args.dtype,
        "attn_implementation": "sdpa",
        "seed": args.seed,
        "seq_len": args.seq_len,
        "gen_tokens": args.gen_tokens,
        "input_ids": ids.tolist(),
        "greedy_len": int(gen.shape[1] - input_ids.shape[1]),
        "num_hidden_states": len(out.hidden_states),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "config": cfg.to_dict(),
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    n_layers = len(out.hidden_states) - 1
    print(
        f"[dump] done -> {out_dir}  ({n_layers} layers + embed, "
        f"hidden {tuple(out.hidden_states[0].shape)}, greedy {meta['greedy_len']} tok)"
    )


if __name__ == "__main__":
    main()