"""P1 · 自研模型 golden dump：与 dump_hf.py 对称，输入从参考 golden 的 meta 读取保证一致。

用法:
  python golden/dump_nano.py --model models/Qwen2.5-1.5B-Instruct \
      --ref golden/Qwen2.5-1.5B-Instruct_bf16 --out golden/nano_bf16
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import platform


import numpy as np
import torch
import transformers

from nano_vllm.model_executor.runner import NanoRunner


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--ref", required=True, help="参考 golden 目录（读取其 meta.json 的输入 ids）")
    p.add_argument("--out", default=None)
    p.add_argument("--gen-tokens", type=int, default=None, help="覆盖 ref 的生成长度（CPU 慢机可调小）")
    args = p.parse_args()

    ref_meta = json.loads((Path(args.ref) / "meta.json").read_text(encoding="utf-8"))
    input_ids = ref_meta["input_ids"]
    gen_tokens = args.gen_tokens or ref_meta["gen_tokens"]

    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    print(f"[dump-nano] model={args.model} device={args.device} dtype={args.dtype} seq={len(input_ids)}")
    runner = NanoRunner(args.model, device=args.device, dtype=dtype)

    t = torch.tensor([input_ids], dtype=torch.long, device=args.device)
    with torch.no_grad():
        logits, all_hidden = runner.model(t, output_hidden_states=True)

    out_dir = Path(args.out or f"golden/nano_{args.dtype}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, h in enumerate(all_hidden):
        np.save(out_dir / f"hidden_{i:02d}.npy", h.float().cpu().numpy())
    np.save(out_dir / "logits.npy", logits.float().cpu().numpy())

    eos = set(runner.eos_ids)
    gc_path = Path(args.model) / "generation_config.json"
    if gc_path.exists():
        gc = json.loads(gc_path.read_text(encoding="utf-8"))
        e = gc.get("eos_token_id")
        eos |= set(e) if isinstance(e, list) else ({e} if e is not None else set())

    ids = list(input_ids)
    gen = []
    for _ in range(gen_tokens):
        last = runner.forward_last_logits(ids)
        tok = int(last.argmax())
        gen.append(tok)
        ids.append(tok)
        if tok in eos:
            break
    np.save(out_dir / "greedy_ids.npy", np.asarray([input_ids + gen], dtype=np.int64))

    meta = {
        "engine": "nano",
        "ref_meta": args.ref,
        "model_path": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "seed": ref_meta["seed"],
        "seq_len": len(input_ids),
        "gen_tokens": gen_tokens,
        "input_ids": input_ids,
        "greedy_len": len(gen),
        "num_hidden_states": len(all_hidden),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[dump-nano] done -> {out_dir}  ({len(all_hidden)} hidden, "
        f"greedy {meta['greedy_len']} tok)"
    )


if __name__ == "__main__":
    main()