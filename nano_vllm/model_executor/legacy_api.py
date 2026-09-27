"""P1/P2 历史入口（对拍基线）：eager 重算 + 连续 KV cache + 静态批拼批。

从 runner.py 拆出（E5），NanoRunner 保留薄委托转发到此。
这些是所有加速比的分母与 P3 完成标准要求的对照物，不可删。

路径说明:
  - generate_eager:   P1 use_cache=False，每步从头重算全部 token（最差基线）
  - generate_cached:  P2 use_cache=True，连续 KV cache prefill+decode
  - generate_batch:   P2 静态批，padding 拼批 + 静态推进（padding 浪费率证据产地）
"""
from __future__ import annotations

import torch

from nano_vllm.kv_cache import KVCache


def forward_last_logits(model, ids: list[int], device: str) -> torch.Tensor:
    """P1 eager 辅助：从头算全部 token，返回最后一个 token 的 logits。"""
    t = torch.tensor([ids], dtype=torch.long, device=device)
    last_idx = torch.tensor([len(ids) - 1], device=device)
    logits, _ = model(t, last_idx=last_idx)
    return logits[0]


def _build_prefill_mask(attn_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    b, s = attn_mask.shape
    causal = torch.tril(torch.ones(s, s, device=attn_mask.device, dtype=torch.bool))
    full = causal[None, None, :, :] & attn_mask[:, None, None, :].bool()
    mask = torch.zeros(b, 1, s, s, device=attn_mask.device, dtype=dtype)
    mask.masked_fill_(~full, float("-inf"))
    return mask


def _build_decode_mask(valid_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    b, total = valid_mask.shape
    mask = torch.zeros(b, 1, 1, total, device=valid_mask.device, dtype=dtype)
    mask.masked_fill_(~valid_mask[:, None, None, :].bool(), float("-inf"))
    return mask


def generate_eager(runner, prompt_ids: list[int], params) -> list[int]:
    """P1 eager：每步从头重算（use_cache=False）。"""
    ids = list(prompt_ids)
    out: list[int] = []
    for _ in range(params.max_new_tokens):
        logits = forward_last_logits(runner.model, ids, runner.device)
        tok = runner.sampler.sample(
            logits, temperature=params.temperature,
            top_k=params.top_k, top_p=params.top_p,
        )
        if tok in runner.eos_ids:
            break
        out.append(tok)
        ids.append(tok)
    return out


def generate_cached(runner, prompt_ids: list[int], params) -> list[int]:
    """P2 连续 KV cache：prefill + decode（use_cache=True）。"""
    kv = runner.kv_cache
    kv.reset()
    t = torch.tensor([prompt_ids], dtype=torch.long, device=runner.device)
    last_idx = torch.tensor([len(prompt_ids) - 1], device=runner.device)
    logits, _ = runner.model(t, kv_cache=kv, is_prefill=True, cache_seq_len=0, last_idx=last_idx)
    kv.seq_len = len(prompt_ids)
    last_logits = logits[0]
    out: list[int] = []
    for _ in range(params.max_new_tokens):
        tok = runner.sampler.sample(
            last_logits, temperature=params.temperature,
            top_k=params.top_k, top_p=params.top_p,
        )
        if tok in runner.eos_ids:
            break
        out.append(tok)
        pos = kv.seq_len
        t = torch.tensor([[tok]], dtype=torch.long, device=runner.device)
        logits, _ = runner.model(t, kv_cache=kv, is_prefill=False, cache_seq_len=pos)
        kv.seq_len = pos + 1
        last_logits = logits[0, -1]
    return out


def generate_batch(runner, prompts: list[list[int]], params) -> tuple[list[list[int]], dict]:
    """P2 静态批：padding 拼批 + 静态推进（generate_batch）。"""
    b = len(prompts)
    prompt_lens = [len(p) for p in prompts]
    max_prompt = max(prompt_lens)
    device = runner.device
    dtype = runner.dtype

    input_ids = torch.full((b, max_prompt), runner.pad_id, dtype=torch.long, device=device)
    attn = torch.zeros(b, max_prompt, device=device, dtype=dtype)
    for i, p in enumerate(prompts):
        input_ids[i, : len(p)] = torch.tensor(p, dtype=torch.long, device=device)
        attn[i, : len(p)] = 1.0

    batch_cache = KVCache(
        num_layers=runner.config.num_hidden_layers,
        num_kv_heads=runner.config.num_key_value_heads,
        head_dim=runner.config.head_dim,
        max_seq_len=runner.kv_cache.max_seq_len,
        batch_size=b,
        dtype=dtype,
        device=device,
    )
    batch_cache.reset()

    prefill_pos = torch.zeros(b, max_prompt, dtype=torch.long, device=device)
    for i, l in enumerate(prompt_lens):
        prefill_pos[i, :l] = torch.arange(l, device=device)

    prefill_mask = _build_prefill_mask(attn, dtype)
    last_idx = torch.tensor([l - 1 for l in prompt_lens], device=device)
    logits, _ = runner.model(
        input_ids, kv_cache=batch_cache, is_prefill=True,
        cache_seq_len=0, attn_mask=prefill_mask, position_ids=prefill_pos,
        last_idx=last_idx,
    )
    batch_cache.seq_len = max_prompt

    last_logits = logits
    valid_mask = attn.clone()
    seq_lens = list(prompt_lens)
    outs: list[list[int]] = [[] for _ in range(b)]
    done = [False] * b
    total_tokens = sum(prompt_lens)

    for _ in range(params.max_new_tokens):
        tokens = []
        for i in range(b):
            if done[i]:
                tokens.append(runner.pad_id)
                continue
            tok = runner.sampler.sample(
                last_logits[i], temperature=params.temperature,
                top_k=params.top_k, top_p=params.top_p,
            )
            if tok in runner.eos_ids:
                done[i] = True
            else:
                outs[i].append(tok)
            tokens.append(tok if not done[i] else runner.pad_id)

        if all(done):
            break

        t = torch.tensor([tokens], dtype=torch.long, device=device).T
        decode_pos = torch.tensor([seq_lens], dtype=torch.long, device=device).T
        new_valid = torch.tensor(
            [[0 if done[i] else 1] for i in range(b)],
            device=device, dtype=dtype,
        )
        valid_mask = torch.cat([valid_mask, new_valid], dim=1)
        total_tokens += sum(0 if d else 1 for d in done)

        decode_mask = _build_decode_mask(valid_mask, dtype)
        logits, _ = runner.model(
            t, kv_cache=batch_cache, is_prefill=False,
            cache_seq_len=batch_cache.seq_len, attn_mask=decode_mask,
            position_ids=decode_pos,
        )
        batch_cache.seq_len += 1
        for i in range(b):
            if not done[i]:
                seq_lens[i] += 1
        last_logits = logits[:, -1]

    waste = 1.0 - total_tokens / (b * batch_cache.seq_len) if batch_cache.seq_len > 0 else 1.0
    stats = {
        "batch_size": b,
        "max_prompt_len": max_prompt,
        "total_tokens": total_tokens,
        "batch_slots": b * batch_cache.seq_len,
        "padding_waste_rate": waste,
        "kv_waste_rate": batch_cache.waste_rate,
    }
    return outs, stats