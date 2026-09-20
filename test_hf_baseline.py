#!/usr/bin/env python3
"""
HF transformers 基线 —— 你自研引擎要打败的参照物。

运行:
    source ~/venvs/dev/bin/activate
    python test_hf_baseline.py

三个产出:
  1. 打印模型结构, 顺便算出 KV Cache 到底占多少显存(分页显存的动机来源)
  2. 对比「每步重算全序列」和「复用 KV Cache」两种解码, 看差距
  3. 给出可复用的吞吐基线数字
"""
from __future__ import annotations

import os

# 国内直连 huggingface.co 会超时, 走镜像。必须在 import transformers 之前设置。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_NEW = int(os.environ.get("MAX_NEW", "64"))

assert torch.cuda.is_available(), "torch 没拿到 GPU"

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()

cfg = model.config
n_layers = cfg.num_hidden_layers
n_heads = cfg.num_attention_heads
n_kv = cfg.num_key_value_heads
head_dim = getattr(cfg, "head_dim", cfg.hidden_size // n_heads)

print("=" * 70)
print(" 1. 模型结构与 KV Cache 规模")
print("=" * 70)
print(f"模型            {MODEL}")
print(f"层数            {n_layers}")
print(f"注意力头 / KV头 {n_heads} / {n_kv}   {'← GQA, KV 头更少' if n_kv < n_heads else ''}")
print(f"hidden / head   {cfg.hidden_size} / {head_dim}")

per_token = n_layers * 2 * n_kv * head_dim * 2  # K 和 V 各一份, bf16 = 2 字节
print(f"\n每 token KV    {per_token / 1024:.1f} KiB  (={n_layers}层 × 2(K,V) × {n_kv}头 × {head_dim}维 × 2字节)")
for L in (512, 2048, 8192):
    print(f"  {L:>5} tokens  →  {L * per_token / 2**20:>7.1f} MiB")
print("\n注意这是【单条序列】。并发 N 条就乘 N —— 不分页的话碎片会吃掉一大半。")

msgs = [{"role": "user", "content": "用一句话解释什么是 KV Cache。"}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt").input_ids.cuda()
print(f"\nprompt tokens: {ids.shape[-1]}")


def gen_naive(input_ids: torch.Tensor, max_new: int) -> tuple[torch.Tensor, float, float]:
    """每步把整个序列重新喂进去。教学对照组 —— 它慢在哪, 就是 KV Cache 存在的理由。"""
    cur = input_ids.clone()
    with torch.no_grad():
        logits = model(cur).logits[:, -1, :]
    first = logits.argmax(-1, keepdim=True)
    cur = torch.cat([cur, first], dim=-1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(max_new - 1):
        with torch.no_grad():
            logits = model(cur).logits[:, -1, :]
        cur = torch.cat([cur, logits.argmax(-1, keepdim=True)], dim=-1)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return cur, 0.0, dt


def gen_cached(input_ids: torch.Tensor, max_new: int) -> tuple[torch.Tensor, float, float]:
    """标准做法: prefill 一次, 之后每步只喂上一步的 token + past_key_values。"""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
    nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
    torch.cuda.synchronize()
    prefill = time.perf_counter() - t0

    past = out.past_key_values
    cur = torch.cat([input_ids, nxt], dim=-1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(max_new - 1):
        with torch.no_grad():
            out = model(nxt, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        cur = torch.cat([cur, nxt], dim=-1)
    torch.cuda.synchronize()
    decode = time.perf_counter() - t0
    return cur, prefill, decode


print("\n" + "=" * 70)
print(f" 2. 有无 KV Cache 对比 (生成 {MAX_NEW} tokens)")
print("=" * 70)

with torch.no_grad():  # 预热, 避免把 CUDA 初始化时间算进去
    model(ids, use_cache=True)

naive_ids, _, naive_dt = gen_naive(ids, MAX_NEW)
cached_ids, prefill_dt, decode_dt = gen_cached(ids, MAX_NEW)

same = torch.equal(naive_ids[0, -MAX_NEW:], cached_ids[0, -MAX_NEW:])
print(f"{'':<14}{'prefill':>10}{'decode':>10}{'合计':>10}{'tok/s':>10}")
print(f"{'不用 cache':<14}{'-':>10}{naive_dt:>9.2f}s{naive_dt:>9.2f}s{MAX_NEW / naive_dt:>10.1f}")
print(f"{'用 KV cache':<14}{prefill_dt:>9.3f}s{decode_dt:>9.2f}s{prefill_dt + decode_dt:>9.2f}s{MAX_NEW / (prefill_dt + decode_dt):>10.1f}")
print(f"\n加速比 {naive_dt / (prefill_dt + decode_dt):.2f}x   输出一致: {same}")

print("\n" + "=" * 70)
print(" 3. 生成结果")
print("=" * 70)
print(tok.decode(cached_ids[0, ids.shape[-1]:], skip_special_tokens=True))
print(f"\n显存峰值 {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")