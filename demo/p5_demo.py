#!/usr/bin/env python3
"""P5 · Serving demo：用 httpx 连真实 uvicorn HTTP server，展示 4 种请求路径。

用法:
    python demo/p5_demo.py --model models/Qwen2.5-1.5B-Instruct

会自动:
    1. 在后台启动 uvicorn HTTP server (localhost:8000)
    2. 发送 4 种请求（completions/chat × stream/non-stream）
    3. 打印响应
    4. 关闭 server
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import asyncio
import json
import threading
import time

import httpx
import uvicorn

from nano_vllm.server.api import create_app


def start_server(app, port: int) -> threading.Thread:
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)

    def run():
        server.run()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    time.sleep(2)
    return t


async def demo_completions_nonstream(client, model: str):
    print("\n=== 1. /v1/completions 非流式 ===")
    resp = await client.post("/v1/completions", json={
        "model": model,
        "prompt": "The capital of France is",
        "max_tokens": 16,
        "temperature": 0.0,
    })
    data = resp.json()
    print(f"  text: {data['choices'][0]['text']!r}")
    print(f"  usage: {data['usage']}")


async def demo_completions_stream(client, model: str):
    print("\n=== 2. /v1/completions 流式 ===")
    resp = await client.post("/v1/completions", json={
        "model": model,
        "prompt": "The capital of France is",
        "max_tokens": 16,
        "temperature": 0.0,
        "stream": True,
    })
    text = ""
    async for line in resp.aiter_lines():
        if line.startswith("data: ") and line != "data: [DONE]":
            chunk = json.loads(line[6:])
            text += chunk["choices"][0]["text"]
    print(f"  text: {text!r}")


async def demo_chat_nonstream(client, model: str):
    print("\n=== 3. /v1/chat/completions 非流式 ===")
    resp = await client.post("/v1/chat/completions", json={
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        "max_tokens": 16,
        "temperature": 0.0,
    })
    data = resp.json()
    print(f"  content: {data['choices'][0]['message']['content']!r}")
    print(f"  usage: {data['usage']}")


async def demo_chat_stream(client, model: str):
    print("\n=== 4. /v1/chat/completions 流式 ===")
    resp = await client.post("/v1/chat/completions", json={
        "model": model,
        "messages": [
            {"role": "user", "content": "What is the capital of France?"},
        ],
        "max_tokens": 16,
        "temperature": 0.0,
        "stream": True,
    })
    text = ""
    async for line in resp.aiter_lines():
        if line.startswith("data: ") and line != "data: [DONE]":
            chunk = json.loads(line[6:])
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                text += delta["content"]
    print(f"  content: {text!r}")


async def main_async(args):
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    app = create_app(
        args.model, device=device, dtype=args.dtype,
        max_seq_len=args.max_seq_len, num_blocks=args.num_blocks,
        max_num_batched_tokens=args.budget,
    )

    print(f"启动 HTTP server @ localhost:{args.port} ...")
    start_server(app, args.port)

    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{args.port}") as client:
        await demo_completions_nonstream(client, args.model)
        await demo_completions_stream(client, args.model)
        await demo_chat_nonstream(client, args.model)
        await demo_chat_stream(client, args.model)

    print("\nDemo 完成。")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--max-seq-len", type=int, default=8192)
    p.add_argument("--num-blocks", type=int, default=512)
    p.add_argument("--budget", type=int, default=2048)
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()