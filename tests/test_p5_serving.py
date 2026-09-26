"""P5 · Serving 正确性测试：HTTP streaming vs 非流式 vs 本地引擎一致。

CPU fp32 + 0.5B 模型，用 httpx ASGITransport 直连 FastAPI app（不起真实 HTTP server）。
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from nano_vllm.server.api import create_app

MODEL = "models/Qwen2.5-0.5B-Instruct"
PROMPT = "The capital of France is"
MAX_TOKENS = 4


class TestP5Serving(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app(
            MODEL, device="cpu", dtype="fp32",
            max_seq_len=512, num_blocks=64, max_num_batched_tokens=512,
        )

    async def _client(self):
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app),
            base_url="http://test",
        )

    async def test_completion_non_stream(self):
        """非流式 /v1/completions 返回完整文本。"""
        async with await self._client() as c:
            resp = await c.post("/v1/completions", json={
                "model": MODEL, "prompt": PROMPT,
                "max_tokens": MAX_TOKENS, "temperature": 0.0,
            })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["object"], "text_completion")
        self.assertEqual(data["choices"][0]["finish_reason"], "stop")
        self.assertTrue(len(data["choices"][0]["text"]) > 0)
        self.assertEqual(data["usage"]["completion_tokens"], MAX_TOKENS)

    async def test_completion_stream_vs_nonstream(self):
        """流式 /v1/completions 拼接 == 非流式输出。"""
        async with await self._client() as c:
            resp_non = await c.post("/v1/completions", json={
                "model": MODEL, "prompt": PROMPT,
                "max_tokens": MAX_TOKENS, "temperature": 0.0,
            })
            non_text = resp_non.json()["choices"][0]["text"]

            resp_stream = await c.post("/v1/completions", json={
                "model": MODEL, "prompt": PROMPT,
                "max_tokens": MAX_TOKENS, "temperature": 0.0,
                "stream": True,
            })
        self.assertEqual(resp_stream.status_code, 200)
        self.assertEqual(resp_stream.headers["content-type"], "text/event-stream; charset=utf-8")

        stream_text = ""
        done = False
        for line in resp_stream.text.split("\n"):
            if line.startswith("data: ") and line != "data: [DONE]":
                chunk = json.loads(line[6:])
                stream_text += chunk["choices"][0]["text"]
                if chunk["choices"][0]["finish_reason"] == "stop":
                    done = True
            elif line == "data: [DONE]":
                done = True

        self.assertTrue(done, "未收到 [DONE]")
        self.assertEqual(stream_text, non_text, f"stream='{stream_text}' != non='{non_text}'")

    async def test_chat_non_stream(self):
        """非流式 /v1/chat/completions 返回 assistant 消息。"""
        async with await self._client() as c:
            resp = await c.post("/v1/chat/completions", json={
                "model": MODEL,
                "messages": [{"role": "user", "content": PROMPT}],
                "max_tokens": MAX_TOKENS, "temperature": 0.0,
            })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["object"], "chat.completion")
        self.assertEqual(data["choices"][0]["message"]["role"], "assistant")
        self.assertTrue(len(data["choices"][0]["message"]["content"]) > 0)

    async def test_chat_stream_vs_nonstream(self):
        """流式 /v1/chat/completions 拼接 == 非流式输出。"""
        async with await self._client() as c:
            resp_non = await c.post("/v1/chat/completions", json={
                "model": MODEL,
                "messages": [{"role": "user", "content": PROMPT}],
                "max_tokens": MAX_TOKENS, "temperature": 0.0,
            })
            non_text = resp_non.json()["choices"][0]["message"]["content"]

            resp_stream = await c.post("/v1/chat/completions", json={
                "model": MODEL,
                "messages": [{"role": "user", "content": PROMPT}],
                "max_tokens": MAX_TOKENS, "temperature": 0.0,
                "stream": True,
            })

        stream_text = ""
        done = False
        for line in resp_stream.text.split("\n"):
            if line.startswith("data: ") and line != "data: [DONE]":
                chunk = json.loads(line[6:])
                delta = chunk["choices"][0]["delta"]
                if "content" in delta:
                    stream_text += delta["content"]
                if chunk["choices"][0]["finish_reason"] == "stop":
                    done = True
            elif line == "data: [DONE]":
                done = True

        self.assertTrue(done, "未收到 [DONE]")
        self.assertEqual(stream_text, non_text, f"stream='{stream_text}' != non='{non_text}'")


if __name__ == "__main__":
    unittest.main()