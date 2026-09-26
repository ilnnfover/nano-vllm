"""P5 · OpenAI 兼容协议模型（pydantic v2）。"""
from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field


class CompletionRequest(BaseModel):
    """POST /v1/completions 请求体。"""
    model: str
    prompt: str
    max_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = -1
    stream: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """POST /v1/chat/completions 请求体。"""
    model: str
    messages: list[ChatMessage]
    max_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = -1
    stream: bool = False


def _ts() -> int:
    return int(time.time())


def _cid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def make_completion_response(text: str, model: str, prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "id": _cid("cmpl"),
        "object": "text_completion",
        "created": _ts(),
        "model": model,
        "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def make_completion_chunk(text: str, model: str, finish: bool) -> dict:
    return {
        "id": _cid("cmpl"),
        "object": "text_completion",
        "created": _ts(),
        "model": model,
        "choices": [{"text": text, "index": 0, "finish_reason": "stop" if finish else None}],
    }


def make_chat_response(text: str, model: str, prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "id": _cid("chatcmpl"),
        "object": "chat.completion",
        "created": _ts(),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def make_chat_chunk(text: str, model: str, finish: bool) -> dict:
    return {
        "id": _cid("chatcmpl"),
        "object": "chat.completion.chunk",
        "created": _ts(),
        "model": model,
        "choices": [{"index": 0, "delta": {"content": text} if text else {}, "finish_reason": "stop" if finish else None}],
    }
