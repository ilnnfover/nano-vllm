"""P5 · AsyncEngineCore：async 引擎循环，后台 step + Queue 分发。

包装同步 EngineCore（不改原生），用 asyncio.to_thread(step) 扔线程池。
后台单 step 循环 task 独立于 generate_stream()，token 通过 asyncio.Queue
分发给各请求——SSE 发送慢不阻塞其他请求的 token 产出。
"""
from __future__ import annotations

import asyncio

from nano_vllm.engine.core import EngineCore
from nano_vllm.engine.sequence import SamplingParams, Sequence


class AsyncEngineCore:
    def __init__(self, engine: EngineCore) -> None:
        self.engine = engine
        self._token_queues: dict[int, asyncio.Queue[int | None]] = {}
        self._step_task: asyncio.Task | None = None

    def _ensure_step_loop(self) -> None:
        if (
            self._step_task is None
            or self._step_task.done()
            or self._step_task.get_loop() is not asyncio.get_running_loop()
        ):
            self._step_task = asyncio.create_task(self._step_loop())

    async def _step_loop(self) -> None:
        """后台单 step 循环：to_thread(step) → token 分发到各请求 queue。"""
        while self.engine.scheduler.has_requests():
            finished, sampled = await asyncio.to_thread(self.engine.step)
            for seq_id, token_id in sampled.items():
                q = self._token_queues.get(seq_id)
                if q is not None:
                    await q.put(token_id)
            for seq in finished:
                q = self._token_queues.get(seq.seq_id)
                if q is not None:
                    await q.put(None)

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: SamplingParams | None = None,
    ) -> list[int]:
        """非流式：提交请求，等完成返回全部 output token。"""
        seq = self.engine.add_request(prompt_ids, sampling_params)
        q: asyncio.Queue[int | None] = asyncio.Queue()
        self._token_queues[seq.seq_id] = q
        self._ensure_step_loop()
        tokens: list[int] = []
        try:
            while True:
                token = await q.get()
                if token is None:
                    break
                tokens.append(token)
        finally:
            self._token_queues.pop(seq.seq_id, None)
        return tokens

    async def generate_stream(
        self,
        prompt_ids: list[int],
        sampling_params: SamplingParams | None = None,
    ):
        """流式：yield 每个 output token，结束时自然退出。"""
        seq = self.engine.add_request(prompt_ids, sampling_params)
        q: asyncio.Queue[int | None] = asyncio.Queue()
        self._token_queues[seq.seq_id] = q
        self._ensure_step_loop()
        try:
            while True:
                token = await q.get()
                if token is None:
                    break
                yield token
        finally:
            self._token_queues.pop(seq.seq_id, None)

    async def aclose(self) -> None:
        if self._step_task is not None and not self._step_task.done():
            self._step_task.cancel()
            try:
                await self._step_task
            except asyncio.CancelledError:
                pass
