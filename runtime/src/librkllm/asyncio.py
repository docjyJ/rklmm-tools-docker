from asyncio import Lock

from . import LLMHandle as _LLMHandle, RKLLMResult, LLMCallState, RKLLMParam

class LLMHandle:
    def __init__(self, param: RKLLMParam):
        self._handle = _LLMHandle(param, self._callback)
        self._buffer: list[RKLLMResult] | None = None
        self._mutex = Lock()

    def _callback(self, result: RKLLMResult, state: LLMCallState):
        if not self._mutex.locked():
            raise RuntimeError('Callback called without lock')
        if self._buffer is None:
            raise RuntimeError('Callback called without buffer')
        if state == LLMCallState.LLM_RUN_NORMAL:
            self._buffer.append(result)
        elif state == LLMCallState.LLM_RUN_FINISH:
            self._buffer.append(result)
            self._mutex.release()

    async def run(self, prompt: str) -> list[RKLLMResult]:
        if self._mutex.locked():
            raise RuntimeError('Handle is already running')
        await self._mutex.acquire()

        self._buffer = []
        self._handle.run(prompt)
        await self._mutex.acquire()
        if self._buffer is None:
            raise RuntimeError('Failed to run LLM handle')
        buffer = self._buffer
        self._buffer = None
        self._mutex.release()
        return buffer

