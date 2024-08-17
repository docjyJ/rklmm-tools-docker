from ctypes import c_char_p, c_void_p
from enum import IntEnum
from typing import Any, Callable

import _binding

class _WithRef:
    __ref__: Any

    @classmethod
    def __from_ref__(cls, ref: Any):
        self = super().__new__(cls)
        self.__ref__ = ref
        return self

class LLMCallState(IntEnum):
    LLM_RUN_NORMAL = 0
    LLM_RUN_FINISH = 1
    LLM_RUN_ERROR = 2

    def __new__(cls, value):
        int_value = int(value)
        if int_value == cls.LLM_RUN_NORMAL:
            return cls.LLM_RUN_NORMAL
        elif int_value == cls.LLM_RUN_FINISH:
            return cls.LLM_RUN_FINISH
        else:
            return cls.LLM_RUN_ERROR

class RKLLMParam(_WithRef):
    model_path: str
    num_npu_core: int
    max_context_len: int
    max_new_tokens: int
    top_k: int
    top_p: float
    temperature: float
    repeat_penalty: float
    frequency_penalty: float
    presence_penalty: float
    mirostat: int
    mirostat_tau: float
    mirostat_eta: float
    logprobs: bool
    top_logprobs: int
    use_gpu: bool

    def __new__(cls, *args, **kwargs):
        if args or kwargs:
            return super().__new__(cls, _binding.RKLLMParam(*args, **kwargs))
        return cls._from_ref(_binding.rkllm_create_default_param())

    def __getattr__(self, key):
        if key == 'model_path':
            model_path = getattr(self.__ref__, 'model_path')
            return None if model_path is None else model_path.decode('utf-8')
        else:
            return getattr(self.__ref__, key)

    def __setattr__(self, key, value):
        if key == 'model_path':
            model_path = None if value is None else value.encode('utf-8')
            setattr(self.__ref__, 'model_path', model_path)
        else:
            setattr(self.__ref__, key, value)


class Token(_WithRef):
    logprob: float
    id: int

    def __new__(cls):
        return super().__new__(cls, _binding.Token())

    def __getattr__(self, key):
        return getattr(self.__ref__, key)

    def __setattr__(self, key, value):
        setattr(self.__ref__, key, value)

class RKLLMResult(_WithRef):
    text: str
    tokens: list[Token]
    num: int

    def __new__(cls):
        return cls._from_ref(_binding.RKLLMResult())

    def __getattr__(self, key):
        if key == 'text':
            text = getattr(self.__ref__, 'text')
            return None if text is None else text.decode('utf-8')
        elif key == 'tokens':
            raise NotImplementedError()
        else:
            return getattr(self.__ref__, key)

    def __setattr__(self, key, value):
        if key == 'text':
            text = None if value is None else value.encode('utf-8')
            setattr(self.__ref__, 'text', text)
        elif key == 'tokens':
            raise NotImplementedError()
        else:
            setattr(self.__ref__, key, value)

def _wrap_callback(callback):
    def _callback(result, _, state):
        callback(RKLLMResult.__from_ref__(result), LLMCallState(state))
    return _callback

class LLMHandle(_WithRef):
    def __init__(self, param: RKLLMParam, callback: Callable[[RKLLMResult, LLMCallState], None]):
        super().__init__()
        if _binding.rkllm_init(self.__ref__, param.__ref__, _wrap_callback(callback)):
            raise RuntimeError('Failed to initialize LLM handle')


    def __new__(cls, *args, **kwargs):
        return cls.__from_ref__(c_void_p())

    def __del__(self) -> None:
        if _binding.rkllm_destroy(self.__ref__):
            raise RuntimeError('Failed to destroy LLM handle')

    def run(self, prompt: str) -> None:
        if _binding.rkllm_run(self.__ref__, c_char_p(prompt.encode('utf-8')), c_void_p()):
            raise RuntimeError('Failed to run LLM handle')

    def abort(self) -> None:
        if _binding.rkllm_abort(self.__ref__):
            raise RuntimeError('Failed to abort LLM handle')
