import os.path
from ctypes import c_int, c_float, c_char_p, c_bool, POINTER, Structure, CFUNCTYPE, c_void_p, CDLL, byref
from typing import Callable


class RKLLMParam(Structure):
    _fields_ = [
        ("model_path", c_char_p),
        ("num_npu_core", c_int),
        ("max_context_len", c_int),
        ("max_new_tokens", c_int),
        ("top_k", c_int),
        ("top_p", c_float),
        ("temperature", c_float),
        ("repeat_penalty", c_float),
        ("frequency_penalty", c_float),
        ("presence_penalty", c_float),
        ("mirostat", c_int),
        ("mirostat_tau", c_float),
        ("mirostat_eta", c_float),
        ("logprobs", c_bool),
        ("top_logprobs", c_int),
        ("use_gpu", c_bool)
    ]

class Token(Structure):
    _fields_ = [
        ("logprob", c_float),
        ("id", c_int)
    ]

class RKLLMResult(Structure):
    _fields_ = [
        ("text", c_char_p),
        ("tokens", POINTER(Token)),
        ("num", c_int)
    ]

_LLMResultCallback = CFUNCTYPE(None, POINTER(RKLLMResult), c_void_p, c_int)

_rkllm_lib = CDLL(os.path.join(os.path.dirname(__file__), 'librkllm-1.0.1-aarch64/librkllmrt.so'))

_rkllm_create_default_param = _rkllm_lib['rkllm_createDefaultParam']
_rkllm_create_default_param.argtypes = []
_rkllm_create_default_param.restype = RKLLMParam

_rkllm_init = _rkllm_lib['rkllm_init']
_rkllm_init.argtypes = [POINTER(c_void_p), POINTER(RKLLMParam), _LLMResultCallback]
_rkllm_init.restype = c_int

_rkllm_destroy = _rkllm_lib['rkllm_destroy']
_rkllm_destroy.argtypes = [c_void_p]
_rkllm_destroy.restype = c_int

_rkllm_run = _rkllm_lib['rkllm_run']
_rkllm_run.argtypes = [c_void_p, c_char_p, c_void_p]
_rkllm_run.restype = c_int

_rkllm_abort = _rkllm_lib['rkllm_abort']
_rkllm_abort.argtypes = [c_void_p]
_rkllm_abort.restype = c_int

def _wrap_callback(callback):
    def _callback(result, userdata, state):
        callback(result.contents if result else None, userdata, state)
    return _callback

def rkllm_create_default_param() -> RKLLMParam:
    return _rkllm_create_default_param()

def rkllm_init(handle: c_void_p, param: RKLLMParam, callback: Callable[[RKLLMResult, c_void_p, c_int], None]) -> c_int:
    return _rkllm_init(byref(handle), byref(param), _LLMResultCallback(callback))

def rkllm_destroy(handle: c_void_p) -> c_int:
    return _rkllm_destroy(handle)

def rkllm_run(handle: c_void_p, prompt: c_char_p, userdata: c_void_p) -> c_int:
    return _rkllm_run(handle, prompt, userdata)

def rkllm_abort(handle: c_void_p) -> c_int:
    return _rkllm_abort(handle)




