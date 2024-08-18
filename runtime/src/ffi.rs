pub type LLMHandle = *mut std::ffi::c_void;


pub type LLMCallState = std::ffi::c_uint;
pub const LLM_RUN_NORMAL: LLMCallState = 0;
pub const LLM_RUN_FINISH: LLMCallState = 1;
pub const LLM_RUN_ERROR: LLMCallState = 2;


#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct RKLLMParam {
    pub model_path: *const std::ffi::c_char,
    pub num_npu_core: i32,
    pub max_context_len: i32,
    pub max_new_tokens: i32,
    pub top_k: i32,
    pub top_p: f32,
    pub temperature: f32,
    pub repeat_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub mirostat: i32,
    pub mirostat_tau: f32,
    pub mirostat_eta: f32,
    pub logprobs: bool,
    pub top_logprobs: i32,
    pub use_gpu: bool,
}


#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct Token {
    pub logprob: f32,
    pub id: std::ffi::c_int,
}


#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct RKLLMResult {
    pub text: *const std::ffi::c_char,
    pub tokens: *mut Token,
    pub num: std::ffi::c_int,
}


type LLMResultCallback = Option<
    unsafe extern "C" fn(
        result: *mut RKLLMResult,
        userdata: *mut std::ffi::c_void,
        state: LLMCallState,
    ),
>;


extern "C" {
    pub fn rkllm_createDefaultParam() -> RKLLMParam;

    pub fn rkllm_init(
        handle: *mut LLMHandle,
        param: RKLLMParam,
        callback: LLMResultCallback,
    ) -> std::ffi::c_int;

    pub fn rkllm_destroy(handle: LLMHandle) -> std::ffi::c_int;

    pub fn rkllm_run(
        handle: LLMHandle,
        prompt: *const std::ffi::c_char,
        userdata: *mut std::ffi::c_void,
    ) -> std::ffi::c_int;

    pub fn rkllm_abort(handle: LLMHandle) -> std::ffi::c_int;
}