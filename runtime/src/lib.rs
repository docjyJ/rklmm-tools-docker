mod ffi;

pub struct LLMHandle {
    inner: ffi::LLMHandle,
}
pub enum LLMCallState {
    LLMRunNormal,
    LLMRunFinish,
    LLMRunError,
}

pub use ffi::RKLLMParam;
pub use ffi::RKLLMResult;
pub use ffi::Token;
type LLMResultCallback = fn(&LLMCallState, &RKLLMResult);


impl TryFrom<u32> for LLMCallState {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            ffi::LLM_RUN_NORMAL => Ok(LLMCallState::LLMRunNormal),
            ffi::LLM_RUN_FINISH => Ok(LLMCallState::LLMRunFinish),
            ffi::LLM_RUN_ERROR => Ok(LLMCallState::LLMRunError),
            _ => Err(()),
        }
    }
}

impl RKLLMParam {
    pub fn new() -> RKLLMParam {
        unsafe { ffi::rkllm_createDefaultParam() }
    }

    pub fn get_model_path(&self) -> Result<&str, std::str::Utf8Error> {
        unsafe { std::ffi::CStr::from_ptr(self.model_path) }.to_str()
    }

    pub fn set_model_path(&mut self, model_path: &str) {
        let c_model_path = std::ffi::CString::new(model_path).unwrap();
        self.model_path = c_model_path.as_ptr();
    }
}

impl RKLLMResult {
    pub fn get_text(&self) -> Result<&str, std::str::Utf8Error> {
        unsafe { std::ffi::CStr::from_ptr(self.text) }.to_str()
    }

    pub fn set_text(&mut self, text: &str) {
        let c_text = std::ffi::CString::new(text).unwrap();
        self.text = c_text.as_ptr();
    }

    pub fn get_tokens(&self) -> &[Token] {
        unsafe { std::slice::from_raw_parts(self.tokens, self.num as usize) }
    }

    pub fn set_tokens(&mut self, tokens: &[Token]) {
        self.tokens = tokens.as_ptr() as *mut Token;
        self.num = tokens.len() as i32;
    }
}

impl LLMHandle {
    pub fn new(param: RKLLMParam) -> Option<LLMHandle> {
        let mut handle = LLMHandle {
            inner: std::ptr::null_mut(),
        };
        if unsafe { ffi::rkllm_init(&mut handle.inner, param, Some(callback_wrapper)) } != 0 {
            None
        } else {
            Some(handle)
        }
    }

    pub fn run(&self, prompt: &str, callback: Option<LLMResultCallback>) -> Option<()> {
        let c_prompt = std::ffi::CString::new(prompt).unwrap();
        let callback_ptr = Box::into_raw(Box::new(callback)) as *mut std::ffi::c_void;
        if unsafe { ffi::rkllm_run(self.inner, c_prompt.as_ptr(), callback_ptr) } != 0 {
            None
        } else {
            Some(())
        }
    }

    pub fn abort(&self) -> Option<()> {
        if unsafe { ffi::rkllm_abort(self.inner) } != 0 {
            None
        } else {
            Some(())
        }
    }
}

impl Drop for LLMHandle {
    fn drop(&mut self) {
        if unsafe { ffi::rkllm_destroy(self.inner) } != 0 {
            panic!("Failed to destroy LLMHandle");
        }

    }
}

extern "C" fn callback_wrapper(result_ptr: *mut RKLLMResult, callback_ptr: *mut std::ffi::c_void, state: ffi::LLMCallState) {
    unsafe { Box::from_raw(callback_ptr as *mut Option<LLMResultCallback>) }.map(
        |callback| LLMCallState::try_from(state).map(
            |state| callback(&state, unsafe { &*result_ptr }),
        ),
    );
}
