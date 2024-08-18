use serde::{Deserialize, Serialize};
use rkllm_runtime::{LLMHandle, RKLLMParam};

#[derive(Deserialize)]
pub struct CompletionParms {
    prompt: String,
    n: i32,
    model: String,
    max_tokens: Option<i32>,
    user: Option<String>,
}

#[derive(Serialize)]
struct CompletionChoice {
    finish_reason: String,
    index: i32,
    logprobs: None,
    choice: String,
}

#[derive(Serialize)]
struct CompletionUsage {
    prompt_tokens: i32,
    completion_tokens: i32,
    total_tokens: i32,
}

#[derive(Serialize)]
struct Completion {
    id: String,
    choices: Vec<String>,
    created: i32,
    model: String,
    system_fingerprint: String,
    object: String,
    usage: String,
}

pub fn get_completions(params: CompletionParms) -> Completion {
    let mut model_param = RKLLMParam::new();
    model_param.set_model_path(&params.model);

    params.max_tokens.map(|max_tokens| model_param.max_new_tokens = max_tokens);
    // TODO model_param.num_npu_core = ?;
    // TODO model_param.use_gpu = ?;

    let handle = LLMHandle::new(model_param).map(|handle| {
        handle.run(&params.prompt, None).map(|_| {
            // TODO handle.run() callback
        });
    });

    Completion {
        id: "123".to_string(),
        choices: vec!["".to_string()],
        created: 0,
        model: "".to_string(),
        system_fingerprint: "".to_string(),
        object: "".to_string(),
        usage: "".to_string(),
    }
}
