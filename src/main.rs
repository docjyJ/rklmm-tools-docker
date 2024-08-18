mod completion;

use axum::{
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use crate::completion::{get_completions, CompletionParms};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let app = Router::new()
        .route("/v1/models", get(models))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/translation/completions", post(translation_completions))
        .route("/v1/transcription/completions", post(transcription_completions));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000")
        .await
        .unwrap();
    tracing::debug!("listening on {}", listener.local_addr().unwrap());
    axum::serve(listener, app).await.unwrap();
}

fn models() -> impl IntoResponse {
    StatusCode::NOT_FOUND
}

fn completions(Json(payload): Json<CompletionParms>) -> impl IntoResponse {
    get_completions(payload)
}

fn chat_completions() -> impl IntoResponse {
    StatusCode::NOT_FOUND
}

fn translation_completions() -> impl IntoResponse {
    StatusCode::NOT_FOUND
}

fn transcription_completions() -> impl IntoResponse {
    StatusCode::NOT_FOUND
}
