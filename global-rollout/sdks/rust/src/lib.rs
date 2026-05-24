//! Official Rust client for the Atherion NRS Platform HTTP API.
//!
//! ```no_run
//! use atherion_nrs::{Client, ChatCompletionRequest, Message};
//!
//! # async fn run() -> anyhow::Result<()> {
//! let client = Client::new("nrs_live_...");
//! let resp = client.chat().completions().create(
//!     ChatCompletionRequest {
//!         model: "nrs-auto".into(),
//!         messages: vec![Message::user("Explain quantum entanglement")],
//!         stream: false,
//!         temperature: None,
//!         max_tokens: None,
//!     }
//! ).await?;
//! println!("{}", resp.choices[0].message.content);
//! # Ok(())
//! # }
//! ```
//!
//! Streaming completions are returned as a `Stream<Item = Result<StreamChunk, Error>>`.

use futures_util::stream::StreamExt;
use futures_util::Stream;
use reqwest::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE, USER_AGENT};
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use thiserror::Error;

/// Production base URL for the Atherion NRS API.
pub const DEFAULT_BASE_URL: &str = "https://api.atheriongroup.com";

/// Top-level client. Cheap to clone — the underlying `reqwest::Client`
/// is reference-counted.
#[derive(Clone)]
pub struct Client {
    api_key: String,
    base_url: String,
    http: reqwest::Client,
}

impl Client {
    /// Construct a client with the production base URL and a 120-second
    /// HTTP timeout. Use [`Client::with_options`] for self-hosted /
    /// on-prem deployments.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_options(api_key, DEFAULT_BASE_URL, None)
    }

    pub fn with_options(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
        http: Option<reqwest::Client>,
    ) -> Self {
        let http = http.unwrap_or_else(|| {
            reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .build()
                .expect("default reqwest client")
        });
        let base_url = base_url.into();
        let base_url = base_url.trim_end_matches('/').to_string();
        Self { api_key: api_key.into(), base_url, http }
    }

    pub fn chat(&self) -> ChatNamespace<'_> {
        ChatNamespace { client: self }
    }
}

pub struct ChatNamespace<'a> {
    client: &'a Client,
}

impl<'a> ChatNamespace<'a> {
    pub fn completions(&self) -> CompletionsApi<'a> {
        CompletionsApi { client: self.client }
    }
}

pub struct CompletionsApi<'a> {
    client: &'a Client,
}

impl<'a> CompletionsApi<'a> {
    /// Issue a non-streaming chat completion. To stream, set
    /// `req.stream = true` and use [`Self::create_stream`] instead.
    pub async fn create(
        &self,
        mut req: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, Error> {
        req.stream = false;
        let resp = self
            .client
            .http
            .post(format!("{}/v1/chat/completions", self.client.base_url))
            .header(AUTHORIZATION, format!("Bearer {}", self.client.api_key))
            .header(CONTENT_TYPE, "application/json")
            .header(ACCEPT, "application/json")
            .header(USER_AGENT, concat!("atherion-nrs-rust/", env!("CARGO_PKG_VERSION")))
            .json(&req)
            .send()
            .await?;
        if !resp.status().is_success() {
            return Err(Error::Api {
                status: resp.status().as_u16(),
                body: resp.text().await.unwrap_or_default(),
            });
        }
        Ok(resp.json::<ChatCompletionResponse>().await?)
    }

    /// Issue a streaming chat completion. Each yielded item is a parsed
    /// SSE `data:` frame (already JSON-decoded into [`StreamChunk`]).
    /// The stream ends when the upstream emits `data: [DONE]`.
    pub async fn create_stream(
        &self,
        mut req: ChatCompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, Error>> + Send>>, Error> {
        req.stream = true;
        let resp = self
            .client
            .http
            .post(format!("{}/v1/chat/completions", self.client.base_url))
            .header(AUTHORIZATION, format!("Bearer {}", self.client.api_key))
            .header(CONTENT_TYPE, "application/json")
            .header(ACCEPT, "text/event-stream")
            .header(USER_AGENT, concat!("atherion-nrs-rust/", env!("CARGO_PKG_VERSION")))
            .json(&req)
            .send()
            .await?;
        if !resp.status().is_success() {
            return Err(Error::Api {
                status: resp.status().as_u16(),
                body: resp.text().await.unwrap_or_default(),
            });
        }

        let byte_stream = resp.bytes_stream();
        let parsed = byte_stream.flat_map(|frame| {
            let frames: Vec<Result<StreamChunk, Error>> = match frame {
                Err(e) => vec![Err(Error::Http(e))],
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes).to_string();
                    let mut out: Vec<Result<StreamChunk, Error>> = Vec::new();
                    for line in text.lines() {
                        let line = line.trim();
                        if line.is_empty() || !line.starts_with("data:") {
                            continue;
                        }
                        let payload = line.trim_start_matches("data:").trim();
                        if payload == "[DONE]" {
                            continue;
                        }
                        match serde_json::from_str::<StreamChunk>(payload) {
                            Ok(chunk) => out.push(Ok(chunk)),
                            Err(_) => continue,
                        }
                    }
                    out
                }
            };
            futures_util::stream::iter(frames)
        });
        Ok(Box::pin(parsed))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: "user".into(), content: content.into() }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: "assistant".into(), content: content.into() }
    }
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: "system".into(), content: content.into() }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    pub index: u32,
    pub message: Message,
    pub finish_reason: Option<String>,
}

/// Token accounting block returned with every completion. These
/// values are also reported to the Stripe Billing Meter for
/// pay-as-you-go and overage billing.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    #[serde(default)]
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
    #[serde(default)]
    pub nrs_meta: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChoice {
    pub index: u32,
    pub delta: StreamDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    pub id: String,
    #[serde(default)]
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
    #[serde(default)]
    pub nrs_meta: serde_json::Value,
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("HTTP transport error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("Atherion NRS API error: HTTP {status}: {body}")]
    Api { status: u16, body: String },
    #[error("decode error: {0}")]
    Decode(#[from] serde_json::Error),
}
