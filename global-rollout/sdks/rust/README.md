# Atherion NRS — Rust SDK

Official Rust client for the Atherion NRS Platform HTTP API.

## Install

```toml
[dependencies]
atherion-nrs = "0.1"
tokio = { version = "1", features = ["full"] }
```

## Quick start

```rust
use atherion_nrs::{Client, ChatCompletionRequest, Message};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new("nrs_live_...");

    let resp = client.chat().completions().create(ChatCompletionRequest {
        model: "nrs-auto".into(),
        messages: vec![Message::user("Explain quantum entanglement")],
        stream: false,
        temperature: None,
        max_tokens: None,
    }).await?;

    println!("{}", resp.choices[0].message.content);
    Ok(())
}
```

## Streaming

```rust
use futures_util::StreamExt;

let mut stream = client.chat().completions().create_stream(req).await?;
while let Some(chunk) = stream.next().await {
    let chunk = chunk?;
    if let Some(delta) = chunk.choices.first().and_then(|c| c.delta.content.as_deref()) {
        print!("{delta}");
    }
}
```

## Billing

Every successful completion records token usage (`usage.prompt_tokens
+ usage.completion_tokens`) to the Stripe Billing Meter for the
authenticated user. See [pricing](https://atheriongroup.com/pricing).
