# Atherion NRS — Go SDK

Official Go client for the Atherion NRS Platform HTTP API.

## Install

```bash
go get github.com/atherion-group/atherion-nrs-go
```

## Quick start

```go
package main

import (
    "context"
    "fmt"
    "log"

    nrs "github.com/atherion-group/atherion-nrs-go"
)

func main() {
    client := nrs.NewClient("nrs_live_...")

    resp, err := client.Chat.Completions.Create(context.Background(),
        nrs.ChatCompletionRequest{
            Model: "nrs-auto",
            Messages: []nrs.Message{
                {Role: "user", Content: "Explain quantum entanglement"},
            },
        },
    )
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println(resp.Choices[0].Message.Content)
}
```

## Streaming

```go
chunks, errs := client.Chat.Completions.CreateStream(ctx,
    nrs.ChatCompletionRequest{
        Model: "nrs-auto",
        Messages: []nrs.Message{{Role: "user", Content: "Write a haiku"}},
        Stream: true,
    },
)
for chunk := range chunks {
    if len(chunk.Choices) > 0 {
        fmt.Print(chunk.Choices[0].Delta.Content)
    }
}
if err := <-errs; err != nil {
    log.Fatal(err)
}
```

## Self-hosted / on-prem

```go
client := nrs.NewClientWithOptions(
    "nrs_live_...",
    "https://nrs.internal.acme.corp",
    nil, // use default http client
)
```

## Billing

Every successful completion records token usage to the Stripe Billing
Meter for the authenticated user. The exact values reported are the
ones in `resp.Usage` (`prompt_tokens + completion_tokens`). On-prem
deployments without Stripe configured no-op silently.

See [https://atheriongroup.com/pricing](https://atheriongroup.com/pricing)
for current per-1M-token pricing across NRS-Core, NRS-Medical,
NRS-Legal, NRS-Financial, NRS-Scientific, and NRS-Custom.
