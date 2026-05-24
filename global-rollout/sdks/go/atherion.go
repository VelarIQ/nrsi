// Package atherion is the official Go client for the Atherion NRS
// Platform HTTP API.
//
// The client mirrors the OpenAI Go shape so existing application code
// can swap providers by changing the import path and base URL only:
//
//	import nrs "github.com/atherion-group/atherion-nrs-go"
//
//	c := nrs.NewClient("nrs_live_...")
//	resp, err := c.Chat.Completions.Create(ctx, nrs.ChatCompletionRequest{
//	    Model: "nrs-auto",
//	    Messages: []nrs.Message{
//	        {Role: "user", Content: "Explain quantum entanglement"},
//	    },
//	})
//
// Streaming via Server-Sent Events is supported by setting Stream=true
// and consuming the returned channel.
package atherion

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

const (
	// DefaultBaseURL is the production Atherion NRS API endpoint.
	DefaultBaseURL = "https://api.atheriongroup.com"

	// DefaultTimeout is the default HTTP timeout for non-streaming
	// requests. Streaming requests intentionally bypass this timeout
	// since SSE responses can run for several minutes.
	DefaultTimeout = 120 * time.Second
)

// Client is the top-level Atherion NRS API client.
type Client struct {
	apiKey  string
	baseURL string
	http    *http.Client
	Chat    *chatNamespace
}

// NewClient constructs a client with the production base URL and
// 120-second timeout. Use NewClientWithOptions for a custom base URL
// or HTTP client (e.g. self-hosted on-prem deployments).
func NewClient(apiKey string) *Client {
	return NewClientWithOptions(apiKey, DefaultBaseURL, &http.Client{
		Timeout: DefaultTimeout,
	})
}

// NewClientWithOptions constructs a fully customized client. Pass an
// empty baseURL to use the production default.
func NewClientWithOptions(apiKey, baseURL string, httpClient *http.Client) *Client {
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}
	if httpClient == nil {
		httpClient = &http.Client{Timeout: DefaultTimeout}
	}
	c := &Client{
		apiKey:  apiKey,
		baseURL: strings.TrimRight(baseURL, "/"),
		http:    httpClient,
	}
	c.Chat = &chatNamespace{client: c, Completions: &completionsAPI{client: c}}
	return c
}

// Message is a single chat-history entry.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatCompletionRequest matches the OpenAI request shape.
type ChatCompletionRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Stream      bool      `json:"stream,omitempty"`
	Temperature *float64  `json:"temperature,omitempty"`
	MaxTokens   *int      `json:"max_tokens,omitempty"`
}

// Choice is one generation in a non-streaming response.
type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

// Usage is the token accounting block returned with every completion.
// These exact values are also reported to the Stripe Billing Meter
// for pay-as-you-go and overage billing.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ChatCompletionResponse is the non-streaming response shape.
type ChatCompletionResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []Choice       `json:"choices"`
	Usage   Usage          `json:"usage"`
	NRSMeta map[string]any `json:"nrs_meta,omitempty"`
}

// StreamChunk is a single SSE delta when Stream=true.
type StreamChunk struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index int `json:"index"`
		Delta struct {
			Role    string `json:"role,omitempty"`
			Content string `json:"content,omitempty"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	NRSMeta map[string]any `json:"nrs_meta,omitempty"`
}

type chatNamespace struct {
	client      *Client
	Completions *completionsAPI
}

type completionsAPI struct{ client *Client }

// Create issues a non-streaming chat completion. To stream, use
// CreateStream and read from the returned channel until it closes.
func (a *completionsAPI) Create(ctx context.Context, req ChatCompletionRequest) (*ChatCompletionResponse, error) {
	req.Stream = false
	body, err := a.client.do(ctx, "POST", "/v1/chat/completions", req)
	if err != nil {
		return nil, err
	}
	defer body.Close()
	out := &ChatCompletionResponse{}
	if err := json.NewDecoder(body).Decode(out); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	return out, nil
}

// CreateStream issues a streaming chat completion and returns a
// channel of SSE chunks. The channel closes when the upstream emits
// `data: [DONE]` or the context is cancelled. errCh receives at most
// one error.
func (a *completionsAPI) CreateStream(ctx context.Context, req ChatCompletionRequest) (<-chan StreamChunk, <-chan error) {
	chunks := make(chan StreamChunk, 16)
	errs := make(chan error, 1)
	req.Stream = true

	go func() {
		defer close(chunks)
		defer close(errs)

		body, err := a.client.do(ctx, "POST", "/v1/chat/completions", req)
		if err != nil {
			errs <- err
			return
		}
		defer body.Close()

		scanner := bufio.NewScanner(body)
		// Allow up to 1 MiB per SSE frame (large NRS metadata blobs).
		scanner.Buffer(make([]byte, 64*1024), 1024*1024)

		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line == "" || !strings.HasPrefix(line, "data:") {
				continue
			}
			payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			if payload == "[DONE]" {
				return
			}
			var c StreamChunk
			if err := json.Unmarshal([]byte(payload), &c); err != nil {
				continue
			}
			select {
			case chunks <- c:
			case <-ctx.Done():
				errs <- ctx.Err()
				return
			}
		}
		if err := scanner.Err(); err != nil && !errors.Is(err, io.EOF) {
			errs <- err
		}
	}()

	return chunks, errs
}

func (c *Client) do(ctx context.Context, method, path string, body any) (io.ReadCloser, error) {
	var reader io.Reader
	if body != nil {
		buf, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("marshal request: %w", err)
		}
		reader = bytes.NewReader(buf)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, reader)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	req.Header.Set("User-Agent", "atherion-nrs-go/0.1.0")

	resp, err := c.http.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 400 {
		defer resp.Body.Close()
		buf, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, &APIError{
			StatusCode: resp.StatusCode,
			Body:       string(buf),
		}
	}
	return resp.Body, nil
}

// APIError is returned for any non-2xx response from the NRS API.
type APIError struct {
	StatusCode int
	Body       string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("atherion-nrs: HTTP %d: %s", e.StatusCode, e.Body)
}
