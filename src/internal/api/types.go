package api

import "time"

// ChatCompletionRequest represents an OpenAI-compatible request
type ChatCompletionRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Stream      bool      `json:"stream,omitempty"`
	MaxPixels   *int      `json:"max_pixels,omitempty"`
	MinPixels   *int      `json:"min_pixels,omitempty"`
	MaxContext  *int      `json:"max_context,omitempty"`
	Profile     string    `json:"profile,omitempty"`
	Temperature *float64  `json:"temperature,omitempty"`
	TopP        *float64  `json:"top_p,omitempty"`
	MaxTokens   *int      `json:"max_tokens,omitempty"`
}

// Message represents a chat message
type Message struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"` // string or []ContentPart
}

// ContentPart represents a multipart content (text + image)
type ContentPart struct {
	Type     string    `json:"type"` // "text" or "image_url"
	Text     string    `json:"text,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
}

// ImageURL contains the base64 image data
type ImageURL struct {
	URL string `json:"url"`
}

// ChatCompletionResponse represents an OpenAI-compatible response
type ChatCompletionResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage,omitempty"`
}

// Choice represents a completion choice
type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
	Delta        *Message `json:"delta,omitempty"` // For streaming
	Coordinates  *Point   `json:"coordinates,omitempty"` // GUI-Actor specific
}

// Point represents normalized coordinates [0, 1]
type Point struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
}

// Usage represents token usage
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ModelInfo represents model status
type ModelInfo struct {
	ID       string  `json:"id"`
	Object   string  `json:"object"`
	Loaded   bool    `json:"loaded"`
	MemoryGB float64 `json:"memory_gb,omitempty"`
}

// ModelsResponse lists available models
type ModelsResponse struct {
	Object string      `json:"object"`
	Data   []ModelInfo `json:"data"`
}

// HealthResponse represents health check response
type HealthResponse struct {
	Status   string  `json:"status"`
	Uptime   int64   `json:"uptime_seconds"`
	MemoryGB float64 `json:"memory_used_gb"`
	Models   int     `json:"loaded_models"`
}

// ErrorResponse represents an error
type ErrorResponse struct {
	Error ErrorDetail `json:"error"`
}

// ErrorDetail contains error information
type ErrorDetail struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code,omitempty"`
}

// NewChatCompletionResponse creates a new response
func NewChatCompletionResponse(model string, choices []Choice) *ChatCompletionResponse {
	return &ChatCompletionResponse{
		ID:      generateID(),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: choices,
	}
}

func generateID() string {
	return "chatcmpl-" + randomString(29)
}

func randomString(n int) string {
	// Simple random string generation
	const letters = "abcdefghijklmnopqrstuvwxyz0123456789"
	// TODO: implement proper random generation
	b := make([]byte, n)
	for i := range b {
		b[i] = letters[i%len(letters)]
	}
	return string(b)
}
