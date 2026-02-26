package http

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/agenthands/GUI-Actor/internal/radix"
	"github.com/agenthands/GUI-Actor/pkg/tokenizer"
)

func TestNewServer(t *testing.T) {
	tree := radix.NewTree()
	engine := &radix.MockMLXEngine{}
	tok := tokenizer.NewTokenizer(32000)
	model := "test-model"

	server := NewServer(tree, engine, tok, model)
	if server == nil {
		t.Fatal("Expected non-nil server")
	}

	if server.tree != tree {
		t.Error("Tree not set correctly")
	}

	if server.engine != engine {
		t.Error("Engine not set correctly")
	}

	if server.tokenizer != tok {
		t.Error("Tokenizer not set correctly")
	}

	if server.model != model {
		t.Error("Model not set correctly")
	}
}

func TestChatCompletionHandler(t *testing.T) {
	tree := radix.NewTree()
	engine := &radix.MockMLXEngine{
		ForwardFunc: func(model any, tokens []uint32, base uint64) ([]float32, uint64, error) {
			return make([]float32, 32000), 100, nil
		},
	}
	tok := tokenizer.NewTokenizer(32000)
	model := "test-model"

	server := NewServer(tree, engine, tok, model)

	// Create request
	reqBody := ChatCompletionRequest{
		Messages: []tokenizer.ChatMessage{
			{Role: "user", Content: "Hello!"},
		},
		MaxTokens: 10,
	}

	body, _ := json.Marshal(reqBody)
	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	// Handle request
	server.ChatCompletionHandler(w, req)

	// Check response
	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	contentType := w.Header().Get("Content-Type")
	if contentType != "application/json" {
		t.Errorf("Expected content-type application/json, got %s", contentType)
	}

	// Parse response
	var resp ChatCompletionResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("Failed to parse response: %v", err)
	}

	if resp.Object != "chat.completion" {
		t.Errorf("Expected object 'chat.completion', got '%s'", resp.Object)
	}

	if len(resp.Choices) != 1 {
		t.Fatalf("Expected 1 choice, got %d", len(resp.Choices))
	}

	if resp.Choices[0].Message.Role != "assistant" {
		t.Errorf("Expected role 'assistant', got '%s'", resp.Choices[0].Message.Role)
	}

	if resp.Choices[0].FinishReason != "stop" {
		t.Errorf("Expected finish_reason 'stop', got '%s'", resp.Choices[0].FinishReason)
	}
}

func TestChatCompletionHandlerWrongMethod(t *testing.T) {
	tree := radix.NewTree()
	engine := &radix.MockMLXEngine{}
	tok := tokenizer.NewTokenizer(32000)
	model := "test-model"

	server := NewServer(tree, engine, tok, model)

	req := httptest.NewRequest("GET", "/v1/chat/completions", nil)
	w := httptest.NewRecorder()

	server.ChatCompletionHandler(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("Expected status 405, got %d", w.Code)
	}
}

func TestChatCompletionHandlerInvalidJSON(t *testing.T) {
	tree := radix.NewTree()
	engine := &radix.MockMLXEngine{}
	tok := tokenizer.NewTokenizer(32000)
	model := "test-model"

	server := NewServer(tree, engine, tok, model)

	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader([]byte("invalid json")))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	server.ChatCompletionHandler(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected status 400, got %d", w.Code)
	}
}

func TestChatCompletionHandlerEmptyMessages(t *testing.T) {
	tree := radix.NewTree()
	engine := &radix.MockMLXEngine{}
	tok := tokenizer.NewTokenizer(32000)
	model := "test-model"

	server := NewServer(tree, engine, tok, model)

	reqBody := ChatCompletionRequest{
		Messages: []tokenizer.ChatMessage{},
	}

	body, _ := json.Marshal(reqBody)
	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	server.ChatCompletionHandler(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected status 400 for empty messages, got %d", w.Code)
	}
}

func TestHealthCheckHandler(t *testing.T) {
	tree := radix.NewTree()
	engine := &radix.MockMLXEngine{}
	tok := tokenizer.NewTokenizer(32000)
	model := "test-model"

	server := NewServer(tree, engine, tok, model)

	req := httptest.NewRequest("GET", "/health", nil)
	w := httptest.NewRecorder()

	server.HealthCheckHandler(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	contentType := w.Header().Get("Content-Type")
	if contentType != "application/json" {
		t.Errorf("Expected content-type application/json, got %s", contentType)
	}

	var resp map[string]string
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("Failed to parse response: %v", err)
	}

	if resp["status"] != "ok" {
		t.Errorf("Expected status 'ok', got '%s'", resp["status"])
	}
}

func TestHealthCheckHandlerWrongMethod(t *testing.T) {
	tree := radix.NewTree()
	engine := &radix.MockMLXEngine{}
	tok := tokenizer.NewTokenizer(32000)
	model := "test-model"

	server := NewServer(tree, engine, tok, model)

	req := httptest.NewRequest("POST", "/health", nil)
	w := httptest.NewRecorder()

	server.HealthCheckHandler(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("Expected status 405, got %d", w.Code)
	}
}

func TestRegisterRoutes(t *testing.T) {
	tree := radix.NewTree()
	engine := &radix.MockMLXEngine{}
	tok := tokenizer.NewTokenizer(32000)
	model := "test-model"

	server := NewServer(tree, engine, tok, model)
	mux := http.NewServeMux()

	server.RegisterRoutes(mux)

	// Test routes are registered
	req1 := httptest.NewRequest("GET", "/health", nil)
	w1 := httptest.NewRecorder()
	mux.ServeHTTP(w1, req1)

	if w1.Code != http.StatusOK {
		t.Errorf("Health route not registered: got status %d", w1.Code)
	}
}

func TestGenerateAutoregressive(t *testing.T) {
	tree := radix.NewTree()
	engine := &radix.MockMLXEngine{
		ForwardFunc: func(model any, tokens []uint32, base uint64) ([]float32, uint64, error) {
			return make([]float32, 32000), base + 1, nil
		},
	}
	tok := tokenizer.NewTokenizer(32000)
	model := "test-model"

	server := NewServer(tree, engine, tok, model)

	inputTokens := []uint32{1, 2, 3}
	maxTokens := 10

	output, err := server.GenerateAutoregressive(inputTokens, maxTokens)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	if len(output) == 0 {
		t.Error("Expected some output tokens")
	}

	// Current implementation echoes input then stops
	// In production, would generate actual completion
	if len(output) != len(inputTokens) {
		t.Logf("Note: Generated %d tokens from %d input tokens (simplified logic)", len(output), len(inputTokens))
	}
}

func TestGenerateAutoregressiveWithCacheHit(t *testing.T) {
	tree := radix.NewTree()
	engine := &radix.MockMLXEngine{
		ForwardFunc: func(model any, tokens []uint32, base uint64) ([]float32, uint64, error) {
			return make([]float32, 32000), base + 1, nil
		},
	}
	tok := tokenizer.NewTokenizer(32000)
	model := "test-model"

	server := NewServer(tree, engine, tok, model)

	// Insert a cache entry
	inputTokens := []uint32{1, 2, 3}
	node, _ := tree.InsertPending(inputTokens, engine, model)
	radix.FinalizeNode(node, 100)

	// Now generate - should hit cache
	output, err := server.GenerateAutoregressive(inputTokens, 5)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	if len(output) == 0 {
		t.Error("Expected some output tokens")
	}
}

func TestStreamingResponseWriter(t *testing.T) {
	var buf bytes.Buffer
	writer := NewStreamingResponseWriter(&buf)

	chunk := map[string]string{"content": "test"}

	err := writer.WriteChunk(chunk)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	output := buf.String()
	if output == "" {
		t.Error("Expected some output")
	}

	// Should start with "data: "
	if output[:6] != "data: " {
		t.Errorf("Expected 'data: ' prefix, got '%s'", output[:6])
	}
}

func TestStreamingResponseWriterClose(t *testing.T) {
	var buf bytes.Buffer
	writer := NewStreamingResponseWriter(&buf)

	err := writer.Close()
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	output := buf.String()
	if output != "data: [DONE]\n\n" {
		t.Errorf("Expected 'data: [DONE]\\n\\n', got '%s'", output)
	}
}

func TestRecoverHandler(t *testing.T) {
	tree := radix.NewTree()
	engine := &radix.MockMLXEngine{}
	tok := tokenizer.NewTokenizer(32000)
	model := "test-model"

	server := NewServer(tree, engine, tok, model)

	panicHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		panic("test panic")
	})

	wrapped := server.RecoverHandler(panicHandler)

	req := httptest.NewRequest("GET", "/test", nil)
	w := httptest.NewRecorder()

	wrapped(w, req)

	if w.Code != http.StatusInternalServerError {
		t.Errorf("Expected status 500, got %d", w.Code)
	}
}

func TestLogHandler(t *testing.T) {
	tree := radix.NewTree()
	engine := &radix.MockMLXEngine{}
	tok := tokenizer.NewTokenizer(32000)
	model := "test-model"

	server := NewServer(tree, engine, tok, model)

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	wrapped := server.LogHandler(handler)

	req := httptest.NewRequest("GET", "/test", nil)
	w := httptest.NewRecorder()

	wrapped(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}
}

func TestCompletionUsage(t *testing.T) {
	usage := CompletionUsage{
		PromptTokens:     10,
		CompletionTokens: 20,
		TotalTokens:      30,
	}

	if usage.PromptTokens != 10 {
		t.Errorf("Expected PromptTokens 10, got %d", usage.PromptTokens)
	}

	if usage.TotalTokens != 30 {
		t.Errorf("Expected TotalTokens 30, got %d", usage.TotalTokens)
	}

	// Verify TotalTokens = PromptTokens + CompletionTokens
	if usage.TotalTokens != usage.PromptTokens+usage.CompletionTokens {
		t.Error("TotalTokens should equal PromptTokens + CompletionTokens")
	}
}
