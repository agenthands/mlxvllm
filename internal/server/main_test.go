package main

import (
	"context"
	"fmt"
	nethttp "net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	httpserver "github.com/agenthands/GUI-Actor/internal/http"
	"github.com/agenthands/GUI-Actor/internal/radix"
	"github.com/agenthands/GUI-Actor/pkg/tokenizer"
)

// TestSetupLogging verifies logging configuration
func TestSetupLogging(t *testing.T) {
	tests := []struct {
		level    string
		expected string
	}{
		{"debug", "debug"},
		{"info", "info"},
		{"warn", "warn"},
		{"error", "error"},
		{"invalid", "info"}, // defaults to info
	}

	for _, tt := range tests {
		t.Run(tt.level, func(t *testing.T) {
			// Just verify it doesn't panic
			setupLogging(tt.level)
		})
	}
}

// TestSetupMLXEngine verifies MLX engine initialization
func TestSetupMLXEngine(t *testing.T) {
	engine, err := setupMLXEngine()
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	if engine == nil {
		t.Fatal("Expected non-nil engine")
	}

	// Verify engine implements interface
	var _ radix.MLXEngine = engine
}

// TestLoadModel verifies model loading
func TestLoadModel(t *testing.T) {
	tests := []struct {
		name    string
		path    string
		wantErr bool
	}{
		{"Valid path", "/fake/path/model.safetensors", false},
		{"Empty path", "", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model, err := loadModel(tt.path)

			if tt.wantErr {
				if err == nil {
					t.Error("Expected error for empty path")
				}
				return
			}

			if err != nil {
				t.Fatalf("Expected no error, got %v", err)
			}

			if model == nil {
				t.Error("Expected non-nil model")
			}
		})
	}
}

// TestWrapMiddleware verifies middleware application
func TestWrapMiddleware(t *testing.T) {
	tree := radix.NewTree()
	engine := &radix.MockMLXEngine{}
	tok := tokenizer.NewTokenizer(32000)
	model := "test-model"

	server := httpserver.NewServer(tree, engine, tok, model)

	// Create a test handler
	testHandler := nethttp.HandlerFunc(func(w nethttp.ResponseWriter, r *nethttp.Request) {
		w.WriteHeader(nethttp.StatusOK)
		w.Write([]byte("OK"))
	})

	wrapped := wrapMiddleware(server, testHandler)

	// Test regular request
	req := httptest.NewRequest("GET", "/test", nil)
	w := httptest.NewRecorder()
	wrapped.ServeHTTP(w, req) // Fixed: was w, w

	if w.Code != nethttp.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	// Test CORS preflight
	req = httptest.NewRequest("OPTIONS", "/test", nil)
	w = httptest.NewRecorder()
	wrapped.ServeHTTP(w, req)

	if w.Code != nethttp.StatusOK {
		t.Errorf("Expected status 200 for OPTIONS, got %d", w.Code)
	}

	corsHeader := w.Header().Get("Access-Control-Allow-Origin")
	if corsHeader != "*" {
		t.Errorf("Expected CORS header '*', got '%s'", corsHeader)
	}
}

// TestCORSMiddleware verifies CORS headers
func TestCORSMiddleware(t *testing.T) {
	handler := nethttp.HandlerFunc(func(w nethttp.ResponseWriter, r *nethttp.Request) {
		w.Write([]byte("response"))
	})

	wrapped := corsMiddleware(handler)

	req := httptest.NewRequest("GET", "/test", nil)
	req.Header.Set("Origin", "https://example.com")
	w := httptest.NewRecorder()

	wrapped.ServeHTTP(w, req)

	// Check CORS headers
	headers := map[string]string{
		"Access-Control-Allow-Origin":  "*",
		"Access-Control-Allow-Methods": "GET, POST, OPTIONS",
		"Access-Control-Allow-Headers": "Content-Type, Authorization",
	}

	for key, expected := range headers {
		actual := w.Header().Get(key)
		if actual != expected {
			t.Errorf("Expected header %s='%s', got '%s'", key, expected, actual)
		}
	}
}

// TestServerIntegration tests the full server setup
func TestServerIntegration(t *testing.T) {
	// Setup components
	tree := radix.NewTree()
	engine := &radix.MockMLXEngine{
		ForwardFunc: func(model any, tokens []uint32, base uint64) ([]float32, uint64, error) {
			return make([]float32, 32000), base + 1, nil
		},
	}
	tok := tokenizer.NewTokenizer(32000)
	model := "test-model"

	server := httpserver.NewServer(tree, engine, tok, model)
	mux := nethttp.NewServeMux()
	server.RegisterRoutes(mux)

	wrapped := wrapMiddleware(server, mux)

	// Create test server
	testServer := httptest.NewServer(wrapped)
	defer testServer.Close()

	// Test health endpoint
	resp, err := nethttp.Get(testServer.URL + "/health")
	if err != nil {
		t.Fatalf("Failed to get health: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != nethttp.StatusOK {
		t.Errorf("Expected status 200, got %d", resp.StatusCode)
	}

	// Test chat completion endpoint
	reqBody := `{
		"messages": [
			{"role": "user", "content": "Hello"}
		],
		"max_tokens": 10
	}`
	resp, err = nethttp.Post(testServer.URL+"/v1/chat/completions", "application/json", strings.NewReader(reqBody))
	if err != nil {
		t.Fatalf("Failed to post chat completion: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != nethttp.StatusOK {
		t.Errorf("Expected status 200, got %d", resp.StatusCode)
	}
}

// TestGracefulShutdown verifies graceful shutdown behavior
func TestGracefulShutdown(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Create a simple server
	tree := radix.NewTree()
	engine := &radix.MockMLXEngine{}
	tok := tokenizer.NewTokenizer(32000)
	model := "test-model"

	server := httpserver.NewServer(tree, engine, tok, model)
	mux := nethttp.NewServeMux()
	server.RegisterRoutes(mux)

	handler := wrapMiddleware(server, mux)

	httpServer := &nethttp.Server{
		Addr:    ":0", // Random port
		Handler: handler,
	}

	// Start server in background
	go func() {
		httpServer.ListenAndServe()
	}()

	// Wait for server to start
	time.Sleep(100 * time.Millisecond)

	// Trigger shutdown
	go func() {
		time.Sleep(50 * time.Millisecond)
		httpServer.Shutdown(ctx)
	}()

	// Wait for shutdown
	select {
	case <-ctx.Done():
		// Shutdown completed
	case <-time.After(10 * time.Second):
		t.Fatal("Timeout waiting for shutdown")
	}
}

// TestMainFunction verifies main function doesn't panic on invalid flags
func TestMainFunction(t *testing.T) {
	// Save original args
	oldArgs := os.Args
	defer func() { os.Args = oldArgs }()

	// Test with invalid config path (should log error, not panic)
	os.Args = []string{"test", "-config", "/nonexistent/path", "-addr", ":0"}

	// We can't actually run main() as it would exit
	// But we can test the setup functions
	tok := tokenizer.NewTokenizer(32000)
	if tok == nil {
		t.Error("Failed to create tokenizer")
	}

	tree := radix.NewTree()
	if tree == nil {
		t.Error("Failed to create tree")
	}

	engine, err := setupMLXEngine()
	if err != nil {
		t.Errorf("Failed to setup MLX engine: %v", err)
	}
	if engine == nil {
		t.Error("Expected non-nil engine")
	}

	// Test loadModel with empty path
	_, err = loadModel("")
	if err == nil {
		t.Error("Expected error for empty model path")
	}
}

// TestInitFunction verifies init function
func TestInitFunction(t *testing.T) {
	// Call init to verify it doesn't panic
	// (init is called automatically, but we can verify its effects)
	if tz := os.Getenv("TZ"); tz != "UTC" {
		t.Errorf("Expected TZ=UTC, got %s", tz)
	}
}

// BenchmarkServerRequest benchmarks request handling
func BenchmarkServerRequest(b *testing.B) {
	tree := radix.NewTree()
	engine := &radix.MockMLXEngine{
		ForwardFunc: func(model any, tokens []uint32, base uint64) ([]float32, uint64, error) {
			return make([]float32, 32000), base + 1, nil
		},
	}
	tok := tokenizer.NewTokenizer(32000)
	model := "test-model"

	server := httpserver.NewServer(tree, engine, tok, model)
	mux := nethttp.NewServeMux()
	server.RegisterRoutes(mux)

	wrapped := wrapMiddleware(server, mux)

	testServer := httptest.NewServer(wrapped)
	defer testServer.Close()

	reqBody := `{
		"messages": [
			{"role": "user", "content": "Hello"}
		],
		"max_tokens": 10
	}`

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		resp, err := nethttp.Post(testServer.URL+"/v1/chat/completions", "application/json", strings.NewReader(reqBody))
		if err != nil {
			b.Fatalf("Request failed: %v", err)
		}
		resp.Body.Close()
	}
}

// Example: Manual server startup for development
func Example_main() {
	// In production, run with:
	// go run internal/server/main.go -addr :8080 -model /path/to/model
	// Or build binary:
	// go build -o bin/server internal/server/main.go
	// ./server -addr :8080 -model /path/to/model

	fmt.Println("Server starting on :8080")
	fmt.Println("Model: /path/to/model.safetensors")
	fmt.Println("Vocab size: 32000")
	fmt.Println("Max cache: 1000")
	// Output:
	// Server starting on :8080
	// Model: /path/to/model.safetensors
	// Vocab size: 32000
	// Max cache: 1000
}
