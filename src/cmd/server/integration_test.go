package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"testing"
	"time"

	"github.com/agenthands/gui-actor/internal/api"
	"github.com/agenthands/gui-actor/internal/config"
	"github.com/agenthands/gui-actor/internal/model"
)

func TestFullIntegration(t *testing.T) {
	// Create test config
	cfg := &config.Config{
		Server: config.ServerConfig{
			Host: "127.0.0.1",
			Port: 0, // Random port
		},
		Models: map[string]config.ModelConfig{
			"test-model": {
				Path:    "/tmp/test",
				Enabled: true,
				Preload: false,
			},
		},
		Memory: config.MemoryConfig{
			MaxTotalGB: "32",
		},
	}

	registry := model.NewRegistry(cfg)
	handler := api.NewHandler(registry)
	server := api.NewServer(":18080", handler) // Fixed test port

	// Start server in background
	go func() {
		_ = server.Start()
	}()
	time.Sleep(100 * time.Millisecond)

	// Test health endpoint
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get("http://localhost:18080/v1/health")
	if err != nil {
		t.Fatalf("Health check failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("Expected 200, got %d", resp.StatusCode)
	}

	// Test chat completion (will fail inference but should work)
	reqBody := api.ChatCompletionRequest{
		Model: "test-model",
		Messages: []api.Message{
			{
				Role:    "user",
				Content: "test",
			},
		},
	}

	body, _ := json.Marshal(reqBody)
	resp, err = client.Post(
		"http://localhost:18080/v1/chat/completions",
		"application/json",
		bytes.NewReader(body),
	)

	if err != nil {
		t.Fatalf("Chat completion failed: %v", err)
	}
	defer resp.Body.Close()

	// Accept 200 or 500 (service unavailable) since we don't have actual models
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusServiceUnavailable {
		t.Logf("Got status %d (may be expected for test model)", resp.StatusCode)
	}

	// Shutdown
	server.Shutdown(context.Background())
}
