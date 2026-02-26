package config

import (
	"os"
	"testing"
)

func TestLoadConfig(t *testing.T) {
	// Write temporary config
	tmpFile := "/tmp/test_config.yaml"
	content := `
server:
  host: "127.0.0.1"
  port: 8080
  default_model: "gui-actor-2b"

models:
  gui-actor-2b:
    path: "./models/gui-actor-2b"
    enabled: true
    preload: true
    min_pixels: 3136
    max_pixels: 5720064
    max_context_length: 8192
`
	os.WriteFile(tmpFile, []byte(content), 0644)

	cfg, err := LoadConfig(tmpFile)
	if err != nil {
		t.Fatalf("Failed to load config: %v", err)
	}

	if cfg.Server.Host != "127.0.0.1" {
		t.Errorf("Expected host 127.0.0.1, got %s", cfg.Server.Host)
	}
	if cfg.Server.Port != 8080 {
		t.Errorf("Expected port 8080, got %d", cfg.Server.Port)
	}
}
