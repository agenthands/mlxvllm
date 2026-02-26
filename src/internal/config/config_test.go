package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadConfig(t *testing.T) {
	// Create temp directory for test files
	tmpDir := t.TempDir()
	t.Cleanup(func() {
		os.RemoveAll(tmpDir)
	})

	tests := []struct {
		name        string
		content     string
		expectError bool
		validate    func(*testing.T, *Config)
	}{
		{
			name: "minimal valid config",
			content: `
server:
  host: "127.0.0.1"
  port: 8080
  default_model: "gui-actor-2b"
models: {}
`,
			expectError: false,
			validate: func(t *testing.T, cfg *Config) {
				if cfg.Server.Host != "127.0.0.1" {
					t.Errorf("Expected host 127.0.0.1, got %s", cfg.Server.Host)
				}
				if cfg.Server.Port != 8080 {
					t.Errorf("Expected port 8080, got %d", cfg.Server.Port)
				}
				if cfg.Server.DefaultModel != "gui-actor-2b" {
					t.Errorf("Expected default_model gui-actor-2b, got %s", cfg.Server.DefaultModel)
				}
			},
		},
		{
			name: "full config with all fields",
			content: `
server:
  host: "0.0.0.0"
  port: 9000
  default_model: "gui-actor-7b"

models:
  gui-actor-2b:
    path: "./models/gui-actor-2b"
    enabled: true
    preload: true
    min_pixels: 3136
    max_pixels: 5720064
    max_context_length: 8192
    memory_limit_gb: 0
  gui-actor-7b:
    path: "./models/gui-actor-7b"
    enabled: true
    preload: false
    min_pixels: 3136
    max_pixels: 12845056
    max_context_length: 24576
    memory_limit_gb: 16

profiles:
  fast:
    max_pixels: 1048576
    max_context_length: 4096
  balanced:
    max_pixels: 5720064
    max_context_length: 8192
  quality:
    max_pixels: 12845056
    max_context_length: 24576

memory:
  max_total_gb: "32"
  unload_strategy: "lru"
  keep_models:
    - "gui-actor-2b"
    - "gui-actor-7b"

logging:
  level: "info"
  format: "json"
`,
			expectError: false,
			validate: func(t *testing.T, cfg *Config) {
				// Server config
				if cfg.Server.Host != "0.0.0.0" {
					t.Errorf("Expected host 0.0.0.0, got %s", cfg.Server.Host)
				}
				if cfg.Server.Port != 9000 {
					t.Errorf("Expected port 9000, got %d", cfg.Server.Port)
				}

				// Models config
				if len(cfg.Models) != 2 {
					t.Errorf("Expected 2 models, got %d", len(cfg.Models))
				}

				model2b := cfg.Models["gui-actor-2b"]
				if !model2b.Enabled {
					t.Error("Expected gui-actor-2b to be enabled")
				}
				if !model2b.Preload {
					t.Error("Expected gui-actor-2b to be preloaded")
				}
				if model2b.MinPixels != 3136 {
					t.Errorf("Expected min_pixels 3136, got %d", model2b.MinPixels)
				}

				// Profiles config
				if len(cfg.Profiles) != 3 {
					t.Errorf("Expected 3 profiles, got %d", len(cfg.Profiles))
				}

				fastProfile := cfg.Profiles["fast"]
				if fastProfile.MaxPixels != 1048576 {
					t.Errorf("Expected fast max_pixels 1048576, got %d", fastProfile.MaxPixels)
				}

				// Memory config
				if cfg.Memory.MaxTotalGB != "32" {
					t.Errorf("Expected max_total_gb 32, got %s", cfg.Memory.MaxTotalGB)
				}
				if cfg.Memory.UnloadStrategy != "lru" {
					t.Errorf("Expected unload_strategy lru, got %s", cfg.Memory.UnloadStrategy)
				}
				if len(cfg.Memory.KeepModels) != 2 {
					t.Errorf("Expected 2 keep_models, got %d", len(cfg.Memory.KeepModels))
				}

				// Logging config
				if cfg.Logging.Level != "info" {
					t.Errorf("Expected log level info, got %s", cfg.Logging.Level)
				}
				if cfg.Logging.Format != "json" {
					t.Errorf("Expected log format json, got %s", cfg.Logging.Format)
				}
			},
		},
		{
			name:        "file not found",
			content:     "",
			expectError: true,
			validate:    nil,
		},
		{
			name: "invalid yaml",
			content: `
server:
  host: [invalid
  port: 8080
`,
			expectError: true,
			validate:    nil,
		},
		{
			name: "empty file",
			content: ``,
			expectError: false,
			validate: func(t *testing.T, cfg *Config) {
				// Empty config should load with zero values
				if cfg.Server.Host != "" {
					t.Errorf("Expected empty host, got %s", cfg.Server.Host)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var tmpFile string

			// Handle "file not found" case specially
			if tt.name == "file not found" {
				tmpFile = filepath.Join(tmpDir, "nonexistent.yaml")
			} else if tt.content == "" && tt.name == "empty file" {
				// Create an actually empty file
				tmpFile = filepath.Join(tmpDir, "empty.yaml")
				if err := os.WriteFile(tmpFile, []byte{}, 0644); err != nil {
					t.Fatalf("Failed to write empty test config: %v", err)
				}
			} else {
				tmpFile = filepath.Join(tmpDir, tt.name+".yaml")
				if err := os.WriteFile(tmpFile, []byte(tt.content), 0644); err != nil {
					t.Fatalf("Failed to write test config: %v", err)
				}
			}

			cfg, err := LoadConfig(tmpFile)

			if tt.expectError {
				if err == nil {
					t.Error("Expected error but got none")
				}
				return
			}

			if err != nil {
				t.Fatalf("Failed to load config: %v", err)
			}

			if tt.validate != nil {
				tt.validate(t, cfg)
			}
		})
	}
}

func TestLoadConfigSpecialCharacters(t *testing.T) {
	tmpDir := t.TempDir()

	tmpFile := filepath.Join(tmpDir, "special.yaml")
	content := `
server:
  host: "::1"  # IPv6
  port: 8080
  default_model: "gui-actor-2b"

models:
  "model-with-dash":
    path: "/path/to/model"
    enabled: true
    preload: false
    min_pixels: 1000
    max_pixels: 5000000
    max_context_length: 4096
    memory_limit_gb: 8
`
	if err := os.WriteFile(tmpFile, []byte(content), 0644); err != nil {
		t.Fatalf("Failed to write test config: %v", err)
	}

	cfg, err := LoadConfig(tmpFile)
	if err != nil {
		t.Fatalf("Failed to load config: %v", err)
	}

	if cfg.Server.Host != "::1" {
		t.Errorf("Expected IPv6 address ::1, got %s", cfg.Server.Host)
	}

	if _, ok := cfg.Models["model-with-dash"]; !ok {
		t.Error("Expected model with dash in name to be loaded")
	}
}
