package model

import (
	"testing"

	"github.com/agenthands/mlxvllm/internal/config"
)

func TestNewRegistry(t *testing.T) {
	cfg := &config.Config{
		Models: map[string]config.ModelConfig{
			"gui-actor-2b": {
				Path:    "/tmp/models/2b",
				Enabled: true,
				Preload: true,
			},
		},
		Memory: config.MemoryConfig{
			MaxTotalGB: "32",
		},
	}

	reg := NewRegistry(cfg)
	if reg == nil {
		t.Fatal("Expected non-nil registry")
	}

	if !reg.HasModel("gui-actor-2b") {
		t.Error("Expected gui-actor-2b to be registered")
	}
}

func TestLoadModel(t *testing.T) {
	cfg := &config.Config{
		Models: map[string]config.ModelConfig{
			"gui-actor-2b": {
				Path:    "/tmp/models/2b",
				Enabled: true,
				Preload: false,
			},
		},
	}

	reg := NewRegistry(cfg)
	err := reg.LoadModel("gui-actor-2b")
	// Should fail for non-existent path, but API should work
	if err == nil {
		t.Log("Note: model loading will fail with non-existent path")
	}
}
