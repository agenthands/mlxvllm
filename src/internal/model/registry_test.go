package model

import (
	"testing"

	"github.com/agenthands/mlxvllm/internal/config"
)

func TestNewRegistry(t *testing.T) {
	tests := []struct {
		name          string
		cfg           *config.Config
		expectModels  []string
		expectCount   int
	}{
		{
			name: "single model",
			cfg: &config.Config{
				Models: map[string]config.ModelConfig{
					"gui-actor-2b": {
						Path:    "/tmp/models/2b",
						Enabled: true,
						Preload: true,
					},
				},
			},
			expectModels: []string{"gui-actor-2b"},
			expectCount:  1,
		},
		{
			name: "multiple models",
			cfg: &config.Config{
				Models: map[string]config.ModelConfig{
					"gui-actor-2b": {
						Path:    "/tmp/models/2b",
						Enabled: true,
					},
					"gui-actor-7b": {
						Path:    "/tmp/models/7b",
						Enabled: true,
					},
				},
			},
			expectModels: []string{"gui-actor-2b", "gui-actor-7b"},
			expectCount:  2,
		},
		{
			name: "disabled model not registered",
			cfg: &config.Config{
				Models: map[string]config.ModelConfig{
					"gui-actor-2b": {
						Path:    "/tmp/models/2b",
						Enabled: true,
					},
					"gui-actor-7b": {
						Path:    "/tmp/models/7b",
						Enabled: false, // Disabled
					},
				},
			},
			expectModels: []string{"gui-actor-2b"},
			expectCount:  1,
		},
		{
			name: "empty config",
			cfg: &config.Config{
				Models: map[string]config.ModelConfig{},
			},
			expectModels: []string{},
			expectCount:  0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reg := NewRegistry(tt.cfg)
			if reg == nil {
				t.Fatal("Expected non-nil registry")
			}

			for _, model := range tt.expectModels {
				if !reg.HasModel(model) {
					t.Errorf("Expected %s to be registered", model)
				}
			}

			if tt.expectCount == 0 && reg.HasModel("anything") {
				t.Error("Expected no models in empty registry")
			}
		})
	}
}

func TestHasModel(t *testing.T) {
	cfg := &config.Config{
		Models: map[string]config.ModelConfig{
			"gui-actor-2b": {
				Path:    "/tmp/models/2b",
				Enabled: true,
			},
		},
	}

	reg := NewRegistry(cfg)

	tests := []struct {
		name      string
		modelID   string
		expectHas bool
	}{
		{"existing model", "gui-actor-2b", true},
		{"non-existing model", "gui-actor-7b", false},
		{"empty string", "", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			has := reg.HasModel(tt.modelID)
			if has != tt.expectHas {
				t.Errorf("HasModel(%s) = %v, want %v", tt.modelID, has, tt.expectHas)
			}
		})
	}
}

func TestLoadModel(t *testing.T) {
	tests := []struct {
		name        string
		modelID     string
		expectError bool
	}{
		{"load 2b model", "gui-actor-2b", false}, // Will succeed with placeholder
		{"load 7b model", "gui-actor-7b", false}, // Will succeed with placeholder
		{"load unknown model", "unknown-model", true},
		{"load empty string", "", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &config.Config{
				Models: map[string]config.ModelConfig{
					"gui-actor-2b": {
						Path:    "/tmp/models/2b",
						Enabled: true,
					},
					"gui-actor-7b": {
						Path:    "/tmp/models/7b",
						Enabled: true,
					},
				},
			}

			reg := NewRegistry(cfg)
			err := reg.LoadModel(tt.modelID)

			if tt.expectError && err == nil {
				t.Errorf("Expected error for %s, got nil", tt.modelID)
			}
			if !tt.expectError && err != nil {
				t.Errorf("Expected no error for %s, got %v", tt.modelID, err)
			}
		})
	}
}

func TestLoadModelTwice(t *testing.T) {
	cfg := &config.Config{
		Models: map[string]config.ModelConfig{
			"gui-actor-2b": {
				Path:    "/tmp/models/2b",
				Enabled: true,
			},
		},
	}

	reg := NewRegistry(cfg)

	// Load once
	err := reg.LoadModel("gui-actor-2b")
	if err != nil {
		t.Fatalf("First load failed: %v", err)
	}

	// Load again - should be idempotent
	err = reg.LoadModel("gui-actor-2b")
	if err != nil {
		t.Errorf("Second load should be idempotent, got error: %v", err)
	}
}

func TestUnloadModel(t *testing.T) {
	cfg := &config.Config{
		Models: map[string]config.ModelConfig{
			"gui-actor-2b": {
				Path:    "/tmp/models/2b",
				Enabled: true,
			},
		},
	}

	reg := NewRegistry(cfg)

	// Unload without loading
	err := reg.UnloadModel("gui-actor-2b")
	if err == nil {
		t.Error("Expected error when unloading non-loaded model")
	}

	// Load then unload
	reg.LoadModel("gui-actor-2b")
	err = reg.UnloadModel("gui-actor-2b")
	if err != nil {
		t.Errorf("Failed to unload loaded model: %v", err)
	}

	// Unload again
	err = reg.UnloadModel("gui-actor-2b")
	if err == nil {
		t.Error("Expected error when unloading already unloaded model")
	}

	// Unload unknown model
	err = reg.UnloadModel("unknown-model")
	if err == nil {
		t.Error("Expected error when unloading unknown model")
	}
}

func TestGetModel(t *testing.T) {
	cfg := &config.Config{
		Models: map[string]config.ModelConfig{
			"gui-actor-2b": {
				Path:    "/tmp/models/2b",
				Enabled: true,
			},
		},
	}

	reg := NewRegistry(cfg)

	// Get without loading
	model, err := reg.GetModel("gui-actor-2b")
	if err == nil {
		t.Error("Expected error when getting non-loaded model")
	}

	// Load then get
	reg.LoadModel("gui-actor-2b")
	model, err = reg.GetModel("gui-actor-2b")
	if err != nil {
		t.Errorf("Failed to get loaded model: %v", err)
	}
	if model == nil {
		t.Error("Expected non-nil model")
	}
	if model.ID() != "gui-actor-2b" {
		t.Errorf("Expected model ID 'gui-actor-2b', got '%s'", model.ID())
	}
	if !model.IsLoaded() {
		t.Error("Expected model to be loaded")
	}

	// Get unknown model
	model, err = reg.GetModel("unknown-model")
	if err == nil {
		t.Error("Expected error when getting unknown model")
	}
}

func TestGUIActorModel(t *testing.T) {
	model := &GUIActorModel{
		name:   "test-model",
		path:   "/tmp/test",
		loaded: true,
	}

	tests := []struct {
		name       string
		method     string
		expectVal  interface{}
	}{
		{"ID method", "ID", "test-model"},
		{"IsLoaded true", "IsLoaded", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			switch tt.method {
			case "ID":
				if id := model.ID(); id != tt.expectVal.(string) {
					t.Errorf("ID() = %s, want %s", id, tt.expectVal)
				}
			case "IsLoaded":
				if loaded := model.IsLoaded(); loaded != tt.expectVal.(bool) {
					t.Errorf("IsLoaded() = %v, want %v", loaded, tt.expectVal)
				}
			}
		})
	}

	// Test Unload
	err := model.Unload()
	if err != nil {
		t.Errorf("Unload() failed: %v", err)
	}
	if model.loaded {
		t.Error("Expected model to be unloaded after Unload()")
	}
	if model.IsLoaded() {
		t.Error("Expected IsLoaded() to return false after Unload()")
	}
}

func TestEstimateMemoryGB(t *testing.T) {
	tests := []struct {
		name      string
		modelName string
		expectGB  float64
	}{
		{"gui-actor-2b", "gui-actor-2b", 4.0},
		{"gui-actor-7b", "gui-actor-7b", 14.0},
		{"unknown model", "unknown-model", 8.0},
		{"empty string", "", 8.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gb := estimateMemoryGB(tt.modelName)
			if gb != tt.expectGB {
				t.Errorf("estimateMemoryGB(%s) = %f, want %f", tt.modelName, gb, tt.expectGB)
			}
		})
	}
}
