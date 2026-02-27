//go:build !mlx_mock

package mlx

import (
	"os"
	"path/filepath"
	"testing"
)

func TestNewRealMLXEngine(t *testing.T) {
	engine := NewRealMLXEngine("/fake/path", 152064)

	if engine == nil {
		t.Fatal("Expected non-nil engine")
	}

	if engine.modelPath != "/fake/path" {
		t.Errorf("Expected modelPath /fake/path, got %s", engine.modelPath)
	}

	if engine.vocabSize != 152064 {
		t.Errorf("Expected vocabSize 152064, got %d", engine.vocabSize)
	}

	if engine.loaded {
		t.Error("Expected engine to not be loaded initially")
	}
}

func TestRealMLXEngine_SliceCache(t *testing.T) {
	// This test will fail without actual MLX library
	// Skip for now, will be tested in integration
	t.Skip("Requires MLX library - tested in integration")
}

func TestRealMLXEngine_FreeCache(t *testing.T) {
	engine := NewRealMLXEngine("/fake/path", 152064)

	// Should not panic
	engine.FreeCache(123)
	engine.FreeCache(0) // Root handle - should also not panic
}

func TestRealMLXEngine_LoadModel(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping real model test in short mode")
	}

	modelPath := os.Getenv("MLX_TEST_MODEL_PATH")
	if modelPath == "" {
		// Try default path
		modelPath = "../models/qwen2-vl/7b"
		absPath, err := filepath.Abs(modelPath)
		if err != nil {
			t.Skip("Could not resolve model path")
		}
		modelPath = absPath
	}

	t.Logf("Loading model from: %s", modelPath)

	engine := NewRealMLXEngine(modelPath, 152064)

	err := engine.LoadModel()
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	if !engine.loaded {
		t.Error("Expected engine.loaded to be true after LoadModel()")
	}

	t.Log("Model loaded successfully!")
}
