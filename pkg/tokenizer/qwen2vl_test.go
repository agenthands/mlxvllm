//go:build !mlx_mock

package tokenizer

import (
	"os"
	"path/filepath"
	"testing"
)

// TestQwen2VLTokenizerLoad tests loading the vocab from model directory
func TestQwen2VLTokenizerLoad(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping real tokenizer test in short mode")
	}

	modelPath := os.Getenv("MLX_TEST_MODEL_PATH")
	if modelPath == "" {
		// Try default path - need to go up more directories from pkg/tokenizer
		absPath, err := filepath.Abs("../../models/qwen2-vl/7b")
		if err != nil {
			t.Skip("Could not resolve model path")
		}
		// Check if path exists
		if _, err := os.Stat(absPath); os.IsNotExist(err) {
			t.Skip("Model directory does not exist: " + absPath)
		}
		modelPath = absPath
	}

	tok := NewQwen2VLTokenizer(modelPath, 151643)
	if err := tok.Load(); err != nil {
		t.Fatalf("Failed to load tokenizer: %v", err)
	}

	t.Log("Vocab loaded successfully")

	// Test decoding some common tokens
	tests := []struct {
		tokenID uint32
		desc    string
	}{
		{0, "First token (exclamation mark)"},
		{151642, "Last valid token (vocab size - 1)"},
	}

	for _, tt := range tests {
		t.Run(tt.desc, func(t *testing.T) {
			text, err := tok.DecodeSingle(tt.tokenID)
			if err != nil {
				t.Fatalf("Failed to decode token %d: %v", tt.tokenID, err)
			}
			t.Logf("Token %d -> %q", tt.tokenID, text)
		})
	}

	// Test encoding a simple string
	t.Run("Encode", func(t *testing.T) {
		text := "Hello"
		tokens, err := tok.Encode(text)
		if err != nil {
			t.Fatalf("Failed to encode: %v", err)
		}
		t.Logf("Encoded %q -> %v", text, tokens)
	})
}
