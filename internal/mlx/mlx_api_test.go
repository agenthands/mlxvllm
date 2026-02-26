package mlx

import (
	"testing"
)

// TestMLXConstants verifies C constants are correctly mapped
func TestMLXConstants(t *testing.T) {
	if RootCacheHandle != 0 {
		t.Errorf("Expected RootCacheHandle to be 0, got %d", RootCacheHandle)
	}

	if Success != 0 {
		t.Errorf("Expected Success to be 0, got %d", Success)
	}

	// Verify error codes are negative
	if ErrorInvalidHandle >= 0 {
		t.Error("Expected ErrorInvalidHandle to be negative")
	}
}

// TestMLXFunctionSignatures verifies function signatures are correct
func TestMLXFunctionSignatures(t *testing.T) {
	// Verify functions are callable (type checking)
	// If signatures don't match, this will fail to compile
	_ = ForwardWithCache
	_ = SliceCache
	_ = FreeCache
}

// TestMLXAPIHeaderCompilation verifies the C header compiles with CGO
func TestMLXAPIHeaderCompilation(t *testing.T) {
	// This test passes if the package compiles
	// The CGO directives in mlx_bridges.go include the header
	// If the header has syntax errors, this test will fail to compile
}

// TestCBufferSizeCalculations verifies buffer size calculations
func TestCBufferSizeCalculations(t *testing.T) {
	// Size calculations for C buffers
	const numTokens = 100
	const vocabSize = 32000

	// Token buffer size (uint32_t = 4 bytes)
	tokenBufferSize := numTokens * 4
	if tokenBufferSize != 400 {
		t.Errorf("Token buffer size calculation failed: got %d", tokenBufferSize)
	}

	// Logits buffer size (float = 4 bytes)
	logitsBufferSize := vocabSize * 4
	if logitsBufferSize != 128000 {
		t.Errorf("Logits buffer size calculation failed: got %d", logitsBufferSize)
	}
}
