//go:build mlx_mock

package mlx

import (
	"errors"
)

// Mock implementations for testing without real MLX library

const (
	RootCacheHandle        = 0
	Success                = 0
	ErrorInvalidHandle     = -1
	ErrorOutOfMemory       = -2
	ErrorInvalidTokens     = -3
	ErrorComputationFailed = -4
	ErrorModelNotLoaded    = -5
)

// ForwardWithCache is a mock implementation
func ForwardWithCache(
	modelHandle uintptr,
	tokens []uint32,
	baseCacheHandle uint64,
) ([]float32, uint64, error) {
	if len(tokens) == 0 {
		return nil, 0, errors.New("empty tokens")
	}

	// Mock: return fake logits and new cache handle
	logits := make([]float32, 32000)
	for i := range logits {
		logits[i] = 0.01
	}

	return logits, baseCacheHandle + 1, nil
}

// SliceCache is a mock implementation
func SliceCache(cacheHandle uint64, keepTokens int) (uint64, error) {
	return cacheHandle + 100, nil
}

// FreeCache is a mock implementation
func FreeCache(cacheHandle uint64) {
	// No-op for mock
}

// FreeError is a mock implementation
func FreeError(errMsg *byte) {
	// No-op for mock
}
