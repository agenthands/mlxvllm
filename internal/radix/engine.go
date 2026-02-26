package radix

// MLXEngine defines the interface for MLX inference operations
// All methods are thread-safe and handle memory management explicitly
type MLXEngine interface {
	// ForwardWithCache executes inference with KV cache
	// Returns logits, new cache handle, or error
	// Caller MUST call FreeLogits on the returned logits
	ForwardWithCache(model any, tokens []uint32, baseHandle uint64) ([]float32, uint64, error)

	// SliceCache creates a zero-copy view of an existing cache
	// O(1) operation using MLX copy-on-write
	SliceCache(handle uint64, keepTokens int) (uint64, error)

	// FreeCache releases a cache handle and associated GPU memory
	// Safe to call multiple times on same handle (idempotent)
	FreeCache(handle uint64)
}

// CacheHandle constants
const (
	RootCacheHandle uint64 = 0 // Represents empty/root cache state
)
