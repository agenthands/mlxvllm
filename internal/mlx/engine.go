//go:build !mlx_mock

package mlx

// RealMLXEngine implements radix.MLXEngine using actual MLX inference
type RealMLXEngine struct {
	loaded    bool
	vocabSize int
	modelPath string
}

// NewRealMLXEngine creates a new MLX engine instance
// Note: Model is not loaded until LoadModel() is called
func NewRealMLXEngine(modelPath string, vocabSize int) *RealMLXEngine {
	return &RealMLXEngine{
		modelPath: modelPath,
		vocabSize: vocabSize,
		loaded:    false,
	}
}

// LoadModel loads the model weights via CGO bridge
// This must be called before ForwardWithCache
// Thread-safe: uses global C++ model state
func (e *RealMLXEngine) LoadModel() error {
	if e.loaded {
		return nil // Already loaded
	}

	if err := LoadModel(e.modelPath, e.vocabSize); err != nil {
		return err
	}

	e.loaded = true
	return nil
}

// ForwardWithCache executes inference with KV cache
// Lock-free: thread safety handled in C++ layer
// Zero-copy: passes Go pointers directly to CGO
func (e *RealMLXEngine) ForwardWithCache(model any, tokens []uint32, baseHandle uint64) ([]float32, uint64, error) {
	// model parameter is ignored - MLX uses global C++ state
	// This is a design simplification for the initial implementation

	// Allocate logits buffer in Go (C++ will write directly to this memory)
	logits := make([]float32, e.vocabSize)

	newHandle, err := forwardWithCacheImpl(tokens, baseHandle, logits)
	if err != nil {
		return nil, 0, err
	}

	return logits, newHandle, nil
}

// forwardWithCacheImpl is the actual CGO bridge call
// separated for easier testing with mocks
func forwardWithCacheImpl(tokens []uint32, baseHandle uint64, logits []float32) (uint64, error) {
	return ForwardWithCache(0, tokens, baseHandle, logits)
}

// SliceCache creates a zero-copy view of existing cache
// O(1) operation using MLX copy-on-write semantics
func (e *RealMLXEngine) SliceCache(handle uint64, keepTokens int) (uint64, error) {
	return SliceCache(handle, keepTokens)
}

// FreeCache releases a cache handle and associated GPU memory
// Thread-safe: idempotent, safe to call multiple times
func (e *RealMLXEngine) FreeCache(handle uint64) {
	FreeCache(handle)
}
