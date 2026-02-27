//go:build mlx_mock

package mlx

import (
	"fmt"

	"github.com/agenthands/GUI-Actor/internal/radix"
)

// MockMLXEngine is a test double
type MockMLXEngine struct {
	// Delegate to existing mock implementation
	*radix.MockMLXEngine
}

func NewRealMLXEngine(modelPath string, vocabSize int) *MockMLXEngine {
	return &MockMLXEngine{
		MockMLXEngine: &radix.MockMLXEngine{},
	}
}

func (e *MockMLXEngine) LoadModel() error {
	return fmt.Errorf("mock: model not loaded")
}

func (e *MockMLXEngine) ForwardWithCache(model any, tokens []uint32, baseHandle uint64) ([]float32, uint64, error) {
	if e.MockMLXEngine != nil && e.MockMLXEngine.ForwardFunc != nil {
		return e.MockMLXEngine.ForwardFunc(model, tokens, baseHandle)
	}
	return nil, 0, fmt.Errorf("mock: ForwardFunc not implemented")
}

func (e *MockMLXEngine) SliceCache(handle uint64, keepTokens int) (uint64, error) {
	if e.MockMLXEngine != nil && e.MockMLXEngine.SliceFunc != nil {
		return e.MockMLXEngine.SliceFunc(handle, keepTokens)
	}
	return 0, fmt.Errorf("mock: SliceFunc not implemented")
}

func (e *MockMLXEngine) FreeCache(handle uint64) {
	if e.MockMLXEngine != nil && e.MockMLXEngine.FreeFunc != nil {
		e.MockMLXEngine.FreeFunc(handle)
	}
}
