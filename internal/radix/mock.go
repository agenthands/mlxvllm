package radix

import "errors"

// MockMLXEngine is a test double for MLXEngine
// This is exported for use in other packages' tests
type MockMLXEngine struct {
	ForwardFunc func(model any, tokens []uint32, base uint64) ([]float32, uint64, error)
	SliceFunc   func(handle uint64, keepTokens int) (uint64, error)
	FreeFunc    func(handle uint64)
}

func (m *MockMLXEngine) ForwardWithCache(model any, tokens []uint32, base uint64) ([]float32, uint64, error) {
	if m.ForwardFunc != nil {
		return m.ForwardFunc(model, tokens, base)
	}
	return nil, 0, errors.New("not implemented")
}

func (m *MockMLXEngine) SliceCache(handle uint64, keepTokens int) (uint64, error) {
	if m.SliceFunc != nil {
		return m.SliceFunc(handle, keepTokens)
	}
	return 0, errors.New("not implemented")
}

func (m *MockMLXEngine) FreeCache(handle uint64) {
	if m.FreeFunc != nil {
		m.FreeFunc(handle)
	}
}
