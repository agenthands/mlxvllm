package radix

import (
	"testing"
)

func TestMLXEngineInterface(t *testing.T) {
	var _ MLXEngine = (*MockMLXEngine)(nil)
}
