package mlx

import (
	"testing"
)

func TestMLXInit(t *testing.T) {
	err := Init()
	if err != nil {
		t.Fatalf("Failed to initialize MLX: %v", err)
	}
	defer Shutdown()

	if !IsInitialized() {
		t.Error("MLX should be initialized")
	}
}

func TestMLXGetDefaultDevice(t *testing.T) {
	err := Init()
	if err != nil {
		t.Fatal(err)
	}
	defer Shutdown()

	device := GetDefaultDevice()
	if device == "" {
		t.Error("Expected non-empty device name")
	}
}
