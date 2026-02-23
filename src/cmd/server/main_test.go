package main

import (
	"os"
	"testing"

	ort "github.com/yalue/onnxruntime_go"
)

func TestNewInferenceEngine(t *testing.T) {
	// Skip if libonnxruntime is not available
	libPath := "/opt/homebrew/opt/onnxruntime/lib/libonnxruntime.dylib"
	if _, err := os.Stat(libPath); os.IsNotExist(err) {
		libPath = "/usr/local/lib/libonnxruntime.dylib"
	}
	if _, err := os.Stat(libPath); os.IsNotExist(err) {
		t.Skip("libonnxruntime.dylib not found")
	}

	ort.SetSharedLibraryPath(libPath)
	err := ort.InitializeEnvironment()
	if err != nil {
		t.Fatalf("Failed to initialize runtime: %v", err)
	}
	defer ort.DestroyEnvironment()

	engine, err := NewInferenceEngine("../../../onnx_models")
	if err != nil {
		t.Fatalf("NewInferenceEngine failed: %v", err)
	}

	if engine == nil {
		t.Fatal("InferenceEngine is nil")
	}

	if engine.visionSession == nil {
		t.Error("visionSession should be initialized")
	}
	if engine.pointerSession == nil {
		t.Error("pointerSession should be initialized")
	}
}
