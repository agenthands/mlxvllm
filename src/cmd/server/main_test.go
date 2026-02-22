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

	engine, err := NewInferenceEngine("./onnx_models")
	if err != nil {
		t.Fatalf("NewInferenceEngine failed: %v", err)
	}

	if engine == nil {
		t.Fatal("InferenceEngine is nil")
	}

	// We expect sessions to be nil for now, but in the future they should be initialized
	// This test will "fail" once we add the requirement for sessions to be non-nil
	if engine.visionSession != nil {
		t.Error("visionSession should be nil for now")
	}
}
