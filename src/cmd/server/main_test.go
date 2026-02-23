package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
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

func TestHandleChat(t *testing.T) {
	engine := &InferenceEngine{} // Mock engine with nil sessions
	
	// Create a sample OpenAI-compatible request
	reqBody := `{"model": "gui-actor", "messages": [{"role": "user", "content": "click the start button"}]}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(reqBody))
	w := httptest.NewRecorder()

	engine.HandleChat(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("Expected status OK, got %v", resp.Status)
	}

	var completion ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&completion); err != nil {
		t.Fatalf("Failed to decode response: %v", err)
	}

	if len(completion.Choices) == 0 {
		t.Fatal("Expected at least one choice in response")
	}
	
	expected := "pyautogui.click(0.5, 0.5)"
	if completion.Choices[0].Message.Content != expected {
		t.Errorf("Expected content %q, got %q", expected, completion.Choices[0].Message.Content)
	}
}
