package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
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

func TestPreprocessImage(t *testing.T) {
	// Generate a 1x1 black PNG
	img := image.NewRGBA(image.Rect(0, 0, 1, 1))
	img.Set(0, 0, color.Black)
	var buf bytes.Buffer
	png.Encode(&buf, img)
	base64Image := base64.StdEncoding.EncodeToString(buf.Bytes())
	
	pixels, width, height, err := PreprocessImage(base64Image)
	if err != nil {
		t.Fatalf("PreprocessImage failed: %v", err)
	}

	if width != 1 || height != 1 {
		t.Errorf("Expected 1x1 image, got %dx%d", width, height)
	}

	if len(pixels) != 3 { // 3 channels (RGB)
		t.Errorf("Expected 3 pixels (RGB), got %d", len(pixels))
	}
}

func TestRunInference(t *testing.T) {
	// Skip if libonnxruntime is not available
	libPath := "/opt/homebrew/opt/onnxruntime/lib/libonnxruntime.dylib"
	if _, err := os.Stat(libPath); os.IsNotExist(err) {
		libPath = "/usr/local/lib/libonnxruntime.dylib"
	}
	if _, err := os.Stat(libPath); os.IsNotExist(err) {
		t.Skip("libonnxruntime.dylib not found")
	}

	ort.SetSharedLibraryPath(libPath)
	_ = ort.InitializeEnvironment()
	defer ort.DestroyEnvironment()

	engine, err := NewInferenceEngine("../../../onnx_models")
	if err != nil {
		t.Fatalf("NewInferenceEngine failed: %v", err)
	}

	pixels := make([]float32, 224*224*3)
	instruction := "click the button"
	
	// Test the core inference orchestration
	result, err := engine.RunInference(pixels, 224, 224, instruction)
	if err != nil {
		t.Fatalf("RunInference failed: %v", err)
	}

	if result == "" {
		t.Error("RunInference returned empty result")
	}
}

func TestGetClickPoint(t *testing.T) {
	// Sample attention scores for a 2x2 grid (4 patches)
	// Scores: [0.1, 0.8, 0.05, 0.05] -> Max is at index 1 (top-right)
	scores := []float32{0.1, 0.8, 0.05, 0.05}
	nWidth, nHeight := 2, 2

	x, y, err := GetClickPoint(scores, nWidth, nHeight)
	if err != nil {
		t.Fatalf("GetClickPoint failed: %v", err)
	}

	// Index 1 in 2x2 grid is x=1, y=0
	// Normalized center of patch (1,0) should be around x=0.75, y=0.25
	expectedX, expectedY := 0.75, 0.25
	if x < expectedX-0.01 || x > expectedX+0.01 || y < expectedY-0.01 || y > expectedY+0.01 {
		t.Errorf("Expected click point (%.2f, %.2f), got (%.2f, %.2f)", expectedX, expectedY, x, y)
	}
}

func TestHandleChatComplex(t *testing.T) {
	engine := &InferenceEngine{}
	
	// Create a complex multi-modal request with base64 image
	img := image.NewRGBA(image.Rect(0, 0, 1, 1))
	var buf bytes.Buffer
	png.Encode(&buf, img)
	base64Image := base64.StdEncoding.EncodeToString(buf.Bytes())

	reqBody := fmt.Sprintf(`{
		"model": "gui-actor",
		"messages": [
			{
				"role": "user",
				"content": [
					{"type": "text", "text": "click here"},
					{"type": "image_url", "image_url": {"url": "data:image/png;base64,%s"}}
				]
			}
		]
	}`, base64Image)

	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(reqBody))
	w := httptest.NewRecorder()

	engine.HandleChat(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("Expected status OK, got %v", resp.Status)
	}
}

func TestHandleChatEndToEnd(t *testing.T) {
	// Skip if libonnxruntime is not available
	libPath := "/opt/homebrew/opt/onnxruntime/lib/libonnxruntime.dylib"
	if _, err := os.Stat(libPath); os.IsNotExist(err) {
		libPath = "/usr/local/lib/libonnxruntime.dylib"
	}
	if _, err := os.Stat(libPath); os.IsNotExist(err) {
		t.Skip("libonnxruntime.dylib not found")
	}

	ort.SetSharedLibraryPath(libPath)
	_ = ort.InitializeEnvironment()
	defer ort.DestroyEnvironment()

	engine, err := NewInferenceEngine("../../../onnx_models")
	if err != nil {
		t.Fatalf("NewInferenceEngine failed: %v", err)
	}

	img := image.NewRGBA(image.Rect(0, 0, 224, 224))
	var buf bytes.Buffer
	png.Encode(&buf, img)
	base64Image := base64.StdEncoding.EncodeToString(buf.Bytes())

	reqBody := fmt.Sprintf(`{
		"model": "gui-actor",
		"messages": [
			{
				"role": "user",
				"content": [
					{"type": "text", "text": "click start"},
					{"type": "image_url", "image_url": {"url": "data:image/png;base64,%s"}}
				]
			}
		]
	}`, base64Image)

	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(reqBody))
	w := httptest.NewRecorder()

	engine.HandleChat(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("Expected status OK, got %v", resp.Status)
	}
}
