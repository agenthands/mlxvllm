package main

import (
	"encoding/json"
	"log"
	"net/http"
	"os"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// OpenAI-compatible types
type ChatCompletionRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
}

type Message struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"` // Can be string or list of content objects
}

type ChatCompletionResponse struct {
	ID      string   `json:"id"`
	Choices []Choice `json:"choices"`
}

type Choice struct {
	Message Message `json:"message"`
}

// InferenceEngine wraps ONNX sessions and ensures they stay in GPU memory
type InferenceEngine struct {
	mu           sync.Mutex
	visionSession *ort.AdvancedSession
	llmSession    *ort.AdvancedSession
	pointerSession *ort.AdvancedSession
}

func NewInferenceEngine(modelPath string) (*InferenceEngine, error) {
	// Initialize ONNX Runtime with CoreML Execution Provider
	options, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	defer options.Destroy()

	// CRITICAL: This line enables Apple Silicon GPU/ANE acceleration
	err = options.AppendExecutionProviderCoreML(0)
	if err != nil {
		log.Printf("Warning: Failed to enable CoreML, falling back to CPU: %v", err)
	}

	// Load sessions - Sessions stay active to keep models in GPU memory
	log.Println("Loading Vision Tower into GPU...")
	// Note: You'll need to define input/output names matching the ONNX export
	// vision, _ := ort.NewAdvancedSession(modelPath+"/vision_tower.onnx", ...)

	return &InferenceEngine{}, nil
}

func (e *InferenceEngine) HandleChat(w http.ResponseWriter, r *http.Request) {
	var req ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// 1. Pre-process image from Message Content
	// 2. Run Vision Session -> image_embeds
	// 3. Run LLM Session -> hidden_states
	// 4. Run Pointer Session -> coordinates

	// Mocking response for structural demonstration
	resp := ChatCompletionResponse{
		ID: "gui-actor-001",
		Choices: []Choice{
			{Message: Message{Role: "assistant", Content: "pyautogui.click(0.5, 0.5)"}},
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func main() {
	// 1. Initialise the global ONNX runtime
	libPath := "/opt/homebrew/opt/onnxruntime/lib/libonnxruntime.dylib"
	if _, err := os.Stat(libPath); os.IsNotExist(err) {
		libPath = "/usr/local/lib/libonnxruntime.dylib"
	}
	ort.SetSharedLibraryPath(libPath)
	if err := ort.InitializeEnvironment(); err != nil {
		log.Fatal(err)
	}
	defer ort.DestroyEnvironment()

	// 2. Initialize the model engine
	engine, err := NewInferenceEngine("./onnx_models")
	if err != nil {
		log.Fatal(err)
	}

	// 3. Setup OpenAI-compatible API
	http.HandleFunc("/v1/chat/completions", engine.HandleChat)

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	log.Printf("GUI-Actor OpenAI-compatible server running on :%s", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}
