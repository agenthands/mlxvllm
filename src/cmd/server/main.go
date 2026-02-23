package main

import (
	"encoding/json"
	"fmt"
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
	mu             sync.Mutex
	visionSession  *ort.DynamicAdvancedSession
	llmSession     *ort.DynamicAdvancedSession
	pointerSession *ort.DynamicAdvancedSession
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

	engine := &InferenceEngine{}

	// Load Vision Tower
	log.Println("Loading Vision Tower...")
	visionPath := modelPath + "/vision_tower.onnx"
	if _, err := os.Stat(visionPath); err == nil {
		engine.visionSession, err = ort.NewDynamicAdvancedSession(
			visionPath,
			[]string{"pixel_values", "grid_thw"},
			[]string{"image_embeds"},
			options,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to load vision session: %v", err)
		}
	} else {
		log.Printf("Warning: Vision Tower not found at %s", visionPath)
	}

	// Load Pointer Head
	log.Println("Loading Pointer Head...")
	pointerPath := modelPath + "/pointer_head.onnx"
	if _, err := os.Stat(pointerPath); err == nil {
		engine.pointerSession, err = ort.NewDynamicAdvancedSession(
			pointerPath,
			[]string{"visual_hidden_states", "target_hidden_states"},
			[]string{"attn_weights", "loss"},
			options,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to load pointer session: %v", err)
		}
	} else {
		log.Printf("Warning: Pointer Head not found at %s", pointerPath)
	}

	return engine, nil
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
