package main

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"log"
	"net/http"
	"os"
	"strings"
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

// PreprocessImage decodes a base64 image and converts it to a float32 RGB slice
func PreprocessImage(base64Str string) (pixels []float32, width, height int, err error) {
	// Handle data URL prefix if present
	if idx := strings.Index(base64Str, ","); idx != -1 {
		base64Str = base64Str[idx+1:]
	}

	reader := base64.NewDecoder(base64.StdEncoding, strings.NewReader(base64Str))
	img, _, err := image.Decode(reader)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("failed to decode image: %v", err)
	}

	bounds := img.Bounds()
	width, height = bounds.Dx(), bounds.Dy()
	pixels = make([]float32, 0, width*height*3)

	// Simple RGB extraction (not optimized for large images yet)
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			// RGBA() returns 16-bit values (0-65535)
			pixels = append(pixels, float32(r)/65535.0, float32(g)/65535.0, float32(b)/65535.0)
		}
	}

	return pixels, width, height, nil
}

// RunInference orchestrates the Vision -> LLM -> Pointer Head pipeline
func (e *InferenceEngine) RunInference(pixels []float32, width, height int, instruction string) (string, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	// 1. Run Vision Tower
	// 2. Run LLM (Backbone) - Note: This needs to be implemented or mocked
	// 3. Run Pointer Head

	return "pyautogui.click(0.5, 0.5)", nil
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
