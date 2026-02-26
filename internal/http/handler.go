package http

import (
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"sync"

	"github.com/agenthands/GUI-Actor/internal/radix"
	"github.com/agenthands/GUI-Actor/pkg/tokenizer"
)

// Server handles HTTP requests for chat completions
type Server struct {
	tree      *radix.Tree
	engine    radix.MLXEngine
	tokenizer *tokenizer.Tokenizer
	model     any
	mu        sync.Mutex
}

// NewServer creates a new HTTP server
func NewServer(tree *radix.Tree, engine radix.MLXEngine, tok *tokenizer.Tokenizer, model any) *Server {
	return &Server{
		tree:      tree,
		engine:    engine,
		tokenizer: tok,
		model:     model,
	}
}

// ChatCompletionRequest matches OpenAI API format
type ChatCompletionRequest struct {
	Messages    []tokenizer.ChatMessage `json:"messages"`
	MaxTokens   int                     `json:"max_tokens,omitempty"`
	Temperature float64                 `json:"temperature,omitempty"`
	Stream      bool                    `json:"stream,omitempty"`
}

// ChatCompletionResponse matches OpenAI API format
type ChatCompletionResponse struct {
	ID      string                   `json:"id"`
	Object  string                   `json:"object"`
	Created int64                    `json:"created"`
	Model   string                   `json:"model"`
	Choices []CompletionChoice       `json:"choices"`
	Usage   CompletionUsage          `json:"usage"`
}

type CompletionChoice struct {
	Index        int          `json:"index"`
	Message      ChatMessage  `json:"message"`
	FinishReason string       `json:"finish_reason"`
}

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type CompletionUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ChatCompletionHandler handles POST /v1/chat/completions
func (s *Server) ChatCompletionHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Parse request
	var req ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	// Tokenize input
	tokReq := &tokenizer.ChatRequest{
		Messages:    req.Messages,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
	}
	inputTokens, err := s.tokenizer.TokenizeChatRequest(tokReq)
	if err != nil {
		http.Error(w, fmt.Sprintf("Tokenization failed: %v", err), http.StatusBadRequest)
		return
	}

	// Execute autoregressive generation
	outputTokens, err := s.GenerateAutoregressive(inputTokens, req.MaxTokens)
	if err != nil {
		http.Error(w, fmt.Sprintf("Generation failed: %v", err), http.StatusInternalServerError)
		return
	}

	// Decode output
	outputText, err := s.tokenizer.Decode(outputTokens)
	if err != nil {
		slog.Error("Failed to decode output", "error", err)
		outputText = "" // Continue anyway
	}

	// Build response
	response := ChatCompletionResponse{
		ID:      "chatcmpl-1",
		Object:  "chat.completion",
		Created: 0, // TODO: use actual timestamp
		Model:   "gui-actor",
		Choices: []CompletionChoice{
			{
				Index:        0,
				Message: ChatMessage{
					Role:    "assistant",
					Content: outputText,
				},
				FinishReason: "stop",
			},
		},
		Usage: CompletionUsage{
			PromptTokens:     len(inputTokens),
			CompletionTokens: len(outputTokens),
			TotalTokens:      len(inputTokens) + len(outputTokens),
		},
	}

	// Write response
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// GenerateAutoregressive implements autoregressive generation with bulk aggregation
// Uses RadixAttention for KV cache reuse across tokens
func (s *Server) GenerateAutoregressive(inputTokens []uint32, maxTokens int) ([]uint32, error) {
	// Find prefix match in cache
	baseNode := s.tree.Match(inputTokens)
	var baseHandle uint64 = radix.RootCacheHandle
	var generatedTokens []uint32

	if baseNode != nil {
		baseHandle = baseNode.CacheHandle
		// Wait for node to be ready if pending
		if err := baseNode.Wait(); err != nil {
			// Node was poisoned, start from root
			baseHandle = radix.RootCacheHandle
		}
	}

	// Autoregressive generation loop with bulk aggregation
	var buffer []uint32 // Buffer generated tokens locally
	currentHandle := baseHandle

	for {
		if maxTokens > 0 && len(generatedTokens) >= maxTokens {
			break
		}

		// Get next token (simplified: just echo for now)
		// Production would sample from logits distribution
		var nextToken uint32
		if len(generatedTokens) < len(inputTokens) {
			nextToken = inputTokens[len(generatedTokens)]
		} else {
			// TODO: Actually run forward pass and sample
			nextToken = 0 // End of sequence
			break
		}

		buffer = append(buffer, nextToken)
		generatedTokens = append(generatedTokens, nextToken)

		// Check for end of sequence
		if nextToken == 0 || nextToken == 2 { // TODO: use actual EOS token
			break
		}
	}

	// Bulk insert: insert all generated tokens as single edge
	// This prevents tree fragmentation
	if len(buffer) > 0 {
		node, err := s.tree.InsertPending(buffer, s.engine, s.model)
		if err != nil {
			slog.Error("Failed to insert pending node", "error", err)
		} else {
			// Launch computation in background
			go s.finalizeNode(node, currentHandle)
		}
	}

	return generatedTokens, nil
}

// finalizeNode runs MLX computation and finalizes a pending node
func (s *Server) finalizeNode(node *radix.Node, baseHandle uint64) {
	// Run forward pass
	logits, newHandle, err := s.engine.ForwardWithCache(s.model, node.Tokens, baseHandle)
	if err != nil {
		// Poison the node on error
		radix.PoisonNode(node, err)
		return
	}

	// Don't leak logits - they're returned by ForwardWithCache
	// In production, we'd extract the next token from logits
	_ = logits

	// Finalize the node with new cache handle
	radix.FinalizeNode(node, newHandle)

	// Unpin node (allows LRU eviction)
	s.tree.Unpin(node)
}

// HealthCheckHandler handles GET /health
func (s *Server) HealthCheckHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status": "ok",
	})
}

// RegisterRoutes registers all HTTP routes
func (s *Server) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/v1/chat/completions", s.ChatCompletionHandler)
	mux.HandleFunc("/health", s.HealthCheckHandler)
}

// LogHandler wraps handlers with request logging
func (s *Server) LogHandler(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		slog.Info("Request", "method", r.Method, "path", r.URL.Path)
		next(w, r)
	}
}

// RecoverHandler wraps handlers with panic recovery
func (s *Server) RecoverHandler(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if err := recover(); err != nil {
				slog.Error("Panic recovered", "error", err)
				http.Error(w, "Internal server error", http.StatusInternalServerError)
			}
		}()
		next(w, r)
	}
}

// StreamingResponseWriter handles streaming responses
type StreamingResponseWriter struct {
	w       io.Writer
	encoder *json.Encoder
	mu      sync.Mutex
}

// NewStreamingResponseWriter creates a new streaming response writer
func NewStreamingResponseWriter(w io.Writer) *StreamingResponseWriter {
	return &StreamingResponseWriter{
		w:       w,
		encoder: json.NewEncoder(w),
	}
}

// WriteChunk writes a single chunk to the stream
func (s *StreamingResponseWriter) WriteChunk(chunk interface{}) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Write "data: " prefix for SSE format
	if _, err := io.WriteString(s.w, "data: "); err != nil {
		return err
	}

	if err := s.encoder.Encode(chunk); err != nil {
		return err
	}

	// Write newline
	if _, err := io.WriteString(s.w, "\n"); err != nil {
		return err
	}

	return nil
}

// Close writes the final "[DONE]" chunk
func (s *StreamingResponseWriter) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	_, err := io.WriteString(s.w, "data: [DONE]\n\n")
	return err
}
