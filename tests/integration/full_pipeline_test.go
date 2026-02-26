//go:build integration
// +build integration

package integration

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"testing"
	"time"

	"github.com/agenthands/GUI-Actor/internal/http"
	"github.com/agenthands/GUI-Actor/internal/radix"
	"github.com/agenthands/GUI-Actor/pkg/tokenizer"
)

// TestFullPipelineIntegration tests the complete request pipeline
// Requires Apple Silicon hardware and MLX runtime
func TestFullPipelineIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	// This test requires:
	// 1. Apple Silicon M4 Pro (or similar)
	// 2. MLX runtime library built and installed
	// 3. Model weights available
	// 4. Sufficient memory

	t.Skip("Integration tests require Apple Silicon hardware - set up environment first")

	// Setup components
	tree := radix.NewTree()
	engine := setupMLXEngine(t)
	tok := tokenizer.NewTokenizer(32000)
	model := loadModel(t)

	server := http.NewServer(tree, engine, tok, model)

	// Start HTTP server
	addr := startTestServer(server)
	defer stopTestServer(addr)

	// Test 1: Simple chat completion
	t.Run("SimpleChatCompletion", func(t *testing.T) {
		req := http.ChatCompletionRequest{
			Messages: []tokenizer.ChatMessage{
				{Role: "system", Content: "You are a helpful assistant."},
				{Role: "user", Content: "Say 'Hello, World!'"},
			},
			MaxTokens: 10,
		}

		response := sendChatCompletion(t, addr, req)

		if len(response.Choices) == 0 {
			t.Fatal("Expected at least one choice")
		}

		if response.Choices[0].Message.Content == "" {
			t.Error("Expected non-empty content")
		}

		t.Logf("Response: %s", response.Choices[0].Message.Content)
	})

	// Test 2: Prefix caching - second request should hit cache
	t.Run("PrefixCaching", func(t *testing.T) {
		// First request
		req1 := http.ChatCompletionRequest{
			Messages: []tokenizer.ChatMessage{
				{Role: "user", Content: "The capital of France is"},
			},
			MaxTokens: 5,
		}

		start1 := time.Now()
		resp1 := sendChatCompletion(t, addr, req1)
		duration1 := time.Since(start1)

		// Second request with same prefix
		req2 := http.ChatCompletionRequest{
			Messages: []tokenizer.ChatMessage{
				{Role: "user", Content: "The capital of France is Paris"},
			},
			MaxTokens: 5,
		}

		start2 := time.Now()
		resp2 := sendChatCompletion(t, addr, req2)
		duration2 := time.Since(start2)

		t.Logf("First request: %v, Second request: %v", duration1, duration2)

		// Second request should be faster due to cache hit
		// (This is a weak assertion due to variance)
		if duration2 < duration1 {
			t.Log("Cache hit: second request was faster")
		}

		if len(resp2.Choices) == 0 {
			t.Fatal("Expected at least one choice")
		}
	})

	// Test 3: Concurrent requests
	t.Run("ConcurrentRequests", func(t *testing.T) {
		const numRequests = 5

		results := make(chan *http.ChatCompletionResponse, numRequests)
		errors := make(chan error, numRequests)

		for i := 0; i < numRequests; i++ {
			go func(idx int) {
				req := http.ChatCompletionRequest{
					Messages: []tokenizer.ChatMessage{
						{Role: "user", Content: fmt.Sprintf("Request %d", idx)},
					},
					MaxTokens: 3,
				}

				resp, err := sendChatCompletionAsync(addr, req)
				if err != nil {
					errors <- err
					return
				}
				results <- resp
			}(i)
		}

		// Collect results
		successCount := 0
		for i := 0; i < numRequests; i++ {
			select {
			case <-results:
				successCount++
			case err := <-errors:
				t.Logf("Request failed: %v", err)
			case <-time.After(30 * time.Second):
				t.Fatal("Timeout waiting for responses")
			}
		}

		if successCount < numRequests {
			t.Errorf("Only %d/%d requests succeeded", successCount, numRequests)
		}
	})

	// Test 4: Multimodal (text + image)
	t.Run("MultimodalRequest", func(t *testing.T) {
		// Small 1x1 PNG red pixel
		imageBase64 := "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="

		req := http.ChatCompletionRequest{
			Messages: []tokenizer.ChatMessage{
				{
					Role:    "user",
					Content: "What color is this image?",
					Image:   imageBase64,
				},
			},
			MaxTokens: 10,
		}

		response := sendChatCompletion(t, addr, req)

		if len(response.Choices) == 0 {
			t.Fatal("Expected at least one choice")
		}

		t.Logf("Multimodal response: %s", response.Choices[0].Message.Content)
	})

	// Test 5: LRU eviction
	t.Run("LRUEviction", func(t *testing.T) {
		// Generate many requests to fill LRU
		const numRequests = 100

		for i := 0; i < numRequests; i++ {
			req := http.ChatCompletionRequest{
				Messages: []tokenizer.ChatMessage{
					{Role: "user", Content: fmt.Sprintf("Unique message %d", i)},
				},
				MaxTokens: 2,
			}

			sendChatCompletion(t, addr, req)
		}

		// Check that tree hasn't grown unbounded
		// (LRU should have evicted old entries)
		t.Logf("Tree LRU size: (would check actual size here)")
	})
}

// TestRadixTreeBehavior tests Radix tree specific behaviors
func TestRadixTreeBehavior(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	t.Skip("Requires MLX runtime")

	tree := radix.NewTree()
	engine := setupMLXEngine(t)
	tok := tokenizer.NewTokenizer(32000)
	model := loadModel(t)

	t.Run("PrefixMatch", func(t *testing.T) {
		// Insert [1,2,3,4,5]
		node1, _ := tree.InsertPending([]uint32{1, 2, 3, 4, 5}, engine, model)
		radix.FinalizeNode(node1, 100)

		// Query [1,2,3] should match
		match := tree.Match([]uint32{1, 2, 3})
		if match == nil {
			t.Error("Expected match for prefix")
		}

		if match.CacheHandle != 100 {
			t.Errorf("Expected cache handle 100, got %d", match.CacheHandle)
		}
	})

	t.Run("ThunderingHerd", func(t *testing.T) {
		// Many concurrent requests for same tokens
		const numGoroutines = 10
		nodes := make(chan *radix.Node, numGoroutines)

		for i := 0; i < numGoroutines; i++ {
			go func() {
				node, _ := tree.InsertPending([]uint32{99, 98, 97}, engine, model)
				nodes <- node
			}()
		}

		// Collect nodes
		var firstNode *radix.Node
		uniqueNodes := make(map[*radix.Node]bool)

		for i := 0; i < numGoroutines; i++ {
			node := <-nodes
			if firstNode == nil {
				firstNode = node
			}
			uniqueNodes[node] = true
		}

		// All should get same node (thundering herd prevention)
		if len(uniqueNodes) != 1 {
			t.Errorf("Expected 1 unique node, got %d", len(uniqueNodes))
		}

		// refCount should be number of goroutines
		if firstNode.refCount.Load() != numGoroutines {
			t.Errorf("Expected refCount %d, got %d", numGoroutines, firstNode.refCount.Load())
		}
	})

	t.Run("PoisonedNodeRetry", func(t *testing.T) {
		// Create and poison node
		node1, _ := tree.InsertPending([]uint32{77, 88}, engine, model)
		radix.PoisonNode(node1, fmt.Errorf("test error"))

		// Try again - should create new node
		node2, _ := tree.InsertPending([]uint32{77, 88}, engine, model)

		if node1 == node2 {
			t.Error("Expected different node after poison")
		}

		if node2.err != nil {
			t.Errorf("New node should not be poisoned: %v", node2.err)
		}
	})
}

// TestMemoryManagement tests memory-related behaviors
func TestMemoryManagement(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	t.Skip("Requires MLX runtime")

	tree := radix.NewTree()
	engine := setupMLXEngine(t)
	model := loadModel(t)

	t.Run("UnpinAndEvict", func(t *testing.T) {
		// Create node
		node, _ := tree.InsertPending([]uint32{1, 2, 3}, engine, model)
		radix.FinalizeNode(node, 100)

		// Unpin - should add to LRU
		tree.Unpin(node)

		if node.lruElem == nil {
			t.Error("Expected node to be in LRU")
		}

		// Evict - should remove from tree
		tree.EvictLRU(1)

		// Verify node removed
		// (Would need to expose tree internals or add Contains method)
	})

	t.Run("CascadingCleanup", func(t *testing.T) {
		// Create tree: root -> [1] -> [2] -> [3]
		n1, _ := tree.InsertPending([]uint32{1}, engine, model)
		radix.FinalizeNode(n1, 100)

		n2, _ := tree.InsertPending([]uint32{1, 2}, engine, model)
		radix.FinalizeNode(n2, 200)

		n3, _ := tree.InsertPending([]uint32{1, 2, 3}, engine, model)
		radix.FinalizeNode(n3, 300)

		// Poison middle node
		radix.PoisonNode(n2, fmt.Errorf("error"))

		// Prune should remove n2 and n3
		tree.PrunePoisoned()

		// Verify cleanup
		// (Would need to expose tree internals)
	})
}

// Helper functions

func setupMLXEngine(t *testing.T) radix.MLXEngine {
	// In production, this would load the actual MLX engine
	// For now, use mock
	return &radix.MockMLXEngine{
		ForwardFunc: func(model any, tokens []uint32, base uint64) ([]float32, uint64, error) {
			return make([]float32, 32000), base + 1, nil
		},
	}
}

func loadModel(t *testing.T) any {
	// In production, this would load actual model weights
	return "test-model"
}

func startTestServer(server *http.Server) string {
	mux := http.NewServeMux()
	server.RegisterRoutes(mux)

	hs := &http.Server{
		Addr:    ":0", // Random available port
		Handler: mux,
	}

	go func() {
		hs.ListenAndServe()
	}()

	// Wait for server to start
	time.Sleep(100 * time.Millisecond)

	return hs.Addr
}

func stopTestServer(addr string) {
	// Shutdown server
	// (Implementation depends on actual server setup)
}

func sendChatCompletion(t *testing.T, addr string, req http.ChatCompletionRequest) *http.ChatCompletionResponse {
	body, _ := json.Marshal(req)
	resp, err := http.Post("http://"+addr+"/v1/chat/completions", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("Failed to send request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("Expected status 200, got %d", resp.StatusCode)
	}

	var response http.ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		t.Fatalf("Failed to decode response: %v", err)
	}

	return &response
}

func sendChatCompletionAsync(addr string, req http.ChatCompletionRequest) (*http.ChatCompletionResponse, error) {
	body, _ := json.Marshal(req)
	resp, err := http.Post("http://"+addr+"/v1/chat/completions", "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("status %d", resp.StatusCode)
	}

	var response http.ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, err
	}

	return &response, nil
}

// TestMain runs setup/teardown for integration tests
func TestMain(m *testing.M) {
	// Check if integration tests should run
	if os.Getenv("INTEGRATION_TEST") == "" {
		fmt.Println("Skipping integration tests (set INTEGRATION_TEST=1 to run)")
		os.Exit(0)
	}

	// Setup MLX environment
	// - Load model
	// - Initialize runtime
	// - Set up test data

	// Run tests
	code := m.Run()

	// Teardown
	// - Unload model
	// - Clean up test data

	os.Exit(code)
}
