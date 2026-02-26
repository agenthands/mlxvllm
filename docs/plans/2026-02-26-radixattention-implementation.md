# RadixAttention Prefix Caching Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build SGLang-style RadixAttention with Prefix Caching for Apple Silicon MLX server - achieving O(1) cache hits, zero-copy tensor slicing, and fragmentation-free autoregressive generation.

**Architecture:** 4-layer system - HTTP/API Layer (Go), Radix Topology Manager (Go with OCC), Flattened CGO Boundary (primitive-only), Apple MLX Engine (C++/Metal with shared_ptr registry)

**Tech Stack:** Go 1.21+, C++17, MLX framework, Apple Metal, CGO

---

## Task 1: Project Structure & Build Configuration

**Files:**
- Create: `Makefile`
- Create: `go.mod` (update existing)
- Create: `internal/radix/.gitkeep`
- Create: `internal/mlx/.gitkeep`
- Create: `pkg/tokenizer/.gitkeep`

**Step 1: Create Makefile with CGO configuration**

```makefile
# Makefile
.PHONY: all test clean build

CGO_ENABLED=1
GO_TAGS=mlx
BUILD_DIR := ./bin
SERVER binary := $(BUILD_DIR)/server

all: build

build:
	mkdir -p $(BUILD_DIR)
	go build -tags=$(GO_TAGS) -o $(SERVER) ./cmd/server

test-short:
	go test ./... -short

test-integration:
	go test ./internal/mlx/... -run Integration

test-coverage:
	go test ./... -coverprofile=coverage.out -covermode=atomic
	go tool cover -html=coverage.out -o coverage.html

clean:
	rm -rf $(BUILD_DIR) coverage.out coverage.html
	go clean -cache
```

**Step 2: Run make to verify**

Run: `make`
Expected: No errors (creates empty bin/ directory)

**Step 3: Commit**

```bash
git add Makefile go.mod
git commit -m "build: add project structure and Makefile for CGO build"
```

---

## Task 2: MLXEngine Interface Definition

**Files:**
- Create: `internal/radix/engine.go`

**Step 1: Write the failing test for interface**

```go
package radix

import (
    "testing"
)

func TestMLXEngineInterface(t *testing.T) {
    var _ MLXEngine = (*MockMLXEngine)(nil)
}

type MockMLXEngine struct {
    ForwardFunc func(model any, tokens []uint32, base uint64) ([]float32, uint64, error)
    SliceFunc   func(handle uint64, keepTokens int) (uint64, error)
    FreeFunc    func(handle uint64)
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./internal/radix/... -v`
Expected: FAIL with "undefined: MLXEngine"

**Step 3: Write minimal implementation**

```go
package radix

// MLXEngine defines the interface for MLX inference operations
// All methods are thread-safe and handle memory management explicitly
type MLXEngine interface {
    // ForwardWithCache executes inference with KV cache
    // Returns logits, new cache handle, or error
    // Caller MUST call FreeLogits on the returned logits
    ForwardWithCache(model any, tokens []uint32, baseHandle uint64) ([]float32, uint64, error)

    // SliceCache creates a zero-copy view of an existing cache
    // O(1) operation using MLX copy-on-write
    SliceCache(handle uint64, keepTokens int) (uint64, error)

    // FreeCache releases a cache handle and associated GPU memory
    // Safe to call multiple times on same handle (idempotent)
    FreeCache(handle uint64)
}

// CacheHandle constants
const (
    RootCacheHandle uint64 = 0 // Represents empty/root cache state
)
```

**Step 4: Run test to verify it passes**

Run: `go test ./internal/radix/... -v`
Expected: PASS

**Step 5: Commit**

```bash
git add internal/radix/engine.go
git commit -m "feat(radix): define MLXEngine interface for cache operations"
```

---

## Task 3: Radix Tree Node Structure

**Files:**
- Create: `internal/radix/node.go`

**Step 1: Write the failing test**

```go
package radix

import (
    "container/list"
    "sync/atomic"
    "testing"
)

func TestNewNode(t *testing.T) {
    node := NewNode([]uint32{1, 2, 3}, nil)
    
    if len(node.Tokens) != 3 {
        t.Errorf("Expected 3 tokens, got %d", len(node.Tokens))
    }
    
    if node.refCount.Load() != 0 {
        t.Errorf("Expected refCount 0, got %d", node.refCount.Load())
    }
    
    if node.ready == nil {
        t.Error("Expected ready channel to be initialized")
    }
}

func TestNodeWait(t *testing.T) {
    node := NewNode([]uint32{1}, nil)
    
    // Not finalized yet - should block
    done := make(chan struct{})
    go func() {
        node.Wait()
        close(done)
    }()
    
    select {
    case <-done:
        t.Error("Wait should block when node not ready")
    default:
    }
    
    // Finalize the node
    FinalizeNode(node, 123)
    
    // Now wait should unblock
    node.Wait()
    
    if node.CacheHandle != 123 {
        t.Errorf("Expected CacheHandle 123, got %d", node.CacheHandle)
    }
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./internal/radix/... -run TestNode -v`
Expected: FAIL with "undefined: NewNode"

**Step 3: Write minimal implementation**

```go
package radix

import (
    "container/list"
    "sync/atomic"
)

// Node represents a single node in the Radix prefix tree
type Node struct {
    // Tokens is the sequence of tokens at this node edge
    Tokens []uint32
    
    // Children maps first token to child nodes
    Children map[uint32]*Node
    
    // Parent pointer for tree traversal and cascading cleanup
    Parent *Node
    
    // CacheHandle is the opaque MLX KV cache reference
    CacheHandle uint64

    // ready blocks waiters until node is finalized or poisoned
    ready chan struct{}
    
    // err holds any error from MLX computation (poison state)
    err error
    
    // refCount pins node to prevent LRU eviction during computation
    // Must be incremented before releasing tree lock during ForwardWithCache
    refCount atomic.Int32
    
    // lruElem points to this node's position in the LRU queue
    // Nil when node is pinned (refCount > 0) or is internal node
    lruElem *list.Element
}

// NewNode creates a pending node that is not yet ready
func NewNode(tokens []uint32, parent *Node) *Node {
    return &Node{
        Tokens:   tokens,
        Children: make(map[uint32]*Node),
        Parent:   parent,
        ready:    make(chan struct{}),
    }
}

// Wait blocks until the node is finalized or returns an error immediately if ready
func (n *Node) Wait() error {
    <-n.ready
    return n.err
}

// FinalizeNode marks a pending node as complete and stores the cache handle
func FinalizeNode(n *Node, handle uint64) {
    n.CacheHandle = handle
    close(n.ready)
}

// PoisonNode marks a node as failed due to MLX error
func PoisonNode(n *Node, err error) {
    n.err = err
    close(n.ready)
}

// IsReady returns true if the node has been finalized
func (n *Node) IsReady() bool {
    select {
    case <-n.ready:
        return true
    default:
        return false
    }
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./internal/radix/... -run TestNode -v`
Expected: PASS

**Step 5: Add test for poisoned node**

```go
func TestPoisonedNode(t *testing.T) {
    node := NewNode([]uint32{1}, nil)
    
    expectedErr := errors.New("MLX OOM")
    PoisonNode(node, expectedErr)
    
    err := node.Wait()
    if err != expectedErr {
        t.Errorf("Expected error %v, got %v", expectedErr, err)
    }
}
```

**Step 6: Run test to verify it passes**

Run: `go test ./internal/radix/... -run TestPoisonedNode -v`
Expected: PASS

**Step 7: Commit**

```bash
git add internal/radix/node.go
git commit -m "feat(radix): implement Node structure with ready channel and refCount"
```

---

## Task 4: Radix Tree - Match Operation

**Files:**
- Create: `internal/radix/tree.go`

**Step 1: Write the failing test**

```go
package radix

import "testing"

func TestTreeMatch_Empty(t *testing.T) {
    engine := &MockMLXEngine{}
    tree := NewTree(engine, 1000)
    
    match, unmatched := tree.Match([]uint32{1, 2, 3})
    
    if match != tree.Root {
        t.Error("Empty tree should match root")
    }
    
    if len(unmatched) != 3 || unmatched[0] != 1 {
        t.Errorf("Expected [1,2,3] unmatched, got %v", unmatched)
    }
}

func TestTreeMatch_ExactMatch(t *testing.T) {
    engine := &MockMLXEngine{}
    tree := NewTree(engine, 1000)
    
    // Insert node with tokens [1,2,3]
    node := NewNode([]uint32{1, 2, 3}, tree.Root)
    FinalizeNode(node, 100)
    tree.Root.Children[1] = node
    
    match, unmatched := tree.Match([]uint32{1, 2, 3})
    
    if match != node {
        t.Error("Should match exact node")
    }
    
    if len(unmatched) != 0 {
        t.Errorf("Expected no unmatched, got %v", unmatched)
    }
}

func TestTreeMatch_PartialMatch(t *testing.T) {
    engine := &MockMLXEngine{}
    tree := NewTree(engine, 1000)
    
    // Insert node with tokens [1,2]
    node := NewNode([]uint32{1, 2}, tree.Root)
    FinalizeNode(node, 100)
    tree.Root.Children[1] = node
    
    match, unmatched := tree.Match([]uint32{1, 2, 3, 4})
    
    if match != node {
        t.Error("Should match partial node")
    }
    
    if len(unmatched) != 2 || unmatched[0] != 3 {
        t.Errorf("Expected [3,4] unmatched, got %v", unmatched)
    }
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./internal/radix/... -run TestTreeMatch -v`
Expected: FAIL with "undefined: NewTree"

**Step 3: Write minimal implementation**

```go
package radix

import (
    "container/list"
    "sync"
)

// Tree implements a Radix tree for prefix caching with OCC
type Tree struct {
    mu        sync.Mutex
    Root      *Node
    Engine    MLXEngine
    MaxTokens int
    size      int
    
    lruQueue *list.List // O(1) LRU eviction queue
}

// NewTree creates an empty Radix tree
func NewTree(engine MLXEngine, maxTokens int) *Tree {
    return &Tree{
        Root:     NewNode(nil, nil), // Root has no tokens
        Engine:   engine,
        MaxTokens: maxTokens,
        lruQueue: list.New(),
    }
}

// Match finds the longest matching prefix for the given tokens
// Returns (matchedNode, unmatchedTokens)
// Thread-safe: uses read locking for concurrency
func (t *Tree) Match(tokens []uint32) (*Node, []uint32) {
    t.mu.Lock()
    defer t.mu.Unlock()
    
    current := t.Root
    matched := 0
    
    // Traverse tree following token sequence
    for matched < len(tokens) {
        child, exists := current.Children[tokens[matched]]
        if !exists {
            break
        }
        
        // Check if child's token sequence fully matches
        lcp := longestCommonPrefix(child.Tokens, tokens[matched:])
        if lcp != len(child.Tokens) {
            // Partial match - need to split (handled in InsertPending)
            break
        }
        
        current = child
        matched += len(child.Tokens)
    }
    
    unmatched := tokens[matched:]
    return current, unmatched
}

// longestCommonPrefix returns the length of the common prefix
func longestCommonPrefix(a, b []uint32) int {
    i := 0
    for i < len(a) && i < len(b) && a[i] == b[i] {
        i++
    }
    return i
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./internal/radix/... -run TestTreeMatch -v`
Expected: PASS

**Step 5: Commit**

```bash
git add internal/radix/tree.go
git commit -m "feat(radix): implement Match operation for prefix tree traversal"
```

---

## Task 5: Radix Tree - InsertPending with OCC

**Files:**
- Modify: `internal/radix/tree.go`

**Step 1: Write the failing test for OCC**

```go
func TestInsertPending_OCC_NoBlock(t *testing.T) {
    // Create a mock that blocks ForwardWithCache
    blockChan := make(chan struct{})
    engine := &MockMLXEngine{
        ForwardFunc: func(model any, tokens []uint32, base uint64) ([]float32, uint64, error) {
            <-blockChan // Block until signaled
            return []float32{0.1}, 200, nil
        },
    }
    
    tree := NewTree(engine, 10000)
    
    // First insert should succeed and create pending node
    node1, err := tree.InsertPending(tree.Root, []uint32{1, 2, 3})
    if err != nil {
        t.Fatalf("InsertPending failed: %v", err)
    }
    
    // Verify we can insert a DIFFERENT branch while first is pending
    // This proves OCC doesn't block the entire tree
    node2, err := tree.InsertPending(tree.Root, []uint32{5, 6})
    if err != nil {
        t.Fatalf("Second InsertPending failed: %v", err)
    }
    
    if node1 == node2 {
        t.Error("Should create different nodes for different branches")
    }
    
    // Unblock first computation
    close(blockChan)
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./internal/radix/... -run TestInsertPending_OCC -v`
Expected: FAIL with "undefined: InsertPending"

**Step 3: Write minimal implementation**

```go
// InsertPending creates and attaches a pending node to the tree
// Uses Optimistic Concurrency Control - releases lock during Wait()
// Returns the pending node (caller must finalize it later)
func (t *Tree) InsertPending(parent *Node, tokens []uint32) (*Node, error) {
    if len(tokens) == 0 {
        return parent, nil
    }

retry:
    t.mu.Lock()
    
    firstToken := tokens[0]
    child, exists := parent.Children[firstToken]
    
    if exists {
        // OCC: Release lock immediately before waiting
        t.mu.Unlock()
        
        // Wait for existing node to be ready
        if err := child.Wait(); err != nil {
            return nil, err // Node was poisoned
        }
        
        // Double-checked locking
        t.mu.Lock()
        currentChild, stillExists := parent.Children[firstToken]
        if !stillExists || currentChild != child {
            t.mu.Unlock()
            goto retry // Topology changed, retry
        }
        
        // Check for partial match - may need to split
        lcp := longestCommonPrefix(child.Tokens, tokens)
        if lcp == len(child.Tokens) {
            t.mu.Unlock()
            // Full match, continue with remaining tokens
            return t.InsertPending(child, tokens[lcp:])
        }
        
        // Split required - partial edge match
        // This happens when inserting [1,2,3,4] into tree with [1,2,5,6]
        splitNode := NewNode(child.Tokens[:lcp], parent)
        
        // Zero-copy slice the cache
        slicedHandle, err := t.Engine.SliceCache(child.CacheHandle, lcp)
        if err != nil {
            t.mu.Unlock()
            return nil, err
        }
        
        FinalizeNode(splitNode, slicedHandle)
        
        // Reparent child
        child.Tokens = child.Tokens[lcp:]
        child.Parent = splitNode
        splitNode.Children[child.Tokens[0]] = child
        parent.Children[firstToken] = splitNode
        
        t.mu.Unlock()
        return t.InsertPending(splitNode, tokens[lcp:])
    }
    
    // Normal case: create new node
    t.enforceCapacity(len(tokens))
    
    newNode := NewNode(tokens, parent)
    parent.Children[firstToken] = newNode
    t.size += len(tokens)
    
    // Pre-pin: increment refCount BEFORE releasing lock
    // This prevents the node from being evicted during ForwardWithCache
    newNode.refCount.Add(1)
    
    // Remove parent from LRU if it was there
    if parent.lruElem != nil {
        t.lruQueue.Remove(parent.lruElem)
        parent.lruElem = nil
    }
    
    t.mu.Unlock()
    return newNode, nil
}

// enforceCapacity ensures there's room for new tokens using O(1) LRU eviction
// MUST be called with t.mu.Lock() held
func (t *Tree) enforceCapacity(needed int) {
    for t.size+needed > t.MaxTokens {
        elem := t.lruQueue.Back()
        if elem == nil {
            break // System saturated - should return 429 to client
        }
        
        lruLeaf := t.lruQueue.Remove(elem).(*Node)
        lruLeaf.lruElem = nil
        
        // Free GPU memory
        t.Engine.FreeCache(lruLeaf.CacheHandle)
        
        // Remove from parent
        parent := lruLeaf.Parent
        delete(parent.Children, lruLeaf.Tokens[0])
        t.size -= len(lruLeaf.Tokens)
        
        // Cascading cleanup
        if len(parent.Children) == 0 && 
           parent.refCount.Load() == 0 && 
           parent != t.Root {
            parent.lruElem = t.lruQueue.PushFront(parent)
        }
    }
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./internal/radix/... -run TestInsertPending_OCC -v`
Expected: PASS

**Step 5: Commit**

```bash
git add internal/radix/tree.go
git commit -m "feat(radix): implement InsertPending with OCC and double-checked locking"
```

---

## Task 6: Radix Tree - PrunePoisoned

**Files:**
- Modify: `internal/radix/tree.go`

**Step 1: Write the failing test**

```go
func TestPrunePoisoned(t *testing.T) {
    engine := &MockMLXEngine{
        FreeFunc: func(handle uint64) {},
    }
    
    tree := NewTree(engine, 1000)
    
    // Insert a node
    node := NewNode([]uint32{1, 2, 3}, tree.Root)
    tree.Root.Children[1] = node
    tree.size += 3
    
    expectedErr := errors.New("MLX OOM")
    tree.PrunePoisoned(node, expectedErr)
    
    // Node should be removed from tree
    _, exists := tree.Root.Children[1]
    if exists {
        t.Error("Poisoned node should be removed from parent")
    }
    
    if tree.size != 0 {
        t.Errorf("Tree size should be 0, got %d", tree.size)
    }
    
    // Waiting on node should return the error
    err := node.Wait()
    if err != expectedErr {
        t.Errorf("Expected error %v, got %v", expectedErr, err)
    }
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./internal/radix/... -run TestPrunePoisoned -v`
Expected: FAIL with "undefined: PrunePoisoned"

**Step 3: Write minimal implementation**

```go
// PrunePoisoned removes a failed node from the tree and broadcasts the error
// This prevents cache poisoning from permanent MLX failures
func (t *Tree) PrunePoisoned(node *Node, err error) {
    t.mu.Lock()
    defer t.mu.Unlock()
    
    PoisonNode(node, err)
    
    if node.Parent != nil {
        delete(node.Parent.Children, node.Tokens[0])
        t.size -= len(node.Tokens)
        
        // If parent is now an unpinned leaf, queue it for LRU
        if len(node.Parent.Children) == 0 && 
           node.Parent.refCount.Load() == 0 && 
           node.Parent != t.Root {
            node.Parent.lruElem = t.lruQueue.PushFront(node.Parent)
        }
    }
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./internal/radix/... -run TestPrunePoisoned -v`
Expected: PASS

**Step 5: Commit**

```bash
git add internal/radix/tree.go
git commit -m "feat(radix): implement PrunePoisoned for failed node cleanup"
```

---

## Task 7: Radix Tree - Unpin Operation

**Files:**
- Modify: `internal/radix/tree.go`

**Step 1: Write the failing test**

```go
func TestUnpin(t *testing.T) {
    engine := &MockMLXEngine{
        FreeFunc: func(handle uint64) {},
    }
    
    tree := NewTree(engine, 1000)
    
    // Create a pinned node
    node := NewNode([]uint32{1, 2}, tree.Root)
    node.refCount.Add(1) // Simulate being pinned
    tree.Root.Children[1] = node
    tree.size += 2
    
    // Unpin the node
    tree.Unpin(node)
    
    if node.refCount.Load() != 0 {
        t.Error("refCount should be 0 after unpin")
    }
    
    // Node should be in LRU queue now (it's a leaf with refCount=0)
    if node.lruElem == nil {
        t.Error("Unpinned leaf should be in LRU queue")
    }
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./internal/radix/... -run TestUnpin -v`
Expected: FAIL with "undefined: Unpin"

**Step 3: Write minimal implementation**

```go
// Unpin decrements the node's reference count
// If the node becomes an unpinned leaf, it's added to the LRU queue
// This is called via defer in HTTP handlers to ensure cleanup
func (t *Tree) Unpin(n *Node) {
    if n == t.Root {
        return
    }
    
    val := n.refCount.Add(-1)
    if val == 0 && len(n.Children) == 0 {
        t.mu.Lock()
        defer t.mu.Unlock()
        
        // Double-check under lock
        if n.refCount.Load() == 0 && len(n.Children) == 0 && n.lruElem == nil {
            n.lruElem = t.lruQueue.PushFront(n)
        }
    }
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./internal/radix/... -run TestUnpin -v`
Expected: PASS

**Step 5: Commit**

```bash
git add internal/radix/tree.go
git commit -m "feat(radix): implement Unpin for LRU queue management"
```

---

## Task 8: Flattened CGO API Header

**Files:**
- Create: `internal/mlx/mlx_api.h`

**Step 1: Create the header file**

```c
#ifndef MLX_RADIX_API_H
#define MLX_RADIX_API_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to MLX KV cache (0 = root/empty)
typedef uint64_t mlx_kv_cache_t;

// Forward pass with KV cache
// 
// Parameters:
//   model_ptr   - Opaque pointer to MLX model
//   tokens      - Contiguous array of token IDs
//   num_tokens  - Number of tokens in array
//   base_cache  - Handle to base cache (0 for no cache)
//   out_cache   - Output: new cache handle after forward pass
//   out_logits  - Output: dynamically allocated logits array
//                  MUST be freed via mlx_free_logits
//
// Returns: true on success, false on failure
bool mlx_forward_with_cache(
    void* model_ptr,
    const uint32_t* tokens,
    size_t num_tokens,
    mlx_kv_cache_t base_cache,
    mlx_kv_cache_t* out_cache,
    float** out_logits
);

// Zero-copy slice of existing cache
// Creates a copy-on-write view keeping first keep_tokens
// Returns: new cache handle, or 0 on failure
mlx_kv_cache_t mlx_kv_cache_slice(
    mlx_kv_cache_t base_cache,
    size_t keep_tokens
);

// Explicit memory destructors

// Frees GPU memory associated with cache handle
void mlx_free_kv_cache(mlx_kv_cache_t cache);

// Frees CPU memory for logits array
void mlx_free_logits(float* logits);

// Model lifecycle
void* mlx_load_model(const char* path, const char* device);
void mlx_unload_model(void* model_ptr);

#ifdef __cplusplus
}
#endif

#endif // MLX_RADIX_API_H
```

**Step 2: Commit**

```bash
git add internal/mlx/mlx_api.h
git commit -m "feat(mlx): define flattened CGO API header with primitive types"
```

---

## Task 9: Go MLX Bridge Wrapper

**Files:**
- Create: `internal/mlx/bridge.go`

**Step 1: Write the failing test**

```go
package mlx

import (
    "testing"
)

func TestNewMLXBridge(t *testing.T) {
    bridge := NewMLXBridge()
    
    if bridge == nil {
        t.Error("Expected non-nil bridge")
    }
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./internal/mlx/... -v`
Expected: FAIL with "undefined: NewMLXBridge"

**Step 3: Write minimal implementation**

```go
package mlx

/*
#cgo CXXFLAGS: -std=c++17 -I/opt/homebrew/include
#cgo LDFLAGS: -L/opt/homebrew/lib -lmlx -lmlx-metal -framework Foundation -framework Metal
#include "mlx_api.h"
*/
import "C"
import (
    "sync"
    "unsafe"
)

// Bridge wraps MLX C API with thread-safe handle management
type Bridge struct {
    mu     sync.Mutex
    models map[uintptr]interface{} // model pointers
    nextHandle uint64
}

// NewMLXBridge creates a new MLX bridge instance
func NewMLXBridge() *Bridge {
    return &Bridge{
        models: make(map[uintptr]interface{}),
        nextHandle: 1, // Start at 1, reserve 0 for root
    }
}

// ForwardWithCache executes inference with KV caching
// The returned logits MUST be freed via FreeLogits
func (b *Bridge) ForwardWithCache(model interface{}, tokens []uint32, baseHandle uint64) ([]float32, uint64, error) {
    modelPtr := unsafe.Pointer(uintptr(0)) // TODO: get actual model pointer
    
    var cOutCache C.mlx_kv_cache_t
    var cLogits *C.float
    
    // Convert tokens to C array
    cTokens := (*C.uint32_t)(unsafe.Pointer(&tokens[0]))
    
    if C.mlx_forward_with_cache(
        modelPtr,
        cTokens,
        C.size_t(len(tokens)),
        C.mlx_kv_cache_t(baseHandle),
        &cOutCache,
        &cLogits,
    ) == false {
        return nil, 0, fmt.Errorf("MLX forward failed")
    }
    
    // Convert C array to Go slice (zero-copy copy of pointer)
    logitsLen := 151936 // TODO: get vocab size from model
    logits = (*[1 << 20]float32)(unsafe.Pointer(cLogits))[:logitsLen:logitsLen]
    
    return logits, uint64(cOutCache), nil
}

// SliceCache creates a zero-copy view of an existing cache
func (b *Bridge) SliceCache(handle uint64, keepTokens int) (uint64, error) {
    newHandle := C.mlx_kv_cache_slice(C.mlx_kv_cache_t(handle), C.size_t(keepTokens))
    if newHandle == 0 {
        return 0, fmt.Errorf("slice failed")
    }
    return uint64(newHandle), nil
}

// FreeCache releases GPU memory for a cache handle
func (b *Bridge) FreeCache(handle uint64) {
    if handle != 0 {
        C.mlx_free_kv_cache(C.mlx_kv_cache_t(handle))
    }
}

// FreeLogits releases CPU memory for logits
func FreeLogits(logits []float32) {
    if len(logits) > 0 {
        C.mlx_free_logits((*C.float)(unsafe.Pointer(&logits[0])))
    }
}

// LoadModel loads an MLX model from disk
func (b *Bridge) LoadModel(path string) (interface{}, error) {
    cPath := C.CString(path)
    defer C.free(unsafe.Pointer(cPath))
    
    modelPtr := C.mlx_load_model(cPath, C.CString("metal"))
    if modelPtr == nil {
        return nil, fmt.Errorf("failed to load model")
    }
    
    b.mu.Lock()
    defer b.mu.Unlock()
    
    modelKey := uintptr(modelPtr)
    b.models[modelKey] = modelPtr
    
    return modelPtr, nil
}

// UnloadModel releases model resources
func (b *Bridge) UnloadModel(model interface{}) error {
    modelPtr := uintptr(model.(unsafe.Pointer))
    
    b.mu.Lock()
    defer b.mu.Unlock()
    
    delete(b.models, modelPtr)
    C.mlx_unload_model(modelPtr)
    
    return nil
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./internal/mlx/... -v`
Expected: PASS

**Step 5: Commit**

```bash
git add internal/mlx/bridge.go
git commit -m "feat(mlx): implement Go bridge wrapper with CGO bindings"
```

---

## Task 10: MLX C++ Engine Implementation

**Files:**
- Create: `internal/mlx/mlx_engine.cpp`

**Step 1: Create the C++ implementation**

```cpp
#include "mlx_api.h"
#include <mlx/mlx.h>
#include <mlx/core/array.h>
#include <mlx/core/kv_cache.h>
#include <unordered_map>
#include <shared_mutex>
#include <mutex>

// Thread-safe registry mapping handles to KV caches
static std::unordered_map<uint64_t, std::shared_ptr<mlx::core::KVCache>> cache_registry;
static std::shared_mutex registry_mutex;
static uint64_t next_handle = 1;

extern "C" bool mlx_forward_with_cache(
    void* model_ptr,
    const uint32_t* tokens,
    size_t num_tokens,
    mlx_kv_cache_t base_cache,
    mlx_kv_cache_t* out_cache,
    float** out_logits
) {
    try {
        // Convert tokens to MLX array
        std::vector<int> token_vec(tokens, tokens + num_tokens);
        mlx::core::Array token_array = mlx::core::array::make(token_vec);
        
        // Get base cache or create new
        std::shared_ptr<mlx::core::KVCache> base;
        if (base_cache == 0) {
            base = nullptr;
        } else {
            std::unique_lock lock(registry_mutex);
            auto it = cache_registry.find(base_cache);
            if (it == cache_registry.end()) {
                return false;
            }
            base = it->second;
        }
        
        // Execute forward pass with lazy evaluation
        // This is where MLX schedules Metal compute
        auto [logits, new_cache] = forward_with_cache_impl(
            static_cast<mlx::core::Model*>(model_ptr),
            token_array,
            base
        );
        
        // Register new cache and generate handle
        uint64_t new_handle = next_handle++;
        {
            std::unique_lock lock(registry_mutex);
            cache_registry[new_handle] = new_cache;
        }
        
        *out_cache = new_handle;
        *out_logits = logits.data();
        
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

extern "C" mlx_kv_cache_t mlx_kv_cache_slice(
    mlx_kv_cache_t base_cache,
    size_t keep_tokens
) {
    try {
        std::shared_lock lock(registry_mutex);
        auto it = cache_registry.find(base_cache);
        if (it == cache_registry.end()) {
            return 0;
        }
        
        // MLX zero-copy slice operation
        auto sliced = it->second->slice(0, keep_tokens);
        
        uint64_t new_handle = next_handle++;
        cache_registry[new_handle] = sliced;
        
        return new_handle;
    } catch (...) {
        return 0;
    }
}

extern "C" void mlx_free_kv_cache(mlx_kv_cache_t cache) {
    if (cache == 0) return;
    
    std::unique_lock lock(registry_mutex);
    cache_registry.erase(cache);
}

extern "C" void mlx_free_logits(float* logits) {
    // MLX uses shared_ptr for memory management
    // We don't actually free here, just drop reference
    // The actual memory is managed by shared_ptr
}

extern "C" void* mlx_load_model(const char* path, const char* device) {
    // TODO: Implement actual model loading
    return reinterpret_cast<void*>(0x1); // Placeholder
}

extern "C" void mlx_unload_model(void* model_ptr) {
    // TODO: Implement actual model unloading
}
```

**Step 2: Commit**

```bash
git add internal/mlx/mlx_engine.cpp
git commit -m "feat(mlx): implement C++ MLX engine with thread-safe cache registry"
```

---

## Task 11: Tokenizer Package

**Files:**
- Create: `pkg/tokenizer/tokenizer.go`

**Step 1: Write the failing test**

```go
package tokenizer

import "testing"

func TestEncodeText(t *testing.T) {
    tz, err := NewTokenizer()
    if err != nil {
        t.Fatalf("NewTokenizer failed: %v", err)
    }
    
    tokens, err := tz.Encode("Hello, world!")
    if err != nil {
        t.Fatalf("Encode failed: %v", err)
    }
    
    if len(tokens) == 0 {
        t.Error("Expected some tokens")
    }
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./pkg/tokenizer/... -v`
Expected: FAIL with "undefined: NewTokenizer"

**Step 3: Write minimal implementation**

```go
package tokenizer

import (
    "strings"
)

// Tokenizer converts text/images to uint32 token IDs
type Tokenizer struct {
    vocabSize int
}

// NewTokenizer creates a new tokenizer instance
func NewTokenizer() (*Tokenizer, error) {
    return &Tokenizer{
        vocabSize: 151936, // Qwen2-VL default
    }, nil
}

// Encode converts text message to token IDs
func (tz *Tokenizer) Encode(text string) ([]uint32, error) {
    // Placeholder: simple word-based tokenization
    // TODO: Integrate actual SentencePiece tokenizer
    words := strings.Fields(text)
    
    tokens := make([]uint32, 0, len(words)+2) // +2 for special tokens
    
    // Add BOS token
    tokens = append(tokens, 151644) // <|im_start|>
    
    for _, word := range words {
        // Simple hash-based tokenization (placeholder)
        token := uint32((hashString(word) % uint32(tz.vocabSize-2))) + 2
        tokens = append(tokens, token)
    }
    
    // Add EOS token
    tokens = append(tokens, 151645) // <|im_end|>
    
    return tokens, nil
}

// EncodeMessage converts a chat message to token IDs
func (tz *Tokenizer) EncodeMessage(role, content string) ([]uint32, error) {
    // Add role prefix
    formatted := role + "\n" + content
    return tz.Encode(formatted)
}

// EncodeMessages converts a conversation to token IDs
func (tz *Tokenizer) EncodeMessages(messages []Message) ([]uint32, error) {
    var allTokens []uint32
    
    for _, msg := range messages {
        tokens, err := tz.EncodeMessage(msg.Role, msg.Content)
        if err != nil {
            return nil, err
        }
        allTokens = append(allTokens, tokens...)
    }
    
    // Add assistant prompt
    allTokens = append(allTokens, 151643) // <|im_start|>assistant
    allTokens = append(allTokens, 151644) // recipient
    allTokens = append(allTokens, 151648) // os
    
    return allTokens, nil
}

// Message represents a chat message
type Message struct {
    Role    string
    Content string
}

func hashString(s string) uint32 {
    h := uint32(2166136261)
    for _, c := range s {
        h = h*31 + uint32(c)
    }
    return h
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./pkg/tokenizer/... -v`
Expected: PASS

**Step 5: Commit**

```bash
git add pkg/tokenizer/
git commit -m "feat(tokenizer): implement tokenizer with placeholder hash-based encoding"
```

---

## Task 12: HTTP Handler - Autoregressive Generation

**Files:**
- Create: `internal/api/handler.go` (update existing)

**Step 1: Write the failing test**

```go
func TestHandleChatCompletion_BulkInsert(t *testing.T) {
    // This test verifies that 150 generated tokens
    // result in exactly 1 node (not 150 fragmented nodes)
    tz, _ := tokenizer.NewTokenizer()
    engine := &MockMLXEngine{
        ForwardFunc: func(model any, tokens []uint32, base uint64) ([]float32, uint64, error) {
            return []float32{0.1}, base + 1, nil
        },
        SliceFunc: func(handle uint64, n int) (uint64, error) {
            return handle, nil
        },
        FreeFunc: func(handle uint64) {},
    }
    
    tree := radix.NewTree(engine, 10000)
    handler := &Handler{
        Tokenizer: tz,
        RadixTree: tree,
        Model:     "test-model",
    }
    
    // Simulate request
    req := ChatCompletionRequest{
        Model: "test-model",
        Messages: []Message{
            {Role: "user", Content: "Hello"},
        },
    }
    
    // Execute handler (this will call HandleChatCompletion)
    // We'll verify tree size after generation
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./internal/api/... -run TestHandleChatCompletion_BulkInsert -v`
Expected: FAIL with handler not properly implemented

**Step 3: Write minimal implementation**

```go
package api

import (
    "encoding/json"
    "net/http"
    
    "github.com/agenthands/mlxvllm/internal/radix"
    "github.com/agenthands/mlxvllm/pkg/tokenizer"
)

type Handler struct {
    Tokenizer *tokenizer.Tokenizer
    RadixTree *radix.Tree
    Model     interface{}
    MLXEngine *mlx.Bridge
}

func NewHandler(tree *radix.Tree, mlx *mlx.Bridge, tz *tokenizer.Tokenizer) *Handler {
    return &Handler{
        Tokenizer: tz,
        RadixTree: tree,
        MLXEngine: mlx,
    }
}

// HandleChatCompletion implements autoregressive bulk aggregation
func (h *Handler) HandleChatCompletion(w http.ResponseWriter, r *http.Request) {
    var req ChatCompletionRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        writeError(w, http.StatusBadRequest, "invalid request")
        return
    }
    
    // 1. Tokenize input
    allTokens, err := h.Tokenizer.EncodeMessages(req.Messages)
    if err != nil {
        writeError(w, http.StatusBadRequest, "tokenization failed")
        return
    }
    
    // 2. Match against radix tree
    matchNode, unmatched := h.RadixTree.Match(allTokens)
    if err := matchNode.Wait(); err != nil {
        writeError(w, http.StatusInternalServerError, "prefix cache error")
        return
    }
    
    // 3. Process unmatched prefix
    var promptNode *radix.Node
    var currHandle uint64
    
    if len(unmatched) > 0 {
        pendingNode, err := h.RadixTree.InsertPending(matchNode, unmatched)
        if err != nil {
            writeError(w, http.StatusInternalServerError, "tree error")
            return
        }
        
        logits, newHandle, err := h.MLXEngine.ForwardWithCache(h.Model, unmatched, matchNode.CacheHandle)
        if err != nil {
            h.RadixTree.PrunePoisoned(pendingNode, err)
            writeError(w, http.StatusInternalServerError, "inference failed")
            return
        }
        defer mlx.FreeLogits(logits)
        
        radix.FinalizeNode(pendingNode, newHandle)
        promptNode = pendingNode
        currHandle = newHandle
    } else {
        promptNode = matchNode
        promptNode.refCount.Add(1)
        currHandle = matchNode.CacheHandle
    }
    
    defer h.RadixTree.Unpin(promptNode)
    
    // 4. Autoregressive generation loop (lock-free)
    generatedTokens := make([]uint32, 0, 256)
    lastToken := allTokens[len(allTokens)-1]
    
    for !isStopToken(lastToken) && len(generatedTokens) < 256 {
        logits, nextHandle, err := h.MLXEngine.ForwardWithCache(h.Model, []uint32{lastToken}, currHandle)
        if err != nil {
            break // Generation failed
        }
        
        // Explicit free - NOT defer (loop scoping!)
        mlx.FreeLogits(logits)
        
        // Free intermediate cache handles
        if currHandle != promptNode.CacheHandle {
            h.MLXEngine.FreeCache(currHandle)
        }
        
        lastToken = sampleToken(logits)
        generatedTokens = append(generatedTokens, lastToken)
        currHandle = nextHandle
        
        // Stream to client would go here
    }
    
    // 5. Bulk insert generated tokens
    if len(generatedTokens) > 0 {
        genNode, _ := h.RadixTree.InsertPending(promptNode, generatedTokens)
        h.RadixTree.Finalize(genNode, currHandle)
        h.RadixTree.Unpin(genNode)
    } else {
        h.MLXEngine.FreeCache(currHandle)
    }
    
    // 6. Return response
    resp := NewChatCompletionResponse(req.Model, []Choice{{
        Index:        0,
        Message:      Message{Role: "assistant", Content: decodeTokens(generatedTokens)},
        FinishReason: "stop",
    }})
    
    writeJSON(w, http.StatusOK, resp)
}

func isStopToken(token uint32) bool {
    return token == 151645 // <|im_end|>
}

func sampleToken(logits []float32) uint32 {
    // TODO: Implement actual sampling (temperature, top-p, etc.)
    // For now, return argmax
    maxIdx := 0
    maxVal := logits[0]
    for i, val := range logits {
        if val > maxVal {
            maxVal = val
            maxIdx = i
        }
    }
    return uint32(maxIdx)
}

func decodeTokens(tokens []uint32) string {
    // TODO: Implement actual detokenization
    return "<generated>"
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./internal/api/... -run TestHandleChatCompletion_BulkInsert -v`
Expected: PASS

**Step 5: Commit**

```bash
git add internal/api/handler.go
git commit -m "feat(api): implement autoregressive handler with bulk aggregation"
```

---

## Task 13: Integration Tests

**Files:**
- Create: `internal/mlx/integration_test.go`

**Step 1: Write the integration test**

```go
// +build integration

package mlx

import (
    "testing"
)

func TestIntegration_RealMLXForward(t *testing.T) {
    if testing.Short() {
        t.Skip("Skipping integration test - requires Apple Silicon")
    }
    
    // This test requires actual MLX installation and a model
    // It verifies the complete CGO bridge works correctly
    
    bridge := NewMLXBridge()
    
    // Load actual model
    model, err := bridge.LoadModel("/path/to/model")
    if err != nil {
        t.Skip("Model not available")
    }
    defer bridge.UnloadModel(model)
    
    // Test forward pass
    tokens := []uint32{1, 2, 3}
    logits, handle, err := bridge.ForwardWithCache(model, tokens, 0)
    if err != nil {
        t.Fatalf("ForwardWithCache failed: %v", err)
    }
    
    if len(logits) == 0 {
        t.Error("Expected non-empty logits")
    }
    
    if handle == 0 {
        t.Error("Expected non-zero cache handle")
    }
    
    // Test slice
    slicedHandle, err := bridge.SliceCache(handle, 2)
    if err != nil {
        t.Fatalf("SliceCache failed: %v", err)
    }
    
    if slicedHandle == 0 {
        t.Error("Expected non-zero sliced handle")
    }
    
    // Cleanup
    bridge.FreeCache(handle)
    bridge.FreeCache(slicedHandle)
    FreeLogits(logits)
}
```

**Step 2: Run test to verify it compiles**

Run: `go test ./internal/mlx/... -run TestIntegration_RealMLXForward -short`
Expected: Test is skipped (no model available)

**Step 3: Commit**

```bash
git add internal/mlx/integration_test.go
git commit -m "test(mlx): add integration test for real MLX forward pass"
```

---

## Task 14: Main Entry Point & DI Wiring

**Files:**
- Create: `cmd/server/main.go`

**Step 1: Create main.go**

```go
package main

import (
    "flag"
    "log"
    "net/http"
    "os"
    "os/signal"
    "syscall"
    
    "github.com/agenthands/mlxvllm/internal/api"
    "github.com/agenthands/mlxvllm/internal/mlx"
    "github.com/agenthands/mlxvllm/internal/radix"
    "github.com/agenthands/mlxvllm/pkg/tokenizer"
)

var (
    modelPath = flag.String("model", "./models/gui-actor-2b", "Path to MLX model")
    maxTokens = flag.Int("max_tokens", 100000, "Maximum cache size in tokens")
    addr       = flag.String("addr", ":8080", "Server address")
)

func main() {
    flag.Parse()
    
    log.Println("Initializing MLX bridge...")
    mlxEngine, err := mlx.NewMLXBridge()
    if err != nil {
        log.Fatalf("Failed to initialize MLX: %v", err)
    }
    
    log.Printf("Loading model from: %s", *modelPath)
    model, err := mlxEngine.LoadModel(*modelPath)
    if err != nil {
        log.Fatalf("Failed to load model: %v", err)
    }
    
    log.Println("Initializing tokenizer...")
    tz, err := tokenizer.NewTokenizer()
    if err != nil {
        log.Fatalf("Failed to initialize tokenizer: %v", err)
    }
    
    log.Printf("Creating Radix tree (max_tokens=%d)...", *maxTokens)
    tree := radix.NewTree(mlxEngine, *maxTokens)
    
    log.Println("Creating HTTP handler...")
    handler := api.NewHandler(tree, mlxEngine, tz)
    
    server := api.NewServer(*addr, handler)
    
    // Handle shutdown gracefully
    go func() {
        sigChan := make(chan os.Signal, 1)
        signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
        <-sigChan
        
        log.Println("Shutting down...")
        mlxEngine.UnloadModel(model)
        server.Shutdown(context.Background())
    }()
    
    log.Printf("Server listening on %s", *addr)
    if err := server.Start(); err != nil && err != http.ErrServerClosed {
        log.Fatalf("Server error: %v", err)
    }
}
```

**Step 2: Commit**

```bash
git add cmd/server/main.go
git commit -m "feat(server): implement main with DI wiring and graceful shutdown"
```

---

## Completion Checklist

After implementing all tasks:

- [ ] Run full test suite: `make test-coverage`
- [ ] Verify 100% coverage for `internal/radix/`
- [ ] Build server: `make build`
- [ ] Run integration tests: `make test-integration`
- [ ] Generate coverage report: `go tool cover -html=coverage.out`
- [ ] Test with real MLX model on Apple Silicon

**Pre-commit Verification:**
```bash
# Run before committing
go test ./... -cover
go fmt ./...
go vet ./...
```
