# RadixAttention Prefix Caching for Apple Silicon MLX

**Date:** 2026-02-26  
**Status:** Approved for Implementation  
**Approach:** SGLang-style RadixAttention with Optimistic Concurrency Control

## Overview

Production-ready implementation of SGLang-style RadixAttention (Prefix Caching) targeting Apple Silicon Unified Memory. The system enforces strict separation of concerns to orchestrate high-throughput multi-modal LLM caching while isolating Go's Garbage Collector from Apple's Metal Unified Memory.

## Architecture

### 4-Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              HTTP / API Layer (Go)                           │
│    Lock-free autoregressive generation, bulk insertion      │
├─────────────────────────────────────────────────────────────┤
│           Radix Topology Manager (Go)                        │
│    OCC + Double-Checked Locking, O(1) Intrusive LRU         │
├─────────────────────────────────────────────────────────────┤
│              Flattened CGO Boundary                          │
│    Primitive-only uint64 handles, explicit destructors       │
├─────────────────────────────────────────────────────────────┤
│           Apple MLX Engine (C++ / Metal)                     │
│    Thread-safe shared_ptr registry, zero-copy slicing       │
└─────────────────────────────────────────────────────────────┘
```

**Data Flow:** HTTP Request → Radix Match → MLX Forward (with cache) → Autoregressive Loop → Bulk Insert

## Component Specifications

### 1. Radix Tree Node (`internal/radix/node.go`)

```go
type Node struct {
    Tokens      []uint32
    Children    map[uint32]*Node
    Parent      *Node
    CacheHandle uint64

    ready chan struct{} // Blocks waiters during computation
    err   error         // Poison state on MLX failure

    refCount atomic.Int32    // Pins node during computation
    lruElem  *list.Element  // O(1) LRU queue pointer
}

func (n *Node) Wait() error {
    <-n.ready
    return n.err
}
```

**Key Design:**
- `ready` channel solves Thundering Herd (multiple requests waiting on same prefix)
- `atomic.Int32` refCount prevents OOM by pinning active nodes
- Intrusive `lruElem` enables O(1) removal from `container/list`

### 2. Tree with OCC (`internal/radix/tree.go`)

```go
type Tree struct {
    mu        sync.Mutex
    Root      *Node
    Engine    MLXEngine
    MaxTokens int
    size      int
    
    lruQueue *list.List // O(1) LRU Eviction Queue
}
```

**Core Operations:**
- `Match()` - Read-only prefix search, returns (node, unmatched)
- `InsertPending()` - OCC write path: releases lock during `Wait()`, double-checked retry
- `Finalize()` - Marks node complete, wakes waiters
- `PrunePoisoned()` - Excises failed nodes, broadcasts error

**OCC Pattern:**
```go
func (t *Tree) InsertPending(parent *Node, unmatched []uint32) (*Node, error) {
    // 1. Acquire lock, check topology
    // 2. If exists, UNLOCK immediately
    // 3. Wait() on existing node
    // 4. Double-checked locking retry
    // 5. Return result
}
```

### 3. O(1) LRU Evictor (`internal/radix/evictor.go`)

Uses Go's `container/list` for constant-time operations:

```go
func (t *Tree) enforceCapacity(needed int) {
    for t.size+needed > t.MaxTokens {
        elem := t.lruQueue.Back() // O(1) pop from tail
        lruLeaf := t.lruQueue.Remove(elem).(*Node)
        
        t.Engine.FreeCache(lruLeaf.CacheHandle)
        delete(lruLeaf.Parent.Children, lruLeaf.Tokens[0])
        
        // Cascading cleanup
        if len(lruLeaf.Parent.Children) == 0 && 
           lruLeaf.Parent.refCount.Load() == 0 {
            lruLeaf.Parent.lruElem = t.lruQueue.PushFront(lruLeaf.Parent)
        }
    }
}
```

### 4. Flattened CGO API (`internal/mlx/mlx_api.h`)

```c
#ifndef MLX_RADIX_API_H
#define MLX_RADIX_API_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

typedef uint64_t mlx_kv_cache_t;

#ifdef __cplusplus
extern "C" {
#endif

bool mlx_forward_with_cache(
    void* model_ptr, 
    const uint32_t* tokens, 
    size_t num_tokens,
    mlx_kv_cache_t base_cache, 
    mlx_kv_cache_t* out_cache,
    float** out_logits 
);

mlx_kv_cache_t mlx_kv_cache_slice(mlx_kv_cache_t base_cache, size_t keep_tokens);
void mlx_free_kv_cache(mlx_kv_cache_t cache);
void mlx_free_logits(float* logits);

#ifdef __cplusplus
}
#endif

#endif
```

**Go Bridge with CGO Directives:**
```go
/*
#cgo CXXFLAGS: -std=c++17 -I/opt/homebrew/include
#cgo LDFLAGS: -L/opt/homebrew/lib -lmlx -lmlx-metal -framework Foundation -framework Metal
#include "mlx_api.h"
*/
import "C"
```

### 5. MLX C++ Engine (`internal/mlx/mlx_engine.cpp`)

Thread-safe registry using `std::shared_ptr`:

```cpp
static std::unordered_map<uint64_t, std::shared_ptr<mlx::core::KVCache>> cache_registry;
static std::mutex registry_mutex;

extern "C" bool mlx_forward_with_cache(
    void* model_ptr,
    const uint32_t* tokens,
    size_t num_tokens,
    mlx_kv_cache_t base_cache,
    mlx_kv_cache_t* out_cache,
    float** out_logits
) {
    // Lazy tensor evaluation
    // Zero-copy slicing for tree branches
    // Returns new opaque handle
}
```

## Data Flow

### Request Processing

1. **Tokenize** - `Tokenizer.Encode(messages) → []uint32`
2. **Match** - `RadixTree.Match(tokens) → (matchNode, unmatched)`
3. **Insert Pending** (if unmatched):
   - `InsertPending(matchNode, unmatched) → pendingNode`
   - `MLX.ForwardWithCache(unmatched, matchHandle) → (logits, newHandle)`
   - `Finalize(pendingNode, newHandle)`
4. **Pin** - `defer Unpin(promptNode)`

### Autoregressive Generation (Lock-Free)

```go
currHandle := promptNode.CacheHandle
generatedTokens := []uint32{} // Local buffer

for !isStopToken(lastToken) {
    logits, nextHandle, err := MLX.ForwardWithCache(model, []uint32{lastToken}, currHandle)
    MLX.FreeLogits(logits) // Explicit free, NOT defer (loop scoping!)
    
    if currHandle != promptNode.CacheHandle {
        MLX.FreeCache(currHandle) // Free intermediate handles
    }
    
    lastToken = sample(logits)
    generatedTokens = append(generatedTokens, lastToken)
    currHandle = nextHandle
    streamToClient(lastToken)
}
```

### Bulk Insertion

```go
if len(generatedTokens) > 0 {
    genNode, _ := RadixTree.InsertPending(promptNode, generatedTokens)
    RadixTree.Finalize(genNode, currHandle)
    RadixTree.Unpin(genNode)
} else {
    MLX.FreeCache(currHandle)
}
```

## File Structure

```
/radix-mlx-server
├── cmd/server/
│   └── main.go                 # App entry point, DI wiring
├── internal/api/
│   ├── handler.go              # Autoregressive generation, HTTP DoS checks
│   └── middleware.go           # Context timeouts, TLS, Panic recovery
├── internal/radix/
│   ├── tree.go                 # OCC Topology
│   ├── node.go                 # Node struct, channels, LRU pointers
│   ├── evictor.go              # O(1) LRU logic & cascading cleanup
│   └── tree_test.go            # TDD Mocking suite
├── internal/mlx/
│   ├── bridge.go               # Go wrapper with defer C.mlx_free_*
│   ├── mlx_api.h               # Flattened C-API boundary
│   └── mlx_engine.cpp          # C++ MLX execution and registry
├── pkg/tokenizer/
│   └── tokenizer.go            # Text/Image to uint32 conversion
├── go.mod
├── go.sum
└── Makefile
```

## Testing Strategy

### Unit Tests (Pure Go, No Hardware Required)

```go
type MockMLXEngine struct {
    ForwardFunc func(model any, tokens []uint32, base uint64) ([]float32, uint64, error)
    SliceFunc   func(handle uint64, keepTokens int) (uint64, error)
    FreeFunc    func(handle uint64)
}

func TestOptimisticConcurrency_WritePath(t *testing.T)
func TestO1_LRU_Cascading(t *testing.T)
func TestPoisonedNodePruning(t *testing.T)
func TestAutoregressive_BulkInsert(t *testing.T)
func TestTreeMatch_TableDriven(t *testing.T)
```

### Integration Tests (Apple Silicon Required)

```go
func TestIntegration_RealMLXForward(t *testing.T) {
    if testing.Short() { t.Skip("Requires Apple Silicon") }
    // Tests actual MLX ForwardWithCache
}
```

## Security & Performance Guarantees

| Guarantee | Implementation |
|-----------|----------------|
| **Deadlock Immunity** | OCC + Double-Checked Locking, lock held <1µs |
| **O(1) LRU Stability** | `container/list` intrusive queue |
| **Zero-Copy Memory** | Flattened CGO with primitive arrays |
| **OOM Firewall** | Atomic refCount pins active computation |
| **Poisoned Node Recovery** | PrunePoisoned excises failed nodes immediately |

## Build Configuration

```makefile
# Makefile
CGO_ENABLED=1
GO_TAGS=mlx

.PHONY: all test clean

all:
	go build -o bin/server ./cmd/server

test:
	go test ./... -short
	go test ./internal/mlx/... -run Integration

clean:
	rm -rf bin/
```

## Implementation Roadmap

1. **Phase 1:** Core Radix Tree with MockMLXEngine
2. **Phase 2:** Flattened CGO Boundary
3. **Phase 3:** MLX C++ Engine
4. **Phase 4:** HTTP Handler Integration
5. **Phase 5:** End-to-End Testing
