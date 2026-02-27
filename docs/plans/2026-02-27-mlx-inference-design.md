# MLX Real Inference Engine Design

**Date:** 2026-02-27
**Author:** Claude + User Collaboration
**Status:** Approved

---

## 1. Problem Statement

The server currently uses a `MockMLXEngine` that returns fake logits. The CGO bridge and C++ MLX engine are fully implemented, but there's no Go wrapper that connects them to the `radix.MLXEngine` interface used by the server.

**Current flow (broken):**
```
HTTP Request → Handler → MockMLXEngine → fake logits
```

**Target flow (working):**
```
HTTP Request → Handler → RealMLXEngine → CGO → C++ MLX → Metal GPU → real logits
```

---

## 2. Solution Overview

Create a `RealMLXEngine` struct in the `mlx` package that:

1. **Implements `radix.MLXEngine` interface** - drops into existing server code
2. **Wraps existing CGO functions** - no changes to C++ code needed
3. **Manages model lifecycle** - load once, use for all requests
4. **Handles vocabulary size** - dynamically allocates logits buffer

**Files to create/modify:**
- `internal/mlx/engine.go` (new) - RealMLXEngine implementation
- `internal/server/main.go` (modify) - Use RealMLXEngine instead of mock

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Go HTTP Server                          │
│  ┌────────────────┐  ┌────────────────┐  ┌─────────────────┐  │
│  │ HTTP Handler   │  │  Tokenizer     │  │ Radix Tree      │  │
│  │ (OpenAI API)   │  │  (SentencePiece)│  │ (KV Cache)      │  │
│  └────────┬───────┘  └────────┬───────┘  └────────┬────────┘  │
│           │                   │                   │             │
│           └───────────────────┼───────────────────┘             │
│                               ▼                                 │
│                    ┌───────────────────┐                        │
│                    │   RealMLXEngine   │                        │
│                    │   (Lock-Free)     │                        │
│                    └─────────┬─────────┘                        │
└──────────────────────────────┼─────────────────────────────────┘
                               │ CGO Boundary (Zero-Copy)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Objective-C++ MLX Engine                     │
│  ┌────────────────┐  ┌────────────────┐  ┌─────────────────┐   │
│  │ Metal Kernels  │  │  Model Loader  │  │ Cache Registry  │   │
│  │ (GPU compute)  │  │  (Weights)     │  │ (Thread-safe)   │   │
│  └────────────────┘  └────────────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Component Design

### 4.1 RealMLXEngine Struct

```go
// internal/mlx/engine.go
package mlx

type RealMLXEngine struct {
    loaded    bool
    vocabSize int
    modelPath string
}

func NewRealMLXEngine(modelPath string, vocabSize int) *RealMLXEngine

func (e *RealMLXEngine) LoadModel() error

func (e *RealMLXEngine) ForwardWithCache(model any, tokens []uint32, baseHandle uint64) ([]float32, uint64, error)

func (e *RealMLXEngine) SliceCache(handle uint64, keepTokens int) (uint64, error)

func (e *RealMLXEngine) FreeCache(handle uint64)
```

**Key design decisions:**
- **Lock-free:** No mutex around forward pass (C++ handles handle lookup with brief mutex)
- **Zero-copy:** Pass Go pointers directly to CGO, no allocation
- **Model parameter ignored:** MLX uses global C++ model state (simplifies design)

### 4.2 Update main.go

```go
// internal/server/main.go

func setupMLXEngine() (radix.MLXEngine, error) {
    if *modelPath == "" {
        // No model path provided, use mock for testing
        return &radix.MockMLXEngine{...}, nil
    }

    // Use real MLX engine
    engine := mlx.NewRealMLXEngine(*modelPath, *vocabSize)
    if err := engine.LoadModel(); err != nil {
        return nil, fmt.Errorf("failed to load MLX model: %w", err)
    }

    return engine, nil
}
```

---

## 5. Data Flow (Corrected)

```
┌─────────────────────────────────────────────────────────────────┐
│ HTTP Request: POST /v1/chat/completions                         │
│ { "messages": [...], "max_tokens": 100 }                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ ChatCompletionHandler                                           │
│  1. Parse request                                               │
│  2. Tokenize messages → []uint32                                │
│  3. Call GenerateAutoregressive()                               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ GenerateAutoregressive()                                        │
│  1. Find prefix match in RadixTree (O(1) Match)                 │
│  2. Get baseHandle from cache                                   │
│  3. Loop (Lock-Free):                                           │
│     - Call engine.ForwardWithCache(tokens, baseHandle)          │
│     - Sample next token from logits                             │
│     - Append to local buffer                                    │
│     - Break on EOS or max_tokens                                │
│  4. BULK INSERT generated buffer into RadixTree (OCC Write)     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ RealMLXEngine.ForwardWithCache() (Lock-Free)                    │
│  1. Pre-allocate logits []float32(vocab_size) in Go             │
│  2. Pass raw Go pointers via unsafe.Pointer                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ CGO Bridge (Zero-Copy)                                          │
│  1. NO ALLOCATION (use unsafe.Pointer)                          │
│  2. Call C.MLXForwardWithCache()                                │
│  3. NO COPYING (results written directly to Go slice)           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ C++ MLX Engine (mlx_engine.mm)                                  │
│  1. Brief mutex for handle lookup → Get base KV Cache           │
│  2. Run forward pass (Metal GPU, lock-free):                    │
│     - Embedding lookup                                          │
│     - 28 transformer layers (Store K/V in KV cache)              │
│     - Final LM head projection → logits                         │
│  3. Brief mutex for handle storage → Store new KV Cache         │
│  4. Write logits directly into Go's memory pointer              │
│  5. Return new cache handle                                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Response                                                         │
│  - Logits sampled → next token                                  │
│  - Token decoded → text                                         │
│  - JSON response returned                                       │
└─────────────────────────────────────────────────────────────────┘

Key performance points:
  1. KV cache handles stay in C++ (no GC tracking overhead)
  2. Zero-copy CGO boundary (no allocation or copying)
  3. Bulk insert prevents tree fragmentation
  4. Lock-free Metal GPU computation across concurrent requests
```

---

## 6. Implementation Steps

1. **Create `internal/mlx/engine.go`**
   - Define `RealMLXEngine` struct
   - Implement `radix.MLXEngine` interface
   - Lock-free, zero-copy design

2. **Update `internal/server/main.go`**
   - Change `setupMLXEngine()` to return `RealMLXEngine`
   - Keep mock fallback for testing

3. **Update imports**
   - Add `github.com/agenthands/GUI-Actor/internal/mlx`

4. **Verify build**
   - Run `go build ./...`
   - Ensure CGO linking works

---

## 7. Testing Strategy

**Unit tests:**
- `internal/mlx/engine_test.go` - Test RealMLXEngine methods
- Mock the CGO calls for unit testing

**Integration tests:**
- `internal/server/integration_test.go` - End-to-end with real model
- Test actual inference flow

**Performance tests:**
- Benchmark concurrent requests
- Verify lock-free behavior
- Measure zero-copy overhead

---

## 8. Error Handling

- **Model loading:** Clear errors for missing paths, bin_weights, vocab mismatch
- **Forward pass:** Invalid cache → restart from root; OOM → clear LRU; Metal fail → poison node
- **Graceful degradation:** MLX load fail → mock fallback; request fail → 500 error

---

## 9. Configuration

```
-model="/path/to/models/qwen2-vl/7b"

Expected structure:
/path/to/models/qwen2-vl/7b/
  ├── config.json
  ├── bin_weights/
  │   ├── embed_tokens.bin
  │   ├── layer0.attn.q_proj.bin
  │   └── ...
```

**Vocabulary sizes:**
- Qwen2-VL-2B: 151936 tokens
- Qwen2-VL-7B: 152064 tokens
- Default: 32000 (for testing)

---

## 10. Success Criteria

✅ Server loads real MLX engine when modelPath provided
✅ Inference returns actual logits (not fake)
✅ Multiple concurrent requests execute lock-free
✅ Zero-copy CGO boundary (no allocation/copying)
✅ Bulk insert prevents tree fragmentation
✅ 100% test coverage for new code
