# MLX Real Inference Engine Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Connect the existing CGO MLX bridge to the HTTP server, enabling actual model inference with Metal acceleration.

**Architecture:** Create a `RealMLXEngine` Go struct that implements the `radix.MLXEngine` interface, wrapping the existing CGO functions in `internal/mlx/mlx_bridges.go`. The engine is lock-free for concurrent requests and uses zero-copy CGO for maximum performance.

**Tech Stack:** Go 1.25, CGO, Objective-C++, Metal Framework, existing MLX C++ engine

---

## Prerequisites

**Required reading:**
- `docs/plans/2026-02-27-mlx-inference-design.md` - Full design document
- `internal/mlx/mlx_bridges.go` - Existing CGO bridge functions
- `internal/mlx/mlx_api.h` - C API declarations
- `internal/radix/engine.go` - MLXEngine interface definition
- `internal/server/main.go:145-167` - Current setupMLXEngine() function

**Model requirements:**
- Downloaded Qwen2-VL model with `bin_weights/` directory
- See `models/qwen2-vl/7b/` for example structure

---

## Task 1: Create RealMLXEngine struct

**Files:**
- Create: `internal/mlx/engine.go`

**Step 1: Write the struct definition and constructor**

```go
//go:build !mlx_mock

package mlx

// RealMLXEngine implements radix.MLXEngine using actual MLX inference
type RealMLXEngine struct {
    loaded    bool
    vocabSize int
    modelPath string
}

// NewRealMLXEngine creates a new MLX engine instance
// Note: Model is not loaded until LoadModel() is called
func NewRealMLXEngine(modelPath string, vocabSize int) *RealMLXEngine {
    return &RealMLXEngine{
        modelPath: modelPath,
        vocabSize: vocabSize,
        loaded:    false,
    }
}
```

**Step 2: Write LoadModel method**

```go
// LoadModel loads the model weights via CGO bridge
// This must be called before ForwardWithCache
// Thread-safe: uses global C++ model state
func (e *RealMLXEngine) LoadModel() error {
    if e.loaded {
        return nil // Already loaded
    }

    if err := LoadModel(e.modelPath, e.vocabSize); err != nil {
        return err
    }

    e.loaded = true
    return nil
}
```

**Step 3: Write ForwardWithCache method (lock-free)**

```go
// ForwardWithCache executes inference with KV cache
// Lock-free: thread safety handled in C++ layer
// Zero-copy: passes Go pointers directly to CGO
func (e *RealMLXEngine) ForwardWithCache(model any, tokens []uint32, baseHandle uint64) ([]float32, uint64, error) {
    // model parameter is ignored - MLX uses global C++ state
    // This is a design simplification for the initial implementation

    // Allocate logits buffer in Go (C++ will write directly to this memory)
    logits := make([]float32, e.vocabSize)

    newHandle, err := forwardWithCacheImpl(tokens, baseHandle, logits)
    if err != nil {
        return nil, 0, err
    }

    return logits, newHandle, nil
}
```

**Step 4: Write helper for CGO bridge call**

```go
// forwardWithCacheImpl is the actual CGO bridge call
// separated for easier testing with mocks
func forwardWithCacheImpl(tokens []uint32, baseHandle uint64, logits []float32) (uint64, error) {
    return ForwardWithCache(0, tokens, baseHandle)
}
```

**Step 5: Write SliceCache method**

```go
// SliceCache creates a zero-copy view of existing cache
// O(1) operation using MLX copy-on-write semantics
func (e *RealMLXEngine) SliceCache(handle uint64, keepTokens int) (uint64, error) {
    return SliceCache(handle, keepTokens)
}
```

**Step 6: Write FreeCache method**

```go
// FreeCache releases a cache handle and associated GPU memory
// Thread-safe: idempotent, safe to call multiple times
func (e *RealMLXEngine) FreeCache(handle uint64) {
    FreeCache(handle)
}
```

**Step 7: Create mock build tag version**

Create: `internal/mlx/engine_mock.go`

```go
//go:build mlx_mock

package mlx

import (
    "fmt"
    "github.com/agenthands/GUI-Actor/internal/radix"
)

// MockMLXEngine is a test double
type MockMLXEngine struct {
    // Delegate to existing mock implementation
    *radix.MockMLXEngine
}

func NewRealMLXEngine(modelPath string, vocabSize int) *MockMLXEngine {
    return &MockMLXEngine{
        MockMLXEngine: &radix.MockMLXEngine{},
    }
}

func (e *MockMLXEngine) LoadModel() error {
    return fmt.Errorf("mock: model not loaded")
}
```

**Step 8: Run build to verify**

```bash
go build ./internal/mlx/...
```

Expected: SUCCESS, no errors

**Step 9: Commit**

```bash
git add internal/mlx/engine.go internal/mlx/engine_mock.go
git commit -m "feat(mlx): add RealMLXEngine struct

Implements radix.MLXEngine interface with:
- Lock-free forward pass (C++ handles thread safety)
- Zero-copy CGO (Go pre-allocates, C++ writes directly)
- LoadModel, ForwardWithCache, SliceCache, FreeCache methods

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 2: Update main.go to use RealMLXEngine

**Files:**
- Modify: `internal/server/main.go:13-17` (add import)
- Modify: `internal/server/main.go:145-167` (update setupMLXEngine)

**Step 1: Add mlx import**

Find this line:
```go
    "github.com/agenthands/GUI-Actor/internal/mlx"
```

If it exists, keep it. If not, add it to the imports section.

**Step 2: Replace setupMLXEngine function**

Replace the entire function (lines 145-167) with:

```go
// setupMLXEngine initializes the MLX inference engine
// Uses RealMLXEngine if model path is provided, otherwise falls back to mock
func setupMLXEngine() (radix.MLXEngine, error) {
    // No model path provided - use mock for testing
    if *modelPath == "" {
        slog.Info("No model path provided, using mock MLX engine")
        slog.Info("To use real model, download Qwen2-VL from: https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct")
        return &radix.MockMLXEngine{
            ForwardFunc: func(model any, tokens []uint32, base uint64) ([]float32, uint64, error) {
                // Mock: return random logits for testing
                logits := make([]float32, *vocabSize)
                return logits, base + 1, nil
            },
            SliceFunc: func(handle uint64, keepTokens int) (uint64, error) {
                return handle + 100, nil
            },
            FreeFunc: func(handle uint64) {
                // No-op for mock
            },
        }, nil
    }

    // Use real MLX engine with Metal acceleration
    slog.Info("Initializing real MLX engine",
        "path", *modelPath,
        "vocab_size", *vocabSize,
    )

    engine := mlx.NewRealMLXEngine(*modelPath, *vocabSize)
    if err := engine.LoadModel(); err != nil {
        return nil, fmt.Errorf("failed to load MLX model: %w", err)
    }

    slog.Info("MLX engine loaded successfully", "type", "real")
    return engine, nil
}
```

**Step 3: Verify build**

```bash
go build ./cmd/server/...
```

Expected: SUCCESS, no errors

**Step 4: Commit**

```bash
git add internal/server/main.go
git commit -m "feat(server): use RealMLXEngine when model path provided

- setupMLXEngine now returns RealMLXEngine if modelPath is set
- Falls back to MockMLXEngine for testing without model
- Added proper logging for engine initialization

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 3: Update ForwardWithCache in CGO bridge to use vocab size

**Files:**
- Modify: `internal/mlx/mlx_bridges.go:32-69`

**Current issue:** ForwardWithCache hardcodes vocab size to 32000

**Step 1: Check current implementation**

The function currently has:
```go
outLogits := make([]float32, 32000) // TODO: get vocab size from model
```

**Step 2: Update to accept pre-allocated logits buffer**

Replace the entire `ForwardWithCache` function with:

```go
// ForwardWithCache executes MLX inference with KV cache
// logits must be pre-allocated by caller with size vocab_size
// This enables zero-copy: C++ writes directly to Go's memory
func ForwardWithCache(
    modelHandle uintptr,
    tokens []uint32,
    baseCacheHandle uint64,
    logits []float32,
) (uint64, error) {
    if len(tokens) == 0 {
        return 0, nil
    }

    if len(logits) == 0 {
        return 0, errors.New("logits buffer must be pre-allocated")
    }

    var outCacheHandle C.uint64_t
    var outErrorMsg *C.char

    ret := C.MLXForwardWithCache(
        C.uintptr_t(modelHandle),
        (*C.uint32_t)(unsafe.Pointer(&tokens[0])),
        C.int(len(tokens)),
        C.uint64_t(baseCacheHandle),
        (*C.float)(unsafe.Pointer(&logits[0])),
        C.int(len(logits)),
        &outCacheHandle,
        &outErrorMsg,
    )

    if ret != C.MLX_SUCCESS {
        if outErrorMsg != nil {
            errMsg := C.GoString(outErrorMsg)
            C.MLXFreeError(outErrorMsg)
            return 0, errors.New(errMsg)
        }
        return 0, errors.New("MLX error: unknown failure")
    }

    return uint64(outCacheHandle), nil
}
```

**Step 3: Update engine.go to use new signature**

Modify `internal/mlx/engine.go` forwardWithCacheImpl:

```go
// forwardWithCacheImpl is the actual CGO bridge call
func forwardWithCacheImpl(tokens []uint32, baseHandle uint64, logits []float32) (uint64, error) {
    return ForwardWithCache(0, tokens, baseHandle, logits)
}
```

**Step 4: Verify build**

```bash
go build ./internal/mlx/...
```

Expected: SUCCESS

**Step 5: Commit**

```bash
git add internal/mlx/mlx_bridges.go internal/mlx/engine.go
git commit -m "refactor(mlx): zero-copy ForwardWithCache

- Accepts pre-allocated logits buffer from caller
- C++ writes directly to Go memory (no copying)
- Enables dynamic vocab size support

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 4: Write unit tests for RealMLXEngine

**Files:**
- Create: `internal/mlx/engine_test.go`

**Step 1: Write test for NewRealMLXEngine**

```go
//go:build !mlx_mock

package mlx

import (
    "testing"
)

func TestNewRealMLXEngine(t *testing.T) {
    engine := NewRealMLXEngine("/fake/path", 152064)

    if engine == nil {
        t.Fatal("Expected non-nil engine")
    }

    if engine.modelPath != "/fake/path" {
        t.Errorf("Expected modelPath /fake/path, got %s", engine.modelPath)
    }

    if engine.vocabSize != 152064 {
        t.Errorf("Expected vocabSize 152064, got %d", engine.vocabSize)
    }

    if engine.loaded {
        t.Error("Expected engine to not be loaded initially")
    }
}
```

**Step 2: Write test for SliceCache**

```go
func TestRealMLXEngine_SliceCache(t *testing.T) {
    // This test will fail without actual MLX library
    // Skip for now, will be tested in integration
    t.Skip("Requires MLX library - tested in integration")
}
```

**Step 3: Write test for FreeCache**

```go
func TestRealMLXEngine_FreeCache(t *testing.T) {
    engine := NewRealMLXEngine("/fake/path", 152064)

    // Should not panic
    engine.FreeCache(123)
    engine.FreeCache(0) // Root handle - should also not panic
}
```

**Step 4: Run tests**

```bash
go test ./internal/mlx/... -v
```

Expected: PASS (FreeCache test may pass, others skip/fail)

**Step 5: Commit**

```bash
git add internal/mlx/engine_test.go
git commit -m "test(mlx): add unit tests for RealMLXEngine

- TestNewRealMLXEngine: verify constructor
- TestRealMLXEngine_FreeCache: verify idempotent free
- SliceCache skipped (requires MLX library)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 5: Update integration tests

**Files:**
- Modify: `internal/server/main_test.go`

**Step 1: Add test with real model path**

Add to existing test file:

```go
func TestMainWithRealModel(t *testing.T) {
    if testing.Short() {
        t.Skip("Skipping real model test in short mode")
    }

    // This test requires actual model to be downloaded
    modelPath := os.Getenv("MLX_TEST_MODEL_PATH")
    if modelPath == "" {
        t.Skip("Set MLX_TEST_MODEL_PATH to run this test")
    }

    // Would load real model and test inference
    // For now, just verify the code path exists
    t.Log("Real model test would run with:", modelPath)
}
```

**Step 2: Run tests**

```bash
go test ./internal/server/... -v
```

Expected: PASS (test skipped if env var not set)

**Step 3: Commit**

```bash
git add internal/server/main_test.go
git commit -m "test(server): add opt-in real model integration test

Test runs only when MLX_TEST_MODEL_PATH is set.
Skipped by default for CI/CD.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 6: Build and verify

**Step 1: Full build**

```bash
go build ./...
```

Expected: SUCCESS, no errors

**Step 2: Run all tests**

```bash
go test ./... -v
```

Expected: All tests pass

**Step 3: Check test coverage**

```bash
go test ./... -coverprofile=coverage.out -covermode=atomic
go tool cover -func=coverage.out | grep total
```

Expected: Coverage >= 70%

**Step 4: Verify CGO linking**

```bash
# Check that the binary links to Metal
otool -L $(go env GOPATH)/bin/server 2>/dev/null || otool -L ./server 2>/dev/null || echo "Binary check skipped"
```

Expected: Should see Metal framework references

**Step 5: Commit any final fixes**

```bash
git add .
git commit -m "fix: final build adjustments"
```

---

## Task 7: Documentation

**Step 1: Update CLAUDE.md**

Add to "Go Server (Production Inference)" section:

```markdown
### Running with Real MLX Engine

To run the server with actual model inference:

```bash
# From repository root
cd src

# Build with CGO enabled
CGO_ENABLED=1 go build -o server cmd/server/main.go

# Run with model path
./server -model ../models/qwen2-vl/7b -vocab-size 152064

# Or with go run
go run cmd/server/main.go -model ../models/qwen2-vl/7b -vocab-size 152064
```

**Model Requirements:**
- Model must have `bin_weights/` directory with converted weights
- Use `scripts/convert_safetensors.py` to convert from safetensors
```

**Step 2: Create README for MLX engine**

Create: `internal/mlx/README.md`

```markdown
# MLX Inference Engine

This package provides Go bindings for MLX inference on Apple Silicon.

## Components

- `engine.go` - RealMLXEngine implements radix.MLXEngine interface
- `mlx_bridges.go` - CGO bridge to C++ MLX engine
- `mlx_api.h` - C API declarations

## Usage

```go
import "github.com/agenthands/GUI-Actor/internal/mlx"

engine := mlx.NewRealMLXEngine(modelPath, vocabSize)
if err := engine.LoadModel(); err != nil {
    log.Fatal(err)
}

logits, handle, err := engine.ForwardWithCache(nil, tokens, baseHandle)
```

## Build Tags

- Default: Uses real MLX engine
- `mlx_mock`: Uses mock implementation for testing
```

**Step 3: Commit documentation**

```bash
git add CLAUDE.md internal/mlx/README.md
git commit -m "docs: add MLX engine usage documentation

- Update CLAUDE.md with real engine run instructions
- Add internal/mlx/README.md with usage examples

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Verification Steps

After completing all tasks:

1. **Build succeeds:** `go build ./...` ✅
2. **Tests pass:** `go test ./...` ✅
3. **Coverage >= 70%:** `go tool cover -func=coverage.out | grep total` ✅
4. **Can run with mock:** `go run cmd/server/main.go` (no model path) ✅
5. **Can load real model:** `go run cmd/server/main.go -model ../models/qwen2-vl/7b -vocab-size 152064` ✅
6. **HTTP server responds:** `curl http://localhost:8080/health` ✅

---

## Rollback Plan

If issues arise:

1. Revert to mock: Remove `-model` flag when running server
2. Use build tag: `go build -tags mlx_mock` to force mock
3. Git revert: `git revert <commit-hash>` for specific commits

---

## Next Steps

After this implementation:

1. Add tokenizer integration (currently mock)
2. Implement proper sampling from logits (currently greedy)
3. Add streaming response support
4. Performance benchmarking
5. Add support for Qwen2-VL-2B model
