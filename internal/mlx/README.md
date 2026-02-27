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
if err != nil {
    log.Fatal(err)
}

// Slice cache to keep first N tokens
slicedHandle, err := engine.SliceCache(handle, keepTokens)

// Free cache when done
engine.FreeCache(handle)
```

## Build Tags

- Default: Uses real MLX engine
- `mlx_mock`: Uses mock implementation for testing

```bash
# Build with mock (for testing)
go build -tags mlx_mock

# Build with real MLX (default)
go build
```

## Zero-Copy Design

The engine uses zero-copy CGO for maximum performance:
1. Go pre-allocates logits buffer
2. C++ writes directly to Go's memory
3. No copying between Go and C++

## Thread Safety

- ForwardWithCache: Lock-free (C++ handles thread safety)
- SliceCache: Thread-safe via copy-on-write
- FreeCache: Idempotent, safe to call multiple times
