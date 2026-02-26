# Integration Tests

These tests verify the complete RadixAttention pipeline with real MLX runtime on Apple Silicon hardware.

## Requirements

- **Hardware**: Apple Silicon (M4 Pro recommended)
- **OS**: macOS 14.0+ with Metal support
- **MLX Runtime**: `libmlx_runtime.dylib` built and installed
- **Model**: Model weights available (Qwen2-VL or similar)
- **Memory**: At least 16GB RAM recommended

## Building MLX Runtime

```bash
# Compile C++ engine
cd internal/mlx/cpp
clang++ -std=c++17 -fPIC -c mlx_engine.cpp -o mlx_engine.o

# Link shared library
clang++ -dynamiclib -o libmlx_runtime.dylib mlx_engine.o -framework Metal

# Install to system path
sudo cp libmlx_runtime.dylib /usr/local/lib/
```

## Running Integration Tests

### Quick Start

```bash
# Set environment variable
export INTEGRATION_TEST=1

# Run integration tests
cd tests/integration
go test -v -tags=integration
```

### Run Specific Test

```bash
go test -v -tags=integration -run TestFullPipelineIntegration/SimpleChatCompletion
```

### Run with Short Mode Skip

```bash
go test -v -tags=integration -short
```

## Test Coverage

### TestFullPipelineIntegration

Tests the complete HTTP request pipeline:

1. **SimpleChatCompletion**: Basic end-to-end request
2. **PrefixCaching**: Verifies cache hits on subsequent requests
3. **ConcurrentRequests**: Multiple simultaneous requests
4. **MultimodalRequest**: Text + image inputs
5. **LRUEviction**: Memory management under load

### TestRadixTreeBehavior

Tests Radix tree specific behaviors:

1. **PrefixMatch**: Longest prefix matching
2. **ThunderingHerd**: Multiple requests for same tokens
3. **PoisonedNodeRetry**: Error recovery and retry

### TestMemoryManagement

Tests memory-related behaviors:

1. **UnpinAndEvict**: LRU eviction
2. **CascadingCleanup**: Parent node cleanup

## Expected Performance

On M4 Pro hardware:

| Operation | Expected Latency |
|-----------|------------------|
| Token generation (cached) | 5-15ms |
| Token generation (uncached) | 20-50ms |
| Cache slice | <1ms |
| HTTP round-trip | +5ms overhead |

## Troubleshooting

### "library not found" Error

```bash
# Set library path
export DYLD_LIBRARY_PATH=/usr/local/lib:$DYLD_LIBRARY_PATH

# Or install to standard location
sudo cp libmlx_runtime.dylib /usr/local/lib/
```

### "Model not loaded" Error

Ensure model weights are available and model path is configured:

```bash
export MLX_MODEL_PATH=/path/to/model
```

### Tests Hang/Timeout

Check Metal is available:

```bash
system_profiler SPDisplaysDataType | grep Metal
```

## CI/CD Integration

Integration tests should run in CI only when:

- Running on Apple Silicon runner
- MLX runtime is pre-built
- Model weights are cached

Example GitHub Actions workflow:

```yaml
name: Integration Tests
on: [push, pull_request]
jobs:
  integration:
    runs-on: [self-hosted, macos, arm64]
    steps:
      - uses: actions/checkout@v3
      - name: Build MLX Runtime
        run: make mlx-runtime
      - name: Run Integration Tests
        run: go test -v -tags=integration ./tests/integration/...
        env:
          INTEGRATION_TEST: 1
```

## Adding New Tests

When adding new integration tests:

1. Add `//go:build integration` tag at top
2. Include `t.Skip("Skipping...")` with clear reason
3. Document requirements in this README
4. Add expected performance metrics
5. Include cleanup in defer blocks

## Status

- [ ] MLX C++ engine integration
- [ ] Model loading
- [ ] End-to-end generation
- [ ] Prefix caching verification
- [ ] LRU eviction testing
- [ ] Concurrent request handling
- [ ] Memory leak detection
- [ ] Performance benchmarking

## Next Steps

1. Implement actual MLX SDK integration
2. Add performance regression tests
3. Add stress tests for high concurrency
4. Add memory profiling
5. Add coverage for edge cases
