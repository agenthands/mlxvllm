# MLX C++ Engine Implementation

## Overview

This directory contains the C++ implementation of the MLX inference engine with RadixAttention support for Apple Silicon.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Go Layer (radix package)                  │
│  Tree, Node, Match, InsertPending, Unpin, EvictLRU          │
└─────────────────────────────┬───────────────────────────────┘
                              │ CGO
┌─────────────────────────────▼───────────────────────────────┐
│              CGO Bridge Layer (mlx package)                  │
│  ForwardWithCache, SliceCache, FreeCache (Go → C bindings)  │
└─────────────────────────────┬───────────────────────────────┘
                              │ C API
┌─────────────────────────────▼───────────────────────────────┐
│                     C++ Layer (this dir)                     │
│  CacheRegistry, KVCache, MLXForwardWithCache implementation │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                    MLX Framework (Metal)                    │
│  GPU computation, KV cache management                       │
└─────────────────────────────────────────────────────────────┘
```

## Files

- `mlx_engine.h`: Header with CacheRegistry, KVCache, C API implementations
- `mlx_engine.cpp`: Implementation file (placeholder for separate compilation)

## Key Components

### CacheRegistry

Thread-safe registry for KV cache entries:
- `Insert(cache)`: Add cache, generate unique ID
- `Get(id)`: Retrieve cache by handle
- `Remove(id)`: Decrement refcount, erase when zero
- `Ref(id)`: Increment refcount

### KVCache

Cache entry representing a KV cache state:
- `id`: Unique handle (uint64_t)
- `tokens`: Token sequence at this node
- `logits`: Computed logits for last token
- `parent`: Parent cache pointer (for zero-copy slicing)
- `ref_count`: Reference count for memory management

### C API Functions

- `MLXForwardWithCache`: Execute inference with base cache
- `MLXSliceCache`: Create zero-copy view (O(1) operation)
- `MLXFreeCache`: Release cache handle
- `MLXFreeError`: Free error message string

## Thread Safety

- CacheRegistry: Protected by `std::mutex`
- Multiple threads can safely call ForwardWithCache on different caches
- NOT safe to call on same cache concurrently

## Memory Management

- Uses `std::shared_ptr` for automatic reference counting
- Parent caches kept alive by child references
- Explicit FreeCache decrements refcount
- Cache freed when refcount reaches zero

## Building

The C++ code must be compiled into `libmlx_runtime.dylib`:

```bash
# Compile C++ source
clang++ -std=c++17 -fPIC -c mlx_engine.cpp -o mlx_engine.o

# Link into shared library
clang++ -dynamiclib -o libmlx_runtime.dylib mlx_engine.o -framework Metal

# Install to library path
cp libmlx_runtime.dylib /usr/local/lib/
```

Then build Go code with:
```bash
go build -tags=mlx ./...
```

## Status

- [x] Header file with complete C++ implementation
- [x] Thread-safe CacheRegistry
- [x] Zero-copy slice implementation
- [x] CGO-compatible C API
- [ ] Actual MLX model integration (requires MLX SDK)
- [ ] Metal shader compilation
- [ ] Performance optimization

## Integration with Radix Tree

```
Go Radix Tree              C++ CacheRegistry
     │                            │
     │ InsertPending(tokens)       │
     ├───────────────────────────>│
     │                            │ Create cache
     │                            │ Generate handle
     │<───────────────────────────┤
     │  cache_handle              │
     │                            │
     │ ForwardWithCache(handle)   │
     ├───────────────────────────>│
     │                            │ Get parent cache
     │                            │ Run MLX inference
     │                            │ Create new cache
     │<───────────────────────────┤
     │  logits, new_handle        │
     │                            │
     │ SliceCache(handle, n)      │
     ├───────────────────────────>│
     │                            │ Get full tokens
     │                            │ Slice first n
     │                            │ Create child cache
     │<───────────────────────────┤
     │  sliced_handle             │
     │                            │
     │ FreeCache(handle)          │
     ├───────────────────────────>│
     │                            │ Decrement refcount
     │                            │ Erase if zero
```

## Next Steps

1. Integrate actual MLX SDK for model loading
2. Implement Metal GPU kernels for transformer layers
3. Add profiler hooks for performance measurement
4. Implement cache warmup strategies
5. Add memory pooling for KV cache entries
