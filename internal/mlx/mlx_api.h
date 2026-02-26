#ifndef MLX_API_H
#define MLX_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// MLX KV Cache Management API
// =============================================================================
// All functions use primitive-only types (uint64_t, uint32_t*, float32*)
// to avoid CGO complexity with Go garbage collector
// All returned handles must be explicitly freed with MLXFreeCache
// =============================================================================

// MLXForwardWithCache executes inference with KV cache
//
// Parameters:
//   model_handle - Opaque pointer to loaded model (cast from uintptr_t)
//   tokens - Array of token IDs (uint32_t*)
//   num_tokens - Length of tokens array
//   base_cache_handle - KV cache to extend from (0 = RootCacheHandle, empty cache)
//   out_logits - Output buffer for logits (pre-allocated by caller, float32*)
//   out_logits_size - Size of output buffer (number of float32 elements)
//   out_cache_handle - Output: new cache handle for KV cache
//   out_error - Output: error message (NULL on success, must be freed with MLXFreeError)
//
// Returns:
//   0 on success, non-zero error code on failure
//
// Thread Safety:
//   Safe to call concurrently on different cache handles
//   NOT safe to call concurrently on same cache handle
//
// Memory Management:
//   Caller must allocate out_logits buffer before call
//   Caller must call MLXFreeCache on out_cache_handle when done
//   Caller must call MLXFreeError on out_error when non-NULL
int MLXForwardWithCache(
    uintptr_t model_handle,
    const uint32_t* tokens,
    int num_tokens,
    uint64_t base_cache_handle,
    float* out_logits,
    int out_logits_size,
    uint64_t* out_cache_handle,
    char** out_error
);

// MLXSliceCache creates a zero-copy view of an existing cache
//
// This is an O(1) operation using MLX copy-on-write semantics
// The new cache handle shares memory with the original
//
// Parameters:
//   cache_handle - Source cache handle
//   keep_tokens - Number of tokens to keep from start of cache
//   out_sliced_handle - Output: new cache handle for sliced view
//   out_error - Output: error message (NULL on success)
//
// Returns:
//   0 on success, non-zero error code on failure
//
// Use Case:
//   When generation completes, slice cache to keep only committed tokens
//   Allows branching: original cache retains full sequence
int MLXSliceCache(
    uint64_t cache_handle,
    int keep_tokens,
    uint64_t* out_sliced_handle,
    char** out_error
);

// MLXFreeCache releases a cache handle and associated GPU memory
//
// Parameters:
//   cache_handle - Cache handle to free
//
// Thread Safety:
//   Safe to call concurrently on different handles
//   Idempotent: safe to call multiple times on same handle
//
// Memory Management:
//   Decrements internal refcount
//   Memory is freed when refcount reaches zero
void MLXFreeCache(uint64_t cache_handle);

// MLXFreeError frees an error message returned by MLX functions
//
// Parameters:
//   error - Error message to free (NULL-safe)
void MLXFreeError(char* error);

// =============================================================================
// Constants
// =============================================================================

#define MLX_ROOT_CACHE_HANDLE 0

// =============================================================================
// Error Codes
// =============================================================================

#define MLX_SUCCESS 0
#define MLX_ERROR_INVALID_HANDLE -1
#define MLX_ERROR_OUT_OF_MEMORY -2
#define MLX_ERROR_INVALID_TOKENS -3
#define MLX_ERROR_COMPUTATION_FAILED -4
#define MLX_ERROR_MODEL_NOT_LOADED -5

#ifdef __cplusplus
}
#endif

#endif // MLX_API_H
