#ifndef MLX_ENGINE_H
#define MLX_ENGINE_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Error codes
#define MLX_SUCCESS 0
#define MLX_ERROR_INVALID_HANDLE -1
#define MLX_ERROR_OUT_OF_MEMORY -2
#define MLX_ERROR_INVALID_TOKENS -3
#define MLX_ERROR_COMPUTATION_FAILED -4
#define MLX_ERROR_MODEL_NOT_LOADED -5

// C API declarations
int MLXLoadModel(const char* model_path, int vocab_size);

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

int MLXSliceCache(
    uint64_t cache_handle,
    int keep_tokens,
    uint64_t* out_sliced_handle,
    char** out_error
);

void MLXFreeCache(uint64_t cache_handle);
void MLXFreeError(char* error);

#ifdef __cplusplus
}
#endif

#endif // MLX_ENGINE_H
