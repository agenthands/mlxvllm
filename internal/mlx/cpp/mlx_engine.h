#ifndef MLX_ENGINE_H
#define MLX_ENGINE_H

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <vector>
#include <string>
#include <stdexcept>

namespace mlx_vllm {

// =============================================================================
// KV Cache Entry
// =============================================================================

struct KVCache {
    uint64_t id;                      // Unique cache handle
    std::vector<float> logits;        // Computed logits (for last token)
    std::vector<uint32_t> tokens;     // Token sequence in this cache
    std::shared_ptr<KVCache> parent;  // Parent cache (for slicing)
    int ref_count;                    // Reference count for memory management

    KVCache(uint64_t _id, const std::vector<uint32_t>& _tokens,
             std::shared_ptr<KVCache> _parent = nullptr)
        : id(_id), tokens(_tokens), parent(_parent), ref_count(1) {}

    // Get full token sequence by walking up parent chain
    std::vector<uint32_t> GetFullTokenSequence() const {
        std::vector<uint32_t> result;
        auto current = this;
        while (current) {
            result.insert(result.begin(), current->tokens.begin(),
                         current->tokens.end());
            current = current->parent.get();
        }
        return result;
    }
};

// =============================================================================
// Thread-Safe Cache Registry
// =============================================================================

class CacheRegistry {
private:
    std::unordered_map<uint64_t, std::shared_ptr<KVCache>> caches_;
    std::mutex mutex_;
    uint64_t next_id_;

public:
    CacheRegistry() : next_id_(1) {}  // 0 is reserved for RootCacheHandle

    // Insert a new cache entry
    uint64_t Insert(std::shared_ptr<KVCache> cache) {
        std::lock_guard<std::mutex> lock(mutex_);
        cache->id = next_id_++;
        caches_[cache->id] = cache;
        return cache->id;
    }

    // Get cache entry by ID
    std::shared_ptr<KVCache> Get(uint64_t id) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = caches_.find(id);
        if (it != caches_.end()) {
            return it->second;
        }
        return nullptr;
    }

    // Remove cache entry (decrements refcount)
    void Remove(uint64_t id) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = caches_.find(id);
        if (it != caches_.end()) {
            it->second->ref_count--;
            if (it->second->ref_count <= 0) {
                caches_.erase(it);
            }
        }
    }

    // Increment refcount
    void Ref(uint64_t id) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = caches_.find(id);
        if (it != caches_.end()) {
            it->second->ref_count++;
        }
    }
};

// =============================================================================
// Global Registry Instance
// =============================================================================

static CacheRegistry g_registry;

// =============================================================================
// MLX Engine API Implementation
// =============================================================================

// Model placeholder (in production, this would hold actual MLX model state)
struct Model {
    int vocab_size;
    // Add actual MLX model fields here
};

// Global model placeholder (will be loaded externally)
static std::shared_ptr<Model> g_model;

// LoadModel sets the global model for inference
extern "C" void MLXLoadModel(uintptr_t model_ptr, int vocab_size) {
    g_model = std::make_shared<Model>();
    g_model->vocab_size = vocab_size;
    // In production, cast model_ptr to actual MLX model
}

// ForwardWithCache executes inference with KV cache
extern "C" int MLXForwardWithCache(
    uintptr_t model_handle,
    const uint32_t* tokens,
    int num_tokens,
    uint64_t base_cache_handle,
    float* out_logits,
    int out_logits_size,
    uint64_t* out_cache_handle,
    char** out_error
) {
    try {
        // Validate inputs
        if (!g_model) {
            *out_error = strdup("Model not loaded");
            return MLX_ERROR_MODEL_NOT_LOADED;
        }

        if (num_tokens <= 0 || !tokens) {
            *out_error = strdup("Invalid tokens");
            return MLX_ERROR_INVALID_TOKENS;
        }

        if (out_logits_size < g_model->vocab_size) {
            *out_error = strdup("Output buffer too small");
            return MLX_ERROR_OUT_OF_MEMORY;
        }

        // Get parent cache if specified
        std::shared_ptr<KVCache> parent_cache;
        if (base_cache_handle != 0) {
            parent_cache = g_registry.Get(base_cache_handle);
            if (!parent_cache) {
                *out_error = strdup("Invalid base cache handle");
                return MLX_ERROR_INVALID_HANDLE;
            }
            g_registry.Ref(base_cache_handle);  // Increment parent refcount
        }

        // Create new cache entry
        std::vector<uint32_t> token_vec(tokens, tokens + num_tokens);
        auto new_cache = std::make_shared<KVCache>(0, token_vec, parent_cache);

        // TODO: Actual MLX forward pass
        // For now, generate fake logits
        for (int i = 0; i < g_model->vocab_size; i++) {
            out_logits[i] = 0.01f;  // Placeholder
        }
        new_cache->logits.assign(out_logits, out_logits + g_model->vocab_size);

        // Register cache
        *out_cache_handle = g_registry.Insert(new_cache);
        *out_error = nullptr;

        return MLX_SUCCESS;

    } catch (const std::exception& e) {
        *out_error = strdup(e.what());
        return MLX_ERROR_COMPUTATION_FAILED;
    }
}

// SliceCache creates a zero-copy view of existing cache
extern "C" int MLXSliceCache(
    uint64_t cache_handle,
    int keep_tokens,
    uint64_t* out_sliced_handle,
    char** out_error
) {
    try {
        auto cache = g_registry.Get(cache_handle);
        if (!cache) {
            *out_error = strdup("Invalid cache handle");
            return MLX_ERROR_INVALID_HANDLE;
        }

        // Get full token sequence
        auto full_tokens = cache->GetFullTokenSequence();

        if (keep_tokens < 0 || keep_tokens > static_cast<int>(full_tokens.size())) {
            *out_error = strdup("keep_tokens out of range");
            return MLX_ERROR_INVALID_TOKENS;
        }

        // Slice tokens (copy-on-write: parent is original cache)
        std::vector<uint32_t> sliced_tokens(
            full_tokens.begin(),
            full_tokens.begin() + keep_tokens
        );

        // Create new cache pointing to parent
        auto sliced_cache = std::make_shared<KVCache>(0, sliced_tokens, cache);

        // Register new cache
        *out_sliced_handle = g_registry.Insert(sliced_cache);
        *out_error = nullptr;

        return MLX_SUCCESS;

    } catch (const std::exception& e) {
        *out_error = strdup(e.what());
        return MLX_ERROR_COMPUTATION_FAILED;
    }
}

// FreeCache releases a cache handle
extern "C" void MLXFreeCache(uint64_t cache_handle) {
    if (cache_handle == 0) {
        return;  // Don't free root handle
    }
    g_registry.Remove(cache_handle);
}

// FreeError frees an error message
extern "C" void MLXFreeError(char* error) {
    if (error) {
        free(error);
    }
}

} // namespace mlx_vllm

#endif // MLX_ENGINE_H
