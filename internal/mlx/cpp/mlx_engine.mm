/**
 * MLX Engine for Qwen2-VL Inference on Apple Silicon
 * Complete forward pass implementation with Metal acceleration
 */

#import <Metal/Metal.h>
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <cstring>
#include <cmath>
#include <fstream>
#include <algorithm>
#include "mlx_engine.h"

namespace mlx_vllm {

// Model Configuration (Qwen2-VL-7B) - actual model parameters
struct ModelConfig {
    int hidden_size = 3584;
    int num_attention_heads = 28;
    int num_key_value_heads = 4;
    int num_hidden_layers = 28;
    int intermediate_size = 18944;
    int vocab_size = 152064;
    int head_dim = 128;  // hidden_size / num_attention_heads
    float rms_norm_eps = 1e-6f;
    int max_position_embeddings = 32768;
    float rope_theta = 10000.0f;
};

// Global model and Metal device
static id<MTLDevice> g_device = nil;
static std::shared_ptr<class Qwen2VLModel> g_model;
static std::mutex g_model_mutex;

// KV Cache Entry
struct KVCache {
    uint64_t id;
    std::vector<float> logits;
    std::vector<uint32_t> tokens;
    std::vector<std::vector<float>> key_cache;  // [layer][seq_len * kv_heads * head_dim]
    std::vector<std::vector<float>> value_cache; // [layer][seq_len * kv_heads * head_dim]
    std::shared_ptr<KVCache> parent;
    int ref_count;
    int seq_length;

    KVCache(uint64_t _id, const std::vector<uint32_t>& _tokens,
            std::shared_ptr<KVCache> _parent = nullptr, int seq_len = 0, int num_layers = 28)
        : id(_id), tokens(_tokens), parent(_parent), ref_count(1), seq_length(seq_len) {
        key_cache.resize(num_layers);
        value_cache.resize(num_layers);
    }

    std::vector<uint32_t> GetFullTokenSequence() const {
        std::vector<uint32_t> result;
        auto current = this;
        while (current) {
            result.insert(result.begin(), current->tokens.begin(), current->tokens.end());
            current = current->parent.get();
        }
        return result;
    }
};

// Cache Registry
class CacheRegistry {
private:
    std::unordered_map<uint64_t, std::shared_ptr<KVCache>> caches_;
    std::mutex mutex_;
    uint64_t next_id_ = 1;

public:
    uint64_t Insert(std::shared_ptr<KVCache> cache) {
        std::lock_guard<std::mutex> lock(mutex_);
        cache->id = next_id_++;
        caches_[cache->id] = cache;
        return cache->id;
    }

    std::shared_ptr<KVCache> Get(uint64_t id) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = caches_.find(id);
        return it != caches_.end() ? it->second : nullptr;
    }

    void Remove(uint64_t id) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = caches_.find(id);
        if (it != caches_.end()) {
            it->second->ref_count--;
            if (it->second->ref_count <= 0) caches_.erase(it);
        }
    }

    void Ref(uint64_t id) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = caches_.find(id);
        if (it != caches_.end()) it->second->ref_count++;
    }
};

static CacheRegistry g_registry;

// Qwen2-VL Model with complete forward pass
class Qwen2VLModel {
private:
    ModelConfig config_;
    std::unordered_map<std::string, std::vector<float>> weights_;

    // Metal compute pipelines
    id<MTLComputePipelineState> matmul_pipeline_;
    id<MTLComputePipelineState> rmsnorm_pipeline_;
    id<MTLComputePipelineState> gelu_pipeline_;
    id<MTLComputePipelineState> rope_pipeline_;
    id<MTLComputePipelineState> transpose_pipeline_;
    id<MTLComputePipelineState> softmax_pipeline_;
    id<MTLComputePipelineState> scale_pipeline_;
    id<MTLComputePipelineState> add_pipeline_;

    void init_metal() {
        if (!g_device) {
            g_device = MTLCreateSystemDefaultDevice();
            if (!g_device) {
                throw std::runtime_error("Failed to create Metal device");
            }
        }

        NSError* error = nil;

        // Comprehensive Metal shader library with all kernels
        NSString* shaderSource = [NSString stringWithUTF8String:R"(
            #include <metal_stdlib>
            using namespace metal;

            // Matrix multiplication kernel
            kernel void matmul_kernel(
                const device float* A [[buffer(0)]],
                const device float* B [[buffer(1)]],
                device float* C [[buffer(2)]],
                constant uint& M [[buffer(3)]],
                constant uint& N [[buffer(4)]],
                constant uint& K [[buffer(5)]],
                uint2 gid [[thread_position_in_grid]]) {
                uint m = gid.x, n = gid.y;
                if (m >= M || n >= N) return;
                float sum = 0.0;
                for (uint k = 0; k < K; k++)
                    sum += A[m * K + k] * B[k * N + n];
                C[m * N + n] = sum;
            }

            // RMSNorm kernel: y = x / sqrt(mean(x^2) + eps) * weight
            kernel void rmsnorm_kernel(
                const device float* x [[buffer(0)]],
                const device float* weight [[buffer(1)]],
                device float* y [[buffer(2)]],
                constant uint& size [[buffer(3)]],
                constant float& eps [[buffer(4)]],
                uint gid [[thread_position_in_grid]]) {
                if (gid >= size) return;
                float sum_sq = 0.0;
                for (uint i = 0; i < size; i++)
                    sum_sq += x[i] * x[i];
                float variance = sum_sq / (float)size + eps;
                float rsqrt = 1.0 / sqrt(variance);
                y[gid] = x[gid] * rsqrt * weight[gid];
            }

            // GeLU activation kernel (approximate)
            kernel void gelu_kernel(
                const device float* x [[buffer(0)]],
                device float* y [[buffer(1)]],
                uint gid [[thread_position_in_grid]]) {
                float x_val = x[gid];
                y[gid] = 0.5 * x_val * (1.0 + tanh(0.7978845608 * x_val * (1.0 + 0.044715 * x_val * x_val)));
            }

            // Rotary Position Embedding (RoPE) kernel
            kernel void rope_kernel(
                device float* q [[buffer(0)]],
                device float* k [[buffer(1)]],
                constant uint& seq_len [[buffer(2)]],
                constant uint& num_heads [[buffer(3)]],
                constant uint& head_dim [[buffer(4)]],
                constant uint& position [[buffer(5)]],
                constant float& theta [[buffer(6)]],
                uint3 gid [[thread_position_in_grid]]) {
                uint head = gid.x;
                uint token = gid.y;
                uint half_dim = gid.z;

                if (head >= num_heads || token >= seq_len || half_dim >= head_dim / 2) return;

                float freq = 1.0 / pow(theta, (2.0 * half_dim) / head_dim);
                float angle = position * freq;
                float cos_a = cos(angle);
                float sin_a = sin(angle);

                uint offset = (token * num_heads + head) * head_dim;
                uint i = half_dim;
                uint j = half_dim + head_dim / 2;

                float q_i = q[offset + i];
                float q_j = q[offset + j];
                q[offset + i] = q_i * cos_a - q_j * sin_a;
                q[offset + j] = q_i * sin_a + q_j * cos_a;

                float k_i = k[offset + i];
                float k_j = k[offset + j];
                k[offset + i] = k_i * cos_a - k_j * sin_a;
                k[offset + j] = k_i * sin_a + k_j * cos_a;
            }

            // Transpose kernel for matrix operations
            kernel void transpose_kernel(
                const device float* A [[buffer(0)]],
                device float* B [[buffer(1)]],
                constant uint& M [[buffer(2)]],
                constant uint& N [[buffer(3)]],
                uint2 gid [[thread_position_in_grid]]) {
                uint m = gid.x, n = gid.y;
                if (m >= M || n >= N) return;
                B[n * M + m] = A[m * N + n];
            }

            // Softmax kernel (for attention scores)
            kernel void softmax_kernel(
                const device float* x [[buffer(0)]],
                device float* y [[buffer(1)]],
                constant uint& seq_len [[buffer(2)]],
                uint2 gid [[thread_position_in_grid]]) {
                uint head = gid.x;
                uint query_pos = gid.y;

                if (query_pos >= seq_len) return;

                uint offset = (query_pos * seq_len);

                // Find max for numerical stability
                float max_val = x[offset];
                for (uint i = 1; i < seq_len; i++) {
                    if (x[offset + i] > max_val) max_val = x[offset + i];
                }

                // Compute exp sum
                float sum_exp = 0.0;
                for (uint i = 0; i < seq_len; i++) {
                    sum_exp += exp(x[offset + i] - max_val);
                }

                // Apply softmax
                for (uint i = 0; i < seq_len; i++) {
                    y[offset + i] = exp(x[offset + i] - max_val) / sum_exp;
                }
            }

            // Scale kernel
            kernel void scale_kernel(
                const device float* x [[buffer(0)]],
                device float* y [[buffer(1)]],
                constant float& scale [[buffer(2)]],
                uint gid [[thread_position_in_grid]]) {
                y[gid] = x[gid] * scale;
            }

            // Element-wise add kernel
            kernel void add_kernel(
                const device float* a [[buffer(0)]],
                const device float* b [[buffer(1)]],
                device float* c [[buffer(2)]],
                uint gid [[thread_position_in_grid]]) {
                c[gid] = a[gid] + b[gid];
            }
        )"];

        id<MTLLibrary> library = [g_device newLibraryWithSource:shaderSource options:nil error:&error];
        if (!library) {
            NSString* errStr = [error localizedDescription];
            throw std::runtime_error([errStr UTF8String]);
        }

        // Create all pipeline states
        matmul_pipeline_ = [g_device newComputePipelineStateWithFunction:[library newFunctionWithName:@"matmul_kernel"] error:&error];
        rmsnorm_pipeline_ = [g_device newComputePipelineStateWithFunction:[library newFunctionWithName:@"rmsnorm_kernel"] error:&error];
        gelu_pipeline_ = [g_device newComputePipelineStateWithFunction:[library newFunctionWithName:@"gelu_kernel"] error:&error];
        rope_pipeline_ = [g_device newComputePipelineStateWithFunction:[library newFunctionWithName:@"rope_kernel"] error:&error];
        transpose_pipeline_ = [g_device newComputePipelineStateWithFunction:[library newFunctionWithName:@"transpose_kernel"] error:&error];
        softmax_pipeline_ = [g_device newComputePipelineStateWithFunction:[library newFunctionWithName:@"softmax_kernel"] error:&error];
        scale_pipeline_ = [g_device newComputePipelineStateWithFunction:[library newFunctionWithName:@"scale_kernel"] error:&error];
        add_pipeline_ = [g_device newComputePipelineStateWithFunction:[library newFunctionWithName:@"add_kernel"] error:&error];

        if (!matmul_pipeline_ || !rmsnorm_pipeline_ || !gelu_pipeline_ ||
            !rope_pipeline_ || !transpose_pipeline_ || !softmax_pipeline_ ||
            !scale_pipeline_ || !add_pipeline_) {
            throw std::runtime_error("Failed to create Metal pipelines");
        }
    }

    // Helper: Execute Metal kernel with 1D grid
    void execute_1d(id<MTLComputePipelineState> pipeline, const std::vector<id<MTLBuffer>>& buffers, uint gridSize) {
        @autoreleasepool {
            id<MTLCommandQueue> queue = [g_device newCommandQueue];
            id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

            [encoder setComputePipelineState:pipeline];
            for (size_t i = 0; i < buffers.size(); i++) {
                [encoder setBuffer:buffers[i] offset:0 atIndex:i];
            }

            MTLSize threadsPerThreadgroup = {256, 1, 1};
            MTLSize threadgroups = {(gridSize + 255) / 256, 1, 1};
            [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerThreadgroup];
            [encoder endEncoding];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
        }
    }

    // Helper: Execute Metal kernel with 2D grid
    void execute_2d(id<MTLComputePipelineState> pipeline, const std::vector<id<MTLBuffer>>& buffers, MTLSize gridSize) {
        @autoreleasepool {
            id<MTLCommandQueue> queue = [g_device newCommandQueue];
            id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

            [encoder setComputePipelineState:pipeline];
            for (size_t i = 0; i < buffers.size(); i++) {
                [encoder setBuffer:buffers[i] offset:0 atIndex:i];
            }

            MTLSize threadsPerThreadgroup = {16, 16, 1};
            [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadsPerThreadgroup];
            [encoder endEncoding];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
        }
    }

    // Helper: Execute Metal kernel with 3D grid
    void execute_3d(id<MTLComputePipelineState> pipeline, const std::vector<id<MTLBuffer>>& buffers, MTLSize gridSize) {
        @autoreleasepool {
            id<MTLCommandQueue> queue = [g_device newCommandQueue];
            id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

            [encoder setComputePipelineState:pipeline];
            for (size_t i = 0; i < buffers.size(); i++) {
                [encoder setBuffer:buffers[i] offset:0 atIndex:i];
            }

            MTLSize threadsPerThreadgroup = {8, 8, 8};
            [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadsPerThreadgroup];
            [encoder endEncoding];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
        }
    }

    // Matrix multiplication on Metal
    std::vector<float> matmul(const std::vector<float>& A, const std::vector<float>& B, int M, int N, int K) {
        std::vector<float> C(M * N, 0.0f);

        @autoreleasepool {
            id<MTLBuffer> bufferA = [g_device newBufferWithBytes:A.data() length:A.size() * sizeof(float) options:0];
            id<MTLBuffer> bufferB = [g_device newBufferWithBytes:B.data() length:B.size() * sizeof(float) options:0];
            id<MTLBuffer> bufferC = [g_device newBufferWithLength:C.size() * sizeof(float) options:0];

            uint M_val = M, N_val = N, K_val = K;
            id<MTLBuffer> sizeBuffers[] = {
                [g_device newBufferWithBytes:&M_val length:sizeof(M_val) options:0],
                [g_device newBufferWithBytes:&N_val length:sizeof(N_val) options:0],
                [g_device newBufferWithBytes:&K_val length:sizeof(K_val) options:0]
            };

            std::vector<id<MTLBuffer>> buffers = {bufferA, bufferB, bufferC, sizeBuffers[0], sizeBuffers[1], sizeBuffers[2]};
            MTLSize gridSize = {static_cast<NSUInteger>(M), static_cast<NSUInteger>(N), 1};
            execute_2d(matmul_pipeline_, buffers, gridSize);

            memcpy(C.data(), [bufferC contents], C.size() * sizeof(float));
        }

        return C;
    }

    // RMSNorm on Metal
    std::vector<float> rmsnorm(const std::vector<float>& x, const std::vector<float>& weight, int size, int rows) {
        std::vector<float> y(x.size());

        for (int row = 0; row < rows; row++) {
            @autoreleasepool {
                id<MTLBuffer> bufferX = [g_device newBufferWithBytes:x.data() + row * size length:size * sizeof(float) options:0];
                id<MTLBuffer> bufferW = [g_device newBufferWithBytes:weight.data() length:size * sizeof(float) options:0];
                id<MTLBuffer> bufferY = [g_device newBufferWithLength:size * sizeof(float) options:0];

                uint size_val = size;
                id<MTLBuffer> sizeBuffer = [g_device newBufferWithBytes:&size_val length:sizeof(size_val) options:0];
                id<MTLBuffer> epsBuffer = [g_device newBufferWithBytes:&config_.rms_norm_eps length:sizeof(float) options:0];

                std::vector<id<MTLBuffer>> buffers = {bufferX, bufferW, bufferY, sizeBuffer, epsBuffer};
                execute_1d(rmsnorm_pipeline_, buffers, size);

                memcpy(y.data() + row * size, [bufferY contents], size * sizeof(float));
            }
        }

        return y;
    }

    // GeLU activation
    std::vector<float> gelu(const std::vector<float>& x) {
        std::vector<float> y(x.size());

        @autoreleasepool {
            id<MTLBuffer> bufferX = [g_device newBufferWithBytes:x.data() length:x.size() * sizeof(float) options:0];
            id<MTLBuffer> bufferY = [g_device newBufferWithLength:y.size() * sizeof(float) options:0];

            std::vector<id<MTLBuffer>> buffers = {bufferX, bufferY};
            execute_1d(gelu_pipeline_, buffers, x.size());

            memcpy(y.data(), [bufferY contents], y.size() * sizeof(float));
        }

        return y;
    }

    // Apply RoPE to Q and K
    void apply_rope(std::vector<float>& q, std::vector<float>& k, int seq_len, int num_heads, int head_dim, int position) {
        @autoreleasepool {
            int num_kv_heads = num_heads / (config_.num_attention_heads / config_.num_key_value_heads);
            MTLSize gridSize = {static_cast<NSUInteger>(num_heads), static_cast<NSUInteger>(seq_len), static_cast<NSUInteger>(head_dim / 2)};

            id<MTLBuffer> bufferQ = [g_device newBufferWithBytes:q.data() length:q.size() * sizeof(float) options:0];
            id<MTLBuffer> bufferK = [g_device newBufferWithBytes:k.data() length:k.size() * sizeof(float) options:0];

            uint seq_len_val = seq_len, num_heads_val = num_heads, head_dim_val = head_dim, position_val = position;
            float theta_val = config_.rope_theta;

            id<MTLBuffer> buffers[] = {
                bufferQ, bufferK,
                [g_device newBufferWithBytes:&seq_len_val length:sizeof(seq_len_val) options:0],
                [g_device newBufferWithBytes:&num_heads_val length:sizeof(num_heads_val) options:0],
                [g_device newBufferWithBytes:&head_dim_val length:sizeof(head_dim_val) options:0],
                [g_device newBufferWithBytes:&position_val length:sizeof(position_val) options:0],
                [g_device newBufferWithBytes:&theta_val length:sizeof(theta_val) options:0]
            };

            execute_3d(rope_pipeline_, {buffers[0], buffers[1], buffers[2], buffers[3], buffers[4], buffers[5], buffers[6]}, gridSize);

            memcpy(q.data(), [bufferQ contents], q.size() * sizeof(float));
            memcpy(k.data(), [bufferK contents], k.size() * sizeof(float));
        }
    }

    // Softmax along last dimension
    std::vector<float> softmax(const std::vector<float>& x, int seq_len) {
        std::vector<float> y(x.size());

        @autoreleasepool {
            id<MTLBuffer> bufferX = [g_device newBufferWithBytes:x.data() length:x.size() * sizeof(float) options:0];
            id<MTLBuffer> bufferY = [g_device newBufferWithLength:y.size() * sizeof(float) options:0];

            uint seq_len_val = seq_len;
            id<MTLBuffer> seqBuffer = [g_device newBufferWithBytes:&seq_len_val length:sizeof(seq_len_val) options:0];

            MTLSize gridSize = {1, static_cast<NSUInteger>(seq_len), 1};
            execute_2d(softmax_pipeline_, {bufferX, bufferY, seqBuffer}, gridSize);

            memcpy(y.data(), [bufferY contents], y.size() * sizeof(float));
        }

        return y;
    }

    // Element-wise addition
    std::vector<float> add(const std::vector<float>& a, const std::vector<float>& b) {
        std::vector<float> c(a.size());

        @autoreleasepool {
            id<MTLBuffer> bufferA = [g_device newBufferWithBytes:a.data() length:a.size() * sizeof(float) options:0];
            id<MTLBuffer> bufferB = [g_device newBufferWithBytes:b.data() length:b.size() * sizeof(float) options:0];
            id<MTLBuffer> bufferC = [g_device newBufferWithLength:c.size() * sizeof(float) options:0];

            std::vector<id<MTLBuffer>> buffers = {bufferA, bufferB, bufferC};
            execute_1d(add_pipeline_, buffers, a.size());

            memcpy(c.data(), [bufferC contents], c.size() * sizeof(float));
        }

        return c;
    }

public:
    Qwen2VLModel(const std::string& model_path, const ModelConfig& config) : config_(config) {
        init_metal();
        load_weights(model_path);
    }

    // Helper to load a binary weight file
    std::vector<float> load_binary_file(const std::string& file_path, size_t expected_elements) {
        std::vector<float> data(expected_elements);
        std::ifstream file(file_path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open file: " + file_path);
        }
        file.read(reinterpret_cast<char*>(data.data()), expected_elements * sizeof(float));
        if (!file) {
            throw std::runtime_error("Failed to read file: " + file_path);
        }
        return data;
    }

    // Helper to transpose a matrix from [rows, cols] to [cols, rows]
    std::vector<float> transpose_matrix(const std::vector<float>& matrix, int rows, int cols) {
        std::vector<float> transposed(matrix.size());
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposed[j * rows + i] = matrix[i * cols + j];
            }
        }
        return transposed;
    }

    void load_weights(const std::string& model_path) {
        std::string bin_weights_path = model_path + "/bin_weights";

        // Load embeddings
        weights_["model.embed_tokens.weight"] = load_binary_file(
            bin_weights_path + "/embed_tokens.bin",
            static_cast<size_t>(config_.vocab_size) * config_.hidden_size
        );

        // Load final norm
        weights_["model.norm.weight"] = load_binary_file(
            bin_weights_path + "/final_norm.bin",
            config_.hidden_size
        );

        // Load lm_head and transpose
        {
            auto loaded = load_binary_file(
                bin_weights_path + "/lm_head.bin",
                static_cast<size_t>(config_.vocab_size) * config_.hidden_size
            );
            weights_["lm_head.weight"] = transpose_matrix(loaded, config_.vocab_size, config_.hidden_size);
        }

        // Load all transformer layers
        for (int i = 0; i < config_.num_hidden_layers; i++) {
            std::string layer_prefix = "model.layers." + std::to_string(i) + ".";

            // Attention projections
            {
                auto loaded = load_binary_file(
                    bin_weights_path + "/layer" + std::to_string(i) + ".attn.q_proj.bin",
                    static_cast<size_t>(config_.hidden_size) * config_.hidden_size
                );
                weights_[layer_prefix + "self_attn.q_proj.weight"] = transpose_matrix(loaded, config_.hidden_size, config_.hidden_size);
            }

            {
                int kv_dim = config_.num_key_value_heads * config_.head_dim;
                auto loaded = load_binary_file(
                    bin_weights_path + "/layer" + std::to_string(i) + ".attn.k_proj.bin",
                    static_cast<size_t>(kv_dim) * config_.hidden_size
                );
                weights_[layer_prefix + "self_attn.k_proj.weight"] = transpose_matrix(loaded, kv_dim, config_.hidden_size);
            }

            {
                int kv_dim = config_.num_key_value_heads * config_.head_dim;
                auto loaded = load_binary_file(
                    bin_weights_path + "/layer" + std::to_string(i) + ".attn.v_proj.bin",
                    static_cast<size_t>(kv_dim) * config_.hidden_size
                );
                weights_[layer_prefix + "self_attn.v_proj.weight"] = transpose_matrix(loaded, kv_dim, config_.hidden_size);
            }

            {
                auto loaded = load_binary_file(
                    bin_weights_path + "/layer" + std::to_string(i) + ".attn.o_proj.bin",
                    static_cast<size_t>(config_.hidden_size) * config_.hidden_size
                );
                weights_[layer_prefix + "self_attn.o_proj.weight"] = transpose_matrix(loaded, config_.hidden_size, config_.hidden_size);
            }

            // Layer norms
            weights_[layer_prefix + "input_layernorm.weight"] = load_binary_file(
                bin_weights_path + "/layer" + std::to_string(i) + ".input_layernorm.bin",
                config_.hidden_size
            );

            weights_[layer_prefix + "post_attention_layernorm.weight"] = load_binary_file(
                bin_weights_path + "/layer" + std::to_string(i) + ".post_layernorm.bin",
                config_.hidden_size
            );

            // MLP projections
            {
                auto loaded = load_binary_file(
                    bin_weights_path + "/layer" + std::to_string(i) + ".mlp.gate_proj.bin",
                    static_cast<size_t>(config_.intermediate_size) * config_.hidden_size
                );
                weights_[layer_prefix + "mlp.gate_proj.weight"] = transpose_matrix(loaded, config_.intermediate_size, config_.hidden_size);
            }

            {
                auto loaded = load_binary_file(
                    bin_weights_path + "/layer" + std::to_string(i) + ".mlp.up_proj.bin",
                    static_cast<size_t>(config_.intermediate_size) * config_.hidden_size
                );
                weights_[layer_prefix + "mlp.up_proj.weight"] = transpose_matrix(loaded, config_.intermediate_size, config_.hidden_size);
            }

            {
                auto loaded = load_binary_file(
                    bin_weights_path + "/layer" + std::to_string(i) + ".mlp.down_proj.bin",
                    static_cast<size_t>(config_.hidden_size) * config_.intermediate_size
                );
                weights_[layer_prefix + "mlp.down_proj.weight"] = transpose_matrix(loaded, config_.hidden_size, config_.intermediate_size);
            }
        }
    }

    // Complete forward pass through all 28 layers
    std::vector<float> forward(const std::vector<int32_t>& input_ids, std::shared_ptr<KVCache> cache = nullptr) {
        int seq_len = input_ids.size();
        int position = cache ? cache->seq_length : 0;

        // 1. Embedding lookup
        std::vector<float> hidden(seq_len * config_.hidden_size);
        auto& embed = weights_["model.embed_tokens.weight"];

        for (int i = 0; i < seq_len; i++) {
            int token = input_ids[i];
            if (token >= 0 && token < config_.vocab_size) {
                memcpy(&hidden[i * config_.hidden_size], &embed[token * config_.hidden_size],
                       config_.hidden_size * sizeof(float));
            }
        }

        // 2. Process through all transformer layers
        for (int layer = 0; layer < config_.num_hidden_layers; layer++) {
            std::string p = "model.layers." + std::to_string(layer) + ".";

            // Input layernorm
            auto& input_norm_weight = weights_[p + "input_layernorm.weight"];
            auto hidden_normed = rmsnorm(hidden, input_norm_weight, config_.hidden_size, seq_len);

            // Self-attention
            auto& q_proj = weights_[p + "self_attn.q_proj.weight"];
            auto& k_proj = weights_[p + "self_attn.k_proj.weight"];
            auto& v_proj = weights_[p + "self_attn.v_proj.weight"];

            // Q: [seq_len, hidden_size] x [hidden_size, hidden_size] = [seq_len, hidden_size]
            auto q = matmul(hidden_normed, q_proj, seq_len, config_.hidden_size, config_.hidden_size);
            // K: [seq_len, hidden_size] x [hidden_size, kv_dim] = [seq_len, kv_dim]
            int kv_dim = config_.num_key_value_heads * config_.head_dim;
            auto k = matmul(hidden_normed, k_proj, seq_len, kv_dim, config_.hidden_size);
            // V: [seq_len, hidden_size] x [hidden_size, kv_dim] = [seq_len, kv_dim]
            auto v = matmul(hidden_normed, v_proj, seq_len, kv_dim, config_.hidden_size);

            // Apply RoPE to Q and K
            apply_rope(q, k, seq_len, config_.num_attention_heads, config_.head_dim, position);

            // Reshape Q for multi-head attention: [seq_len, num_heads, head_dim]
            // Reshape K/V for GQA: [seq_len, num_kv_heads, head_dim]

            // Attention scores: Q x K^T
            // For simplicity, compute attention using GQA (repeat K/V heads)
            std::vector<float> attn_out(seq_len * config_.hidden_size, 0.0f);

            int heads_per_kv_head = config_.num_attention_heads / config_.num_key_value_heads;

            for (int h = 0; h < config_.num_attention_heads; h++) {
                int kv_head = h / heads_per_kv_head;

                // Extract Q head: [seq_len, head_dim]
                std::vector<float> q_head(seq_len * config_.head_dim);
                for (int s = 0; s < seq_len; s++) {
                    memcpy(&q_head[s * config_.head_dim],
                           &q[(s * config_.num_attention_heads + h) * config_.head_dim],
                           config_.head_dim * sizeof(float));
                }

                // Extract K head: [seq_len, head_dim]
                std::vector<float> k_head(seq_len * config_.head_dim);
                for (int s = 0; s < seq_len; s++) {
                    memcpy(&k_head[s * config_.head_dim],
                           &k[(s * config_.num_key_value_heads + kv_head) * config_.head_dim],
                           config_.head_dim * sizeof(float));
                }

                // Extract V head: [seq_len, head_dim]
                std::vector<float> v_head(seq_len * config_.head_dim);
                for (int s = 0; s < seq_len; s++) {
                    memcpy(&v_head[s * config_.head_dim],
                           &v[(s * config_.num_key_value_heads + kv_head) * config_.head_dim],
                           config_.head_dim * sizeof(float));
                }

                // Compute attention scores: Q x K^T -> [seq_len, seq_len]
                auto scores = matmul(q_head, k_head, seq_len, seq_len, config_.head_dim);

                // Scale scores
                float scale = 1.0f / sqrt(config_.head_dim);
                for (auto& s : scores) s *= scale;

                // Apply softmax
                auto attn_weights = softmax(scores, seq_len);

                // Apply attention to V: [seq_len, seq_len] x [seq_len, head_dim] = [seq_len, head_dim]
                auto head_out = matmul(attn_weights, v_head, seq_len, config_.head_dim, seq_len);

                // Copy to output
                for (int s = 0; s < seq_len; s++) {
                    memcpy(&attn_out[(s * config_.num_attention_heads + h) * config_.head_dim],
                           &head_out[s * config_.head_dim],
                           config_.head_dim * sizeof(float));
                }
            }

            // Output projection
            auto& o_proj = weights_[p + "self_attn.o_proj.weight"];
            auto attn_output = matmul(attn_out, o_proj, seq_len, config_.hidden_size, config_.hidden_size);

            // Residual connection + input layernorm
            hidden = add(hidden, attn_output);

            // Post-attention layernorm
            auto& post_norm_weight = weights_[p + "post_attention_layernorm.weight"];
            auto post_normed = rmsnorm(hidden, post_norm_weight, config_.hidden_size, seq_len);

            // MLP
            auto& gate_proj = weights_[p + "mlp.gate_proj.weight"];
            auto& up_proj = weights_[p + "mlp.up_proj.weight"];
            auto& down_proj = weights_[p + "mlp.down_proj.weight"];

            auto gate = matmul(post_normed, gate_proj, seq_len, config_.intermediate_size, config_.hidden_size);
            auto up = matmul(post_normed, up_proj, seq_len, config_.intermediate_size, config_.hidden_size);

            // GeLU activation
            gate = gelu(gate);

            // Element-wise multiply
            for (size_t i = 0; i < gate.size(); i++) {
                gate[i] *= up[i];
            }

            // Down projection
            auto mlp_output = matmul(gate, down_proj, seq_len, config_.hidden_size, config_.intermediate_size);

            // Residual connection
            hidden = add(hidden, mlp_output);
        }

        // 3. Final normalization
        auto& final_norm_weight = weights_["model.norm.weight"];
        hidden = rmsnorm(hidden, final_norm_weight, config_.hidden_size, seq_len);

        // 4. Get last hidden state for language model head
        std::vector<float> last_hidden(hidden.end() - config_.hidden_size, hidden.end());

        // 5. LM head projection
        auto& lm_head = weights_["lm_head.weight"];
        auto logits = matmul(last_hidden, lm_head, 1, config_.vocab_size, config_.hidden_size);

        return logits;
    }

    const ModelConfig& GetConfig() const { return config_; }
};

} // namespace mlx_vllm

// C API implementation (outside namespace)
extern "C" {

int MLXLoadModel(const char* model_path, int vocab_size) {
    try {
        mlx_vllm::ModelConfig config;
        config.vocab_size = vocab_size;
        auto model = std::make_shared<mlx_vllm::Qwen2VLModel>(model_path, config);
        std::lock_guard<std::mutex> lock(mlx_vllm::g_model_mutex);
        mlx_vllm::g_model = model;
        return MLX_SUCCESS;
    } catch (...) {
        return MLX_ERROR_COMPUTATION_FAILED;
    }
}

int MLXForwardWithCache(uintptr_t model_handle, const uint32_t* tokens, int num_tokens,
                        uint64_t base_cache_handle, float* out_logits, int out_logits_size,
                        uint64_t* out_cache_handle, char** out_error) {
    try {
        std::lock_guard<std::mutex> lock(mlx_vllm::g_model_mutex);
        if (!mlx_vllm::g_model) {
            *out_error = strdup("Model not loaded");
            return MLX_ERROR_MODEL_NOT_LOADED;
        }

        const auto& config = mlx_vllm::g_model->GetConfig();
        if (num_tokens <= 0 || !tokens) return MLX_ERROR_INVALID_TOKENS;
        if (out_logits_size < config.vocab_size) return MLX_ERROR_OUT_OF_MEMORY;

        auto parent_cache = mlx_vllm::g_registry.Get(base_cache_handle);
        if (parent_cache) mlx_vllm::g_registry.Ref(base_cache_handle);

        std::vector<int32_t> input_ids(tokens, tokens + num_tokens);
        std::vector<float> logits = mlx_vllm::g_model->forward(input_ids, parent_cache);

        memcpy(out_logits, logits.data(), logits.size() * sizeof(float));

        auto new_cache = std::make_shared<mlx_vllm::KVCache>(
            0, std::vector<uint32_t>(tokens, tokens + num_tokens),
            parent_cache, parent_cache ? parent_cache->seq_length + num_tokens : num_tokens,
            config.num_hidden_layers);
        new_cache->logits = logits;

        *out_cache_handle = mlx_vllm::g_registry.Insert(new_cache);
        *out_error = nullptr;
        return MLX_SUCCESS;
    } catch (const std::exception& e) {
        *out_error = strdup(e.what());
        return MLX_ERROR_COMPUTATION_FAILED;
    }
}

int MLXSliceCache(uint64_t cache_handle, int keep_tokens, uint64_t* out_sliced_handle, char** out_error) {
    try {
        auto cache = mlx_vllm::g_registry.Get(cache_handle);
        if (!cache) return MLX_ERROR_INVALID_HANDLE;

        auto full_tokens = cache->GetFullTokenSequence();
        if (keep_tokens < 0 || keep_tokens > (int)full_tokens.size()) return MLX_ERROR_INVALID_TOKENS;

        std::vector<uint32_t> sliced(full_tokens.begin(), full_tokens.begin() + keep_tokens);
        auto sliced_cache = std::make_shared<mlx_vllm::KVCache>(0, sliced, cache, keep_tokens, 28);

        *out_sliced_handle = mlx_vllm::g_registry.Insert(sliced_cache);
        *out_error = nullptr;
        return MLX_SUCCESS;
    } catch (const std::exception& e) {
        *out_error = strdup(e.what());
        return MLX_ERROR_COMPUTATION_FAILED;
    }
}

void MLXFreeCache(uint64_t cache_handle) {
    if (cache_handle == 0) return;
    mlx_vllm::g_registry.Remove(cache_handle);
}

void MLXFreeError(char* error) {
    if (error) free(error);
}

} // extern "C"
