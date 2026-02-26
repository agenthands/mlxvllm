# GUI-Actor Architecture Status

**Date:** 2026-02-26
**Repository:** github.com/agenthands/mlxvllm

## Overview

GUI-Actor is a Vision-Language Model (VLM) enhanced by an attention-based action head for coordinate-free GUI grounding. The repository contains:

1. **Python training/evaluation code** - Full implementation for model training and benchmark evaluation
2. **Go production server** - OpenAI-compatible API server (in progress)

## Python Implementation (Complete)

### Core Model (`src/gui_actor/`)

| File | Purpose |
|------|---------|
| `modeling.py` | `Qwen2VLForConditionalGenerationWithPointer` - main model with VisionHead_MultiPatch |
| `modeling_qwen25vl.py` | Qwen2.5-VL variant |
| `inference.py` | `ForceFollowTokensLogitsProcessor`, `inference()` function |
| `constants.py` | Special tokens: `<|pointer_start\|>`, `<|pointer_pad\|>`, `<|pointer_end\|>` |
| `trainer.py` | Training logic (warmup + full-parameter SFT) |
| `dataset.py` | Data loading for GUI grounding tasks |

### Training Scripts

| Script | Purpose |
|--------|---------|
| `scripts/warmup.sh` | Stage 1: Warmup (trains pointer head + new tokens only) |
| `scripts/train.sh` | Stage 2: Full-parameter training (SFT) |
| `scripts/zero3.json` | DeepSpeed ZeRO-3 configuration |

### Evaluation

| Script | Benchmark |
|--------|-----------|
| `eval/screenSpot.py` | ScreenSpot v1 |
| `eval/screenSpot_v2.py` | ScreenSpot v2 |
| `eval/screenSpot_pro.py` | ScreenSpot-Pro |

### Demo

| File | Purpose |
|------|---------|
| `demo/app.py` | Gradio web interface for interactive GUI grounding |

## Go Server Implementation (In Progress)

### Completed Components

#### HTTP API Layer (`src/internal/api/`)

```
handler.go       - OpenAI-compatible request handlers
server.go        - HTTP server with gorilla/mux routing
types.go         - Request/response type definitions
```

**Endpoints:**
- `GET /v1/health` - Health check + uptime
- `GET /v1/models` - List available models
- `GET /v1/models/{id}` - Get model status
- `POST /v1/models/{id}/load` - Load model into memory
- `DELETE /v1/models/{id}` - Unload model
- `POST /v1/chat/completions` - Inference request

#### Configuration (`src/internal/config/`)

```
config.go        - YAML config loader with server, model, profile configs
```

**Config Structure:**
```yaml
server:
  host: "127.0.0.1"
  port: 8080
  default_model: "gui-actor-2b"

models:
  gui-actor-2b:
    path: "./models/gui-actor-2b"
    enabled: true
    preload: true
    min_pixels: 3136
    max_pixels: 5720064
    max_context_length: 8192

profiles:
  fast: {max_pixels: 1048576}
  balanced: {max_pixels: 5720064}
  quality: {max_pixels: 12845056}
```

#### Model Registry (`src/internal/model/`)

```
registry.go      - Memory-aware model loading/unloading with LRU eviction
```

**Features:**
- Dynamic model loading
- Memory tracking (2B: ~4GB, 7B: ~14GB)
- LRU eviction when memory limit reached

#### Image Preprocessing (`src/internal/mlx/preprocessing.go`)

```
SmartResize()    - Resize to fit within [minPixels, maxPixels]
CalculateGrid()  - Return grid dimensions (28px patch size)
```

**Algorithm:**
1. Calculate scale factor to fit pixel constraints
2. Resize image maintaining aspect ratio
3. Align dimensions to 28px grid (MergePatchSize)
4. Return grid dimensions: `grid_w = W // 28`, `grid_h = H // 28`

### In Progress Components

#### MLX Runtime (`src/internal/mlx/`)

```
mlx.h           - C API header
mlx.c           - C implementation (placeholder)
mlx.go          - Go cgo bindings
```

**Current Status:** Placeholder implementation

**Required Integration:**
1. Link to MLX C API (`#include <mlx/mlx.h>`)
2. Implement model loading using mlx-vlm
3. Implement forward pass for inference

**C Interface:**
```c
int mlx_init(void);
void mlx_shutdown(void);
mlx_model_t mlx_load_model(const char* path, const char* device);
mlx_array_t* mlx_forward(mlx_model_t model, mlx_array_t** inputs, int num_inputs);
```

### Not Started Components

| Component | Description |
|-----------|-------------|
| Tokenizer | SentencePiece wrapper for Qwen2-VL tokenization |
| Inference Pipeline | Full pipeline: vision → LLM → pointer head → post-processing |
| Post-processing | Connected component analysis for coordinate extraction |

## Model Specifications

### GUI-Actor-2B

| Parameter | Value |
|-----------|-------|
| Backbone | Qwen2-VL-2B |
| hidden_size | 2048 |
| num_layers | 36 |
| num_attention_heads | 16 |
| vocab_size | 151936 |
| patch_size | 14 |
| merge_patch_size | 28 |
| min_pixels | 3136 (56×56) |
| max_pixels | 5720064 (~3192×1792) |
| max_context_length | 8192 tokens |
| Memory (est.) | ~4 GB |

### GUI-Actor-7B

| Parameter | Value |
|-----------|-------|
| Backbone | Qwen2-VL-7B |
| hidden_size | 3584 |
| num_layers | 28 |
| num_attention_heads | 28 |
| vocab_size | 151936 |
| patch_size | 14 |
| merge_patch_size | 28 |
| min_pixels | 3136 (56×56) |
| max_pixels | 12845056 (~7B supports higher) |
| max_context_length | 24576 tokens |
| Memory (est.) | ~14 GB |

## Inference Pipeline (Planned)

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. PREPROCESSING (Go)                                               │
│  - Decode base64 image                                               │
│  - Smart resize to [minPixels, maxPixels]                           │
│  - Align to 28px grid                                               │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  2. TOKENIZATION (Go + SentencePiece)                               │
│  - Construct prompt with special tokens                             │
│  - Tokenize text + image placeholder                                │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  3. MLX INFERENCE                                                   │
│  - Vision Tower: image → image_embeds                               │
│  - Qwen2-VL: tokens + image_embeds → hidden_states                 │
│  - Pointer Head: hidden_states → attn_scores[grid_h × grid_w]      │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  4. POST-PROCESSING (Go)                                            │
│  - Threshold: score > 0.3 × max_score                              │
│  - BFS: Group connected patches into regions                        │
│  - Return: Center of highest-scored region (x, y) in [0,1]         │
└─────────────────────────────────────────────────────────────────────┘
```

## Special Tokens

| Token | Purpose |
|-------|---------|
| `<|im_start|>`, `<|im_end|>` | Message boundaries |
| `<|vision_start|>`, `<|vision_end|>` | Image content boundaries |
| `<|image_pad|>` | Image placeholder in token sequence |
| `<|pointer_start|>` | Triggers pointer head activation |
| `<|pointer_pad|>` | Query token for pointer head |
| `<|pointer_end|>` | End marker for pointer sequence |

## Benchmarks (ScreenSpot-Pro with Qwen2-VL)

| Method | Parameters | Score |
|--------|------------|-------|
| UI-TARS-72B | 72B | 38.1 |
| GUI-Actor-7B | 7B | **40.7** |
| GUI-Actor-7B + Verifier | 7B | **44.2** |
| GUI-Actor-2B | 2B | **36.7** |
| GUI-Actor-2B + Verifier | 2B | **41.8** |

## References

- [Paper](https://arxiv.org/abs/2506.03143): GUI-Actor: Coordinate-Free Visual Grounding for GUI Agents
- [Models](https://huggingface.co/microsoft/GUI-Actor-7B-Qwen2-VL): Hugging Face model checkpoints
- [MLX](https://github.com/ml-explore/mlx): Apple Silicon ML framework
- [mlx-vlm](https://github.com/ml-explore/mlx-vlm): VLM support for MLX
