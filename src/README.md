# GUI-Actor Go Server

OpenAI-compatible API server for GUI-Actor models.

## Overview

A Go HTTP server providing an OpenAI-compatible API for GUI-Actor inference. Currently implements the HTTP API layer and infrastructure; MLX runtime integration is in progress.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Go HTTP Server                               │
│                       (localhost:8080)                               │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────────┐   │
│  │   HTTP       │  │    API       │  │     Model               │   │
│  │   Server     │→ │   Handler    │→ │     Registry            │   │
│  │  (gorilla/)  │  │  (OpenAI)    │  │  (memory-managed)       │   │
│  └──────────────┘  └──────────────┘  └─────────────────────────┘   │
│                                                  ↓                   │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                   MLX Runtime (via cgo)                         ││
│  │  ┌─────────────────────────────────────────────────────────┐   ││
│  │  │  Status: ⚠️ Placeholder implementation                  │   ││
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  │   ││
│  │  │  │ Vision       │  │ Qwen2-VL     │  │ Pointer     │  │   ││
│  │  │  │ Encoder      │→ │ 2B/7B LLM    │→ │ Head        │  │   ││
│  │  │  └──────────────┘  └──────────────┘  └─────────────┘  │   ││
│  │  │         ↓                  ↓                 ↓          │   ││
│  │  │    image_embeds      hidden_states    attn_scores     │   ││
│  │  └─────────────────────────────────────────────────────────┘   ││
│  │                          ↓                                     ││
│  │              Metal Performance Shaders                         ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

## Features

- **OpenAI-Compatible API**: Drop-in replacement for OpenAI chat completions API
- **Multi-Model Support**: GUI-Actor 2B and 7B models
- **Memory Management**: Dynamic model loading/unloading with LRU eviction
- **Image Preprocessing**: Smart resize with pixel constraints and grid alignment
- **Graceful Shutdown**: Clean model unloading on termination
- **Health Monitoring**: Built-in health check and status endpoints

## Quick Start

### Prerequisites

- Go 1.21+
- Apple Silicon (M1/M2/M3/M4)
- MLX framework (optional, currently uses placeholder)

### Installation

```bash
# Clone the repository
git clone git@github.com:agenthands/mlxvllm.git
cd mlxvllm/src

# Install dependencies
go mod download

# Run the server
go run cmd/server/main.go
```

### Configuration

Create `models/config.yaml`:

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
  fast:
    max_pixels: 1048576
  balanced:
    max_pixels: 5720064
  quality:
    max_pixels: 12845056
```

### Usage

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gui-actor-2b",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Click the submit button"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
      ]
    }]
  }'
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/v1/health` | Health check + uptime |
| GET | `/v1/models` | List available models |
| GET | `/v1/models/{id}` | Get model status |
| POST | `/v1/models/{id}/load` | Load model into memory |
| DELETE | `/v1/models/{id}` | Unload model |
| POST | `/v1/chat/completions` | Inference request |

## Project Structure

```
src/
├── cmd/server/              # Application entry point
│   ├── main.go              # Main with graceful shutdown
│   └── integration_test.go  # End-to-end tests
├── internal/
│   ├── api/                 # HTTP API layer
│   │   ├── handler.go       # Request handlers
│   │   ├── server.go        # HTTP server setup
│   │   └── types.go         # OpenAI-compatible types
│   ├── config/              # Configuration management
│   │   └── config.go        # YAML config loader
│   ├── mlx/                 # MLX cgo bindings
│   │   ├── mlx.h            # C API header
│   │   ├── mlx.c            # C implementation
│   │   ├── mlx.go           # cgo bindings
│   │   └── preprocessing.go # Image preprocessing
│   └── model/               # Model registry
│       └── registry.go      # Model loading/unloading
└── go.mod                   # Go module definition
```

## ONNX Model Export

The repository includes scripts to export GUI-Actor model components to ONNX format:

```bash
# Export vision tower and pointer head
python scripts/export_onnx.py microsoft/GUI-Actor-7B-Qwen2-VL
```

**Exported Components:**
- `onnx_models/vision_tower.onnx` - Vision encoder (ViT)
- `onnx_models/pointer_head.onnx` - Multi-patch pointer head (attention-based action head)

**Note:** The full Qwen2-VL LLM (7B parameters) requires separate export using `optimum-cli` due to memory considerations.

## MLX Interface

The MLX cgo bindings provide the following interface:

```c
// Context management
int mlx_init(void);
void mlx_shutdown(void);
int mlx_is_initialized(void);
int mlx_get_default_device(char* device, int max_len);

// Model loading
mlx_model_t mlx_load_model(const char* path, const char* device);
void mlx_unload_model(mlx_model_t model);

// Inference
mlx_array_t* mlx_forward(mlx_model_t model, mlx_array_t** inputs, int num_inputs);
void mlx_free_array(mlx_array_t* arr);
```

## Development

### Running Tests

```bash
# Run all tests
go test ./...

# Run specific package tests
go test ./internal/api/...
go test ./internal/mlx/...

# Run with coverage
go test -cover ./...
```

### Code Quality

```bash
# Format code
go fmt ./...

# Run linter
go vet ./...
```

## Implementation Status

### Completed ✅

| Component | Status | Notes |
|-----------|--------|-------|
| Configuration management | ✅ Complete | YAML config loader with profiles |
| HTTP server | ✅ Complete | gorilla/mux routing, graceful shutdown |
| OpenAI-compatible API | ✅ Complete | All endpoints implemented |
| Model registry | ✅ Complete | Memory-aware loading/unloading |
| Image preprocessing | ✅ Complete | Smart resize with 28px grid alignment |
| Integration tests | ✅ Complete | End-to-end test coverage |

### In Progress ⚠️

| Component | Status | Notes |
|-----------|--------|-------|
| MLX cgo bindings | ⚠️ Placeholder | `mlx.c` has stub implementation |
| Tokenizer | ⚠️ Not Started | SentencePiece wrapper needed |
| Inference pipeline | ⚠️ Not Started | Vision → LLM → Pointer head |
| Model loading | ⚠️ Not Started | Actual model weight loading |

### Architecture Notes

The server is designed to integrate with Apple's MLX framework for Metal-accelerated inference. The current implementation provides the complete HTTP API and infrastructure layer. The MLX runtime integration (`src/internal/mlx/`) requires:

1. Linking to MLX C API (`#include <mlx/mlx.h>`)
2. Implementing `mlx_load_model()` using mlx-vlm loader
3. Implementing `mlx_forward()` for inference
4. Adding SentencePiece tokenizer for tokenization

See [docs/plans/2026-02-26-mlx-go-server-design.md](../docs/plans/2026-02-26-mlx-go-server-design.md) for the full design specification.

## Roadmap

### Phase 1: MLX Integration (Next)
- [ ] Link to MLX C API (`#include <mlx/mlx.h>`)
- [ ] Implement `mlx_load_model()` using mlx-vlm
- [ ] Implement `mlx_forward()` for inference
- [ ] Add memory allocation for model weights

### Phase 2: Tokenizer & Preprocessing
- [ ] SentencePiece tokenizer wrapper
- [ ] Prompt template construction
- [ ] Special token handling (`<|pointer_start|>`, etc.)

### Phase 3: Full Inference Pipeline
- [ ] Vision tower forward pass
- [ ] Qwen2-VL LLM forward pass
- [ ] Pointer head attention scoring
- [ ] Connected component post-processing

### Phase 4: Production Features
- [ ] Streaming responses
- [ ] Model quantization (INT4/INT8)
- [ ] Metrics and observability
- [ ] Docker support for Linux deployments

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.
