# MLX-VLLM Server

MLX-based Vision-Language Model server for GUI grounding with OpenAI-compatible API.

## Overview

A high-performance Go HTTP server that runs GUI-Actor models using Apple's MLX framework for native Metal acceleration on Apple Silicon. The server provides an OpenAI-compatible API for vision-language inference with coordinate-free GUI grounding.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Go HTTP Server                           │
│                   (localhost:8080)                           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Request   │  │  Handler    │  │    Inference        │ │
│  │   Handler   │→ │  (OpenAI)   │→ │    Pipeline         │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│                                              ↓              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              MLX Engine (via cgo)                       ││
│  │  ┌──────────┐  ┌──────────┐  ┌─────────────┐          ││
│  │  │ Vision   │  │ Qwen2-VL │  │  Pointer    │          ││
│  │  │ Encoder  │→ │  2B/7B   │→ │   Head      │ → coords ││
│  │  └──────────┘  └──────────┘  └─────────────┘          ││
│  │        ↑             ↑              ↑                   ││
│  │        └─────────────┴──────────────┘                   ││
│  │              Metal (via MLX)                             ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
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

## Current Status

- ✅ Configuration management with YAML
- ✅ MLX cgo bindings (placeholder implementation)
- ✅ Image preprocessing with smart resize
- ✅ Model registry with memory management
- ✅ OpenAI-compatible API handlers
- ✅ HTTP server with routing
- ⚠️ MLX inference integration (requires actual MLX models)
- ⚠️ Tokenizer integration (SentencePiece)

## Roadmap

1. Complete MLX model loading integration
2. Add SentencePiece tokenizer
3. Implement full inference pipeline
4. Add streaming support
5. Model quantization for faster inference
6. Docker support for Linux

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.
