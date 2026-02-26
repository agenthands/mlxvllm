# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**GUI-Actor** is a Vision-Language Model (VLM) enhanced by an attention-based action head (pointer head) for coordinate-free GUI grounding. Unlike traditional methods that generate text coordinates, GUI-Actor predicts interaction points by attending directly to visual regions.

### Key Technologies
- **Backbone Models:** Qwen2-VL and Qwen2.5-VL
- **Custom Architecture:** `Qwen2VLForConditionalGenerationWithPointer` adds a `VisionHead_MultiPatch` (pointer head) that computes attention scores over image patches
- **Grounding Verifier:** A secondary model that evaluates and selects the most plausible action region from candidate points

## Build and Development Commands

### Python Environment Setup
```bash
conda create -n gui_actor python=3.10
conda activate gui_actor
conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia
pip install -e .
```

### Training
Two-stage training process:
```bash
# Stage 1: Warmup (trains pointer head and new tokens only)
bash scripts/warmup.sh

# Stage 2: Full-parameter training (SFT)
bash scripts/train.sh
```

### Evaluation
```bash
# ScreenSpot benchmarks
python eval/screenSpot.py
python eval/screenSpot_v2.py
python eval/screenSpot_pro.py --save_path <path> --data_path <path>
```

### Demo
```bash
python demo/app.py
```

### Linting
```bash
ruff check .
ruff format .
```

### Go Server (Production Inference)

A Go HTTP server with OpenAI-compatible API for running GUI-Actor models:

```bash
# From repository root
cd src

# Install dependencies
go mod download

# Run server
go run cmd/server/main.go -config ./models/config.yaml

# Run tests
go test ./src/cmd/server/...
go test ./src/internal/api/...
go test ./src/internal/mlx/...
go test ./src/internal/model/...
go test ./src/internal/config/...
```

**Current Implementation Status:**
- ✅ Configuration management (YAML config loader)
- ✅ HTTP server with OpenAI-compatible endpoints
- ✅ Model registry with memory management
- ✅ Image preprocessing (smart resize with grid alignment)
- ⚠️ MLX runtime integration (placeholder - requires linking to MLX C API)
- ⚠️ Tokenizer integration (SentencePiece wrapper needed)
- ⚠️ Full inference pipeline (vision tower → LLM → pointer head)

## Architecture

### Core Components

```
Python Package:
src/gui_actor/           # Python package
  modeling.py            # VisionHead_MultiPatch + Qwen2VLForConditionalGenerationWithPointer
  modeling_qwen25vl.py   # Qwen2.5-VL variant
  inference.py           # ForceFollowTokensLogitsProcessor + inference()
  constants.py           # Special tokens (<|pointer_start|>, <|pointer_pad|>, <|pointer_end|>)
  trainer.py             # Training logic
  dataset.py             # Data loading

Go Server (OpenAI-Compatible API):
src/cmd/server/          # Go HTTP server entry point
  main.go                # Server with graceful shutdown
  integration_test.go    # End-to-end integration tests
src/internal/api/        # HTTP API layer
  handler.go             # OpenAI-compatible request handlers
  server.go              # HTTP server setup with gorilla/mux
  types.go               # Request/response type definitions
src/internal/config/     # Configuration management
  config.go              # YAML config loader
src/internal/model/      # Model registry
  registry.go            # Model loading/unloading with memory management
src/internal/mlx/        # MLX runtime bindings (via cgo)
  mlx.go                 # Go bindings for MLX C API
  mlx.h / mlx.c          # C bridge layer (placeholder for MLX linking)
  preprocessing.go       # Image smart resize with grid alignment

Other:
verifier/                # Grounding verifier model
  verifier_model.py      # Verifier architecture
  eval_ss_with_verifier.py
train.py                 # Main training entry point
scripts/                 # Shell scripts + DeepSpeed configs (zero3.json)
eval/                    # Benchmark evaluation scripts
demo/                    # Gradio web demo
data/                    # data_config.yaml for training datasets
onnx_models/             # Exported ONNX models (vision_tower, pointer_head)
```

### Model Flow

1. **Vision Tower**: Processes screenshot → visual embeddings
2. **LLM Backbone**: Generates text response with special pointer tokens
3. **Pointer Head**: Cross-attention between visual patches and query hidden states → attention scores → click coordinates
4. **Verifier** (optional): Takes candidate point + instruction → validates correctness

### Special Tokens
The model uses placeholder tokens to trigger the pointer head during generation:
- `<|pointer_start|>` - Triggers pointer head activation
- `<|pointer_pad|>` - Padding token
- `<|pointer_end|>` - End marker

The `ForceFollowTokensLogitsProcessor` ensures these tokens appear in the correct sequence after the model generates `<|pointer_start|>`.

## Production Deployment (Apple Silicon)

The Go server (`src/cmd/server/main.go`) provides an OpenAI-compatible API:

**Server Features:**
- OpenAI-compatible `/v1/chat/completions` endpoint
- Multi-model support (GUI-Actor-2B, GUI-Actor-7B)
- Memory-aware model management with LRU eviction
- Health monitoring and status endpoints
- Graceful shutdown with model cleanup

**Client Usage:**
```go
// Using OpenAI Go SDK
client := openai.NewClient("http://localhost:8080/v1")
resp, err := client.CreateChatCompletion(ctx, req)
```

**Configuration (`src/models/config.yaml`):**
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
    min_pixels: 3136      # 56x56 minimum
    max_pixels: 5720064   # ~3192x1792
    max_context_length: 8192
  
  gui-actor-7b:
    path: "./models/gui-actor-7b"
    enabled: true
    preload: false
    min_pixels: 3136
    max_pixels: 12845056  # 7B supports higher resolution
    max_context_length: 24576

profiles:
  fast:
    max_pixels: 1048576    # ~1024x1024, low latency
  balanced:
    max_pixels: 5720064    # Default
  quality:
    max_pixels: 12845056   # Maximum accuracy
```

## Training Data Configuration

Datasets are configured in `data/data_config.yaml`. Each entry specifies:
- `json_path`: Path to annotation JSON
- `images_folder`: Path to images directory
- `sampling_strategy`: Typically "all"

Training uses multiple datasets including UGround, GUIEnv, GUIAct, AMEX, AndroidControl, and Wave-UI.

## Verifier Training and Evaluation

The grounding verifier is trained to validate whether a candidate click position matches an instruction:
1. Prepare data from OS-Atlas dataset using `verifier/verifier_data_generation.py`
2. Fine-tune using UITARS-2B-SFT as base model
3. Evaluate with `verifier/run_ss_v1.sh`, `run_ss_v2.sh`, `run_ss_pro.sh`

## Code Conventions

- **Python**: Uses ruff for linting (config in `pyproject.toml`), numpy-style docstrings
- **Go**: Standard Go testing with `testing` package
- **Model modifications**: Extend `src/gui_actor/modeling.py` or `modeling_qwen25vl.py`
- **New datasets**: Add entries to `data/data_config.yaml`

## Test Coverage Requirements

**MANDATORY: 100% test coverage for ALL new Go code.**

### Current Coverage Status (2026-02-26)

| Package | Coverage | Status |
|---------|----------|--------|
| `cmd/server` | 0.0% | ❌ Needs tests |
| `internal/api` | 31.6% | ⚠️ Needs more tests |
| `internal/config` | 71.4% | ⚠️ Needs more tests |
| `internal/mlx` | 65.1% | ⚠️ Needs more tests |
| `internal/model` | 44.9% | ⚠️ Needs more tests |
| **Overall** | **41.2%** | ❌ Below target |

### Test Coverage Commands

```bash
# From src directory
cd src

# Run tests with coverage
go test ./... -coverprofile=coverage.out -covermode=atomic

# View overall coverage
go tool cover -func=coverage.out | grep total

# View coverage by package
go test ./... -cover

# Generate HTML coverage report
go tool cover -html=coverage.out -o coverage.html
open coverage.html  # macOS
```

### Test Coverage Rules

1. **Every new feature MUST have 100% test coverage**
2. **Use table-driven tests for multiple test cases**
3. **Test both success and error paths**
4. **Mock external dependencies (MLX, file system, network)**
5. **Run coverage check before committing**

### Test File Naming

- Test files MUST be named `*_test.go`
- Test files MUST be in the same package as the code they test
- Integration tests go in `integration_test.go`

### Pre-commit Checklist

Before committing any Go code changes:
- [ ] All tests pass: `go test ./...`
- [ ] Coverage is 100% for modified packages
- [ ] Code is formatted: `go fmt ./...`
- [ ] No vet warnings: `go vet ./...`

### Example: Table-Driven Test

```go
func TestCalculateGrid(t *testing.T) {
    tests := []struct {
        name      string
        w, h      int
        expectGW  int
        expectGH  int
    }{
        {"112x224", 112, 224, 4, 8},
        {"224x224", 224, 224, 8, 8},
        {"56x56", 56, 56, 2, 2},
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            gw, gh := CalculateGrid(tt.w, tt.h)
            if gw != tt.expectGW || gh != tt.expectGH {
                t.Errorf("CalculateGrid(%d,%d) = (%d,%d), want (%d,%d)",
                    tt.w, tt.h, gw, gh, tt.expectGW, tt.expectGH)
            }
        })
    }
}
```
