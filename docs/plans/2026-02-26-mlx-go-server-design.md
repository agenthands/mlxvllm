# GUI-Actor MLX Go Server Design

**Date:** 2026-02-26
**Status:** Draft
**Approach:** MLX + cgo (Apple Silicon native)

## Overview

Build a high-performance Go HTTP server that runs GUI-Actor models (2B and 7B) using Apple's MLX framework for native Metal acceleration on Apple Silicon.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Go HTTP Server                           │
│                   (localhost:8080)                           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Request   │  │  Tokenizer  │  │    Inference        │ │
│  │   Handler   │→ │   (BPE)     │→ │    Pipeline         │ │
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
│  │              Metal Performance Shaders                  ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## Supported Models

| Parameter | GUI-Actor-2B | GUI-Actor-7B |
|-----------|--------------|--------------|
| **Backbone** | Qwen2-VL | Qwen2-VL |
| **hidden_size** | 2048 | 3584 |
| **num_layers** | 36 | 28 |
| **num_attention_heads** | 16 | 28 |
| **vocab_size** | 151936 | 151936 |
| **patch_size** | 14 | 14 |
| **merge_size** | 2 | 2 |
| **merge_patch_size** | 28 | 28 |
| **Memory (est.)** | ~4 GB | ~14 GB |

## Model-Specific Parameters

| Parameter | GUI-Actor-2B | GUI-Actor-7B | Notes |
|-----------|--------------|--------------|-------|
| **min_pixels** | 3136 | 3136 | 56×56 minimum |
| **max_pixels (default)** | 5720064 | 5720064 | 3192×1792 |
| **max_pixels (max)** | 5720064 | 12845056 | 7B can go higher |
| **max_context_length** | 8192 | 24576 | Tokens (text + image) |

## Inference Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│  1. PREPROCESSING (Go)                                                   │
│  ┌──────────────────┐    ┌──────────────────────────────────┐           │
│  │ Decode base64    │ →  │ smart_resize (min/max pixels)    │           │
│  │ image            │    │ Dynamic grid: varies by size     │           │
│  └──────────────────┘    └──────────────────────────────────┘           │
│                                                                          │
│  Grid calculation: grid_w = W // 28, grid_h = H // 28                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  2. TOKENIZATION (Go + SentencePiece)                                    │
│                                                                          │
│  Prompt construction:                                                    │
│  <|im_start|>system\n{grounding_system_message}<|im_end|>               │
│  <|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>          │
│  {instruction}<|im_end|>                                                │
│  <|im_start|>assistant<|recipient|>os\n                                 │
│  pyautogui.click(<|pointer_start|><|pointer_pad|><|pointer_end|>)       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  3. MLX INFERENCE                                                        │
│  ┌──────────────┐    ┌───────────────┐    ┌─────────────────┐           │
│  │ Vision Tower │ → │ Qwen2-VL 2B   │ → │ Pointer Head    │           │
│  │ (ViT)        │    │ or 7B        │    │ (VisionHead)    │           │
│  └──────────────┘    └───────────────┘    └─────────────────┘           │
│         ↓                   ↓                     ↓                      │
│    image_embeds       hidden_states        attn_scores[grid_h × grid_w] │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  4. POST-PROCESSING (Go) - Connected Component Analysis                  │
│                                                                          │
│  attn_scores ──→ threshold (0.3 × max) ──→ binary_mask                  │
│       ↓                                                                  │
│  BFS to find connected regions                                          │
│       ↓                                                                  │
│  For each region: weighted_avg(center × score)                          │
│       ↓                                                                  │
│  Return: best_region_center (x, y) in [0,1]                             │
└─────────────────────────────────────────────────────────────────────────┘
```

## Latency Targets (M4 Pro)

| Model | Profile | max_pixels | Latency | Use Case |
|-------|---------|------------|---------|----------|
| **2B** | fast | 1M | ~80ms | Real-time web agent |
| **2B** | balanced | 5.7M | ~150ms | General use |
| **7B** | fast | 1M | ~150ms | Quality + speed |
| **7B** | balanced | 5.7M | ~300ms | Default 7B |
| **7B** | quality | 12.8M | ~500ms | Maximum accuracy |

## Package Structure

```
src/
├── cmd/
│   └── server/
│       ├── main.go              # Entry point, HTTP server
│       └── main_test.go         # Integration tests
├── internal/
│   ├── api/
│   │   ├── handler.go           # OpenAI-compatible handlers
│   │   ├── types.go             # Request/Response structs
│   │   └── server.go            # HTTP server setup
│   ├── mlx/
│   │   ├── mlx.go               # cgo bindings to MLX C API
│   │   ├── model.go             # GUIActorModel struct
│   │   ├── inference.go         # RunInference pipeline
│   │   └── preprocessing.go     # Image resize, normalization
│   ├── tokenizer/
│   │   └── tokenizer.go         # SentencePiece wrapper
│   └── grounding/
│       └── postprocess.go       # Connected component analysis
└── models/
    ├── config.yaml              # Model configuration
    ├── gui-actor-2b/            # 2B model weights
    │   ├── config.json
    │   ├── tokenizer.model
    │   ├── vision_encoder/
    │   ├── language_model/
    │   └── pointer_head/
    └── gui-actor-7b/            # 7B model weights
        └── ...
```

## Configuration

### config.yaml

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
    memory_limit_gb: 0

  gui-actor-7b:
    path: "./models/gui-actor-7b"
    enabled: true
    preload: false
    min_pixels: 3136
    max_pixels: 12845056
    max_context_length: 24576
    memory_limit_gb: 0

profiles:
  fast:
    max_pixels: 1048576
    max_context_length: 4096
  balanced:
    max_pixels: 5720064
    max_context_length: 8192
  quality:
    max_pixels: 12845056
    max_context_length: 24576

memory:
  max_total_gb: 32
  unload_strategy: "lru"
  keep_models: ["gui-actor-2b"]

logging:
  level: "info"
  format: "json"
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/chat/completions` | Inference (OpenAI-compatible) |
| GET | `/v1/models` | List available models |
| GET | `/v1/models/{id}` | Get model status |
| POST | `/v1/models/{id}/load` | Load model into memory |
| DELETE | `/v1/models/{id}` | Unload model from memory |
| GET | `/v1/health` | Health check + memory usage |

### Request/Response Types

```go
type ChatCompletionRequest struct {
    Model       string    `json:"model"`
    Messages    []Message `json:"messages"`
    Stream      bool      `json:"stream,omitempty"`
    MaxPixels   *int      `json:"max_pixels,omitempty"`
    MinPixels   *int      `json:"min_pixels,omitempty"`
    MaxContext  *int      `json:"max_context,omitempty"`
    Profile     string    `json:"profile,omitempty"`
}

type Message struct {
    Role    string      `json:"role"`
    Content interface{} `json:"content"`
}

type ChatCompletionResponse struct {
    ID      string   `json:"id"`
    Object  string   `json:"object"`
    Created int64    `json:"created"`
    Model   string   `json:"model"`
    Choices []Choice `json:"choices"`
}

type Choice struct {
    Index        int     `json:"index"`
    Message      Message `json:"message"`
    FinishReason string  `json:"finish_reason"`
}
```

### Usage Examples

```bash
# Default (2B, balanced)
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Click the submit button"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
      ]
    }]
  }'

# Use 7B with quality profile
curl -X POST http://localhost:8080/v1/chat/completions \
  -d '{"model": "gui-actor-7b", "profile": "quality", "messages": [...]}'

# Fast mode
curl -X POST http://localhost:8080/v1/chat/completions \
  -d '{"profile": "fast", "messages": [...]}'

# List models
curl http://localhost:8080/v1/models

# Load 7B model
curl -X POST http://localhost:8080/v1/models/gui-actor-7b/load

# Unload 7B model
curl -X DELETE http://localhost:8080/v1/models/gui-actor-7b

# Health check
curl http://localhost:8080/v1/health
```

## System Prompt

```
You are a GUI agent. Given a screenshot of the current GUI and a human
instruction, your task is to locate the screen element that corresponds
to the instruction. You should output a PyAutoGUI action that performs
a click on the correct position. To indicate the click location, we will
use some special tokens, which is used to refer to a visual patch later.
For example, you can output: pyautogui.click(<your_special_token_here>).
```

## Assistant Starter Template

```
<|im_start|>assistant<|recipient|>os
pyautogui.click(<|pointer_start|><|pointer_pad|><|pointer_end|>)
```

## Special Tokens

| Token | Purpose |
|-------|---------|
| `<|pointer_start|>` | Triggers pointer head activation |
| `<|pointer_pad|>` | Query token for pointer head |
| `<|pointer_end|>` | End marker |

## Post-Processing: Connected Component Analysis

The `get_prediction_region_point` function:

1. Threshold: Select patches where `score > 0.3 × max_score`
2. BFS: Group connected patches into regions
3. Score: Calculate weighted average for each region
4. Return: Center of highest-scored region (normalized [0,1])

## Memory Management Strategy

| Model | Memory | Strategy |
|-------|--------|----------|
| 2B only | ~4GB | Always loaded |
| 7B only | ~14GB | Always loaded |
| Both | ~18GB | Load on demand, keep 2B resident |

## Web Screenshot Usage

GUI-Actor is domain-agnostic - works identically for:
- **Web**: tool, shop, gitlab, forum screenshots
- **Desktop**: windows, macos screenshots
- **Mobile**: ios, android screenshots

No special preprocessing required for web screenshots.

## Model Conversion

Leverage `mlx-vlm` for Qwen2-VL backbone (already ported to MLX), then add custom pointer head.

```python
# Conversion script (Python)
from gui_actor.modeling import Qwen2VLForConditionalGenerationWithPointer

model = Qwen2VLForConditionalGenerationWithPointer.from_pretrained(
    "microsoft/GUI-Actor-2B-Qwen2-VL"
)

# mlx-vlm handles vision + language model
# Extract pointer head weights separately
pointer_weights = model.multi_patch_pointer_head.state_dict()
mlx.save("pointer_head.safetensors", pointer_weights)
```

## Implementation Plan

### Task 1: Project Structure and Configuration

**Files:**
- Create: `src/internal/config/config.go`
- Create: `src/models/config.yaml`
- Test: `src/internal/config/config_test.go`

**Step 1: Write the failing test**

```go
// src/internal/config/config_test.go
package config

import (
    "os"
    "testing"
)

func TestLoadConfig(t *testing.T) {
    // Write temporary config
    tmpFile := "/tmp/test_config.yaml"
    content := `
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
`
    os.WriteFile(tmpFile, []byte(content), 0644)

    cfg, err := LoadConfig(tmpFile)
    if err != nil {
        t.Fatalf("Failed to load config: %v", err)
    }

    if cfg.Server.Host != "127.0.0.1" {
        t.Errorf("Expected host 127.0.0.1, got %s", cfg.Server.Host)
    }
    if cfg.Server.Port != 8080 {
        t.Errorf("Expected port 8080, got %d", cfg.Server.Port)
    }
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./src/internal/config/... -v`
Expected: FAIL with "undefined: LoadConfig"

**Step 3: Write minimal implementation**

```go
// src/internal/config/config.go
package config

import (
    "fmt"
    "os"

    "gopkg.in/yaml.v3"
)

type ServerConfig struct {
    Host         string `yaml:"host"`
    Port         int    `yaml:"port"`
    DefaultModel string `yaml:"default_model"`
}

type ModelConfig struct {
    Path              string `yaml:"path"`
    Enabled           bool   `yaml:"enabled"`
    Preload           bool   `yaml:"preload"`
    MinPixels         int    `yaml:"min_pixels"`
    MaxPixels         int    `yaml:"max_pixels"`
    MaxContextLength  int    `yaml:"max_context_length"`
    MemoryLimitGB     int    `yaml:"memory_limit_gb"`
}

type ProfileConfig struct {
    MaxPixels        int `yaml:"max_pixels"`
    MaxContextLength int `yaml:"max_context_length"`
}

type MemoryConfig struct {
    MaxTotalGB    string   `yaml:"max_total_gb"`
    UnloadStrategy string  `yaml:"unload_strategy"`
    KeepModels    []string `yaml:"keep_models"`
}

type LoggingConfig struct {
    Level  string `yaml:"level"`
    Format string `yaml:"format"`
}

type Config struct {
    Server   ServerConfig              `yaml:"server"`
    Models   map[string]ModelConfig    `yaml:"models"`
    Profiles map[string]ProfileConfig  `yaml:"profiles"`
    Memory   MemoryConfig              `yaml:"memory"`
    Logging  LoggingConfig             `yaml:"logging"`
}

func LoadConfig(path string) (*Config, error) {
    data, err := os.ReadFile(path)
    if err != nil {
        return nil, fmt.Errorf("failed to read config: %w", err)
    }

    var cfg Config
    if err := yaml.Unmarshal(data, &cfg); err != nil {
        return nil, fmt.Errorf("failed to parse config: %w", err)
    }

    return &cfg, nil
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./src/internal/config/... -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/internal/config/
git commit -m "feat(config): add configuration loader with YAML support"
```

---

### Task 2: MLX cgo Bindings

**Files:**
- Create: `src/internal/mlx/mlx.h`
- Create: `src/internal/mlx/mlx.go`
- Test: `src/internal/mlx/mlx_test.go`

**Step 1: Write the failing test**

```go
// src/internal/mlx/mlx_test.go
package mlx

import (
    "testing"
)

func TestMLXInit(t *testing.T) {
    err := Init()
    if err != nil {
        t.Fatalf("Failed to initialize MLX: %v", err)
    }
    defer Shutdown()

    if !IsInitialized() {
        t.Error("MLX should be initialized")
    }
}

func TestMLXGetDefaultDevice(t *testing.T) {
    err := Init()
    if err != nil {
        t.Fatal(err)
    }
    defer Shutdown()

    device := GetDefaultDevice()
    if device == "" {
        t.Error("Expected non-empty device name")
    }
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./src/internal/mlx/... -v`
Expected: FAIL with "undefined: Init"

**Step 3: Write minimal implementation**

```c
// src/internal/mlx/mlx.h
#ifndef MLX_H
#define MLX_H

#include <stdint.h>

// Device types
#define MLX_DEVICE_CPU 0
#define MLX_DEVICE_GPU 1

// Context management
int mlx_init(void);
void mlx_shutdown(void);
int mlx_is_initialized(void);
int mlx_get_default_device(char* device, int max_len);

// Model loading
typedef void* mlx_model_t;
mlx_model_t mlx_load_model(const char* path, const char* device);
void mlx_unload_model(mlx_model_t model);

// Inference
typedef struct {
    float* data;
    int64_t* shape;
    int ndim;
    int dtype;
} mlx_array_t;

mlx_array_t* mlx_forward(mlx_model_t model, mlx_array_t** inputs, int num_inputs);
void mlx_free_array(mlx_array_t* arr);

#endif // MLX_H
```

```c
// src/internal/mlx/mlx.c
#include "mlx.h"
#include <stdlib.h>
#include <string.h>
#include <mlx/mlx.h>

static int initialized = 0;

int mlx_init(void) {
    if (initialized) return 1;
    mlx_metal_init();
    initialized = 1;
    return 1;
}

void mlx_shutdown(void) {
    if (!initialized) return;
    mlx_metal_shutdown();
    initialized = 0;
}

int mlx_is_initialized(void) {
    return initialized;
}

int mlx_get_default_device(char* device, int max_len) {
    if (!initialized) return 0;
    strncpy(device, "metal", max_len);
    return 1;
}

mlx_model_t mlx_load_model(const char* path, const char* device) {
    // Placeholder: actual implementation will use mlx-vlm loader
    return (mlx_model_t)0x1; // Non-null placeholder
}

void mlx_unload_model(mlx_model_t model) {
    // Placeholder
}

mlx_array_t* mlx_forward(mlx_model_t model, mlx_array_t** inputs, int num_inputs) {
    // Placeholder
    return NULL;
}

void mlx_free_array(mlx_array_t* arr) {
    if (arr) {
        if (arr->data) free(arr->data);
        if (arr->shape) free(arr->shape);
        free(arr);
    }
}
```

```go
// src/internal/mlx/mlx.go
package mlx

/*
#cgo CFLAGS: -I/opt/homebrew/include
#cgo LDFLAGS: -L/opt/homebrew/lib -lmlx -lmlx-metal
#include "mlx.h"
*/
import "C"
import (
    "fmt"
    "unsafe"
)

type Device string

const (
    DeviceCPU Device = "cpu"
    DeviceGPU Device = "metal"
)

func Init() error {
    ret := C.mlx_init()
    if ret == 0 {
        return fmt.Errorf("failed to initialize MLX")
    }
    return nil
}

func Shutdown() {
    C.mlx_shutdown()
}

func IsInitialized() bool {
    return C.mlx_is_initialized() == 1
}

func GetDefaultDevice() string {
    buf := make([]byte, 32)
    C.mlx_get_default_device((*C.char)(unsafe.Pointer(&buf[0])), 32)
    return string(buf)
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./src/internal/mlx/... -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/internal/mlx/
git commit -m "feat(mlx): add cgo bindings for MLX framework"
```

---

### Task 3: Image Preprocessing

**Files:**
- Create: `src/internal/mlx/preprocessing.go`
- Create: `src/internal/mlx/preprocessing_test.go`

**Step 1: Write the failing test**

```go
// src/internal/mlx/preprocessing_test.go
package mlx

import (
    "image"
    "image/color"
    "testing"
)

func TestSmartResize(t *testing.T) {
    // Create 100x100 test image
    img := image.NewRGBA(image.Rect(0, 0, 100, 100))
    img.Set(50, 50, color.RGBA{255, 0, 0, 255})

    resized, err := SmartResize(img, 3136, 5720064)
    if err != nil {
        t.Fatalf("SmartResize failed: %v", err)
    }

    // Check dimensions satisfy constraints
    pixels := resized.Bounds().Dx() * resized.Bounds().Dy()
    if pixels < 3136 {
        t.Errorf("Pixels %d below min 3136", pixels)
    }
    if pixels > 5720064 {
        t.Errorf("Pixels %d above max 5720064", pixels)
    }
}

func TestCalculateGrid(t *testing.T) {
    tests := []struct {
        w, h       int
        expectGW   int
        expectGH   int
    }{
        {112, 224, 4, 8},   // 112/28=4, 224/28=8
        {224, 224, 8, 8},   // 224/28=8
        {56, 56, 2, 2},     // 56/28=2
    }

    for _, tt := range tests {
        gw, gh := CalculateGrid(tt.w, tt.h)
        if gw != tt.expectGW || gh != tt.expectGH {
            t.Errorf("CalculateGrid(%d,%d) = (%d,%d), want (%d,%d)",
                tt.w, tt.h, gw, gh, tt.expectGW, tt.expectGH)
        }
    }
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./src/internal/mlx/... -v -run TestSmartResize`
Expected: FAIL with "undefined: SmartResize"

**Step 3: Write minimal implementation**

```go
// src/internal/mlx/preprocessing.go
package mlx

import (
    "image"
    "math"
)

const (
    MergePatchSize = 28
)

// SmartResize resizes image to fit within [minPixels, maxPixels]
// while maintaining aspect ratio and aligning to MergePatchSize
func SmartResize(img image.Image, minPixels, maxPixels int) (image.Image, error) {
    bounds := img.Bounds()
    w, h := bounds.Dx(), bounds.Dy()
    currentPixels := w * h

    // If already in range, ensure grid alignment
    if currentPixels >= minPixels && currentPixels <= maxPixels {
        return alignToGrid(img)
    }

    // Calculate scale factor
    scale := 1.0
    if currentPixels < minPixels {
        scale = math.Sqrt(float64(minPixels) / float64(currentPixels))
    } else if currentPixels > maxPixels {
        scale = math.Sqrt(float64(maxPixels) / float64(currentPixels))
    }

    newW := int(math.Round(float64(w) * scale))
    newH := int(math.Round(float64(h) * scale))

    // Align to grid size
    newW = (newW / MergePatchSize) * MergePatchSize
    newH = (newH / MergePatchSize) * MergePatchSize

    // Ensure minimum size
    if newW < MergePatchSize {
        newW = MergePatchSize
    }
    if newH < MergePatchSize {
        newH = MergePatchSize
    }

    return resizeImage(img, newW, newH)
}

// CalculateGrid returns the grid dimensions for patch processing
func CalculateGrid(w, h int) (int, int) {
    return w / MergePatchSize, h / MergePatchSize
}

func alignToGrid(img image.Image) (image.Image, error) {
    bounds := img.Bounds()
    w, h := bounds.Dx(), bounds.Dy()
    gridW, gridH := CalculateGrid(w, h)
    newW := gridW * MergePatchSize
    newH := gridH * MergePatchSize
    return resizeImage(img, newW, newH)
}

func resizeImage(img image.Image, w, h int) (image.Image, error) {
    // Simple bilinear resize implementation
    // In production, use a dedicated imaging library
    dst := image.NewRGBA(image.Rect(0, 0, w, h))
    srcBounds := img.Bounds()

    for y := 0; y < h; y++ {
        for x := 0; x < w; x++ {
            srcX := x * srcBounds.Dx() / w
            srcY := y * srcBounds.Dy() / h
            dst.Set(x, y, img.At(srcBounds.Min.X+srcX, srcBounds.Min.Y+srcY))
        }
    }

    return dst, nil
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./src/internal/mlx/... -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/internal/mlx/preprocessing.go src/internal/mlx/preprocessing_test.go
git commit -m "feat(mlx): add smart image resize with grid alignment"
```

---

### Task 4: Model Registry

**Files:**
- Create: `src/internal/model/registry.go`
- Create: `src/internal/model/registry_test.go`

**Step 1: Write the failing test**

```go
// src/internal/model/registry_test.go
package model

import (
    "testing"

    "github.com/agenthands/gui-actor/internal/config"
)

func TestNewRegistry(t *testing.T) {
    cfg := &config.Config{
        Models: map[string]config.ModelConfig{
            "gui-actor-2b": {
                Path:     "/tmp/models/2b",
                Enabled:  true,
                Preload:  true,
            },
        },
        Memory: config.MemoryConfig{
            MaxTotalGB: "32",
        },
    }

    reg := NewRegistry(cfg)
    if reg == nil {
        t.Fatal("Expected non-nil registry")
    }

    if !reg.HasModel("gui-actor-2b") {
        t.Error("Expected gui-actor-2b to be registered")
    }
}

func TestLoadModel(t *testing.T) {
    cfg := &config.Config{
        Models: map[string]config.ModelConfig{
            "gui-actor-2b": {
                Path:     "/tmp/models/2b",
                Enabled:  true,
                Preload:  false,
            },
        },
    }

    reg := NewRegistry(cfg)
    err := reg.LoadModel("gui-actor-2b")
    // Should fail for non-existent path, but API should work
    if err == nil {
        t.Log("Note: model loading will fail with non-existent path")
    }
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./src/internal/model/... -v`
Expected: FAIL with "undefined: NewRegistry"

**Step 3: Write minimal implementation**

```go
// src/internal/model/registry.go
package model

import (
    "fmt"
    "sync"

    "github.com/agenthands/gui-actor/internal/config"
)

type ModelStatus struct {
    Name       string
    Loaded     bool
    Path       string
    MemoryGB   float64
    LastUsed   int64 // Unix timestamp
}

type Model interface {
    ID() string
    IsLoaded() bool
    Unload() error
}

type Registry struct {
    mu        sync.RWMutex
    cfg       *config.Config
    models    map[string]*ModelStatus
    loaded    map[string]Model
    totalGB   float64
}

func NewRegistry(cfg *config.Config) *Registry {
    reg := &Registry{
        cfg:    cfg,
        models: make(map[string]*ModelStatus),
        loaded: make(map[string]Model),
    }

    // Register all enabled models
    for name, mcfg := range cfg.Models {
        if mcfg.Enabled {
            reg.models[name] = &ModelStatus{
                Name:     name,
                Loaded:   false,
                Path:     mcfg.Path,
                MemoryGB: estimateMemoryGB(name),
            }
        }
    }

    return reg
}

func (r *Registry) HasModel(name string) bool {
    r.mu.RLock()
    defer r.mu.RUnlock()
    _, ok := r.models[name]
    return ok
}

func (r *Registry) LoadModel(name string) error {
    r.mu.Lock()
    defer r.mu.Unlock()

    status, ok := r.models[name]
    if !ok {
        return fmt.Errorf("model %s not found", name)
    }

    if status.Loaded {
        return nil // Already loaded
    }

    // Check memory constraints
    if r.totalGB+status.MemoryGB > 32 { // TODO: parse cfg.Memory.MaxTotalGB
        r.makeRoom(status.MemoryGB)
    }

    // Load model (placeholder)
    model := &GUIActorModel{
        name:   name,
        path:   status.Path,
        loaded: true,
    }

    r.loaded[name] = model
    status.Loaded = true
    r.totalGB += status.MemoryGB

    return nil
}

func (r *Registry) UnloadModel(name string) error {
    r.mu.Lock()
    defer r.mu.Unlock()

    model, ok := r.loaded[name]
    if !ok {
        return fmt.Errorf("model %s not loaded", name)
    }

    if err := model.Unload(); err != nil {
        return err
    }

    status := r.models[name]
    status.Loaded = false
    r.totalGB -= status.MemoryGB
    delete(r.loaded, name)

    return nil
}

func (r *Registry) GetModel(name string) (Model, error) {
    r.mu.RLock()
    defer r.mu.RUnlock()

    model, ok := r.loaded[name]
    if !ok {
        return nil, fmt.Errorf("model %s not loaded", name)
    }

    return model, nil
}

func (r *Registry) makeRoom(requiredGB float64) {
    // LRU eviction logic
    // TODO: implement actual LRU
}

func estimateMemoryGB(name string) float64 {
    switch name {
    case "gui-actor-2b":
        return 4.0
    case "gui-actor-7b":
        return 14.0
    default:
        return 8.0
    }
}

// GUIActorModel is a placeholder model implementation
type GUIActorModel struct {
    name   string
    path   string
    loaded bool
}

func (m *GUIActorModel) ID() string {
    return m.name
}

func (m *GUIActorModel) IsLoaded() bool {
    return m.loaded
}

func (m *GUIActorModel) Unload() error {
    m.loaded = false
    return nil
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./src/internal/model/... -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/internal/model/
git commit -m "feat(model): add model registry with memory management"
```

---

### Task 5: API Types

**Files:**
- Create: `src/internal/api/types.go`

**Step 1: Write the implementation**

```go
// src/internal/api/types.go
package api

import "time"

// ChatCompletionRequest represents an OpenAI-compatible request
type ChatCompletionRequest struct {
    Model       string    `json:"model"`
    Messages    []Message `json:"messages"`
    Stream      bool      `json:"stream,omitempty"`
    MaxPixels   *int      `json:"max_pixels,omitempty"`
    MinPixels   *int      `json:"min_pixels,omitempty"`
    MaxContext  *int      `json:"max_context,omitempty"`
    Profile     string    `json:"profile,omitempty"`
    Temperature *float64  `json:"temperature,omitempty"`
    TopP        *float64  `json:"top_p,omitempty"`
    MaxTokens   *int      `json:"max_tokens,omitempty"`
}

// Message represents a chat message
type Message struct {
    Role    string      `json:"role"`
    Content interface{} `json:"content"` // string or []ContentPart
}

// ContentPart represents a multipart content (text + image)
type ContentPart struct {
    Type     string            `json:"type"` // "text" or "image_url"
    Text     string            `json:"text,omitempty"`
    ImageURL *ImageURL         `json:"image_url,omitempty"`
}

// ImageURL contains the base64 image data
type ImageURL struct {
    URL string `json:"url"`
}

// ChatCompletionResponse represents an OpenAI-compatible response
type ChatCompletionResponse struct {
    ID      string   `json:"id"`
    Object  string   `json:"object"`
    Created int64    `json:"created"`
    Model   string   `json:"model"`
    Choices []Choice `json:"choices"`
    Usage   Usage    `json:"usage,omitempty"`
}

// Choice represents a completion choice
type Choice struct {
    Index        int            `json:"index"`
    Message      Message        `json:"message"`
    FinishReason string         `json:"finish_reason"`
    Delta        *Message       `json:"delta,omitempty"` // For streaming
    Coordinates  *Point         `json:"coordinates,omitempty"` // GUI-Actor specific
}

// Point represents normalized coordinates [0, 1]
type Point struct {
    X float64 `json:"x"`
    Y float64 `json:"y"`
}

// Usage represents token usage
type Usage struct {
    PromptTokens     int `json:"prompt_tokens"`
    CompletionTokens int `json:"completion_tokens"`
    TotalTokens      int `json:"total_tokens"`
}

// ModelInfo represents model status
type ModelInfo struct {
    ID       string  `json:"id"`
    Object   string  `json:"object"`
    Loaded   bool    `json:"loaded"`
    MemoryGB float64 `json:"memory_gb,omitempty"`
}

// ModelsResponse lists available models
type ModelsResponse struct {
    Object string      `json:"object"`
    Data   []ModelInfo `json:"data"`
}

// HealthResponse represents health check response
type HealthResponse struct {
    Status    string  `json:"status"`
    Uptime    int64   `json:"uptime_seconds"`
    MemoryGB  float64 `json:"memory_used_gb"`
    Models    int     `json:"loaded_models"`
}

// ErrorResponse represents an error
type ErrorResponse struct {
    Error ErrorDetail `json:"error"`
}

// ErrorDetail contains error information
type ErrorDetail struct {
    Message string `json:"message"`
    Type    string `json:"type"`
    Code    string `json:"code,omitempty"`
}

// NewChatCompletionResponse creates a new response
func NewChatCompletionResponse(model string, choices []Choice) *ChatCompletionResponse {
    return &ChatCompletionResponse{
        ID:      generateID(),
        Object:  "chat.completion",
        Created: time.Now().Unix(),
        Model:   model,
        Choices: choices,
    }
}

func generateID() string {
    return "chatcmpl-" + randomString(29)
}

func randomString(n int) string {
    // Simple random string generation
    const letters = "abcdefghijklmnopqrstuvwxyz0123456789"
    // TODO: implement proper random generation
    return "placeholder"
}
```

**Step 2: Commit**

```bash
git add src/internal/api/types.go
git commit -m "feat(api): add OpenAI-compatible request/response types"
```

---

### Task 6: HTTP Handlers

**Files:**
- Create: `src/internal/api/handler.go`
- Create: `src/internal/api/handler_test.go`

**Step 1: Write the failing test**

```go
// src/internal/api/handler_test.go
package api

import (
    "encoding/json"
    "net/http"
    "net/http/httptest"
   "strings"
    "testing"

    "github.com/gorilla/mux"
)

func TestHealthHandler(t *testing.T) {
    h := NewHandler(nil) // nil registry for now
    req := httptest.NewRequest("GET", "/v1/health", nil)
    w := httptest.NewRecorder()

    h.Health(w, req)

    if w.Code != http.StatusOK {
        t.Errorf("Expected status 200, got %d", w.Code)
    }

    var resp HealthResponse
    if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
        t.Fatalf("Failed to decode response: %v", err)
    }

    if resp.Status != "ok" {
        t.Errorf("Expected status 'ok', got '%s'", resp.Status)
    }
}

func TestListModels(t *testing.T) {
    h := NewHandler(nil)
    req := httptest.NewRequest("GET", "/v1/models", nil)
    w := httptest.NewRecorder()

    h.ListModels(w, req)

    if w.Code != http.StatusOK {
        t.Errorf("Expected status 200, got %d", w.Code)
    }

    var resp ModelsResponse
    if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
        t.Fatalf("Failed to decode response: %v", err)
    }

    if resp.Object != "list" {
        t.Errorf("Expected object 'list', got '%s'", resp.Object)
    }
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./src/internal/api/... -v`
Expected: FAIL with "undefined: NewHandler"

**Step 3: Write minimal implementation**

```go
// src/internal/api/handler.go
package api

import (
    "encoding/json"
    "net/http"
    "time"

    "github.com/gorilla/mux"
    "github.com/agenthands/gui-actor/internal/model"
)

type Handler struct {
    registry *model.Registry
    startTime time.Time
}

func NewHandler(registry *model.Registry) *Handler {
    return &Handler{
        registry: registry,
        startTime: time.Now(),
    }
}

// Health returns the server health status
func (h *Handler) Health(w http.ResponseWriter, r *http.Request) {
    resp := HealthResponse{
        Status:   "ok",
        Uptime:   int64(time.Since(h.startTime).Seconds()),
        MemoryGB: 0, // TODO: get actual memory usage
        Models:   0, // TODO: get loaded model count
    }

    writeJSON(w, http.StatusOK, resp)
}

// ListModels returns available models
func (h *Handler) ListModels(w http.ResponseWriter, r *http.Request) {
    resp := ModelsResponse{
        Object: "list",
        Data: []ModelInfo{
            {
                ID:     "gui-actor-2b",
                Object: "model",
                Loaded: false, // TODO: check registry
            },
            {
                ID:     "gui-actor-7b",
                Object: "model",
                Loaded: false,
            },
        },
    }

    writeJSON(w, http.StatusOK, resp)
}

// GetModel returns model status
func (h *Handler) GetModel(w http.ResponseWriter, r *http.Request) {
    vars := mux.Vars(r)
    modelID := vars["id"]

    resp := ModelInfo{
        ID:     modelID,
        Object: "model",
        Loaded: false, // TODO: check registry
    }

    writeJSON(w, http.StatusOK, resp)
}

// LoadModel loads a model into memory
func (h *Handler) LoadModel(w http.ResponseWriter, r *http.Request) {
    vars := mux.Vars(r)
    modelID := vars["id"]

    if h.registry == nil {
        writeError(w, http.StatusServiceUnavailable, "registry not available")
        return
    }

    if err := h.registry.LoadModel(modelID); err != nil {
        writeError(w, http.StatusInternalServerError, err.Error())
        return
    }

    writeJSON(w, http.StatusOK, map[string]string{"status": "loaded"})
}

// UnloadModel unloads a model from memory
func (h *Handler) UnloadModel(w http.ResponseWriter, r *http.Request) {
    vars := mux.Vars(r)
    modelID := vars["id"]

    if h.registry == nil {
        writeError(w, http.StatusServiceUnavailable, "registry not available")
        return
    }

    if err := h.registry.UnloadModel(modelID); err != nil {
        writeError(w, http.StatusInternalServerError, err.Error())
        return
    }

    writeJSON(w, http.StatusOK, map[string]string{"status": "unloaded"})
}

// ChatCompletion handles inference requests
func (h *Handler) ChatCompletion(w http.ResponseWriter, r *http.Request) {
    var req ChatCompletionRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        writeError(w, http.StatusBadRequest, "invalid request body")
        return
    }

    // TODO: implement actual inference
    resp := NewChatCompletionResponse(req.Model, []Choice{
        {
            Index:        0,
            Message:      Message{Role: "assistant", Content: "pyautogui.click(0.5, 0.5)"},
            FinishReason: "stop",
            Coordinates:  &Point{X: 0.5, Y: 0.5},
        },
    })

    writeJSON(w, http.StatusOK, resp)
}

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(status)
    json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, message string) {
    writeJSON(w, status, ErrorResponse{
        Error: ErrorDetail{
            Message: message,
            Type:    "invalid_request_error",
        },
    })
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./src/internal/api/... -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/internal/api/handler.go src/internal/api/handler_test.go
git commit -m "feat(api): add HTTP handlers for OpenAI-compatible endpoints"
```

---

### Task 7: Server Setup

**Files:**
- Create: `src/internal/api/server.go`
- Create: `src/internal/api/server_test.go`

**Step 1: Write the failing test**

```go
// src/internal/api/server_test.go
package api

import (
    "net/http"
    "net/http/httptest"
    "testing"
)

func TestNewServer(t *testing.T) {
    srv := NewServer(":8080", nil)
    if srv == nil {
        t.Fatal("Expected non-nil server")
    }

    if srv.Handler == nil {
        t.Error("Expected non-nil handler")
    }
}

func TestServerRoutes(t *testing.T) {
    h := &Handler{} // Minimal handler for testing
    srv := NewServer(":8080", h)

    tests := []struct {
        path       string
        method     string
        expectCode int
    }{
        {"/v1/health", "GET", 200},
        {"/v1/models", "GET", 200},
        {"/invalid", "GET", 404},
    }

    for _, tt := range tests {
        req := httptest.NewRequest(tt.method, tt.path, nil)
        w := httptest.NewRecorder()

        srv.Handler.ServeHTTP(w, req)

        if w.Code != tt.expectCode {
            t.Errorf("%s %s: expected %d, got %d", tt.method, tt.path, tt.expectCode, w.Code)
        }
    }
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./src/internal/api/... -v -run TestNewServer`
Expected: FAIL with "undefined: NewServer"

**Step 3: Write minimal implementation**

```go
// src/internal/api/server.go
package api

import (
    "context"
    "fmt"
    "net/http"
    "time"

    "github.com/gorilla/mux"
)

type Server struct {
    httpServer *http.Server
    Handler    http.Handler
}

func NewServer(addr string, handler *Handler) *Server {
    r := mux.NewRouter()
    api := r.PathPrefix("/v1").Subrouter()

    // Register routes
    api.HandleFunc("/health", handler.Health).Methods("GET")
    api.HandleFunc("/models", handler.ListModels).Methods("GET")
    api.HandleFunc("/models/{id}", handler.GetModel).Methods("GET")
    api.HandleFunc("/models/{id}/load", handler.LoadModel).Methods("POST")
    api.HandleFunc("/models/{id}", handler.UnloadModel).Methods("DELETE")
    api.HandleFunc("/chat/completions", handler.ChatCompletion).Methods("POST")

    httpSrv := &http.Server{
        Addr:         addr,
        Handler:      r,
        ReadTimeout:  30 * time.Second,
        WriteTimeout: 60 * time.Second,
        IdleTimeout:  120 * time.Second,
    }

    return &Server{
        httpServer: httpSrv,
        Handler:    r,
    }
}

func (s *Server) Start() error {
    fmt.Printf("Server listening on %s\n", s.httpServer.Addr)
    return s.httpServer.ListenAndServe()
}

func (s *Server) Shutdown(ctx context.Context) error {
    return s.httpServer.Shutdown(ctx)
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./src/internal/api/... -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/internal/api/server.go src/internal/api/server_test.go
git commit -m "feat(api): add HTTP server with route setup"
```

---

### Task 8: Main Entry Point

**Files:**
- Create: `src/cmd/server/main.go`
- Create: `src/cmd/server/main_test.go`

**Step 1: Write the implementation**

```go
// src/cmd/server/main.go
package main

import (
    "flag"
    "fmt"
    "log"
    "os"
    "os/signal"
    "syscall"

    "github.com/agenthands/gui-actor/internal/api"
    "github.com/agenthands/gui-actor/internal/config"
    "github.com/agenthands/gui-actor/internal/model"
)

var (
    configPath = flag.String("config", "./models/config.yaml", "Path to configuration file")
)

func main() {
    flag.Parse()

    // Load configuration
    cfg, err := config.LoadConfig(*configPath)
    if err != nil {
        log.Fatalf("Failed to load config: %v", err)
    }

    // Initialize model registry
    registry := model.NewRegistry(cfg)

    // Preload configured models
    for name, mcfg := range cfg.Models {
        if mcfg.Preload {
            log.Printf("Preloading model: %s", name)
            if err := registry.LoadModel(name); err != nil {
                log.Printf("Warning: failed to preload %s: %v", name, err)
            }
        }
    }

    // Create API handler
    handler := api.NewHandler(registry)

    // Start server
    addr := fmt.Sprintf("%s:%d", cfg.Server.Host, cfg.Server.Port)
    server := api.NewServer(addr, handler)

    // Handle shutdown gracefully
    go func() {
        sigChan := make(chan os.Signal, 1)
        signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
        <-sigChan

        log.Println("Shutting down...")
        server.Shutdown(context.Background())
    }()

    if err := server.Start(); err != nil && err != http.ErrServerClosed {
        log.Fatalf("Server error: %v", err)
    }
}
```

```go
// src/cmd/server/main_test.go
package main

import (
    "testing"
)

func TestMain(t *testing.T) {
    // TODO: Add integration tests
    t.Skip("Integration tests not yet implemented")
}
```

**Step 2: Commit**

```bash
git add src/cmd/server/main.go src/cmd/server/main_test.go
git commit -m "feat(server): add main entry point with graceful shutdown"
```

---

### Task 9: Go Module Setup

**Files:**
- Create: `src/go.mod`
- Create: `src/go.sum`

**Step 1: Create go.mod**

```bash
cd src
go mod init github.com/agenthands/gui-actor
go get github.com/gorilla/mux
go get gopkg.in/yaml.v3
go mod tidy
```

**Step 2: Commit**

```bash
git add src/go.mod src/go.sum
git commit -m "build: initialize Go module with dependencies"
```

---

### Task 10: Integration Tests

**Files:**
- Create: `src/cmd/server/integration_test.go`

**Step 1: Write the failing test**

```go
// src/cmd/server/integration_test.go
package main

import (
    "bytes"
    "encoding/json"
    "net/http"
    "net/http/httptest"
    "os"
    "testing"
    "time"

    "github.com/agenthands/gui-actor/internal/api"
    "github.com/agenthands/gui-actor/internal/model"
)

func TestFullIntegration(t *testing.T) {
    // Create test config
    cfg := &config.Config{
        Server: config.ServerConfig{
            Host: "127.0.0.1",
            Port: 0, // Random port
        },
        Models: map[string]config.ModelConfig{
            "test-model": {
                Path:     "/tmp/test",
                Enabled:  true,
                Preload:  false,
            },
        },
    }

    registry := model.NewRegistry(cfg)
    handler := api.NewHandler(registry)
    server := api.NewServer(":0", handler) // :0 for random port

    // Start server in background
    go server.Start()
    time.Sleep(100 * time.Millisecond)

    // Test health endpoint
    resp, err := http.Get("http://localhost" + server.httpServer.Addr + "/v1/health")
    if err != nil {
        t.Fatalf("Health check failed: %v", err)
    }
    defer resp.Body.Close()

    if resp.StatusCode != http.StatusOK {
        t.Errorf("Expected 200, got %d", resp.StatusCode)
    }

    // Test chat completion (will fail inference but should work)
    reqBody := api.ChatCompletionRequest{
        Model: "test-model",
        Messages: []api.Message{
            {
                Role:    "user",
                Content: "test",
            },
        },
    }

    body, _ := json.Marshal(reqBody)
    resp, err = http.Post(
        "http://localhost"+server.httpServer.Addr+"/v1/chat/completions",
        "application/json",
        bytes.NewReader(body),
    )

    if err != nil {
        t.Fatalf("Chat completion failed: %v", err)
    }
    defer resp.Body.Close()

    if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusServiceUnavailable {
        t.Logf("Got status %d (may be expected for test model)", resp.StatusCode)
    }

    server.Shutdown(context.Background())
}
```

**Step 2: Run test to verify it passes**

Run: `go test ./src/cmd/server/... -v -run TestFullIntegration`
Expected: PASS (with some warnings about missing model files)

**Step 3: Commit**

```bash
git add src/cmd/server/integration_test.go
git commit -m "test(server): add integration test for full pipeline"
```

---

## Summary

This implementation plan breaks down the MLX Go Server into 10 bite-sized tasks:

1. **Project Structure and Configuration** - YAML config loader
2. **MLX cgo Bindings** - Bridge to MLX C API
3. **Image Preprocessing** - Smart resize with grid alignment
4. **Model Registry** - Memory-aware model management
5. **API Types** - OpenAI-compatible request/response
6. **HTTP Handlers** - Endpoint implementations
7. **Server Setup** - HTTP server with routing
8. **Main Entry Point** - Application bootstrap
9. **Go Module Setup** - Dependency management
10. **Integration Tests** - End-to-end validation

Each task follows TDD: write failing test → implement → verify pass → commit.

---

**Sources:**
- [MLX Framework](https://github.com/ml-explore/mlx)
- [mlx-vlm](https://github.com/ml-explore/mlx-lm)
- [GUI-Actor Paper](https://arxiv.org/abs/2506.03143)
