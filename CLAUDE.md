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
```bash
# Ensure libonnxruntime is available
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/onnxruntime/lib:$DYLD_LIBRARY_PATH

# Run tests
go test ./src/cmd/server/...

# Run server
go run src/cmd/server/main.go
```

## Architecture

### Core Components

```
src/gui_actor/           # Python package
  modeling.py            # VisionHead_MultiPatch + Qwen2VLForConditionalGenerationWithPointer
  modeling_qwen25vl.py   # Qwen2.5-VL variant
  inference.py           # ForceFollowTokensLogitsProcessor + inference()
  constants.py           # Special tokens (<|pointer_start|>, <|pointer_pad|>, <|pointer_end|>)
  trainer.py             # Training logic
  dataset.py             # Data loading

src/cmd/server/          # Go inference server (OpenAI-compatible API)
  main.go                # ONNX Runtime with CoreML acceleration

verifier/                # Grounding verifier model
  verifier_model.py      # Verifier architecture
  eval_ss_with_verifier.py

train.py                 # Main training entry point
scripts/                 # Shell scripts + DeepSpeed configs (zero3.json)
eval/                    # Benchmark evaluation scripts
demo/                    # Gradio web demo
data/                    # data_config.yaml for training datasets
onnx_models/             # Exported ONNX models for production
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

The Go server (`src/cmd/server/main.go`) provides an OpenAI-compatible endpoint at `/v1/chat/completions`:
- Uses ONNX Runtime with CoreML execution provider for M4 Pro hardware acceleration
- Models stay loaded in GPU memory via `ort.DynamicAdvancedSession`
- Handles image encoding (base64) and multi-turn chat history

```go
// Client usage
client := openai.NewClient("http://localhost:8080/v1")
resp, err := client.CreateChatCompletion(ctx, req)
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
