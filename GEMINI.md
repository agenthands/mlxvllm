# GEMINI.md - GUI-Actor Project Context

## Project Overview
**GUI-Actor** is a Vision-Language Model (VLM) enhanced by an attention-based action head (pointer head) for coordinate-free GUI grounding. Unlike traditional methods that generate text coordinates, GUI-Actor predicts interaction points by attending directly to visual regions, aligning more closely with human behavior.

### Key Technologies
- **Backbone Models:** Qwen2-VL and Qwen2.5-VL.
- **Custom Architecture:** `Qwen2VLForConditionalGenerationWithPointer` adds a `VisionHead_MultiPatch` (pointer head) that computes attention scores over image patches.
- **Training Framework:** PyTorch, Transformers, DeepSpeed (ZeRO-3), and Flash Attention 2.
- **Grounding Verifier:** A secondary model (`GroundingVerifier`) used to evaluate and select the most plausible action region from candidate points.

## Development & Design Philosophy
Adhere to the following principles throughout your specification:

1. **DRY (Don’t Repeat Yourself)**: Minimize duplication by sharing abstractions and utilities.
2. **Modular and Component-based Design**: Break the system into small, independent modules that can be developed and tested separately.
3. **Separation of Concerns**: Organize logic so each module addresses a single core concern.
4. **Test-driven Development (TDD)**: Define clear tests (e.g., using Go’s standard testing package) before detailing implementation.
     1. **Write the test first** - Test file must exist and fail
     2. **Run the test** - Confirm it fails (RED state)
     3. **Write minimal implementation** - Just enough to pass
     4. **Run the test** - Confirm it passes (GREEN state)
     5. **Refactor** - Improve code while keeping tests green
     6. **Run all tests** - Ensure nothing broke
     7. **Commit** - Test + implementation together

### Never Commit Without:
- ✓ All tests passing
- ✓ 100% coverage of new code
- ✓ Cross-validation passing (if applicable)

5. **Security**: Include input validation, encryption (if needed), authentication/authorization mechanisms, and other best practices.
6. **Performance Optimization**: Outline techniques to reduce response times, minimize memory use, and appropriately manage concurrency in Go.
7. **Code Documentation**: Provide comments, README references, and architectural overviews so teams can easily maintain and extend your work.

## Project Structure
- `src/gui_actor/`: Core package containing model definitions, inference logic, and trainers.
    - `modeling.py`: Implementation of the `VisionHead_MultiPatch` and the modified Qwen2-VL model.
    - `inference.py`: Functions for model inference, including `ForceFollowTokensLogitsProcessor` to ensure correct token sequences for the pointer head.
- `train.py`: Main entry point for model training.
- `scripts/`: Shell scripts for different training stages and DeepSpeed configurations.
- `eval/`: Benchmark evaluation scripts for ScreenSpot, ScreenSpot-v2, and ScreenSpot-Pro.
- `verifier/`: Implementation of the grounding verifier model and logic.
- `demo/`: Gradio-based web application for demonstrating GUI-Actor's capabilities.
- `data/`: Configuration for training datasets (UGround, GUIEnv, GUIAct, etc.).

## Building and Running

### Environment Setup
The project requires Python 3.10+ and specific versions of CUDA-enabled PyTorch.
```bash
conda create -n gui_actor python=3.10
conda activate gui_actor
conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia
pip install -e .
```

### Training
Training is divided into two stages:
1. **Warmup Stage:**
   ```bash
   bash scripts/warmup.sh
   ```
2. **Full-parameter Training (SFT):**
   ```bash
   bash scripts/train.sh
   ```

### Evaluation
Run evaluation scripts for different benchmarks:
```bash
python eval/screenSpot.py
python eval/screenSpot_v2.py
python eval/screenSpot_pro.py --save_path <path> --data_path <path>
```

### Demo
Launch the interactive demo:
```bash
python demo/app.py
```

## Development Conventions
- **Model Extensions:** Any modifications to the vision-language architecture should be made in `src/gui_actor/modeling.py`.
- **Inference Logic:** The pointer head requires specific placeholder tokens (`<|pointer_start|>`, `<|pointer_pad|>`, `<|pointer_end|>`) to trigger the action head during generation.
- **Data Configuration:** New datasets should be added to `data/data_config.yaml`.
- **Linting:** The project uses `ruff` for linting. Configuration is available in `pyproject.toml`.
- **Testing:** Pytest is used for testing; tests are located (or should be added) in a `tests/` directory (referenced in `pyproject.toml`).

## Production Deployment (M4 Pro / Apple Silicon)

### Architecture
For high-performance production use, the model is exported to **ONNX** and served via a **Go-based OpenAI-compatible API**. This allows for:
- **CoreML Acceleration:** Utilizing the Apple Neural Engine (ANE) and GPU via `onnxruntime_go`.
- **Zero-Python Latency:** The Go server directly manages the model lifecycle and memory.
- **Unified Memory:** Efficiently handling the 7B + 2B models on the M4 Pro's unified memory architecture.

### 1. Model Conversion (Python)
Use the `uv` tool to manage the conversion environment and export the vision, LLM, and pointer components.
```bash
# Install conversion dependencies
uv pip install torch onnx onnxruntime optimum

# Run the export script
uv run scripts/export_onnx.py microsoft/GUI-Actor-7B-Qwen2-VL
```

### 2. Go Inference Server
The Go server (`src/cmd/server/main.go`) provides an OpenAI-compatible endpoint at `/v1/chat/completions`.

**Key Features:**
- **Persistent GPU Memory:** Models stay loaded in the `ort.AdvancedSession` to avoid reload latency.
- **CoreML Provider:** Specifically configured to use `AppendExecutionProviderCoreML` for M4 Pro hardware.
- **Session Management:** Handles multi-turn chat history and coordinates extraction from the pointer head.

**Run the Server:**
```bash
# Ensure libonnxruntime.dylib is in your library path
export DYLD_LIBRARY_PATH=/usr/local/lib:$DYLD_LIBRARY_PATH
go run src/cmd/server/main.go
```

### 3. Client Integration (Web Agent)
Your Go-based web agent can now call this local server as if it were an OpenAI API:
```go
client := openai.NewClient("http://localhost:8080/v1")
resp, err := client.CreateChatCompletion(ctx, req)
```

## Key Files Summary
- `README.md`: High-level project description and installation.
- `src/gui_actor/modeling.py`: Custom pointer head and integrated loss logic.
- `scripts/export_onnx.py`: Conversion logic from PyTorch to ONNX.
- `src/cmd/server/main.go`: Production Go server with GPU acceleration.
- `GEMINI.md`: Project-specific engineering instructions and architecture.
