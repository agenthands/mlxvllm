# Technology Stack

GUI-Actor is a high-performance system for visual grounding, leveraging modern Vision-Language Models (VLMs) and optimized for both training and production-ready inference.

## 1. Programming Languages
- **Python (3.10+):** The primary language for model development, training, and standard inference.
- **Go:** Used for the production-grade, OpenAI-compatible inference server, providing high concurrency and low-latency performance.

## 2. Machine Learning Core
- **Frameworks:** **PyTorch** for deep learning logic and **Transformers** (Hugging Face) for model management.
- **Models:** **Qwen2-VL** and **Qwen2.5-VL** as the backbone VLMs for superior visual-semantic understanding.
- **Custom Heads:** Attention-based "Pointer Head" (Action Head) for coordinate-free grounding.

## 3. Training & Optimization
- **DeepSpeed (ZeRO-3):** Enables training of large VLMs by partitioning model states across available GPUs.
- **Flash Attention 2:** Significantly reduces attention computation time and memory usage.
- **WandB:** Used for experiment tracking and model monitoring.

## 4. Production Inference
- **ONNX Runtime:** The primary production engine, allowing for efficient, cross-platform inference.
- **CoreML Acceleration:** Specifically leveraged on Apple Silicon (M4 Pro) for GPU and Neural Engine acceleration.
- **OpenAI-Compatible API:** A Go-based server that provides standard endpoints for seamless integration with external agents.

## 5. Deployment & Data
- **Conda/UV:** Used for environment and dependency management (Python).
- **Go Toolchain:** For building and deploying the high-performance inference server.
- **Hugging Face Datasets:** For managing training and evaluation data (e.g., ScreenSpot, ScreenSpot-Pro).
