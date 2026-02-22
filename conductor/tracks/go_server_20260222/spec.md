# Track Specification: Complete the Go-based OpenAI-compatible inference server implementation

## Overview
This track focuses on building a high-performance, Go-based inference server for GUI-Actor. The server will leverage ONNX Runtime with CoreML acceleration (for Apple Silicon/M4 Pro) and provide an OpenAI-compatible API at `/v1/chat/completions`.

## Core Requirements
1.  **OpenAI Compatibility:** Implement the Chat Completion API format (request and response).
2.  **Inference Pipeline:**
    -   Initialize and persist ONNX sessions for Vision Tower, LLM, and Pointer Head.
    -   Enable CoreML Execution Provider for hardware acceleration.
    -   Process image inputs from chat messages.
    -   Execute the multi-stage inference sequence (Vision -> LLM -> Pointer Head).
3.  **Performance:**
    -   Keep models in memory to avoid reload latency.
    -   Optimize concurrency for multi-user support.
4.  **Verification:**
    -   Implement comprehensive unit and integration tests with >80% coverage.
    -   Perform manual verification of API responses against the original Python implementation.

## Tech Stack
-   **Language:** Go
-   **Runtime:** ONNX Runtime Go bindings (`github.com/yalue/onnxruntime_go`)
-   **Acceleration:** CoreML Execution Provider
-   **Protocol:** HTTP/JSON (REST)
