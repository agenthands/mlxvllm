# Implementation Plan: Complete the Go-based OpenAI-compatible inference server implementation

This plan details the steps to implement a high-performance Go-based inference server for GUI-Actor.

## Phase 1: Architecture & Model Loading
Initialize the Go project and set up persistent ONNX sessions with CoreML acceleration.

- [ ] Task: Initialize Go module and setup ONNX session management infrastructure
    - [ ] Write Tests: Define tests for session initialization and CoreML provider availability
    - [ ] Implement Feature: Create the base `InferenceEngine` struct and session loading logic
- [ ] Task: Implement Vision Tower and Pointer Head session loading
    - [ ] Write Tests: Verify that ONNX files are correctly loaded and sessions are ready
    - [ ] Implement Feature: Add model-specific loading logic to `NewInferenceEngine`
- [ ] Task: Conductor - User Manual Verification 'Architecture & Model Loading' (Protocol in workflow.md)

## Phase 2: Request Handling & Preprocessing
Implement image decoding and the OpenAI-compatible request format.

- [ ] Task: Define OpenAI-compatible request/response structures and handler
    - [ ] Write Tests: Verify JSON unmarshaling and error handling for malformed requests
    - [ ] Implement Feature: Create `ChatCompletionRequest` and `ChatCompletionResponse` structs and the initial handler
- [ ] Task: Implement image extraction and preprocessing for the vision model
    - [ ] Write Tests: Verify that base64/URL images are correctly decoded and resized/normalized
    - [ ] Implement Feature: Add `PreprocessImage` utility to prepare data for the Vision Tower
- [ ] Task: Conductor - User Manual Verification 'Request Handling & Preprocessing' (Protocol in workflow.md)

## Phase 3: Inference Pipeline
Orchestrate the full inference flow from vision to action.

- [ ] Task: Orchestrate the sequence: Vision -> LLM -> Pointer Head
    - [ ] Write Tests: Mock session outputs to verify the orchestration logic and data flow
    - [ ] Implement Feature: Integrate all components into a single `RunInference` pipeline
- [ ] Task: Extract coordinates from Pointer Head attention scores
    - [ ] Write Tests: Verify that attention scores are correctly transformed into click points
    - [ ] Implement Feature: Port coordinate extraction logic from Python's `get_prediction_region_point`
- [ ] Task: Conductor - User Manual Verification 'Inference Pipeline' (Protocol in workflow.md)

## Phase 4: OpenAI Compatibility & Finalization
Finalize the API response format and perform end-to-end testing.

- [ ] Task: Implement final Chat Completion response format
    - [ ] Write Tests: Verify the structure of the final JSON response matches OpenAI standards
    - [ ] Implement Feature: Complete the handler logic to return formatted results
- [ ] Task: End-to-end integration tests and performance check
    - [ ] Write Tests: Execute full cycles from request to response against a local mock/test model
    - [ ] Implement Feature: Add logging, metrics, and ensure concurrent safety
- [ ] Task: Conductor - User Manual Verification 'OpenAI Compatibility' (Protocol in workflow.md)
