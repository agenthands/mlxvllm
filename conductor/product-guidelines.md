# Product Guidelines

GUI-Actor is a high-precision tool for automated GUI interaction. These guidelines ensure consistency, reliability, and technical excellence across all project artifacts.

## 1. Technical Precision & Communication
- **Accuracy First:** All documentation and model descriptions must prioritize technical accuracy. Avoid over-promising on model capabilities.
- **Explicit Terminology:** Use consistent terminology for GUI components (e.g., "target element", "candidate region", "patch token").
- **Clear Logic:** When describing model behavior or inference steps, ensure the logic is explicit and reproducible.

## 2. Design & UX Principles for Agents
- **Human-Centric Interaction:** Model interactions should mimic human perception—attending to elements rather than calculating pixels.
- **Predictable Behavior:** Ensure that agent actions are deterministic where possible, with clear error handling for ambiguous GUI states.
- **Visual Feedback:** Any user-facing interface (like the demo) should provide clear visual indicators of where the model is looking and why.

## 3. Reliability & Security
- **Fail-Safe Operations:** Critical GUI actions should be verified by the specialized verifier model before execution.
- **Privacy by Design:** Emphasize local execution on unified memory hardware (M4 Pro) to keep user data and screenshots private.
- **Input Validation:** All instructions and screenshots must be validated to prevent prompt injection or malicious input.

## 4. Documentation Standards
- **Standard Markdown:** All project documents must use clean, standard Markdown with clear hierarchical headers.
- **Code Examples:** Provide well-commented, runnable code examples (Python and Go) for common use cases.
- **Architectural Clarity:** Maintain high-level architectural overviews that clearly separate the training, inference, and verification modules.
