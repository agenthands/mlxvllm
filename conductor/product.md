# Initial Concept
GUI-Actor provides coordinate-free visual grounding for GUI agents, leveraging state-of-the-art VLMs (Qwen2-VL, Qwen2.5-VL) to interact with digital interfaces by perceiving target elements directly, rather than calculating precise screen coordinates.

# Product Definition
GUI-Actor aims to empower **App Developers** and **Agent Builders** by providing a robust, high-accuracy foundation for digital assistants and autonomous agents. By shifting from coordinate-generation to attention-based visual grounding, the project offers a more human-like, intuitive way for models to navigate complex GUIs on **Native Desktop**, **Scalable Cloud**, and **Automation Frameworks**.

## Target Users
- **App Developers:** Creating sophisticated digital assistants that require reliable interaction with any software interface.
- **Agent Builders:** Developing autonomous systems capable of executing complex web or desktop tasks with high precision.

## Core Value Proposition
- **State-of-the-Art Accuracy:** Surpassing traditional coordinate-based methods by using direct visual attention for grounding.
- **Regional Reasoning:** Generating multiple candidate regions in a single forward pass, allowing for advanced search and verification strategies.
- **On-Device Performance:** Optimized for local execution on high-end hardware like Apple's M4 Pro, ensuring fast, private, and reliable inference.

## Key Features & Focus Areas
- **Advanced Model Support:** Continuous integration of the latest VLMs (e.g., Qwen2.5-VL) to maintain leadership in accuracy.
- **Production Infrastructure:** A high-performance, Go-based, OpenAI-compatible inference server for seamless integration into production systems.
- **Grounding Verification:** A specialized verifier model that refines predictions to achieve near-perfect precision for critical operations.

## Deployment Strategy
GUI-Actor is designed to be versatile:
- **Native Desktop Control:** Direct interaction with local OS environments for desktop automation.
- **Scalable Cloud API:** Providing a standard API for remote browser and application control.
- **Automation Integration:** Seamlessly plugging into existing tools (e.g., Playwright, Selenium) to enhance their visual perception.
