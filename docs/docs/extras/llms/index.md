# LLM Integration

## Overview

Bridgic's model integration module (`bridgic.llms`) employs a **Provider-centric** integration strategy, adapting different model service providers and inference frameworks as independent integration packages. Each package adheres to the same interface standards, ensuring model neutrality and consistent interface definitions.

## Design Philosophy

### Provider-Centric Integration

Bridgic designs model integration with **Providers** as the fundamental unit:

- **Independent Packaging**: Each model provider (such as OpenAI, vLLM, other model vendors, or inference frameworks) has its own dedicated integration package
- **Unified Interface**: All packages implement the same core interface to ensure consistent user experience
- **On-Demand Installation**: Developers only need to install the required Provider packages, avoiding unnecessary dependencies

### Model Neutrality

Through Provider-centric design, Bridgic achieves true **model neutrality**:

- **No Vendor Lock-in**: Application code is not tied to specific model providers, enabling easy switching
- **Seamless Switching**: When changing model providers, application code requires minimal changes—simply swap the model object

### Protocol-Driven Design

Each Provider declares its capabilities by implementing predefined protocols:

- **Capability Declaration**: Protocol based implementations clearly declare the supported functional features
- **Progressive Enhancement**: Providers can implement partial protocols to support incremental enhancement
- **Extensibility**: New Providers can be seamlessly integrated by implementing the appropriate protocols

## Available Integrations

| Provider |        Package              | Description                                                          |
|:--------:|:---------------------------:|:---------------------------------------------------------------------|
|  OpenAI  | `bridgic-llms-openai`       | Official OpenAI API integration for GPT-4, GPT-3.5, and other models |
|  vLLM    | `bridgic-llms-vllm`         | Self-hosted large language model inference engine                    |
|    -     | `bridgic-llms-openai-like`  | Thin wrapper for OpenAI compatible model services                    |
