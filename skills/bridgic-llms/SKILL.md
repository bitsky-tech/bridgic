---
name: bridgic-llms
description: |
  LLM provider initialization for bridgic projects. Use when: (1) initializing
  OpenAILlm, OpenAILikeLlm, or VllmServerLlm, (2) configuring OpenAIConfiguration
  (model, temperature, max_tokens, timeout), (3) choosing the right provider package
  for a task, (4) using chat/stream interfaces or advanced protocols (StructuredOutput,
  ToolSelection).
---

# Bridgic LLMs

Model-neutral LLM integration with protocol-driven capability declaration.

## Dependencies

| Package | `BaseLlm` | `StructuredOutput` | `ToolSelection` |
|---------|:---------:|:------------------:|:---------------:|
| `bridgic-llms-openai` | yes | yes | yes |
| `bridgic-llms-openai-like` | yes | no | no |
| `bridgic-llms-vllm` | yes | yes | yes |
| `python-dotenv` | — | — | — |

Install only the LLM provider package you need. `python-dotenv` is required for loading `.env` configuration.

**Installation**: Run the install script to set up all dependencies:

```bash
bash "skills/bridgic-llms/scripts/install-deps.sh" "$PWD"
```

Supported providers: `openai` (default), `openai-like`, `vllm`. The script checks uv availability, initializes a uv project if needed, and installs only missing packages.

## Quick Start

```python
import os
from dotenv import load_dotenv
from bridgic.llms.openai import OpenAILlm, OpenAIConfiguration

load_dotenv()

llm = OpenAILlm(
    api_key=os.environ.get("LLM_API_KEY"),
    api_base=os.environ.get("LLM_API_BASE"),
    configuration=OpenAIConfiguration(
        model=os.environ.get("LLM_MODEL", "gpt-4o"),
        temperature=0.0,
        max_tokens=16384,
    ),
    timeout=180.0,
)
```

## Provider Selection Guide

| Provider | When to Use |
|----------|-------------|
| `OpenAILlm` | Production use, need structured output or tool calling. Works with OpenAI API. |
| `OpenAILikeLlm` | Third-party OpenAI-compatible APIs (DashScope, etc.), only need basic chat/stream. |
| `VllmServerLlm` | Self-hosted vLLM inference server, full capability. |

**Common pitfall**: Do NOT use `OpenAILikeLlm` when you need structured output or tool selection — it does not implement those protocols. Use `OpenAILlm` instead.

## Basic Interfaces

All providers implement `BaseLlm`:

```python
from bridgic.core.model.types import Message, Role

messages = [
    Message.from_text("You are a helpful assistant.", role=Role.SYSTEM),
    Message.from_text("Hello!", role=Role.USER),
]

# Chat — complete response
response = llm.chat(messages=messages, model="gpt-4o", temperature=0.7)
print(response.message.content)

# Stream — real-time chunks
for chunk in llm.stream(messages=messages, model="gpt-4o"):
    print(chunk.delta, end="", flush=True)
```

## Advanced Protocols

See [references/llm-integration.md](references/llm-integration.md) for:
- `StructuredOutput` — generate Pydantic model instances or JSON schema conformant output
- `ToolSelection` — function/tool calling with Tool definitions
- Full code examples for all providers

## Reference Files

| Scenario | Load |
|----------|------|
| Full API details, all providers, advanced protocols | [llm-integration.md](references/llm-integration.md) |
