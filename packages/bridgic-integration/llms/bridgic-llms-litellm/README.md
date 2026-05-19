# bridgic-llms-litellm

LiteLLM adapters for [Bridgic](https://github.com/bitsky-tech/bridgic), enabling connectivity with 100+ LLM providers through a single unified interface.

## Installation

```shell
pip install bridgic-llms-litellm
```

## Usage

```python
from bridgic.llms.litellm import LiteLLMLlm
from bridgic.core.model.types import Message, Role

# API keys are read from environment variables (e.g. OPENAI_API_KEY)
llm = LiteLLMLlm()

response = llm.chat(
    model="openai/gpt-4o",
    messages=[Message.from_text("Hello!", role=Role.USER)],
)
print(response.message.content)
```

See https://docs.litellm.ai/docs/providers for the full provider list.
