"""
The LiteLLM integration module provides access to 100+ LLM providers through
a single unified interface.

Supported providers include OpenAI, Anthropic, Google, Groq, Together AI,
AWS Bedrock, Azure, Mistral, and many more.  Uses provider-prefixed model
names, e.g. ``openai/gpt-4o``, ``anthropic/claude-sonnet-4-6``.

See https://docs.litellm.ai/docs/providers for the full provider list.

You can install the LiteLLM integration package for Bridgic by running:

```shell
pip install bridgic-llms-litellm
```
"""

from importlib.metadata import version
from ._litellm_llm import LiteLLMConfiguration, LiteLLM

__version__ = version("bridgic-llms-litellm")
__all__ = ["LiteLLMConfiguration", "LiteLLM", "__version__"]
