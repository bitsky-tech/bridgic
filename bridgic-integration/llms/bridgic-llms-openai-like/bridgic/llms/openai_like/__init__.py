"""
The OpenAI-Like integration module provides support for third-party services 
compatible with the OpenAI API.

This package is a thin wrapper for the OpenAI API, designed to meet the needs 
for calling third-party model services compatible with the OpenAI API.

Note that this integration does not adapt to specific model providers, but 
provides general-purpose interfaces. Therefore, it is not fully comprehensive 
in functionality and only supports basic chat/stream operations and their 
corresponding async interfaces.

You can install the OpenAI-Like integration package for Bridgic by running:

```shell
pip install bridgic-llms-openai-like
```
"""

from importlib.metadata import version
from .openai_like_llm import OpenAILikeConfiguration, OpenAILikeLlm

__version__ = version("bridgic-llms-openai-like")
__all__ = ["OpenAILikeConfiguration", "OpenAILikeLlm", "__version__"]