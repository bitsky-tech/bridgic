"""
The OpenAI integration module provides support for the OpenAI API.

This module implements integration interfaces with OpenAI language models, supporting 
calls to large language models provided by OpenAI such as the GPT series, and provides 
several wrappers for advanced functionality.

You can install the OpenAI integration package for Bridgic by running:

```shell
pip install bridgic-llms-openai
```
"""

from .openai_llm import OpenAIConfiguration, OpenAILlm

__all__ = ["OpenAIConfiguration", "OpenAILlm"]