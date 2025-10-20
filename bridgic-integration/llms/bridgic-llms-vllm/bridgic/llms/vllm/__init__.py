"""
The vLLM integration module provides support for the vLLM inference engine.

This module implements communication interfaces with vLLM inference services, supporting 
highly reliable calls to large language models deployed via vLLM, and provides several 
encapsulations for common seen high-level functionality.

You can install the vLLM integration package for Bridgic by running:

```shell
pip install bridgic-llms-vllm
```
"""

from importlib.metadata import version
from .vllm_server_llm import VllmServerLlm, VllmServerConfiguration

__version__ = version("bridgic-llms-vllm")
__all__ = ["VllmServerConfiguration", "VllmServerLlm", "__version__"]