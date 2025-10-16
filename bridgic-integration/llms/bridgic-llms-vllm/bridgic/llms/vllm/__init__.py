"""
The vLLM integration module provides support for the vLLM inference engine.

This module implements communication interfaces with vLLM inference services, supporting 
highly reliable calls to large language models deployed via vLLM, and provides several 
encapsulations for common seen high-level functionality.
"""

from .vllm_server_llm import VllmServerLlm, VllmServerConfiguration

__all__ = ["VllmServerConfiguration", "VllmServerLlm"]