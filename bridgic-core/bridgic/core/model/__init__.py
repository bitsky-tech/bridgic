"""
The Model module provides core abstraction entities for LLMs (Large Language Models).

This module defines core abstraction entities for interacting with models, providing 
foundational type abstractions for different model implementations.
"""

from bridgic.core.model._base_llm import BaseLlm

__all__ = [
    "BaseLlm",
]