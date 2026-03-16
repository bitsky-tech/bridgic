"""
The Model module provides core abstraction entities for LLMs (Large Language Models).

This module defines core abstraction entities for interacting with models, providing 
foundational type abstractions for different model implementations.
"""

from bridgic.core.model._base_llm import BaseLlm
from bridgic.core.model._model_retry import (
    RetryPolicyConfig,
    is_recoverable_exception,
    retryable_model_call,
)
from bridgic.core.model._model_error import (
    ModelRetryLimitError,
    ModelUnrecoverableError,
)

__all__ = [
    "BaseLlm",
    "retryable_model_call",
    "is_recoverable_exception",
    "RetryPolicyConfig",
    "ModelRetryLimitError",
    "ModelUnrecoverableError",
]