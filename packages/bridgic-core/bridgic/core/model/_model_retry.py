import asyncio
import random
import time
from functools import wraps
from typing import Any, Callable, Optional
from typing_extensions import ParamSpec, TypeVar
from pydantic import BaseModel, Field

from bridgic.core.model._model_error import ModelRetryLimitError, ModelUnrecoverableError


P = ParamSpec("P")
R = TypeVar("R")


class RetryPolicyConfig(BaseModel):
    """
    Retry policy for decorating model invocation methods.
    """

    max_attempts: int = Field(default=3, ge=1)
    base_delay: float = Field(default=0.2, ge=0.0)
    max_delay: float = Field(default=2.0, ge=0.0)
    exponential_base: float = Field(default=2.0, ge=1.0)
    jitter_ratio: float = Field(default=0.2, ge=0.0, le=1.0)


def is_recoverable_exception(exc: Exception) -> bool:
    """
    Heuristic classifier for retryable exceptions.
    """
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True

    text = (
        f"{exc.__class__.__name__} "
        f"{exc.__class__.__module__} "
        f"{str(exc)}"
    ).lower()
    retry_markers = (
        "429",
        "timeout",
        "timed out",
        "limit",
        "exceeded",
        "overloaded",
        "too many requests",
        "network",
        "connection",
        "harmful",
        "policy",
        "violated",
        "violation",
        "not allowed",
    )
    return any(marker in text for marker in retry_markers)


def retryable_model_call(
    config: Optional[RetryPolicyConfig] = None,
    recoverable_checker: Optional[Callable[[Exception], bool]] = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator for model non-streaming methods.

    Behavior:
    - Retry recoverable exceptions up to max attempts.
    - Raise `ModelUnrecoverableError` immediately for non-recoverable exceptions.
    - Raise `ModelRetryLimitError` after retry attempts are exhausted.
    """
    config = config or RetryPolicyConfig()
    checker = recoverable_checker or is_recoverable_exception

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                op = func.__name__
                last_exc: Optional[Exception] = None
                for attempt in range(1, config.max_attempts + 1):
                    try:
                        return await func(*args, **kwargs)
                    except Exception as exc:
                        if not checker(exc):
                            raise ModelUnrecoverableError(
                                f"Model operation `{op}` failed with non-recoverable error",
                                operation=op,
                                original_exception=exc,
                            ) from exc
                        else:
                            last_exc = exc
                            if attempt < config.max_attempts:
                                await asyncio.sleep(_backoff_delay(attempt, config))
                raise ModelRetryLimitError(
                    (
                        f"Model operation `{op}` exceeded retry attempts "
                        f"({config.max_attempts} attempts)"
                    ),
                    operation=op,
                    original_exception=last_exc,
                ) from last_exc

            return async_wrapper

        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
            op = func.__name__
            last_exc: Optional[Exception] = None
            for attempt in range(1, config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    if not checker(exc):
                        raise ModelUnrecoverableError(
                            f"Model operation `{op}` failed with non-recoverable error",
                            operation=op,
                            original_exception=exc,
                        ) from exc
                    else:
                        last_exc = exc
                        if attempt < config.max_attempts:
                            delay = _backoff_delay(attempt, config)
                            if delay > 0:
                                time.sleep(delay)
            raise ModelRetryLimitError(
                (
                    f"Model operation `{op}` exceeded retry attempts "
                    f"({config.max_attempts} attempts)"
                ),
                operation=op,
                original_exception=last_exc,
            ) from last_exc

        return sync_wrapper

    return decorator


def _backoff_delay(attempt: int, config: RetryPolicyConfig) -> float:
    if config.base_delay <= 0:
        return 0.0
    delay = min(
        config.base_delay * (config.exponential_base ** max(0, attempt - 1)),
        config.max_delay,
    )
    if config.jitter_ratio:
        delay = delay + delay * random.uniform(-config.jitter_ratio, config.jitter_ratio)
    return delay
