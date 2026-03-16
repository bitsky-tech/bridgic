import pytest

from bridgic.core.model import (
    ModelRetryLimitError,
    ModelUnrecoverableError,
    RetryPolicyConfig,
    retryable_model_call,
)
from bridgic.core.model.types import Message, Response, Role


class DecoratedChatClient:
    def __init__(self, failures_before_success: int = 0, error_factory=None, response_text: str = "ok"):
        self.failures_before_success = failures_before_success
        self.error_factory = error_factory or (lambda: TimeoutError("timed out"))
        self.response_text = response_text
        self.chat_calls = 0
        self.achat_calls = 0

    @retryable_model_call(
        RetryPolicyConfig(
            max_attempts=3,
            base_delay=0.0,
            jitter_ratio=0.0,
        )
    )
    def chat(self, messages):
        self.chat_calls += 1
        if self.chat_calls <= self.failures_before_success:
            raise self.error_factory()
        return Response(message=Message.from_text(text=self.response_text, role=Role.AI), raw=None)

    @retryable_model_call(
        RetryPolicyConfig(
            max_attempts=3,
            base_delay=0.0,
            jitter_ratio=0.0,
        )
    )
    async def achat(self, messages):
        self.achat_calls += 1
        if self.achat_calls <= self.failures_before_success:
            raise self.error_factory()
        return Response(message=Message.from_text(text=self.response_text, role=Role.AI), raw=None)


def test_decorator_chat_retries_then_succeeds():
    client = DecoratedChatClient(failures_before_success=2, error_factory=lambda: TimeoutError("request timeout"))
    result = client.chat([Message.from_text("hello", role=Role.USER)])
    assert result.message is not None
    assert result.message.content == "ok"
    assert client.chat_calls == 3


def test_decorator_chat_exhausted_raises_rate_limit_error():
    client = DecoratedChatClient(
        failures_before_success=5,
        error_factory=lambda: ConnectionError("network disconnected"),
    )
    with pytest.raises(ModelRetryLimitError) as exc_info:
        client.chat([Message.from_text("hello", role=Role.USER)])
    assert exc_info.value.operation == "chat"


def test_decorator_chat_non_retryable_raises_unrecoverable_error():
    client = DecoratedChatClient(
        failures_before_success=1,
        error_factory=lambda: ValueError("invalid request payload"),
    )
    with pytest.raises(ModelUnrecoverableError) as exc_info:
        client.chat([Message.from_text("hello", role=Role.USER)])
    assert exc_info.value.operation == "chat"


@pytest.mark.asyncio
async def test_decorator_achat_retries_then_succeeds():
    client = DecoratedChatClient(
        failures_before_success=2,
        error_factory=lambda: TimeoutError("async timeout"),
        response_text="async-ok",
    )
    result = await client.achat([Message.from_text("hello", role=Role.USER)])
    assert result.message is not None
    assert result.message.content == "async-ok"
    assert client.achat_calls == 3


@pytest.mark.asyncio
async def test_decorator_achat_non_retryable_raises_unrecoverable_error():
    client = DecoratedChatClient(
        failures_before_success=1,
        error_factory=lambda: RuntimeError("fatal validation failed"),
    )
    with pytest.raises(ModelUnrecoverableError):
        await client.achat([Message.from_text("hello", role=Role.USER)])
