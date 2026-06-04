import pytest
import os
import sys
import types
from unittest import mock

from bridgic.core.model.types import *


# ---------------------------------------------------------------------------
# Unit tests (no API key required — litellm is mocked)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_litellm():
    """Install a fake litellm module so LiteLLM can be imported without the real package."""
    fake = types.ModuleType("litellm")

    fake_usage = mock.MagicMock()
    fake_usage.prompt_tokens = 10
    fake_usage.completion_tokens = 5
    fake_usage.total_tokens = 15

    fake_message = mock.MagicMock()
    fake_message.content = "Hello from LiteLLM!"
    fake_message.refusal = None

    fake_choice = mock.MagicMock()
    fake_choice.message = fake_message

    fake_response = mock.MagicMock()
    fake_response.choices = [fake_choice]
    fake_response.usage = fake_usage
    fake_response.model = "openai/gpt-4o-mini"

    fake.completion = mock.MagicMock(return_value=fake_response)
    fake.acompletion = mock.AsyncMock(return_value=fake_response)

    sys.modules["litellm"] = fake
    yield fake
    sys.modules.pop("litellm", None)


@pytest.fixture
def llm_instance(mock_litellm):
    from bridgic.llms.litellm import LiteLLM, LiteLLMConfiguration
    config = LiteLLMConfiguration(model="openai/gpt-4o-mini")
    return LiteLLM(configuration=config)


def test_chat_basic(llm_instance, mock_litellm):
    response = llm_instance.chat(
        messages=[Message.from_text("Hello!", role=Role.USER)],
    )
    assert response.message.role == Role.AI
    assert response.message.content == "Hello from LiteLLM!"
    assert response.usage is not None
    assert response.usage.prompt_tokens == 10
    assert response.usage.completion_tokens == 5
    assert response.usage.total_tokens == 15

    call_kwargs = mock_litellm.completion.call_args[1]
    assert call_kwargs["model"] == "openai/gpt-4o-mini"
    assert call_kwargs["drop_params"] is True


def test_chat_model_override(llm_instance, mock_litellm):
    llm_instance.chat(
        messages=[Message.from_text("Hi", role=Role.USER)],
        model="anthropic/claude-haiku-4-5",
    )
    call_kwargs = mock_litellm.completion.call_args[1]
    assert call_kwargs["model"] == "anthropic/claude-haiku-4-5"


def test_api_key_forwarded(mock_litellm):
    from bridgic.llms.litellm import LiteLLM
    llm = LiteLLM(api_key="sk-test-123")
    llm.chat(
        messages=[Message.from_text("Hi", role=Role.USER)],
        model="openai/gpt-4o",
    )
    call_kwargs = mock_litellm.completion.call_args[1]
    assert call_kwargs["api_key"] == "sk-test-123"


def test_api_key_omitted_when_none(llm_instance, mock_litellm):
    llm_instance.chat(
        messages=[Message.from_text("Hi", role=Role.USER)],
    )
    call_kwargs = mock_litellm.completion.call_args[1]
    assert "api_key" not in call_kwargs


def test_api_base_forwarded(mock_litellm):
    from bridgic.llms.litellm import LiteLLM
    llm = LiteLLM(api_base="http://localhost:4000")
    llm.chat(
        messages=[Message.from_text("Hi", role=Role.USER)],
        model="openai/gpt-4o",
    )
    call_kwargs = mock_litellm.completion.call_args[1]
    assert call_kwargs["api_base"] == "http://localhost:4000"


def test_timeout_forwarded(mock_litellm):
    from bridgic.llms.litellm import LiteLLM
    llm = LiteLLM(timeout=30.0)
    llm.chat(
        messages=[Message.from_text("Hi", role=Role.USER)],
        model="openai/gpt-4o",
    )
    call_kwargs = mock_litellm.completion.call_args[1]
    assert call_kwargs["timeout"] == 30.0


def test_drop_params_default_true(llm_instance, mock_litellm):
    llm_instance.chat(
        messages=[Message.from_text("Hi", role=Role.USER)],
    )
    call_kwargs = mock_litellm.completion.call_args[1]
    assert call_kwargs["drop_params"] is True


def test_message_conversion_roles(llm_instance, mock_litellm):
    messages = [
        Message.from_text("System prompt", role=Role.SYSTEM),
        Message.from_text("User message", role=Role.USER),
        Message.from_text("AI response", role=Role.AI),
    ]
    llm_instance.chat(messages=messages)
    call_kwargs = mock_litellm.completion.call_args[1]
    assert call_kwargs["messages"][0]["role"] == "system"
    assert call_kwargs["messages"][1]["role"] == "user"
    assert call_kwargs["messages"][2]["role"] == "assistant"


def test_message_conversion_tool_blocks(llm_instance, mock_litellm):
    messages = [
        Message(
            role=Role.AI,
            blocks=[
                TextBlock(text="Checking weather."),
                ToolCallBlock(id="call_1", name="get_weather", arguments={"city": "Tokyo"}),
            ],
        ),
        Message.from_tool_result(tool_id="call_1", content="22°C sunny"),
    ]
    llm_instance.chat(messages=messages)
    call_kwargs = mock_litellm.completion.call_args[1]
    ai_msg = call_kwargs["messages"][0]
    assert "Tool call:" in ai_msg["content"]
    assert "get_weather" in ai_msg["content"]
    tool_msg = call_kwargs["messages"][1]
    assert "Tool result: 22°C sunny" in tool_msg["content"]


def test_configuration_defaults_merge(mock_litellm):
    from bridgic.llms.litellm import LiteLLM, LiteLLMConfiguration
    config = LiteLLMConfiguration(
        model="openai/gpt-4o-mini",
        temperature=0.5,
        max_tokens=100,
    )
    llm = LiteLLM(configuration=config)
    llm.chat(messages=[Message.from_text("Hi", role=Role.USER)])

    call_kwargs = mock_litellm.completion.call_args[1]
    assert call_kwargs["temperature"] == 0.5
    assert call_kwargs["max_tokens"] == 100


def test_call_time_overrides_config(mock_litellm):
    from bridgic.llms.litellm import LiteLLM, LiteLLMConfiguration
    config = LiteLLMConfiguration(temperature=0.5)
    llm = LiteLLM(configuration=config)
    llm.chat(
        messages=[Message.from_text("Hi", role=Role.USER)],
        model="openai/gpt-4o",
        temperature=0.9,
    )
    call_kwargs = mock_litellm.completion.call_args[1]
    assert call_kwargs["temperature"] == 0.9


@pytest.mark.asyncio
async def test_achat_basic(llm_instance, mock_litellm):
    response = await llm_instance.achat(
        messages=[Message.from_text("Hello!", role=Role.USER)],
    )
    assert response.message.role == Role.AI
    assert response.message.content == "Hello from LiteLLM!"
    mock_litellm.acompletion.assert_called_once()


def test_serialization_roundtrip(mock_litellm):
    from bridgic.llms.litellm import LiteLLM, LiteLLMConfiguration
    config = LiteLLMConfiguration(model="openai/gpt-4o", temperature=0.7)
    llm = LiteLLM(api_key="sk-test", api_base="http://proxy:4000", timeout=60.0, configuration=config)

    state = llm.dump_to_dict()
    assert state["api_key"] == "sk-test"
    assert state["api_base"] == "http://proxy:4000"
    assert state["timeout"] == 60.0
    assert state["configuration"]["model"] == "openai/gpt-4o"
    assert state["configuration"]["temperature"] == 0.7

    new_llm = LiteLLM()
    new_llm.load_from_dict(state)
    assert new_llm.api_key == "sk-test"
    assert new_llm.api_base == "http://proxy:4000"
    assert new_llm.timeout == 60.0
    assert new_llm.configuration.model == "openai/gpt-4o"
    assert new_llm.configuration.temperature == 0.7


# ---------------------------------------------------------------------------
# Edge case tests — exception handling, malformed responses, streaming
# ---------------------------------------------------------------------------

def test_auth_error_raises_unrecoverable(mock_litellm):
    """Invalid/expired API key → ModelUnrecoverableError (not retried)."""
    from bridgic.llms.litellm import LiteLLM
    from bridgic.core.model._model_error import ModelUnrecoverableError

    exc_mod = types.ModuleType("litellm.exceptions")
    exc_mod.AuthenticationError = type("AuthenticationError", (Exception,), {})
    sys.modules["litellm.exceptions"] = exc_mod

    mock_litellm.completion.side_effect = exc_mod.AuthenticationError(
        "Invalid API key"
    )
    llm = LiteLLM(api_key="sk-invalid")
    with pytest.raises(ModelUnrecoverableError):
        llm.chat(
            messages=[Message.from_text("Hi", role=Role.USER)],
            model="openai/gpt-4o",
        )
    assert mock_litellm.completion.call_count == 1

    sys.modules.pop("litellm.exceptions", None)


def test_rate_limit_retried(mock_litellm):
    """429 rate limit → retried by retryable_model_call (has '429' in text)."""
    from bridgic.llms.litellm import LiteLLM
    from bridgic.core.model._model_error import ModelRetryLimitError

    rate_exc = Exception("Rate limit exceeded (429)")
    mock_litellm.completion.side_effect = rate_exc
    llm = LiteLLM()
    with pytest.raises(ModelRetryLimitError):
        llm.chat(
            messages=[Message.from_text("Hi", role=Role.USER)],
            model="openai/gpt-4o",
        )
    assert mock_litellm.completion.call_count == 3


def test_timeout_retried(mock_litellm):
    """Timeout → retried by retryable_model_call."""
    from bridgic.llms.litellm import LiteLLM
    from bridgic.core.model._model_error import ModelRetryLimitError

    mock_litellm.completion.side_effect = TimeoutError("Request timed out")
    llm = LiteLLM(timeout=5.0)
    with pytest.raises(ModelRetryLimitError):
        llm.chat(
            messages=[Message.from_text("Hi", role=Role.USER)],
            model="openai/gpt-4o",
        )
    assert mock_litellm.completion.call_count == 3


def test_empty_response_content(mock_litellm):
    """Provider returns empty content → handled gracefully (empty string)."""
    from bridgic.llms.litellm import LiteLLM

    mock_litellm.completion.return_value.choices[0].message.content = None
    mock_litellm.completion.return_value.choices[0].message.refusal = None
    llm = LiteLLM()
    response = llm.chat(
        messages=[Message.from_text("Hi", role=Role.USER)],
        model="openai/gpt-4o",
    )
    assert response.message.content == ""


def test_no_usage_in_response(mock_litellm):
    """Provider returns no usage data → usage is None, not crash."""
    from bridgic.llms.litellm import LiteLLM

    mock_litellm.completion.return_value.usage = None
    mock_litellm.completion.return_value.choices[0].message.content = "OK"
    mock_litellm.completion.return_value.choices[0].message.refusal = None
    llm = LiteLLM()
    response = llm.chat(
        messages=[Message.from_text("Hi", role=Role.USER)],
        model="openai/gpt-4o",
    )
    assert response.usage is None
    assert response.message.content == "OK"


def test_refusal_emits_warning(mock_litellm):
    """Provider refusal → RuntimeWarning emitted."""
    from bridgic.llms.litellm import LiteLLM
    import warnings

    mock_litellm.completion.return_value.choices[0].message.content = ""
    mock_litellm.completion.return_value.choices[0].message.refusal = "Content policy violation"
    llm = LiteLLM()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        llm.chat(
            messages=[Message.from_text("Hi", role=Role.USER)],
            model="openai/gpt-4o",
        )
        assert len(w) == 1
        assert "Content policy violation" in str(w[0].message)
        assert issubclass(w[0].category, RuntimeWarning)


def test_stream_partial_chunks(mock_litellm):
    """Streaming: partial chunks, some with empty delta."""
    from bridgic.llms.litellm import LiteLLM

    chunk1 = mock.MagicMock()
    chunk1.choices = [mock.MagicMock()]
    chunk1.choices[0].delta.content = "Hello"

    chunk2 = mock.MagicMock()
    chunk2.choices = [mock.MagicMock()]
    chunk2.choices[0].delta.content = None

    chunk3 = mock.MagicMock()
    chunk3.choices = [mock.MagicMock()]
    chunk3.choices[0].delta.content = " world"

    chunk4 = mock.MagicMock()
    chunk4.choices = []

    mock_litellm.completion.return_value = iter([chunk1, chunk2, chunk3, chunk4])
    llm = LiteLLM()
    result = ""
    for chunk in llm.stream(
        messages=[Message.from_text("Hi", role=Role.USER)],
        model="openai/gpt-4o",
    ):
        result += chunk.delta
    assert result == "Hello world"


@pytest.mark.asyncio
async def test_astream_partial_chunks(mock_litellm):
    """Async streaming: partial chunks, some with empty delta."""
    from bridgic.llms.litellm import LiteLLM

    chunk1 = mock.MagicMock()
    chunk1.choices = [mock.MagicMock()]
    chunk1.choices[0].delta.content = "Async"

    chunk2 = mock.MagicMock()
    chunk2.choices = [mock.MagicMock()]
    chunk2.choices[0].delta.content = None

    chunk3 = mock.MagicMock()
    chunk3.choices = [mock.MagicMock()]
    chunk3.choices[0].delta.content = " hello"

    async def async_iter():
        for c in [chunk1, chunk2, chunk3]:
            yield c

    mock_litellm.acompletion = mock.AsyncMock(return_value=async_iter())
    llm = LiteLLM()
    result = ""
    async for chunk in llm.astream(
        messages=[Message.from_text("Hi", role=Role.USER)],
        model="openai/gpt-4o",
    ):
        result += chunk.delta
    assert result == "Async hello"


def test_provider_prefixed_model_string(llm_instance, mock_litellm):
    """Model string uses provider-prefix format consistently."""
    llm_instance.chat(
        messages=[Message.from_text("Hi", role=Role.USER)],
        model="anthropic/claude-sonnet-4-6",
    )
    call_kwargs = mock_litellm.completion.call_args[1]
    assert call_kwargs["model"] == "anthropic/claude-sonnet-4-6"
    assert "/" in call_kwargs["model"]


def test_missing_model_raises(mock_litellm):
    """No model in config or call → validation error."""
    from bridgic.llms.litellm import LiteLLM

    llm = LiteLLM()
    with pytest.raises(Exception):
        llm.chat(messages=[Message.from_text("Hi", role=Role.USER)])


def test_extra_kwargs_forwarded(llm_instance, mock_litellm):
    """Extra kwargs (seed, user, etc.) are forwarded to litellm.completion."""
    llm_instance.chat(
        messages=[Message.from_text("Hi", role=Role.USER)],
        seed=42,
        user="test-user",
    )
    call_kwargs = mock_litellm.completion.call_args[1]
    assert call_kwargs["seed"] == 42
    assert call_kwargs["user"] == "test-user"


# ---------------------------------------------------------------------------
# Integration tests (require API key — skipped in CI)
# ---------------------------------------------------------------------------

_api_key = os.environ.get("LITELLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
_model_name = os.environ.get("LITELLM_MODEL") or "openai/gpt-4o-mini"


@pytest.fixture
def live_llm():
    from bridgic.llms.litellm import LiteLLM, LiteLLMConfiguration
    config = LiteLLMConfiguration(model=_model_name)
    return LiteLLM(api_key=_api_key, configuration=config)


@pytest.mark.skipif(
    _api_key is None,
    reason="LITELLM_API_KEY or OPENAI_API_KEY is not set",
)
def test_live_chat(live_llm):
    response = live_llm.chat(
        messages=[Message.from_text(text="Say 'OK' and nothing else.", role=Role.USER)],
    )
    assert response.message.role == Role.AI
    assert response.message.content is not None
    assert response.usage is not None
    assert response.usage.prompt_tokens > 0


@pytest.mark.skipif(
    _api_key is None,
    reason="LITELLM_API_KEY or OPENAI_API_KEY is not set",
)
def test_live_stream(live_llm):
    result = ""
    for chunk in live_llm.stream(
        messages=[Message.from_text(text="Say 'OK' and nothing else.", role=Role.USER)],
    ):
        result += chunk.delta
        assert chunk.raw is not None
    assert len(result) > 0


@pytest.mark.skipif(
    _api_key is None,
    reason="LITELLM_API_KEY or OPENAI_API_KEY is not set",
)
@pytest.mark.asyncio
async def test_live_achat(live_llm):
    response = await live_llm.achat(
        messages=[Message.from_text(text="Say 'OK' and nothing else.", role=Role.USER)],
    )
    assert response.message.role == Role.AI
    assert response.message.content is not None


@pytest.mark.skipif(
    _api_key is None,
    reason="LITELLM_API_KEY or OPENAI_API_KEY is not set",
)
@pytest.mark.asyncio
async def test_live_astream(live_llm):
    result = ""
    async for chunk in live_llm.astream(
        messages=[Message.from_text(text="Say 'OK' and nothing else.", role=Role.USER)],
    ):
        result += chunk.delta
        assert chunk.raw is not None
    assert len(result) > 0
