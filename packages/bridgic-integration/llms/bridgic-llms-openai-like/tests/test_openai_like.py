import pytest
import os
import httpx_aiohttp
from types import SimpleNamespace

from bridgic.core.model.types import *
from bridgic.core.model import ModelRetryLimitError, ModelUnrecoverableError
from bridgic.core.utils._console import printer
from bridgic.llms.openai_like import OpenAILikeLlm

_api_base = os.environ.get("OPENAI_LIKE_API_BASE")
_api_key = os.environ.get("OPENAI_LIKE_API_KEY")
_model_name = os.environ.get("OPENAI_LIKE_MODEL_NAME")


def _mock_chat_completion(content: str):
    """Build the minimal OpenAI-like response shape used by OpenAILikeLlm."""
    message = SimpleNamespace(content=content, refusal=None)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def test_openai_like_chat_recoverable_error(monkeypatch):
    llm = OpenAILikeLlm(api_base="http://test.local", api_key="test-key")
    attempts = {"count": 0}

    def fake_create(**kwargs):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise TimeoutError("temporary timeout")
        return _mock_chat_completion("ok")

    monkeypatch.setattr(llm.client.chat.completions, "create", fake_create)

    response = llm.chat(
        messages=[Message.from_text(text="hello", role=Role.USER)],
        model="test-model",
    )
    assert response.message is not None
    assert response.message.content == "ok"
    assert attempts["count"] == 3


def test_openai_like_chat_unrecoverable_error(monkeypatch):
    llm = OpenAILikeLlm(api_base="http://test.local", api_key="test-key")

    def fake_create(**kwargs):
        raise ValueError("invalid request")

    monkeypatch.setattr(llm.client.chat.completions, "create", fake_create)

    with pytest.raises(ModelUnrecoverableError):
        llm.chat(
            messages=[Message.from_text(text="hello", role=Role.USER)],
            model="test-model",
        )


@pytest.mark.asyncio
async def test_openai_like_achat_recoverable_error(monkeypatch):
    llm = OpenAILikeLlm(api_base="http://test.local", api_key="test-key")
    attempts = {"count": 0}

    async def fake_create(**kwargs):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise ConnectionError("temporary connection issue")
        return _mock_chat_completion("ok-async")

    monkeypatch.setattr(llm.async_client.chat.completions, "create", fake_create)

    response = await llm.achat(
        messages=[Message.from_text(text="hello", role=Role.USER)],
        model="test-model",
    )
    assert response.message is not None
    assert response.message.content == "ok-async"
    assert attempts["count"] == 3


@pytest.mark.asyncio
async def test_openai_like_achat_retry_limit_error(monkeypatch):
    llm = OpenAILikeLlm(api_base="http://test.local", api_key="test-key")

    async def fake_create(**kwargs):
        raise TimeoutError("still timing out")

    monkeypatch.setattr(llm.async_client.chat.completions, "create", fake_create)

    with pytest.raises(ModelRetryLimitError):
        await llm.achat(
            messages=[Message.from_text(text="hello", role=Role.USER)],
            model="test-model",
        )

@pytest.fixture
def llm():
    llm = OpenAILikeLlm(
        api_base=_api_base,
        api_key=_api_key,
        timeout=5,
    )
    state_dict = llm.dump_to_dict()
    del llm
    llm = OpenAILikeLlm.__new__(OpenAILikeLlm)
    llm.load_from_dict(state_dict)
    return llm

@pytest.mark.skipif(
    (_api_key is None) or (_api_base is None) or (_model_name is None),
    reason="OPENAI_LIKE_API_KEY or OPENAI_LIKE_API_BASE or OPENAI_LIKE_MODEL_NAME is not set",
)
def test_openai_like_chat(llm):
    response = llm.chat(
        model=_model_name,
        messages=[Message.from_text(text="Hello, how are you?", role=Role.USER)],
    )
    printer.print(response)
    assert response.message.role == Role.AI
    assert response.message.content is not None

@pytest.mark.skipif(
    (_api_key is None) or (_api_base is None) or (_model_name is None),
    reason="OPENAI_LIKE_API_KEY or OPENAI_LIKE_API_BASE or OPENAI_LIKE_MODEL_NAME is not set",
)
def test_openai_like_stream(llm):
    response = llm.stream(
        model=_model_name,
        messages=[Message.from_text(text="Hello, how are you?", role=Role.USER)],
    )
    result = ""
    for chunk in response:
        result += chunk.delta
        assert chunk.delta is not None
        assert chunk.raw is not None
    assert len(result) > 0

@pytest.mark.skipif(
    (_api_key is None) or (_api_base is None) or (_model_name is None),
    reason="OPENAI_LIKE_API_KEY or OPENAI_LIKE_API_BASE or OPENAI_LIKE_MODEL_NAME is not set",
)
@pytest.mark.asyncio
async def test_openai_like_achat(llm):
    response = await llm.achat(
        model=_model_name,
        messages=[Message.from_text(text="Hello, how are you?", role=Role.USER)],
    )
    printer.print(response)
    assert response.message.role == Role.AI
    assert response.message.content is not None

@pytest.mark.skipif(
    (_api_key is None) or (_api_base is None) or (_model_name is None),
    reason="OPENAI_LIKE_API_KEY or OPENAI_LIKE_API_BASE or OPENAI_LIKE_MODEL_NAME is not set",
)
@pytest.mark.asyncio
async def test_openai_like_astream(llm):
    response = llm.astream(
        model=_model_name,
        messages=[Message.from_text(text="Hello, how are you?", role=Role.USER)],
    )
    result = ""
    async for chunk in response:
        result += chunk.delta
        assert chunk.delta is not None
        assert chunk.raw is not None
    assert len(result) > 0

@pytest.mark.skipif(
    (_api_key is None) or (_api_base is None) or (_model_name is None),
    reason="OPENAI_LIKE_API_KEY or OPENAI_LIKE_API_BASE or OPENAI_LIKE_MODEL_NAME is not set",
)
@pytest.mark.asyncio
async def test_openai_like_achat_with_aiohttp():
    async with httpx_aiohttp.HttpxAiohttpClient() as aio_client:
        llm = OpenAILikeLlm(
            api_base=_api_base,
            api_key=_api_key,
            http_async_client=aio_client,
        )
        response = await llm.achat(
            model=_model_name,
            messages=[Message.from_text(text="Hello, how are you?", role=Role.USER)],
        )
        printer.print(response)
        assert response.message.role == Role.AI
        assert response.message.content is not None

@pytest.mark.skipif(
    (_api_key is None) or (_api_base is None) or (_model_name is None),
    reason="OPENAI_LIKE_API_KEY or OPENAI_LIKE_API_BASE or OPENAI_LIKE_MODEL_NAME is not set",
)
@pytest.mark.asyncio
async def test_openai_like_astream_with_aiohttp():
    async with httpx_aiohttp.HttpxAiohttpClient() as aio_client:
        llm = OpenAILikeLlm(
            api_base=_api_base,
            api_key=_api_key,
            http_async_client=aio_client,
        )
        response = llm.astream(
            model=_model_name,
            messages=[Message.from_text(text="Hello, how are you?", role=Role.USER)],
        )
        result = ""
        async for chunk in response:
            result += chunk.delta
            assert chunk.delta is not None
            assert chunk.raw is not None
        assert len(result) > 0