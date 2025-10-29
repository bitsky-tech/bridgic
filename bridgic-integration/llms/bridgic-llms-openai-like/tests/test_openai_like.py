import pytest
import os
import httpx_aiohttp

from bridgic.core.model.types import *
from bridgic.core.utils._console import printer
from bridgic.llms.openai_like import OpenAILikeLlm

_api_base = os.environ.get("OPENAI_LIKE_API_BASE")
_api_key = os.environ.get("OPENAI_LIKE_API_KEY")
_model_name = os.environ.get("OPENAI_LIKE_MODEL_NAME")

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