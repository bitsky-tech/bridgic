import os
import pytest
import pytest_asyncio


@pytest.mark.asyncio
async def test_mcp_server_stdio_connection(mock_writer_stdio_connection):
    assert mock_writer_stdio_connection._session is not None
    assert mock_writer_stdio_connection._session._request_id > 0
    assert mock_writer_stdio_connection.is_connected == True

@pytest.mark.asyncio
async def test_mcp_server_streamable_http_connection(mock_writer_streamable_http_connection):
    assert mock_writer_streamable_http_connection._session is not None
    assert mock_writer_streamable_http_connection._session._request_id > 0
    assert mock_writer_streamable_http_connection.is_connected == True

@pytest.mark.asyncio
async def test_mcp_server_stdio_connection_list_prompts(mock_writer_stdio_connection):
    result = mock_writer_stdio_connection.list_prompts()
    assert result is not None
    assert len(result) > 0

@pytest.mark.asyncio
async def test_mcp_server_streamable_http_connection_list_prompts(mock_writer_streamable_http_connection):
    result = mock_writer_streamable_http_connection.list_prompts()
    assert result is not None
    assert len(result) > 0

@pytest.mark.asyncio
async def test_mcp_server_stdio_connection_get_prompt(mock_writer_stdio_connection):
    result = mock_writer_stdio_connection.get_prompt(
        prompt_name="ask_for_creative",
        arguments={"topic": "Product Launch", "description": "A new innovative product that combines AI and design"},
    )
    assert result is not None
    assert len(result.messages) > 0

@pytest.mark.asyncio
async def test_mcp_server_streamable_http_connection_get_prompt(mock_writer_streamable_http_connection):
    result = mock_writer_streamable_http_connection.get_prompt(
        prompt_name="ask_for_creative",
        arguments={"topic": "Product Launch", "description": "A new innovative product that combines AI and design"},
    )
    assert result is not None
    assert len(result.messages) > 0

