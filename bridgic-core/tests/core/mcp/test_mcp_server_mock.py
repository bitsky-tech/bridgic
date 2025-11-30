import os
import pytest
import pytest_asyncio

from bridgic.core.mcp._mcp_server_connection import (
    McpServerConnectionStdio,
    McpServerConnectionStreamableHttp,
)
from tests.core.mcp.mock_servers._server_process import McpHttpServerProcess

@pytest_asyncio.fixture
async def mock_writer_stdio_connection():
    connection = McpServerConnectionStdio(
        name="writer-mcp-stdio",
        command="python",
        args=["tests/core/mcp/mock_servers/mcp_server_writer.py", "--transport", "stdio"],
        request_timeout=5,
    )
    connection.connect()
    yield connection

@pytest_asyncio.fixture
async def mock_writer_streamable_http_connection():
    with McpHttpServerProcess(
        server_script="tests/core/mcp/mock_servers/mcp_server_writer.py",
        transport="streamable_http",
        host="127.0.0.1",
        port=1997,
        startup_timeout=10.0,
    ) as server:
        connection = McpServerConnectionStreamableHttp(
            name="writer-mcp-streamable-http",
            url=server.url,
            request_timeout=5,
        )
        connection.connect()
        yield connection

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