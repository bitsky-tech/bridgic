import pytest
import pytest_asyncio
from mcp.types import Prompt

from bridgic.core.mcp._mcp_server_connection import (
    McpServerConnectionStdio,
    McpServerConnectionStreamableHttp
)
from bridgic.core.prompt._mcp_template import McpPromptTemplate
from bridgic.core.model.types import Message, Role
from tests.core.mcp.mock_servers._server_process import McpHttpServerProcess


@pytest_asyncio.fixture
async def mock_writer_stdio_connection():
    connection = McpServerConnectionStdio(
        name="writer-mcp-stdio",
        command="python",
        args=["tests/core/mcp/mock_servers/mcp_server_writer.py", "--transport", "stdio"],
        request_timeout=8,
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
            request_timeout=30,
        )
        connection.connect()
        yield connection

@pytest.mark.asyncio
async def test_mcp_prompt_template_stdio(mock_writer_stdio_connection):
    template = McpPromptTemplate(
        prompt_name="ask_for_creative",
        prompt_info=Prompt(name="ask_for_creative", description="Ask for a creative idea"),
        server_connection=mock_writer_stdio_connection,
    )

    messages = template.format_messages(
        topic="Product Launch",
        description="A new innovative product that combines AI and design",
    )

    assert isinstance(messages, list)
    assert len(messages) > 0

    for msg in messages:
        assert isinstance(msg, Message)
        assert msg.role in [Role.SYSTEM, Role.USER, Role.AI]
        assert len(msg.blocks) > 0

@pytest.mark.asyncio
async def test_mcp_prompt_template_streamable_http(mock_writer_streamable_http_connection):
    template = McpPromptTemplate(
        prompt_name="ask_for_creative",
        prompt_info=Prompt(name="ask_for_creative", description="Ask for a creative idea"),
        server_connection=mock_writer_streamable_http_connection,
    )

    messages = template.format_messages(
        topic="Product Launch",
        description="A new innovative product that combines AI and design",
    )

    assert isinstance(messages, list)
    assert len(messages) > 0

    for msg in messages:
        assert isinstance(msg, Message)
        assert msg.role in [Role.SYSTEM, Role.USER, Role.AI]
        assert len(msg.blocks) > 0