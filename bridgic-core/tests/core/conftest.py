"""
Shared pytest fixtures for core tests.

This module provides shared fixtures for connecting to mock MCP servers,
avoiding connection name conflicts when multiple test files run together.
"""
import pytest
import pytest_asyncio

from bridgic.core.mcp._mcp_server_connection import (
    McpServerConnectionStdio,
    McpServerConnectionStreamableHttp,
)
from tests.core.mcp.mock_servers._server_process import McpHttpServerProcess


@pytest_asyncio.fixture(scope="session")
async def mock_writer_stdio_connection():
    connection = McpServerConnectionStdio(
        name="writer-mcp-stdio",
        command="python",
        args=["tests/core/mcp/mock_servers/mcp_server_writer.py", "--transport", "stdio"],
        request_timeout=5,
    )
    connection.connect()
    yield connection


@pytest_asyncio.fixture(scope="session")
async def mock_writer_streamable_http_connection():
    with McpHttpServerProcess(
        server_script="tests/core/mcp/mock_servers/mcp_server_writer.py",
        transport="streamable_http",
        host="127.0.0.1",
        port=1997,
        startup_timeout=5.0,
    ) as server:
        connection = McpServerConnectionStreamableHttp(
            name="writer-mcp-streamable-http",
            url=server.url,
            request_timeout=5,
        )
        connection.connect()
        yield connection


@pytest_asyncio.fixture(scope="session")
async def mock_crawler_streamable_http_connection():
    with McpHttpServerProcess(
        server_script="tests/core/mcp/mock_servers/mcp_server_crawler.py",
        transport="streamable_http",
        host="127.0.0.1",
        port=1998,
        startup_timeout=5.0,
    ) as server:
        connection = McpServerConnectionStreamableHttp(
            name="crawler-mcp-streamable-http",
            url=server.url,
            request_timeout=5,
        )
        connection.connect()
        yield connection

