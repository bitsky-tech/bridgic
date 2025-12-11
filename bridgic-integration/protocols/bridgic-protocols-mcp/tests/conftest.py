"""
Shared pytest fixtures for MCP protocol tests.

This module provides shared fixtures for connecting to mock MCP servers,
avoiding connection name conflicts when multiple test files run together.
"""
import os
import shutil
import pytest
import pytest_asyncio

from bridgic.protocols.mcp import (
    McpServerConnectionStdio,
    McpServerConnectionStreamableHttp,
)
from tests.protocols.mcp.mock_servers._server_process import McpHttpServerProcess


@pytest_asyncio.fixture(scope="session")
async def mock_writer_stdio_connection():
    connection = McpServerConnectionStdio(
        name="writer-mcp-stdio",
        command="python",
        args=["tests/protocols/mcp/mock_servers/mcp_server_writer.py", "--transport", "stdio"],
        request_timeout=5,
    )
    connection.connect()
    yield connection
    connection.close()


@pytest_asyncio.fixture(scope="session")
async def mock_writer_streamable_http_connection():
    with McpHttpServerProcess(
        server_script="tests/protocols/mcp/mock_servers/mcp_server_writer.py",
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
        connection.close()


@pytest_asyncio.fixture(scope="session")
async def mock_crawler_streamable_http_connection():
    with McpHttpServerProcess(
        server_script="tests/protocols/mcp/mock_servers/mcp_server_crawler.py",
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
        connection.close()


@pytest.fixture
def has_npx():
    return shutil.which("npx") is not None


@pytest.fixture
def github_token():
    return os.environ.get("GITHUB_TOKEN")


@pytest.fixture
def github_mcp_url():
    return os.environ.get("GITHUB_MCP_HTTP_URL")


@pytest_asyncio.fixture
async def github_mcp_stdio_connection(has_npx, github_token):
    if not has_npx:
        pytest.skip("npx is not available")

    if not github_token:
        pytest.skip("GITHUB_TOKEN environment variable not set")

    connection = McpServerConnectionStdio(
        name="github-mcp-stdio",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        env={"GITHUB_TOKEN": github_token},
        request_timeout=5,
    )

    connection.connect()
    yield connection


@pytest_asyncio.fixture
async def github_mcp_streamable_http_connection(github_mcp_url, github_token):
    if not github_mcp_url:
        pytest.skip("GITHUB_MCP_HTTP_URL environment variable not set")

    if not github_token:
        pytest.skip("GITHUB_TOKEN environment variable not set")

    connection = McpServerConnectionStreamableHttp(
        name="github-mcp-streamable-http",
        url=github_mcp_url,
        headers={"Authorization": f"Bearer {github_token}"},
        request_timeout=10,
    )

    connection.connect()
    yield connection

