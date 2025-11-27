import os
import pytest
import pytest_asyncio
import shutil

from bridgic.core.mcp._mcp_server_connection import (
    McpServerConnectionStdio,
    McpServerConnectionStreamableHttp,
)


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
        request_timeout=30,
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
        request_timeout=30,
    )

    connection.connect()
    yield connection


@pytest.mark.asyncio
async def test_github_mcp_server_stdio_connection(github_mcp_stdio_connection):
    assert github_mcp_stdio_connection._session is not None
    assert github_mcp_stdio_connection._session._request_id > 0
    assert github_mcp_stdio_connection.is_connected == True


@pytest.mark.asyncio
async def test_github_mcp_server_streamable_http_connection(github_mcp_streamable_http_connection):
    assert github_mcp_streamable_http_connection._session is not None
    assert github_mcp_streamable_http_connection._session._request_id > 0
    assert github_mcp_streamable_http_connection.is_connected == True
