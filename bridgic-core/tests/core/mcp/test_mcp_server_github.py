import pytest
import pytest_asyncio


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
