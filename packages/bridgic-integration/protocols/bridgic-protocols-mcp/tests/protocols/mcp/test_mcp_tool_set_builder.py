"""
Tests for McpToolSetBuilder.
"""
import pytest

from bridgic.core.config import HttpClientConfig, HttpClientTimeoutConfig
from bridgic.protocols.mcp import McpToolSetBuilder
from bridgic.protocols.mcp._mcp_tool_spec import McpToolSpec


@pytest.mark.asyncio
async def test_mcp_toolset_builder_stdio():
    """Test McpToolSetBuilder with stdio connection."""
    # Create builder for all tools
    builder = McpToolSetBuilder.stdio(
        command="python",
        args=["tests/protocols/mcp/mock_servers/mcp_server_writer.py", "--transport", "stdio"],
        request_timeout=5,
    )
    
    # Build tool specs
    response = builder.build()
    
    # Verify response structure
    assert "tool_specs" in response
    assert "extras" in response
    assert "mcp_server_connection_name" in response["extras"]
    
    tool_specs = response["tool_specs"]
    assert len(tool_specs) > 0
    
    # Verify all tool specs are from builder
    for tool_spec in tool_specs:
        assert isinstance(tool_spec, McpToolSpec)
        assert tool_spec._from_builder is True


@pytest.mark.asyncio
async def test_mcp_toolset_builder_streamable_http(mock_crawler_http_server):
    """Test McpToolSetBuilder with streamable HTTP connection."""
    # Create builder for all tools
    builder = McpToolSetBuilder.streamable_http(
        url=mock_crawler_http_server.url,
        http_client_config=HttpClientConfig(
            timeout=HttpClientTimeoutConfig(read=5),
        ),
        request_timeout=5,
    )
    
    # Build tool specs
    response = builder.build()
    
    # Verify response structure
    assert "tool_specs" in response
    assert "extras" in response
    assert "mcp_server_connection_name" in response["extras"]
    
    tool_specs = response["tool_specs"]
    assert len(tool_specs) > 0
    
    # Verify all tool specs are from builder
    for tool_spec in tool_specs:
        assert isinstance(tool_spec, McpToolSpec)
        assert tool_spec._from_builder is True
