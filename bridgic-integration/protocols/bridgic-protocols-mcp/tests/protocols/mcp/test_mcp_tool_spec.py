import pytest
import pytest_asyncio

from bridgic.protocols.mcp import McpToolSpec
from bridgic.protocols.mcp._mcp_tool_worker import McpToolWorker
from bridgic.core.utils._msgpackx import dump_bytes, load_bytes


@pytest_asyncio.fixture
async def get_weather_tool_spec(mock_crawler_streamable_http_connection):
    """Create an McpToolSpec for the get_weather tool from the crawler server."""
    return McpToolSpec.from_raw(
        tool_name="get_weather",
        server_connection=mock_crawler_streamable_http_connection,
    )

@pytest.mark.asyncio
async def test_mcp_tool_spec(get_weather_tool_spec, mock_crawler_streamable_http_connection):
    """Test that McpToolSpec can create a tool and execute it."""
    # Test to_tool() method
    tool = get_weather_tool_spec.to_tool()
    assert tool.name == "get_weather"
    assert tool.description is not None
    assert tool.parameters is not None
    assert "properties" in tool.parameters
    assert "date" in tool.parameters["properties"]
    assert "city" in tool.parameters["properties"]

    # Test create_worker() method
    my_worker = get_weather_tool_spec.create_worker()
    assert isinstance(my_worker, McpToolWorker)

    # Test that the tool can be executed through the connection
    result = mock_crawler_streamable_http_connection.call_tool(
        tool_name="get_weather",
        arguments={"date": "2024-01-01", "city": "New York"}
    )
    assert result is not None
    assert result.content is not None
    # Verify the result contains expected fields
    assert "temperature" in str(result.content) or "date" in str(result.content)

@pytest.mark.asyncio
async def test_mcp_tool_spec_deserialization(get_weather_tool_spec, mock_crawler_streamable_http_connection):
    """Test that McpToolSpec can be serialized and deserialized."""
    # Serialize
    data = dump_bytes(get_weather_tool_spec)
    assert type(data) is bytes

    # Deserialize
    obj = load_bytes(data)
    assert type(obj) is McpToolSpec
    assert obj.tool_name == "get_weather"
    assert obj.tool_description is not None
    assert obj.tool_parameters is not None
    assert "properties" in obj.tool_parameters
    assert "date" in obj.tool_parameters["properties"]
    assert "city" in obj.tool_parameters["properties"]

    # Verify the deserialized object works correctly
    tool = obj.to_tool()
    assert tool.name == "get_weather"

    # Verify the connection can still be accessed after deserialization
    assert obj.server_connection is mock_crawler_streamable_http_connection
    assert obj.server_connection.name == "crawler-mcp-streamable-http"

    # Verify the tool_info is not None
    assert obj.tool_info is not None    

