"""
Integration tests for ReActAutoma with GitHub MCP Server.

These tests require:
- OPENAI_API_KEY and OPENAI_MODEL_NAME environment variables
- GITHUB_TOKEN and GITHUB_MCP_HTTP_URL environment variables
"""
import pytest

from bridgic.core.agentic import ReActAutoma
from bridgic.core.automa import RunningOptions
from bridgic.core.utils._console import printer


@pytest.fixture
def react_automa_with_github_mcp(openai_llm, github_mcp_streamable_http_connection):
    """Create a ReActAutoma instance with OpenAI LLM and GitHub MCP tools."""
    # Get all tools from the MCP server connection
    mcp_tools = github_mcp_streamable_http_connection.list_tools()

    # Create a ReActAutoma instance with the LLM and MCP tools
    return ReActAutoma(
        llm=openai_llm,
        system_prompt="You are a helpful assistant that can help users query information about GitHub repositories.",
        tools=mcp_tools,
        running_options=RunningOptions(debug=True),
    )


@pytest.mark.asyncio
async def test_query_for_pull_requests(
    react_automa_with_github_mcp,
    openai_api_key,
    openai_model_name,
    github_token,
    github_mcp_url,
):
    """Test ReActAutoma querying GitHub pull requests via MCP."""
    if not openai_api_key:
        pytest.skip("OPENAI_API_KEY environment variable must be set")
    if not openai_model_name:
        pytest.skip("OPENAI_MODEL_NAME environment variable must be set")
    if not github_token:
        pytest.skip("GITHUB_TOKEN environment variable must be set")
    if not github_mcp_url:
        pytest.skip("GITHUB_MCP_HTTP_URL environment variable must be set")

    result = await react_automa_with_github_mcp.arun(
        user_msg="Please show me the recent pull requests for the bitsky-tech/bridgic repository."
    )

    assert bool(result)
    assert isinstance(result, str)
    printer.print(f"\nTest result: {result}")








