"""
Integration tests for ReCentAutoma with GitHub MCP Server and Browser-Use MCP Server.

These tests require:
- OPENAI_API_KEY and OPENAI_MODEL_NAME environment variables
- GITHUB_TOKEN and GITHUB_MCP_HTTP_URL environment variables (for GitHub MCP)
- Chrome browser installed (for Browser-Use MCP)
"""
import pytest

from bridgic.core.agentic.recent import ReCentAutoma, ReCentMemoryConfig
from bridgic.core.automa import RunningOptions
from bridgic.core.utils._console import printer


@pytest.fixture
def recent_automa_with_github_and_browser_use_mcp(
    openai_llm,
    vllm_llm,
    github_mcp_streamable_http_connection,
    playwright_mcp_stdio_connection,
):
    """Create a ReCentAutoma instance with OpenAI LLM, GitHub MCP tools, and Browser-Use MCP tools."""
    # Get tools from both MCP servers
    github_tools = github_mcp_streamable_http_connection.list_tools()
    browser_use_tools = playwright_mcp_stdio_connection.list_tools()
    all_tools = github_tools + browser_use_tools
    
    # Create a ReCentAutoma instance with the LLM and combined MCP tools
    return ReCentAutoma(
        llm=vllm_llm,
        tools=all_tools,
        memory_config=ReCentMemoryConfig(llm=openai_llm, max_node_size=10, max_token_size=1024 * 8),
        running_options=RunningOptions(debug=True, verbose=True),
    )


@pytest.mark.asyncio
async def test_check_github_stars(
    recent_automa_with_github_and_browser_use_mcp,
    openai_api_key,
    openai_model_name,
    github_token,
    github_mcp_url,
):
    """Test ReCentAutoma checking GitHub repository stars via MCP."""
    if not openai_api_key:
        pytest.skip("OPENAI_API_KEY environment variable must be set")
    if not openai_model_name:
        pytest.skip("OPENAI_MODEL_NAME environment variable must be set")
    if not github_token:
        pytest.skip("GITHUB_TOKEN environment variable must be set")
    if not github_mcp_url:
        pytest.skip("GITHUB_MCP_HTTP_URL environment variable must be set")

    result = await recent_automa_with_github_and_browser_use_mcp.arun(
        goal="Please check how many stars bitsky-tech/bridgic currently has.",
        guidance=None,
    )

    assert bool(result)
    assert isinstance(result, str)
    printer.print(f"\n{result}")


@pytest.mark.asyncio
async def test_search_and_summarize_ai_products(
    recent_automa_with_github_and_browser_use_mcp,
    openai_api_key,
    openai_model_name,
    github_token,
    github_mcp_url,
):
    """Test ReCentAutoma searching Google and summarizing AI products via MCP."""
    if not openai_api_key:
        pytest.skip("OPENAI_API_KEY environment variable must be set")
    if not openai_model_name:
        pytest.skip("OPENAI_MODEL_NAME environment variable must be set")
    if not github_token:
        pytest.skip("GITHUB_TOKEN environment variable must be set")
    if not github_mcp_url:
        pytest.skip("GITHUB_MCP_HTTP_URL environment variable must be set")

    result = await recent_automa_with_github_and_browser_use_mcp.arun(
        goal="Search for the most popular AI product in 2025 on Google and analyze it in: features and market performance.",
        guidance=(
            "## Initial Search\n"
            "1. Input the query \"the most popular AI products in 2025\" in Google Search.\n"
            "2. Press the Enter key to initiate the search.\n"
            "3. Identify candidate AI products that appear by browsering the top related results.\n"
            "\n"
            "## Deep Research\n"
            "Open three separate tabs for the identified product and search as follows:\n"
            "1. Query: \"[Product name] features 2025\". Focus on its key capabilities, target users, and unique selling points.\n"
            "2. Query: \"[Product name] market performance 2025\". Gather data on financial performance, user growth, and market share.\n"
            "\n"
            "## Synthesis\n"
            "Gather the information and present the summary in a clear, concise format, covering:\n"
            "a. Overview of the product.\n"
            "b. Key features.\n"
            "c. Core technology.\n"
            "d. Revenue/market highlights.\n"
        ),
    )

    assert bool(result)
    assert isinstance(result, str)
    printer.print(f"\n{result}")

