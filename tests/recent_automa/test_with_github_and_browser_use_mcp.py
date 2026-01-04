"""
Integration tests for ReCentAutoma with GitHub MCP Server and Browser-Use MCP Server.

These tests require:
- OPENAI_API_KEY and OPENAI_MODEL_NAME environment variables
- GITHUB_TOKEN and GITHUB_MCP_HTTP_URL environment variables (for GitHub MCP)
- Chrome browser installed (for Browser-Use MCP)
"""
from bridgic.core.agentic.tool_specs import AutomaToolSpec, as_tool
import pytest

from bridgic.core.automa import RunningOptions
from bridgic.core.agentic.recent import (
    ReCentAutoma,
    StopCondition,
    ReCentMemoryConfig,
    ObservationTaskConfig,
    ToolTaskConfig,
)
from bridgic.core.utils._console import printer


@pytest.fixture
def research_automa_tool_spec(
    openai_llm,
    playwright_mcp_stdio_connection,
) -> AutomaToolSpec:
    def deep_search(goal: str, guidance: str):
        """
        This tool is a dedicated research agent. It opens a single browser tab and then performs the 
        searching and browsing process step by step, using web resources interactively to ultimately 
        solve a given goal.

        Parameters
        ----------
        goal : str
            The concrete research objective or target to achieve which specify what information the research must obtain.
        guidance : str
            Additional context, preferences, restrictions, or step-by-step instructions for how the 
            research should be conducted. This may include preferred sources, search strategies, 
            comparison criteria, or special constraints.
        """
        pass

    @as_tool(deep_search)
    class ResearchAutoma(ReCentAutoma):
        pass

    all_tools = playwright_mcp_stdio_connection.list_tools()
    return AutomaToolSpec.from_raw(
        ResearchAutoma,
        llm=openai_llm,
        tools=all_tools,
        running_options=RunningOptions(debug=True, verbose=False),
    )


@pytest.fixture
def recent_automa_with_github_and_browser_use_mcp(
    openai_llm,
    github_mcp_streamable_http_connection,
    playwright_mcp_stdio_connection,
    research_automa_tool_spec,
):
    """Create a ReCentAutoma instance with OpenAI LLM, GitHub MCP tools, and Browser-Use MCP tools."""
    # Get tools from both MCP servers
    github_tools = github_mcp_streamable_http_connection.list_tools()
    browser_use_tools = playwright_mcp_stdio_connection.list_tools()
    # all_tools = github_tools + browser_use_tools + [research_automa_tool_spec]
    all_tools = [research_automa_tool_spec]
    
    # Create a ReCentAutoma instance with the LLM and combined MCP tools
    return ReCentAutoma(
        llm=openai_llm,
        tools=all_tools,
        stop_condition=StopCondition(max_iteration=-1, max_consecutive_no_tool_selected=3),
        memory_config=ReCentMemoryConfig(llm=openai_llm),
        observation_task_config=ObservationTaskConfig(llm=openai_llm),
        tool_task_config=ToolTaskConfig(llm=openai_llm),
        running_options=RunningOptions(debug=True, verbose=False),
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
        goal="Try to research the most popular AI product in 2025 on Google.",
        guidance=(
            "## Initial Search\n"
            "1. Search on Google for \"the most popular AI products in 2025\".\n"
            "2. Try to determine which is the most popular one among the candidates.\n"
            "\n"
            "## Research More\n"
            "1. Open a new browser tab and then research on \"[The product name] features 2025\".\n"
            "2. Open a new browser tab and then research on \"[The product name] market performance 2025\".\n"
            "\n"
            "## Synthesis\n"
            "Gather the information and present the summary in a clear, concise format, covering:\n"
            "a. Overview of the product.\n"
            "b. Key features.\n"
            "c. Core technology.\n"
            "d. Revenue/market highlights.\n"
            "\n"
            "## Notes\n"
            "If human validation is needed, please wait for 3 seconds and then recheck the result."
        ),
    )

    assert bool(result)
    assert isinstance(result, str)
    printer.print(f"\n{result}")
