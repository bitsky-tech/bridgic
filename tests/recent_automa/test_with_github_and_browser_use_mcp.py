"""
Integration tests for ReCentAutoma with GitHub MCP Server and Browser-Use MCP Server.

These tests require:
- OPENAI_API_KEY and OPENAI_MODEL_NAME environment variables
- GITHUB_TOKEN and GITHUB_MCP_HTTP_URL environment variables (for GitHub MCP)
- Chrome browser installed (for Playwright MCP Server)
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
def browser_task_automa_tool_spec(
    openai_llm,
    playwright_mcp_stdio_connection_toolset_builder,
) -> AutomaToolSpec:
    def solve_with_browser(goal: str, guidance: str):
        """
        A specialized tool that use a browser to solve tasks exclusively using browser-based operations.

        IMPORTANT: This tool is designed for browser-solvable tasks only. For complex tasks that involve
        multiple steps or require non-browser capabilities, it is recommended to break down the task into
        browser-solvable sub-tasks first, and then use this tool with each sub-task as the goal.

        Parameters
        ----------
        goal : str
            A specific, browser-solvable sub-task goal described in one or two sentences. The goal should 
            describe a concrete objective that can be achieved through browser operations (e.g., "Search 
            for X on Google and find Y", "Navigate to website Z and extract information about W"). For 
            complex tasks, break them down into smaller browser-solvable sub-tasks and call this tool 
            multiple times with different goals.
        guidance : str
            Additional context, preferences, restrictions, or step-by-step instructions for how the task
            should be accomplished using browser operations. This may include preferred websites, search
            strategies, specific pages to visit, information to extract, or interaction patterns.
            It is recommended to write the guidance in a clear, structured format.
        """
        pass

    @as_tool(solve_with_browser)
    class BrowserTaskAutoma(ReCentAutoma):
        async def arun(self, goal: str, guidance: str, **kwargs):
            guidance += (
                "\n\nImportant: You should not call multiple tools at once. "
                "Always use one tool at a time and proceed step by step. "
                "For multi-step tasks, solve them as a series of single-tool actions."
            )
            return await super().arun(goal=goal, guidance=guidance)

    return AutomaToolSpec.from_raw(
        BrowserTaskAutoma,
        llm=openai_llm,
        tools_builders=[
            playwright_mcp_stdio_connection_toolset_builder,
        ],
        running_options=RunningOptions(debug=True, verbose=False),
    )


@pytest.fixture
def recent_automa_with_github_mcp_and_browser_task_automa(
    openai_llm,
    github_mcp_streamable_http_connection,
    browser_task_automa_tool_spec,
):
    """Create a ReCentAutoma instance with OpenAI LLM, GitHub MCP tools, and Browser-Use MCP tools."""
    # Get tools from both MCP servers
    all_tools = []
    all_tools.extend(github_mcp_streamable_http_connection.list_tools())
    all_tools.extend([browser_task_automa_tool_spec])

    # Create a ReCentAutoma instance with the LLM and combined MCP tools
    return ReCentAutoma(
        llm=openai_llm,
        tools=all_tools,
        stop_condition=StopCondition(max_iteration=1, max_consecutive_no_tool_selected=3),
        memory_config=ReCentMemoryConfig(llm=openai_llm),
        observation_task_config=ObservationTaskConfig(llm=openai_llm),
        tool_task_config=ToolTaskConfig(llm=openai_llm),
        running_options=RunningOptions(debug=True, verbose=False),
    )


@pytest.mark.asyncio
async def test_check_github_stars(
    recent_automa_with_github_mcp_and_browser_task_automa,
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

    result = await recent_automa_with_github_mcp_and_browser_task_automa.arun(
        goal="Please check how many stars bitsky-tech/bridgic currently has.",
        guidance=None,
    )

    assert bool(result)
    assert isinstance(result, str)
    printer.print(f"\n{result}")


@pytest.mark.asyncio
async def test_research_on_revenue_comparison(
    recent_automa_with_github_mcp_and_browser_task_automa,
    openai_api_key,
    openai_model_name,
    github_token,
    github_mcp_url,
):
    """Test research on different technologies in the specified field."""
    if not openai_api_key:
        pytest.skip("OPENAI_API_KEY environment variable must be set")
    if not openai_model_name:
        pytest.skip("OPENAI_MODEL_NAME environment variable must be set")
    if not github_token:
        pytest.skip("GITHUB_TOKEN environment variable must be set")
    if not github_mcp_url:
        pytest.skip("GITHUB_MCP_HTTP_URL environment variable must be set")

    result = await recent_automa_with_github_mcp_and_browser_task_automa.arun(
        goal=(
            "Research on Google Search Engine about the revenue and market performance of:\n"
            "- LangChain/LangGraph\n"
            "- LlamaIndex\n"
            "- CrewAI"
        ),
        guidance="Open more than one browser to solve this complex task.",
    )

    assert bool(result)
    assert isinstance(result, str)
    printer.print(f"\n{result}")
