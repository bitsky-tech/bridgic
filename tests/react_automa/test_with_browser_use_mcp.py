"""
Integration tests for ReActAutoma with Browser-Use MCP Server.

These tests require:
- OPENAI_API_KEY and OPENAI_MODEL_NAME environment variables
"""
import pytest

from bridgic.core.agentic import ReActAutoma
from bridgic.core.automa import RunningOptions
from bridgic.core.utils._console import printer


@pytest.fixture
def react_automa_with_playwright_mcp_stdio(openai_llm, playwright_mcp_stdio_connection):
    """Create a ReActAutoma instance with OpenAI LLM and Browser-Use MCP tools."""
    forbidden_tools = ["browser_evaluate"]
    available_tools = [tool for tool in playwright_mcp_stdio_connection.list_tools() if tool.tool_name not in forbidden_tools]

    return ReActAutoma(
        llm=openai_llm,
        system_prompt=(
            "You are a helpful assistant. You are supposed to use the available tools to help answer the user's questions. "
        ),
        tools=available_tools,
        running_options=RunningOptions(debug=True),
        max_iterations=50,
    )


@pytest.mark.asyncio
async def test_browser_use_ask_deepseek(
    react_automa_with_playwright_mcp_stdio,
    openai_api_key,
    openai_model_name,
):
    """Test ReActAutoma asking DeepSeek a question and getting the answer."""
    if not openai_api_key:
        pytest.skip("OPENAI_API_KEY environment variable must be set")
    if not openai_model_name:
        pytest.skip("OPENAI_MODEL_NAME environment variable must be set")

    result = await react_automa_with_playwright_mcp_stdio.arun(
        user_msg=(
            "Task Objective:\n"
            "Log in to DeepSeek at https://chat.deepseek.com/ and ask the following questions one by one, then summarize and report the answers:\n"
            "- What is the capital city of France?\n"
            "- What is the weather like in that city?\n"
            "\n"
            "Step-by-step Instructions:\n"
            "1. Open the website https://chat.deepseek.com/.\n"
            "2. Repeat the following steps until login is complete:\n"
            "  a. Take a snapshot of the page.\n"
            "  b. Determine if login is required:\n"
            "    i. If login is needed, please wait a moment and then recheck the login status (I will handle any verification during this period).\n"
            "    ii. If no login is required, exit the loop and proceed to the next step.\n"
            "3. After login, adjust the QA mode as needed:\n"
            "  a. Do not activate the 'DeepThink' mode.\n"
            "  b. Ensure 'Internet Search' mode is activated.\n"
            "4. For each question, carry out the following actions:\n"
            "  a. Find the input box (the empty input box will show 'Message DeepSeek'), input the question to be asked.\n"
            "  b. Press Enter to send the question.\n"
            "  c. After sending, the sending button will be changed to like \"Stop\". Please do not click it but wait.\n"
            "  d. Patiently wait for DeepSeek to complete its streaming reply. Every 10 seconds, take a simple snapshot and check if the answer is complete. When buttons like 'Copy', 'Regenerate', 'Like', 'Dislike', or 'Share' (or similar) appear on the page, it indicates the streaming reply is complete.\n"
            "  e. Extract and save the complete answer, then proceed to the next question.\n"
            "5. After all questions have been answered, organize and summarize the results, then report back to me.\n"
            "\n"
            "Notes:\n"
            "1. Try to avoid using verbose mode for page snapshots each time\n"
            "2. Use at most one browser tool at a time; strictly perform operations serially, do not use multiple tools concurrently\n"
            "3. For any wait, keep the waiting time under 5 seconds.\n"
            "4. For each question, make sure the answer is fully generated in the current round before entering the next question.\n"
        )
    )

    assert bool(result)
    assert isinstance(result, str)
    printer.print(f"\n{result}")


