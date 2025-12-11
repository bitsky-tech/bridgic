import os
import pytest

from bridgic.core.agentic import ReActAutoma
from bridgic.core.automa import RunningOptions
from bridgic.core.utils._console import printer
from bridgic.llms.openai import OpenAILlm, OpenAIConfiguration


@pytest.fixture
def openai_api_key():
    return os.environ.get("OPENAI_API_KEY")


@pytest.fixture
def openai_model_name():
    return os.environ.get("OPENAI_MODEL_NAME")


@pytest.fixture
def llm(openai_api_key, openai_model_name):
    if not openai_api_key:
        pytest.skip("OPENAI_API_KEY environment variable not set")

    if not openai_model_name:
        pytest.skip("OPENAI_MODEL_NAME environment variable not set")

    return OpenAILlm(
        api_key=openai_api_key,
        configuration=OpenAIConfiguration(model=openai_model_name),
        timeout=30,
    )


@pytest.fixture
def react_automa_with_github_mcp(llm, github_mcp_streamable_http_connection):
    # Get all tools from the MCP server connection
    mcp_tools = github_mcp_streamable_http_connection.list_tools()

    # Create a ReActAutoma instance with the LLM and MCP tools
    return ReActAutoma(
        llm=llm,
        system_prompt="You are a helpful assistant that can help users query information about GitHub repositories.",
        tools=mcp_tools,
        running_options=RunningOptions(debug=True),
    )


@pytest.mark.asyncio
async def test_react_automa_github_pull_requests(
    react_automa_with_github_mcp,
    openai_api_key,
    openai_model_name,
    github_token,
    github_mcp_url,
):
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

