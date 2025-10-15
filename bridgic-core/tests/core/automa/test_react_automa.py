import pytest
from enum import Enum
import os

from bridgic.core.automa import GraphAutoma, worker
from bridgic.core.model.protocol import ToolSelection
from bridgic.core.agentic.tool import as_tool
from bridgic.core.agentic.react import ReActAutoma
from bridgic.llms.openai.openai_llm import OpenAILlm, OpenAIConfiguration
from bridgic.llms.vllm.vllm_server_llm import VllmServerLlm, OpenAILikeConfiguration
from tests.core.intelligence.mock_llm import MockLlm

_openai_api_key = os.environ.get("OPENAI_API_KEY")
_openai_model_name = os.environ.get("OPENAI_MODEL_NAME", default="gpt-5-mini")

_vllm_api_base = os.environ.get("VLLM_SERVER_API_BASE")
_vllm_api_key = os.environ.get("VLLM_SERVER_API_KEY", default="EMPTY")
_vllm_model_name = os.environ.get("VLLM_SERVER_MODEL_NAME", default="Qwen/Qwen3-4B-Instruct-2507")

@pytest.fixture
def llm() -> ToolSelection:
    # Use OpenAI LLM by setting environment variables:
    # export OPENAI_API_KEY="xxx"
    # export OPENAI_MODEL_NAME="xxx"
    if _openai_api_key:
        print(f"\nUsing `OpenAILlm` ({_openai_model_name}) to test ReactAutoma...")
        return OpenAILlm(
            api_key=_openai_api_key,
            configuration=OpenAIConfiguration(model=_openai_model_name),
            timeout=10,
        )
    # Use VLLM Server LLM by setting environment variables:
    # export VLLM_SERVER_API_KEY="xxx"
    # export VLLM_SERVER_API_BASE="xxx"
    # export VLLM_SERVER_MODEL_NAME="xxx"
    if _vllm_api_base:
        print(f"\nUsing `VllmServerLlm` ({_vllm_model_name}) to test ReactAutoma...")
        return VllmServerLlm(
            api_key=_vllm_api_key,
            api_base=_vllm_api_base,
            configuration=OpenAILikeConfiguration(model=_vllm_model_name),
            timeout=10,
        )

    print(f"\nUsing `MockLlm` to test ReactAutoma...")
    return MockLlm()


################################################################################
# Test Case 1.
# Configurations:
# - single tool `get_weather`.
# - tools are provided at initialization.
# - inputs are whole `messages` list.
# - functional tool `get_weather`, async def.
################################################################################

async def get_weather(
    city: str,
) -> str:
    """
    Retrieves current weather for the given city.

    Parameters
    ----------
    city : str
        The city to get the weather of, e.g. New York.
    
    Returns
    -------
    str
        The weather for the given city.
    """
    # Mock the weather API call.
    return f"The weather in {city} is sunny today and the temperature is 20 degrees Celsius."

@pytest.fixture
def react_automa_1(llm: ToolSelection) -> ReActAutoma:
    return ReActAutoma(
        llm=llm,
        tools=[get_weather],
    )

@pytest.mark.asyncio
async def test_react_automa_case_1(react_automa_1: ReActAutoma):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "What is the weather in Tokyo?",
        }
    ]
    result = await react_automa_1.arun(messages=messages)
    assert bool(result)
    assert "sunny" in result
    assert "20" in result
    assert "not able to select tools" not in result


################################################################################
# Test Case 2.
# Configurations:
# - single tool `multiply`.
# - tools are provided at runtime.
# - inputs use `user_msg` and `chat_history`.
# - Automa tool, non-async method `multiply()`.
################################################################################

def multiply(x: int, y: int) -> int:
    """
    This function is used to multiply two numbers.

    Parameters
    ----------
    x : int
        The first number to multiply
    y : int
        The second number to multiply

    Returns
    -------
    int
        The product of the two numbers
    """
    # Note: this function need not to be implemented.
    ...

@as_tool(multiply)
class MultiplyAutoma(GraphAutoma):
    @worker(is_start=True, is_output=True)
    def multiply(self, x: int, y: int):
        return x * y

@pytest.fixture
def react_automa_2(llm: ToolSelection) -> ReActAutoma:
    return ReActAutoma(
        llm=llm,
        system_prompt="You are a helpful assistant that are good at calculating by using tools.",
    )

@pytest.mark.asyncio
async def test_react_automa_case_2(react_automa_2: ReActAutoma):
    result = await react_automa_2.arun(
        user_msg="What is 235 * 4689?",
        chat_history=[
            {
                "role": "user",
                "content": "Could you help me to do some calculations?",
            },
            {
                "role": "assistant",
                "content": "Of course, I can help you with that.",
            }
        ],
        # tools are provided at runtime here.
        tools=[MultiplyAutoma],
    )
    assert bool(result)
    assert "1101915" in result or "1,101,915" in result
    assert "not able to select tools" not in result
