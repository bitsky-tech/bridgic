import pytest
from enum import Enum
import os

from bridgic.core.automa import GraphAutoma, worker
from bridgic.core.intelligence import as_tool
from bridgic.core.intelligence.protocol import ToolSelection
from bridgic.core.intelligence.react import ReActAutoma
from bridgic.llms.openai.openai_llm import OpenAILlm, OpenAIConfiguration
from tests.core.intelligence.mock_llm import MockLlm

_api_key = os.environ.get("OPENAI_API_KEY")
_model_name = os.environ.get("OPENAI_MODEL_NAME", default="gpt-5-mini")

@pytest.fixture
def llm() -> ToolSelection:
    if _api_key:
        print(f"\nUsing `OpenAILlm` ({_model_name}) to test ReactAutoma...")
        return OpenAILlm(
            api_key=_api_key,
            configuration=OpenAIConfiguration(model=_model_name),
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
            "name": "Jack"
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
