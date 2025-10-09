import pytest
from enum import Enum

from bridgic.core.intelligence.react import ReActAutoma


class WeatherUnit(Enum):
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"

async def get_weather(
    location: str,
    unit: WeatherUnit,
) -> str:
    """
    Retrieves current weather for the given location.

    Parameters
    ----------
    location : str
        City and country e.g. Bogotá, Colombia.
    unit : WeatherUnit
        Units the temperature will be returned in.
    
    Returns
    -------
    str
        The weather for the given location.
    """
    ...


@pytest.fixture
def react_automa_1() -> ReActAutoma:
    return ReActAutoma(
        llm=None,
        tools=[get_weather],
        system_prompt="You are a helpful assistant in a customer service scenario."
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
            "content": "Hello, how are you?",
            "name": "Jack"
        },
        {
            "role": "assistant",
            "content": "I am a helpful assistant."
        },
        {
            "role": "user",
            "content": "What is the weather in Tokyo?",
            "name": "Jack"
        }
    ]
    result = await react_automa_1.arun(messages=messages)

@pytest.mark.asyncio
async def test_react_automa_case_2(react_automa_1: ReActAutoma):
    result = await react_automa_1.arun(
        user_msg={
            "role": "user",
            "content": "Hello, how are you?",
            "name": "user_1"
        },
        chat_history=[
            {
                "role": "user",
                "content": "Do you know the weather in Tokyo?",
                "name": "user_1"
            },
            {
                "role": "assistant",
                "content": "It's sunny in Tokyo today.",
                "name": "assistant_1"
            }
        ]
    )
