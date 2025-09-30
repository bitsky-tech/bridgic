import pytest

from bridgic.core.intelligence.react import ReActAutoma


@pytest.fixture
def react_automa_1() -> ReActAutoma:
    return ReActAutoma(
        llm=None,
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
