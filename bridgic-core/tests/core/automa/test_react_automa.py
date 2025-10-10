# import pytest
# from enum import Enum
# import os

# from bridgic.core.intelligence.react import ReActAutoma
# from bridgic.llms.openai.openai_llm import OpenAILlm

# _api_key = os.environ.get("OPENAI_API_KEY")

# @pytest.fixture
# def llm():
#     return OpenAILlm(
#         api_key=_api_key,
#         timeout=10,
#     )

# async def get_weather(
#     city: str,
# ) -> str:
#     """
#     Retrieves current weather for the given city.

#     Parameters
#     ----------
#     city : str
#         The city to get the weather of, e.g. New York.
    
#     Returns
#     -------
#     str
#         The weather for the given city.
#     """
#     # Mock the weather API call.
#     return f"The weather in {city} is sunny today and the temperature is 20 degrees Celsius."


# @pytest.fixture
# def react_automa_1(llm: OpenAILlm) -> ReActAutoma:
#     return ReActAutoma(
#         llm=llm,
#         tools=[get_weather],
#         system_prompt="You are a helpful assistant in a customer service scenario."
#     )

# @pytest.mark.asyncio
# async def test_react_automa_case_1(react_automa_1: ReActAutoma):
#     messages = [
#         {
#             "role": "system",
#             "content": "You are a helpful assistant."
#         },
#         # {
#         #     "role": "user",
#         #     "content": "Hello, how are you?",
#         #     "name": "Jack"
#         # },
#         # {
#         #     "role": "assistant",
#         #     "content": "I am a helpful assistant."
#         # },
#         {
#             "role": "user",
#             "content": "What is the weather in Tokyo?",
#             "name": "Jack"
#         }
#     ]
#     result = await react_automa_1.arun(messages=messages)

# @pytest.mark.asyncio
# async def test_react_automa_case_2(react_automa_1: ReActAutoma):
#     result = await react_automa_1.arun(
#         user_msg={
#             "role": "user",
#             "content": "Hello, how are you?",
#             "name": "user_1"
#         },
#         chat_history=[
#             {
#                 "role": "user",
#                 "content": "Do you know the weather in Tokyo?",
#                 "name": "user_1"
#             },
#             {
#                 "role": "assistant",
#                 "content": "It's sunny in Tokyo today.",
#                 "name": "assistant_1"
#             }
#         ]
#     )
