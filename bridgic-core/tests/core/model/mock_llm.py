from typing import List, Tuple, Optional, Dict, Any
from typing_extensions import override
import json

from bridgic.core.model.types import *
from bridgic.core.model import BaseLlm
from bridgic.core.model.protocols import ToolSelection
from bridgic.core.model.types import Message


class MockLlm(BaseLlm, ToolSelection):

    @override
    def dump_to_dict(self) -> Dict[str, Any]:
        return {}

    @override
    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        ...

    @override
    def chat(self, messages: List[Message], **kwargs) -> Response:
        return Response(message=Message.from_text(text="Hello! I am a Mock LLM.", role=Role.AI))

    @override
    def stream(self, messages: List[Message], **kwargs) -> StreamResponse:
        content = "Hello! I am a Mock LLM."
        for chunk in content.split(" "):
            yield MessageChunk(delta=chunk, raw=chunk)

    @override
    async def achat(self, messages: List[Message], **kwargs) -> Response:
        return Response(message=Message.from_text(text="Hello! I am a Mock LLM.", role=Role.AI))

    @override
    async def astream(self, messages: List[Message], **kwargs) -> AsyncStreamResponse:
        content = "Hello! I am a Mock LLM."
        for chunk in content.split(" "):
            yield MessageChunk(delta=chunk, raw=chunk)

    @override
    def select_tool(
        self,
        messages: List[Message],
        tools: List[Tool],
        **kwargs,
    ) -> Tuple[List[ToolCall], Optional[str]]:

        if len(tools) > 0 and tools[0].name == "get_weather":
            if self.message_matched(
                patterns=["What is the weather in Tokyo?", "get_weather", "The weather in Tokyo is sunny today"], 
                messages=messages
            ):
                # To match the input messages and tools:
                # 
                # messages: [Message(role=<Role.SYSTEM: 'system'>, blocks=[TextBlock(block_type='text', text='You are a helpful assistant.')], extras={}), Message(role=<Role.USER: 'user'>, blocks=[TextBlock(block_type='text', text='What is the weather in Tokyo?')], extras={'name': 'Jack'}), Message(role=<Role.AI: 'assistant'>, blocks=[ToolCallBlock(block_type='tool_call', id='call_cPRd8n0oX4sb9VbN6i4UFTyI', name='get_weather', arguments={'city': 'Tokyo'})], extras={}), Message(role=<Role.TOOL: 'tool'>, blocks=[ToolResultBlock(block_type='tool_result', id='call_cPRd8n0oX4sb9VbN6i4UFTyI', content='The weather in Tokyo is sunny today and the temperature is 20 degrees Celsius.')], extras={})]
                # tools: [Tool(name='get_weather', description='Retrieves current weather for the given city.', parameters={'properties': {'city': {'description': 'The city to get the weather of, e.g. New York.', 'title': 'City', 'type': 'string'}}, 'required': ['city'], 'title': 'get_weather', 'type': 'object'})]

                return ([], "It's sunny in Tokyo today with a temperature of 20°C.")
            elif self.message_matched(
                patterns=["What is the weather in Tokyo?"], 
                messages=messages
            ):
                # To match the input messages and tools:
                # 
                # messages: [Message(role=<Role.SYSTEM: 'system'>, blocks=[TextBlock(block_type='text', text='You are a helpful assistant.')], extras={}), Message(role=<Role.USER: 'user'>, blocks=[TextBlock(block_type='text', text='What is the weather in Tokyo?')], extras={'name': 'Jack'})]
                # tools: [Tool(name='get_weather', description='Retrieves current weather for the given city.', parameters={'properties': {'city': {'description': 'The city to get the weather of, e.g. New York.', 'title': 'City', 'type': 'string'}}, 'required': ['city'], 'title': 'get_weather', 'type': 'object'})]

                return ([ToolCall(id="call_cPRd8n0oX4sb9VbN6i4UFTyI", name="get_weather", arguments={"city": "Tokyo"})], None)
        elif len(tools) > 0 and tools[0].name == "multiply":
            if self.message_matched(
                patterns=["Could you help me to do some calculations?", "What is 235 * 4689?", "multiply", "1101915"], 
                messages=messages
            ):
                # To match the input messages and tools:
                # 
                # messages: [Message(role=<Role.SYSTEM: 'system'>, blocks=[TextBlock(block_type='text', text='You are a helpful assistant that are good at calculating by using tools.')], extras={}), Message(role=<Role.USER: 'user'>, blocks=[TextBlock(block_type='text', text='Could you help me to do some calculations?')], extras={}), Message(role=<Role.AI: 'assistant'>, blocks=[TextBlock(block_type='text', text='Of course, I can help you with that.')], extras={}), Message(role=<Role.USER: 'user'>, blocks=[TextBlock(block_type='text', text='What is 235 * 4689?')], extras={}), Message(role=<Role.AI: 'assistant'>, blocks=[ToolCallBlock(block_type='tool_call', id='call_VOG6ymnapGEN3LrLdXkTOdwc', name='multiply', arguments={'x': 235, 'y': 4689})], extras={}), Message(role=<Role.TOOL: 'tool'>, blocks=[ToolResultBlock(block_type='tool_result', id='call_VOG6ymnapGEN3LrLdXkTOdwc', content='1101915')], extras={})]
                # tools: [Tool(name='multiply', description='This function is used to multiply two numbers.', parameters={'properties': {'x': {'description': 'The first number to multiply', 'title': 'X', 'type': 'integer'}, 'y': {'description': 'The second number to multiply', 'title': 'Y', 'type': 'integer'}}, 'required': ['x', 'y'], 'title': 'multiply', 'type': 'object'})]

                return ([], "235 multiplied by 4689 equals 1,101,915.")
            elif self.message_matched(
                patterns=["Could you help me to do some calculations?", "What is 235 * 4689?"], 
                messages=messages
            ):
                # To match the input messages and tools:
                # 
                # messages: [Message(role=<Role.SYSTEM: 'system'>, blocks=[TextBlock(block_type='text', text='You are a helpful assistant that are good at calculating by using tools.')], extras={}), Message(role=<Role.USER: 'user'>, blocks=[TextBlock(block_type='text', text='Could you help me to do some calculations?')], extras={}), Message(role=<Role.AI: 'assistant'>, blocks=[TextBlock(block_type='text', text='Of course, I can help you with that.')], extras={}), Message(role=<Role.USER: 'user'>, blocks=[TextBlock(block_type='text', text='What is 235 * 4689?')], extras={})]
                # tools: [Tool(name='multiply', description='This function is used to multiply two numbers.', parameters={'properties': {'x': {'description': 'The first number to multiply', 'title': 'X', 'type': 'integer'}, 'y': {'description': 'The second number to multiply', 'title': 'Y', 'type': 'integer'}}, 'required': ['x', 'y'], 'title': 'multiply', 'type': 'object'})]

                return ([ToolCall(id="call_VOG6ymnapGEN3LrLdXkTOdwc", name="multiply", arguments={'x': 235, 'y': 4689})], None)

        return ([], "Hello! I am a Mock LLM, not able to select tools.")

    @override
    async def aselect_tool(
        self,
        messages: List[Message],
        tools: List[Tool],
        **kwargs,
    ) -> Tuple[List[ToolCall], Optional[str]]:
        return self.select_tool(messages, tools, **kwargs)
    
    def message_matched(
        self,
        patterns: List[str],
        messages: List[Message],
    ) -> bool:
        match_results_each_pattern = []
        for pattern in patterns:
            match_results_each_message = []
            for message in messages:
                content1 = message.content
                id1 = "\n".join([block.id for block in message.blocks if isinstance(block, ToolCallBlock)])
                name = "\n".join([block.name for block in message.blocks if isinstance(block, ToolCallBlock)])
                arguments = "\n".join([json.dumps(block.arguments) for block in message.blocks if isinstance(block, ToolCallBlock)])
                id2 = "\n".join([block.id for block in message.blocks if isinstance(block, ToolResultBlock)])
                content2 = "\n".join([block.content for block in message.blocks if isinstance(block, ToolResultBlock)])

                fields = [id1, name, arguments, id2, content1, content2]
                match_results_each_message.append(any(pattern in field for field in fields))
            match_results_each_pattern.append(any(match_results_each_message))

        return all(match_results_each_pattern)