from typing import List
from bridgic.core.intelligence.base_llm import *

class MockLlm(BaseLlm):
    def chat(self, messages: List[Message], **kwargs) -> Response:
        return Response(message=Message(role=Role.AI, content="Hello! I am a Mock LLM."))

    def stream(self, messages: List[Message], **kwargs) -> StreamResponse:
        content = "Hello! I am a Mock LLM."
        for chunk in content.split(" "):
            yield MessageChunk(delta=chunk, raw=chunk)

    async def achat(self, messages: List[Message], **kwargs) -> Response:
        return Response(message=Message(role=Role.AI, content="Hello! I am a Mock LLM."))

    async def astream(self, messages: List[Message], **kwargs) -> AsyncStreamResponse:
        content = "Hello! I am a Mock LLM."
        for chunk in content.split(" "):
            yield MessageChunk(delta=chunk, raw=chunk)