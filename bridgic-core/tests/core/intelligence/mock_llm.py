from typing import List
from bridgic.core.intelligence.base_llm import *

class MockLlm(BaseLlm):
    def chat(self, messages: List[Message], **kwargs) -> Response:
        return Response(message=Message.from_text(text="Hello! I am a Mock LLM.", role=Role.AI))

    def stream(self, messages: List[Message], **kwargs) -> StreamResponse:
        content = "Hello! I am a Mock LLM."
        for chunk in content.split(" "):
            yield MessageChunk(delta=chunk, raw=chunk)

    async def achat(self, messages: List[Message], **kwargs) -> Response:
        return Response(message=Message.from_text(text="Hello! I am a Mock LLM.", role=Role.AI))

    async def astream(self, messages: List[Message], **kwargs) -> AsyncStreamResponse:
        content = "Hello! I am a Mock LLM."
        for chunk in content.split(" "):
            yield MessageChunk(delta=chunk, raw=chunk)