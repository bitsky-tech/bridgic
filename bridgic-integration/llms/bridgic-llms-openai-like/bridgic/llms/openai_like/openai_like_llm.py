from typing import List
from openai import OpenAI, AsyncOpenAI

from bridgic.core.intelligence.base_llm import *

class OpenAILikeLlm(BaseLlm):
    """
    OpenAILikeLlm is a thin wrapper around the LLM providers that makes it compatible with the 
    services that provide OpenAI compatible API.
    """

    def __init__(self, api_base: str, api_key: str):
        self.client = OpenAI(api_base=api_base, api_key=api_key)
        self.async_client = AsyncOpenAI(api_base=api_base, api_key=api_key)

    def chat(self, messages: List[Message], **kwargs) -> Response:
        pass

    def stream(self, messages: List[Message], **kwargs) -> StreamResponse:
        pass

    async def achat(self, messages: List[Message], **kwargs) -> Response:
        pass

    async def astream(self, messages: List[Message], **kwargs) -> AsyncStreamResponse:
        pass