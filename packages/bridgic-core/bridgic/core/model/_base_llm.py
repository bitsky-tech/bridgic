from abc import ABC, abstractmethod
from typing import List

from bridgic.core.types._serialization import Serializable
from bridgic.core.model.types import *

class BaseLlm(ABC, Serializable):
    """
    Base class for Large Language Model implementations.
    """

    @abstractmethod
    def chat(self, messages: List[Message], **kwargs) -> Response:
        ...

    @abstractmethod
    def stream(self, messages: List[Message], **kwargs) -> StreamResponse:
        ...

    @abstractmethod
    async def achat(self, messages: List[Message], **kwargs) -> Response:
        ...

    @abstractmethod
    async def astream(self, messages: List[Message], **kwargs) -> AsyncStreamResponse:
        ...
