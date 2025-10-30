from abc import ABC, abstractmethod
from typing import List
import time

from bridgic.core.types._serialization import Serializable
from bridgic.core.model.types import *
from bridgic.core.logging import get_model_logger
from bridgic.core.constants import DEFAULT_MODEL_SOURCE_PREFIX
class BaseLlm(ABC, Serializable):
    """
    Base class for Large Language Model implementations.
    """
    
    def __init__(self):
        # Initialize logger
        self._logger = get_model_logger(self.__class__.__name__, source=f"{DEFAULT_MODEL_SOURCE_PREFIX}-{self.__class__.__name__}")

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
