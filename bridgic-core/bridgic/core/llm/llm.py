from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Generator

class LLM(ABC):
    """
    Abstract base class for Large Language Model implementations.
    """

    @abstractmethod
    def chat(self, prompt: str, **kwargs) -> str:
        pass

    @abstractmethod
    def stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        pass

    @abstractmethod
    def structured(self, prompt: str, schema: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        pass


class MockLLM(LLM):

    def chat(self, prompt: str, **kwargs) -> str:
        pass
    def stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        pass
    def structured(self, prompt: str, schema: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        pass