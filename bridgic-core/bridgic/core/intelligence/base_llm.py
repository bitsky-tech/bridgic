from abc import ABC, abstractmethod
from typing import List, Generator, AsyncGenerator
from enum import Enum

from bridgic.core.intelligence.content import *

class Role(str, Enum):
    """
    Message role.
    """
    SYSTEM = "system"
    USER = "user"
    AI = "assistant"
    TOOL = "tool"

    @classmethod
    def get_all_roles(cls) -> List[str]:
        return [role.value for role in Role]

class Message(BaseModel):
    """
    LLM message.
    """
    role: Role = Field(default=Role.USER)
    blocks: List[ContentBlock] = Field(default=[])
    extras: Dict[str, Any] = Field(default={})

    @classmethod
    def from_text(
        cls,
        text: str,
        role: Union[Role, str] = Role.USER,
        extras: Optional[Dict[str, Any]] = {},
    ) -> "Message":
        if isinstance(role, str):
            role = Role(role)
        return cls(role=role, blocks=[TextBlock(text=text)], extras=extras)

    @property
    def content(self) -> str:
        return "\n\n".join([block.text for block in self.blocks if isinstance(block, TextBlock)])

    @content.setter
    def content(self, text: str):
        if not self.blocks:
            self.blocks = [TextBlock(text=text)]
        elif len(self.blocks) == 1 and isinstance(self.blocks[0], TextBlock):
            self.blocks = [TextBlock(text=text)]
        else:
            raise ValueError(
                "Message contains multiple blocks or contains a non-text block, thus it could not be "
                "easily set by the property \"Message.content\". Use \"Message.blocks\" instead."
            )

class Response(BaseModel):
    """
    LLM response.
    """
    message: Optional[Message] = None
    raw: Optional[Any] = None

class MessageChunk(BaseModel):
    """
    Stream chunk.
    """
    delta: Optional[str] = None
    raw: Optional[Any] = None

StreamResponse = Generator[MessageChunk, None, None]
AsyncStreamResponse = AsyncGenerator[MessageChunk, None]

class BaseLlm(ABC):
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
