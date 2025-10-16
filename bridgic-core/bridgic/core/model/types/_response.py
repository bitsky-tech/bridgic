from pydantic import BaseModel
from typing import Optional, Any, Generator, AsyncGenerator

from bridgic.core.model.types._message import Message, MessageChunk

class Response(BaseModel):
    """
    LLM response.
    """
    message: Optional[Message] = None
    raw: Optional[Any] = None

StreamResponse = Generator[MessageChunk, None, None]
AsyncStreamResponse = AsyncGenerator[MessageChunk, None]
