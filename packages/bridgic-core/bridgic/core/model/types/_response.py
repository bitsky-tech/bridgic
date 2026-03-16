from pydantic import BaseModel
from typing import Optional, Any, Generator, AsyncGenerator

from bridgic.core.model.types._message import Message, MessageChunk
from bridgic.core.model.types._usage import TokenUsage

class Response(BaseModel):
    """
    LLM response container for model outputs.

    Represents the complete response from a language model, containing both
    the message content and the raw response data from the underlying model
    provider.

    Attributes
    ----------
    message : Optional[Message]
        The structured message containing the model's response content.
    usage : Optional[TokenUsage]
        Token usage information for this call.
    raw : Optional[Any]
        Raw response data from the LLM provider for debugging or custom processing.
    """
    message: Optional[Message] = None
    usage: Optional[TokenUsage] = None
    raw: Optional[Any] = None

StreamResponse = Generator[MessageChunk, None, None]
AsyncStreamResponse = AsyncGenerator[MessageChunk, None]
