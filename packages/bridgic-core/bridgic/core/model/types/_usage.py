from datetime import datetime
from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    """
    Token usage information for a single LLM call.

    Attributes
    ----------
    model : str
        Model identifier used for the call.
    prompt_tokens : int
        Number of tokens in the prompt.
    completion_tokens : int
        Number of tokens generated in the completion.
    total_tokens : int
        Total tokens used (prompt + completion).
    timestamp : datetime
        When this usage occurred.
    """
    model: str
    """Model identifier used for the call."""

    prompt_tokens: int = 0
    """Number of tokens in the prompt."""

    completion_tokens: int = 0
    """Number of tokens generated in the completion."""

    total_tokens: int = 0
    """Total tokens used (prompt + completion)."""

    timestamp: datetime = Field(default_factory=datetime.now)
    """When this usage occurred."""
