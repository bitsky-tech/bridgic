from typing import Optional, Literal, TypedDict, Required
from typing_extensions import Required


class UserTextMessage(TypedDict, total=False):
    """Messages sent by an end user, containing prompts."""
    
    role: Required[Literal["user"]]
    """The role of the messages author, in this case `user`."""
    content: Required[str]
    """The content of the user message, which is a text."""
    name: Optional[str]
    """An optional name for the participant. Provides the model information to differentiate between participants of the same role."""

class AssistantTextMessage(TypedDict, total=False):
    """Messages sent by the model in response to user messages."""
    
    role: Required[Literal["user"]]
    """The role of the messages author, in this case `assistant`."""
    content: Optional[str]
    """The content of the assistant message, which is a text. Required unless `tool_calls` is specified."""
    name: Optional[str]
    """An optional name for the participant. Provides the model information to differentiate between participants of the same role."""
