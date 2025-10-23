from typing import List, Dict, Any, Optional, Union, TYPE_CHECKING
from pydantic import BaseModel, Field
from enum import Enum

from bridgic.core.model.types._content_block import ContentBlock, TextBlock, ToolCallBlock, ToolResultBlock

if TYPE_CHECKING:
    from bridgic.core.model.types._tool_use import ToolCall, ToolCallDict

class Role(str, Enum):
    """
    Message role enumeration for LLM conversations.

    Defines the different roles that can be assigned to messages in a conversation
    with language models, following standard chat completion formats.

    Attributes
    ----------
    SYSTEM : str
        System role for providing instructions or context to the model.
    USER : str
        User role for human input and queries.
    AI : str
        Assistant role for model responses and outputs.
    TOOL : str
        Tool role for tool execution results and responses.
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
    LLM message container for conversation exchanges.

    Represents a single message in a conversation with language models, containing
    role information, content blocks, and optional metadata. Supports various
    content types including text, tool calls, and tool results.

    Attributes
    ----------
    role : Role
        The role of the message sender (system, user, assistant, or tool).
    blocks : List[ContentBlock]
        List of content blocks containing the actual message data.
    extras : Dict[str, Any]
        Additional metadata and custom fields for the message.
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

    @classmethod
    def from_tool_call(
        cls,
        tool_calls: Union[
            "ToolCallDict", 
            List["ToolCallDict"], 
            "ToolCall",
            List["ToolCall"]
        ],
        text: Optional[str] = None,
        extras: Optional[Dict[str, Any]] = {},
    ) -> "Message":
        """
        Create a message with tool call blocks and optional text content.
        
        Parameters
        ----------
        tool_calls : Union[ToolCallDict, List[ToolCallDict], ToolCall, List[ToolCall]]
            Tool call data in various formats:
            - Single tool call dict: {"id": "call_123", "name": "get_weather", "arguments": {...}}
            - List of tool call dicts: [{"id": "call_123", ...}, {"id": "call_124", ...}]
            - Single ToolCall instance
            - List of ToolCall instances
        text : Optional[str], optional
            Optional text content to include in the message

        extras : Optional[Dict[str, Any]], optional
            Additional metadata for the message
            
        Returns
        -------
        Message
            A message containing the tool call blocks and optional text
            
        Examples
        --------
        >>> # Build from single tool call dict.
        ... message = Message.from_tool_call(
        ...     tool_calls={
        ...         "id": "call_id_123",
        ...         "name": "get_weather",
        ...         "arguments": {"city": "Tokyo", "unit": "celsius"}
        ...     },
        ...     text="I will check the weather for you."
        ... )
        
        >>> # Build from multiple tool call dicts.
        ... message = Message.from_tool_call(
        ...     tool_calls=[
        ...         {"id": "call_id_123", "name": "get_weather", "arguments": {"city": "Tokyo"}},
        ...         {"id": "call_id_456", "name": "get_news", "arguments": {"topic": "weather"}},
        ...     ],
        ...     text="I will get weather and news for you."
        ... )
        
        >>> # Build from single ToolCall object.
        ... tool_call = ToolCall(id="call_123", name="get_weather", arguments={"city": "Tokyo"})
        ... message = Message.from_tool_call(tool_calls=tool_call, text="I will check the weather.")
        
        >>> # Build from multiple ToolCall objects.
        ... tool_calls = [
        ...     ToolCall(id="call_id_123", name="get_weather", arguments={"city": "Tokyo"}),
        ...     ToolCall(id="call_id_456", name="get_news", arguments={"topic": "weather"}),
        ... ]
        ... message = Message.from_tool_call(tool_calls=tool_calls, text="I will get weather and news.")
        """
        role = Role(Role.AI)
        blocks = []
        
        # Add text content if provided
        if text:
            blocks.append(TextBlock(text=text))
        
        # Handle different tool_calls formats
        if isinstance(tool_calls, dict):
            # Single tool call dict
            tool_calls = [tool_calls]
        if isinstance(tool_calls, list):
            # List of tool calls (dicts or ToolCall)
            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    # Tool call dict
                    blocks.append(ToolCallBlock(
                        id=tool_call["id"],
                        name=tool_call["name"],
                        arguments=tool_call["arguments"]
                    ))
                elif hasattr(tool_call, 'id') and hasattr(tool_call, 'name') and hasattr(tool_call, 'arguments'):
                    blocks.append(ToolCallBlock(
                        id=tool_call.id,
                        name=tool_call.name,
                        arguments=tool_call.arguments
                    ))
                else:
                    raise ValueError(f"Invalid tool call format: {tool_call}")
        elif hasattr(tool_calls, 'id') and hasattr(tool_calls, 'name') and hasattr(tool_calls, 'arguments'):
            blocks.append(ToolCallBlock(
                id=tool_calls.id,
                name=tool_calls.name,
                arguments=tool_calls.arguments
            ))
        else:
            raise ValueError(f"Invalid tool_calls format: {type(tool_calls)}")
        
        return cls(role=role, blocks=blocks, extras=extras)

    @classmethod
    def from_tool_result(
        cls,
        tool_id: str,
        content: str,
        extras: Optional[Dict[str, Any]] = {},
    ) -> "Message":
        """
        Create a message with a tool result block.
        
        Parameters
        ----------
        tool_id : str
            The ID of the tool call that this result corresponds to
        content : str
            The result content from the tool execution
        extras : Optional[Dict[str, Any]], optional
            Additional metadata for the message
            
        Returns
        -------
        Message
            A message containing the tool result block
            
        Examples
        --------
        >>> message = Message.from_tool_result(
        ...     tool_id="call_id_123",
        ...     content="The weather in Tokyo is 22°C and sunny."
        ... )
        """
        role = Role(Role.TOOL)
        return cls(
            role=role, 
            blocks=[ToolResultBlock(id=tool_id, content=content)], 
            extras=extras
        )

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

class MessageChunk(BaseModel):
    """
    Streaming message chunk for real-time LLM responses.

    Represents a partial message chunk received during streaming responses from
    language models, allowing for real-time processing of incremental content.

    Attributes
    ----------
    delta : Optional[str]
        The incremental text content of this chunk.
    raw : Optional[Any]
        Raw response data from the LLM provider.
    """
    delta: Optional[str] = None
    raw: Optional[Any] = None
