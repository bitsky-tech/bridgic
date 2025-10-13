from abc import ABC, abstractmethod
from typing import List, Generator, AsyncGenerator, TYPE_CHECKING
from enum import Enum

from bridgic.core.intelligence.content import *
if TYPE_CHECKING:
    from bridgic.core.intelligence.protocol import ToolCall
from bridgic.core.types.serialization import Serializable

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

    @classmethod
    def from_tool_call(
        cls,
        tool_calls: Union[
            Dict[str, Any], 
            List[Dict[str, Any]], 
            "ToolCall",
            List["ToolCall"]
        ],
        text: Optional[str] = None,
        extras: Optional[Dict[str, Any]] = {},
    ) -> "Message":
        """
        Create a message with tool call blocks and optional text content.
        
        Parameters
        ----
        tool_calls : Union[Dict[str, Any], List[Dict[str, Any]], ToolCall, List[ToolCall]]
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
        ----
        Message
            A message containing the tool call blocks and optional text
            
        Examples
        -----
        # Single tool call dict
        >>> message = Message.from_tool_call(
        ...     tool_calls={
        ...         "id": "call_123",
        ...         "name": "get_weather",
        ...         "arguments": {"city": "Tokyo", "unit": "celsius"}
        ...     },
        ...     text="I will check the weather for you."
        ... )
        
        # Multiple tool call dicts
        >>> message = Message.from_tool_call(
        ...     tool_calls=[
        ...         {"id": "call_123", "name": "get_weather", "arguments": {"city": "Tokyo"}},
        ...         {"id": "call_124", "name": "get_news", "arguments": {"topic": "weather"}}
        ...     ],
        ...     text="I will get weather and news for you."
        ... )
        
        # Single ToolCall
        >>> tool_call = ToolCall(id="call_123", name="get_weather", arguments={"city": "Tokyo"})
        >>> message = Message.from_tool_call(tool_calls=tool_call, text="I will check the weather.")
        
        # Multiple ToolCall
        >>> tool_calls = [
        ...     ToolCall(id="call_123", name="get_weather", arguments={"city": "Tokyo"}),
        ...     ToolCall(id="call_124", name="get_news", arguments={"topic": "weather"})
        ... ]
        >>> message = Message.from_tool_call(tool_calls=tool_calls, text="I will get weather and news.")
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
        ----
        tool_id : str
            The ID of the tool call that this result corresponds to
        content : str
            The result content from the tool execution
        extras : Optional[Dict[str, Any]], optional
            Additional metadata for the message
            
        Returns
        ----
        Message
            A message containing the tool result block
            
        Examples
        -----
        >>> message = Message.from_tool_result(
        ...     tool_id="call_123",
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
