from typing import List, Protocol, Any, Dict, Type, Literal, Union, Optional, ClassVar, Tuple
from pydantic import BaseModel, Field

from bridgic.core.model.types import Message, Tool, ToolCall

class ToolSelection(Protocol):
    """
    ToolSelection is a protocol that defines the interface for LLM providers that can select tools 
    to use and decide their specific parameters.
    """

    def select_tool(
        self,
        messages: List[Message],
        tools: List[Tool],
        **kwargs,
    ) -> Tuple[List[ToolCall], Optional[str]]: ...

    async def aselect_tool(
        self,
        messages: List[Message],
        tools: List[Tool],
        **kwargs,
    ) -> Tuple[List[ToolCall], Optional[str]]: ...