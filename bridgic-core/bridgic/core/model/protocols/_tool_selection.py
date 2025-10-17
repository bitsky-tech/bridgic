from typing import List, Protocol, Any, Dict, Type, Literal, Union, Optional, ClassVar, Tuple
from pydantic import BaseModel, Field

from bridgic.core.model.types import Message, Tool, ToolCall

class ToolSelection(Protocol):
    """
    Protocol for LLM providers that support tool selection and parameter determination.

    ToolSelection defines the interface for language models that can intelligently 
    select appropriate tools from a given tools and determine the specific parameters 
    needed for tool execution.

    Methods
    -------
    select_tool
        Synchronous method for tool selection based on conversation context.
    aselect_tool
        Asynchronous method for tool selection based on conversation context.

    Notes
    ----
    1. Both synchronous and asynchronous methods must be implemented
    2. Tool selection should be based on conversation context and available tools
    3. Return value includes both selected tool calls and optional response text
    """

    def select_tool(
        self,
        messages: List[Message],
        tools: List[Tool],
        **kwargs,
    ) -> Tuple[List[ToolCall], Optional[str]]:
        """
        Select appropriate tools and determine their parameters based on conversation context.

        Parameters
        ----------
        messages : List[Message]
            The conversation history and current context.
        tools : List[Tool]
            Available tools that can be selected for use.
        **kwargs
            Additional keyword arguments for tool selection configuration.

        Returns
        -------
        Tuple[List[ToolCall], Optional[str]]
            A tuple containing:
            - List of selected tool calls with determined parameters
            - Optional response text from the LLM
        """
        ...

    async def aselect_tool(
        self,
        messages: List[Message],
        tools: List[Tool],
        **kwargs,
    ) -> Tuple[List[ToolCall], Optional[str]]:
        """
        Asynchronously select appropriate tools and determine their parameters.

        Parameters
        ----------
        messages : List[Message]
            The conversation history and current context.
        tools : List[Tool]
            Available tools that can be selected for use.
        **kwargs
            Additional keyword arguments for tool selection configuration.

        Returns
        -------
        Tuple[List[ToolCall], Optional[str]]
            A tuple containing:
            - List of selected tool calls with determined parameters
            - Optional response text from the LLM
        """
        ...