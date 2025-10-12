from typing import List, Tuple, Union, Optional

from bridgic.core.automa.worker import Worker
from bridgic.core.intelligence.base_llm import Message, Role
from bridgic.core.prompt.chat_message import ChatMessage
from bridgic.core.intelligence.protocol import Tool, ToolSelection, ToolCall
from bridgic.core.prompt.utils.prompt_utils import transform_chat_message_to_llm_message

class ToolSelectionWorker(Worker):
    """
    A worker that calls an LLM to select tools and/or generate a response.
    """

    # Note: the ToolSelection LLM instance need support serialization and deserialization.
    _tool_selection_llm: ToolSelection
    """The LLM to be used for tool selection."""

    def __init__(self, tool_selection_llm: ToolSelection):
        """
        Parameters
        ----------
        tool_selection_llm: ToolSelect
            The LLM to be used for tool selection.
        """
        super().__init__()
        self._tool_selection_llm = tool_selection_llm

    async def arun(
        self,
        messages: List[Union[ChatMessage, Message]],
        tools: List[Tool],
    ) -> Tuple[List[ToolCall], Optional[str]]:
        """
        Run the worker.

        Parameters
        ----------
        messages: List[Union[ChatMessage, Message]]
            The messages to send to the LLM.
        tools: List[Tool]
            The tool list for the LLM to select from.

        Returns
        -------
        Tuple[List[ToolCall], Optional[str]]
            * The first element is a list of `ToolCall` that the LLM selected.
            * The second element is the text response from the LLM.
        """
        # Validate and transform the input messages and tools to the format expected by the LLM.
        llm_messages: List[Message] = []
        for message in messages:
            if isinstance(message, dict):
                llm_messages.append(transform_chat_message_to_llm_message(message))
            elif isinstance(message, Message):
                llm_messages.append(message)
            else:
                raise TypeError(f"Invalid `messages` type: {type(message)}, expected `ChatMessage` or `Message`.")
        model_name = "gpt-5-mini"
        print(f"\n******* ToolSelectionWorker.arun *******\n")
        print(f"model_name: {model_name}")
        print(f"messages: {llm_messages}")
        print(f"tools: {tools}")
        # TODO: 
        tool_calls, llm_response = await self._tool_selection_llm.aselect_tool(
            model=model_name,
            messages=llm_messages, 
            tools=tools, 
        )
        return tool_calls, llm_response
