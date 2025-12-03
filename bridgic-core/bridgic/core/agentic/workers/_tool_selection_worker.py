from typing import Any, Dict, List, Tuple, Optional
from typing_extensions import override

from bridgic.core.automa.worker import Worker
from bridgic.core.model.types import Message, Tool, ToolCall
from bridgic.core.model.protocols import ToolSelection
from bridgic.core.agentic.types._chat_message import ChatMessage
from bridgic.core.prompt.utils._prompt_utils import transform_chat_message_to_llm_message

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
        messages: List[ChatMessage],
        tools: List[Tool],
    ) -> Tuple[List[ToolCall], Optional[str]]:
        """
        Run the worker.

        Parameters
        ----------
        messages: List[ChatMessage]
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
            llm_messages.append(transform_chat_message_to_llm_message(message))
        # print(f"\n******* ToolSelectionWorker.arun *******\n")
        # print(f"messages: {llm_messages}")
        # print(f"tools: {tools}")
        tool_calls, llm_response = await self._tool_selection_llm.aselect_tool(
            messages=llm_messages, 
            tools=tools, 
        )
        return tool_calls, llm_response

    @override
    def dump_to_dict(self) -> Dict[str, Any]:
        state_dict = super().dump_to_dict()
        state_dict["tool_selection_llm"] = self._tool_selection_llm
        return state_dict

    @override
    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_from_dict(state_dict)
        self._tool_selection_llm = state_dict["tool_selection_llm"]