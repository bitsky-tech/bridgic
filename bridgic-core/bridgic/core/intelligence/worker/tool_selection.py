from typing import Any, Dict, List, Tuple, Optional, cast

from bridgic.core.automa.worker import Worker
from bridgic.core.intelligence.base_llm import Message
from bridgic.core.intelligence.protocol import Tool, ToolSelect, ToolCall
from bridgic.core.types.error import WorkerRuntimeError

class ToolSelectionWorker(Worker):
    """
    A worker that calls an LLM to select tools and/or generate a response.
    """

    # TODO: How to serialize the ToolSelect LLM instance?
    _tool_selection_llm: ToolSelect
    """The LLM to be used for tool selection."""

    def __init__(self, tool_selection_llm: ToolSelect):
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
        messages: List[Message],
        tools: List[Tool],
        **kwargs,
    ) -> Tuple[str, List[ToolCall]]:
        """
        Run the worker.

        Parameters
        ----------
        messages: List[Message]
            The messages to send to the LLM.
        tools: List[Tool]
            The tool list for the LLM to select from.

        Returns
        -------
        Tuple[str, List[ToolCall]]
            * The first element is the text response from the LLM.
            * The second element is a list of `ToolCall` that the LLM selected.

        **kwargs: Any
            The keyword arguments passed through to the LLM. It depends on the LLM's implementation.
        """
        return await self._tool_selection_llm.atool_select(messages, tools, **kwargs)
