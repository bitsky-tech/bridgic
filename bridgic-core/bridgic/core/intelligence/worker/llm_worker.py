from typing import Any, Dict, List, Optional, cast


from bridgic.core.automa.worker import Worker
from bridgic.core.intelligence.base_llm import BaseLlm, Message
from bridgic.core.intelligence.protocol import Tool, ToolSelect, StructuredOutput, Constraint
from bridgic.core.types.error import WorkerRuntimeError

class LlmWorker(Worker):
    """
    A worker that calls an LLM to generate a response.
    """

    ##### Variables that need to be serialized #####
    _llm: BaseLlm
    _support_tool_select: bool
    _support_structured_output: bool

    @property
    def llm(self) -> BaseLlm:
        return self._llm

    @llm.setter
    def llm(self, llm: BaseLlm) -> None:
        """
        Set the LLM to be used by the worker. The LLM must be set before the worker is used.

        Parameters
        ----------
        llm : BaseLlm
            The LLM to be used by the worker.
        """
        self._llm = llm
        # Cache the capabilities of the LLM, partly because of the performance concerns about the isinstance() check against a runtime-checkable protocol.
        # Refer to: https://docs.python.org/3/library/typing.html#typing.runtime_checkable
        self._support_tool_select = isinstance(llm, ToolSelect)
        self._support_structured_output = isinstance(llm, StructuredOutput)

    async def arun(
        self,
        messages: List[Message],
        *,
        tools: Optional[List[Tool]] = None,
        constraint: Optional[Constraint] = None,
        **kwargs: Dict[str, Any]
    ) -> Any:
        """
        Run the worker.

        Parameters
        ----------
        messages: List[Message]
            The messages to send to the LLM.
        tools: Optional[List[Tool]]
            The tool list used for tool calling.
        constraint: Optional[Constraint]
            The constraint used for structured output.
        **kwargs: Dict[str, Any]
            Other keyword arguments passed through to the LLM. It depends on the LLM's implementation.

        Returns
        -------
        Any
            The return type is based on the LLM's type.
            * If the LLM supports `ToolSelect` protocol, return a list of `ToolCall`.
            * If the LLM supports `StructuredOutput` protocol, return a `BaseModel` or a `Dict[str, Any]` or a `str`. TODO: check the return types.
            * Otherwise, return an AsyncGenerator of `MessageChunk`.
        """
        if tools:
            if not self._support_tool_select:
                raise WorkerRuntimeError("The LLM does not support tool selection. Please set a LLM that supports `ToolSelect` protocol.")
            _tool_select_llm = cast(ToolSelect, self._llm)
            return await _tool_select_llm.atool_select(messages, tools, **kwargs)
        if constraint:
            if not self._support_structured_output:
                raise WorkerRuntimeError("The LLM does not support structured output. Please set a LLM that supports `StructuredOutput` protocol.")
            _structured_output_llm = cast(StructuredOutput, self._llm)
            return await _structured_output_llm.astructured_output(messages, constraint, **kwargs)
        # Note: It does not need to await self._llm.astream here because it returns an AsyncGenerator.
        return self._llm.astream(messages, **kwargs)
