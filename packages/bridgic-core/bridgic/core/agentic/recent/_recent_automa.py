import uuid
from typing import Optional, List, Any, Dict, Tuple, Union, Callable
from typing_extensions import override
from concurrent.futures import ThreadPoolExecutor

from bridgic.core.automa import GraphAutoma, Automa, worker, RunningOptions
from bridgic.core.model import BaseLlm
from bridgic.core.model.types import Message, Role, ToolCall
from bridgic.core.agentic.recent._recent_memory_manager import ReCentMemoryManager
from bridgic.core.agentic.recent._recent_memory_config import ReCentMemoryConfig
from bridgic.core.agentic.tool_specs import ToolSpec, FunctionToolSpec, AutomaToolSpec


class ReCentAutoma(GraphAutoma):
    """
    ReCentAutoma is an automa that implements a ReAct-like process, leveraging the ReCENT memory 
    algorithm to support stronger autonomous next-step planning, thus better achieving the pre-set goal.

    This automa extends GraphAutoma to provide a memory-aware agentic automa that:
    - Maintains episodic memory with compression capabilities
    - Supports goal-oriented task execution
    - Dynamically creates tool workers based on LLM decisions
    - Manages memory compression to prevent context explosion

    Parameters
    ----------
    llm : BaseLlm
        LLM instance used for thinking, planning and answering.
    tools : Optional[List[Union[Callable, Automa, ToolSpec]]]
        List of tools available to the automa. Can be functions, Automa instances, or ToolSpec instances.
    memory_config : Optional[ReCentMemoryConfig]
        Memory configuration for ReCent memory management. If None, a default config will be created using the provided llm.
    name : Optional[str]
        The name of the automa instance.
    thread_pool : Optional[ThreadPoolExecutor]
        The thread pool for parallel execution of I/O-bound or CPU-bound tasks.
    running_options : Optional[RunningOptions]
        The running options for the automa instance.
    """

    _llm: BaseLlm
    """The main LLM which is used for thinking, planning and answering."""

    _tool_specs: Optional[List[ToolSpec]]
    """List of tool specifications available to the automa."""

    _memory_manager: ReCentMemoryManager
    """Memory manager instance for managing episodic memory."""

    def __init__(
        self,
        llm: BaseLlm,
        tools: Optional[List[Union[Callable, Automa, ToolSpec]]] = None,
        memory_config: Optional[ReCentMemoryConfig] = None,
        name: Optional[str] = None,
        thread_pool: Optional[ThreadPoolExecutor] = None,
        running_options: Optional[RunningOptions] = None,
    ):
        super().__init__(name=name, thread_pool=thread_pool, running_options=running_options)
        self._llm = llm
        self._tool_specs = [self._ensure_tool_spec(tool) for tool in tools or []]

        memory_config = memory_config or ReCentMemoryConfig(llm=llm)
        self._memory_manager = ReCentMemoryManager(compression_config=memory_config)

    def _ensure_tool_spec(self, tool: Union[Callable, Automa, ToolSpec]) -> ToolSpec:
        if isinstance(tool, ToolSpec):
            return tool
        elif isinstance(tool, type) and issubclass(tool, Automa):
            return AutomaToolSpec.from_raw(tool)
        elif isinstance(tool, Callable):
            # Note: this test against `Callable` should be placed at last.
            return FunctionToolSpec.from_raw(tool)
        else:
            raise TypeError(f"Invalid tool type: {type(tool)} detected, expected `Callable`, `Automa`, or `ToolSpec`.")

    @worker(is_start=True)
    async def init_task_goal(
        self,
        goal: str,
        guidance: Optional[str] = None,
    ):
        """
        Initialize the goal of the task and start the automa.

        This worker is the entry point of the automa. It creates a goal node as the first episodic 
        node in the memory sequence and optionally pushes initial user messages.

        Parameters
        ----------
        goal : str
            The task goal.
        guidance : Optional[str]
            The guidance for achieving the task goal.
        """
        self._memory_manager.create_goal(goal, guidance)
        self.ferry_to("think_and_plan")

    @worker()
    async def think_and_plan(self):
        """
        Think and plan the next step based on current context.

        This worker builds the context from the current memory, calls the LLM to decide on
        the next action (tool call or final answer), and routes accordingly.
        """
        # TODO: Implement think and plan logic
        # 1. Call context = await self._memory_manager.abuild_context() to get context
        # 2. Build LLM message list (task goal, history messages, tool specifications)
        # 3. Call LLM: response = await self._llm.achat(messages)
        # 4. Parse LLM response:
        #    - If tool_calls exist: extract tool calls list and push Assistant message to memory: self._memory_manager.push_messages([assistant_msg])
        #    - If no tool_calls: extract final answer text
        # 5. Dynamic routing:
        #    - If tool calls:
        #      a. Match tools: matched_list = self._match_tool_calls_and_tool_specs(tool_calls, self._tool_specs)
        #      b. Create tool workers dynamically:
        #         for tool_call, tool_spec in matched_list:
        #             tool_worker = tool_spec.create_worker()
        #             tool_worker_key = f"tool_{tool_call.name}_{uuid.uuid4().hex[:8]}"
        #             self.add_worker(key=tool_worker_key, worker=tool_worker)
        #             tool_worker_keys.append(tool_worker_key)
        #             # Push ToolCall message to memory
        #             self._memory_manager.push_messages([tool_call_message])
        #             # ferry_to tool worker
        #             self.ferry_to(tool_worker_key, **tool_call.arguments)
        #      c. Create collect_results worker dynamically:
        #         collect_worker_key = f"collect_results_{uuid.uuid4().hex[:8]}"
        #         self.add_func_as_worker(
        #             key=collect_worker_key,
        #             func=self._collect_tools_results,
        #             dependencies=tool_worker_keys,  # 依赖所有工具 Worker
        #             args_mapping_rule=ArgsMappingRule.MERGE,
        #         )
        #    - If final answer: ferry_to("finalize_answer", answer=final_answer)

        # Placeholder implementation
        # context = await self._memory_manager.abuild_context()
        # messages = self._build_llm_messages(context)
        # response = await self._llm.achat(messages)
        # ... (rest of logic)
        pass

    @worker()
    async def check_compression(self):
        """
        Check whether memory compression is needed.

        This worker checks if the compression conditions are met and decides which nodes to route to.
        Either compress_memory or think_and_plan will inevitably be chosen to execute in the next step.
        """
        # TODO: Implement compression check logic
        # 1. Call should_compress = self._memory_manager.should_trigger_compression() to check if compression is needed
        # 2. Dynamic routing:
        #    - If compression needed: self.ferry_to("compress_memory")
        #    - If not needed: self.ferry_to("think_and_plan")

        # Placeholder implementation
        # should_compress = self._memory_manager.should_trigger_compression()
        # if should_compress:
        #     self.ferry_to("compress_memory")
        # else:
        #     self.ferry_to("think_and_plan")
        pass

    @worker()
    async def compress_memory(self) -> int:
        """
        Compress early memory nodes.

        This worker triggers memory compression by creating a compression node
        that summarizes earlier episodic nodes.

        Returns
        -------
        int
            The timestep of the compression node (optional, not passed to subsequent nodes).
        """
        # TODO: Implement memory compression logic
        # 1. Call timestep = await self._memory_manager.create_compression() (async)
        # 2. Route back to think_and_plan: self.ferry_to("think_and_plan")
        # 3. Return compression timestep (optional)

        # Placeholder implementation
        # timestep = await self._memory_manager.create_compression()
        # self.ferry_to("think_and_plan")
        # return timestep
        pass

    @worker(is_output=True)
    async def finalize_answer(self, answer: str) -> str:
        """
        Generate the final answer and complete the workflow.

        This worker is the output node of the automa. It receives the final answer
        from think_and_plan and optionally pushes it to memory.

        Parameters
        ----------
        answer : str
            The final answer content (from think_and_plan ferry_to).

        Returns
        -------
        str
            The final answer string.
        """
        # TODO: Implement final answer logic
        # 1. Optionally push final answer to memory
        # 2. Return the final answer

        # Placeholder implementation
        # Optionally: self._memory_manager.push_messages([final_answer_msg])
        return answer

    def _collect_tools_results(
        self,
        tool_calls: Optional[List[ToolCall]],
        tool_results: List[Any],
    ) -> None:
        """
        Collect results from tool executions and push to memory.

        This method is used as a dynamic worker function to collect results from multiple tool workers. 
        It converts tool results to ToolResult messages and pushes them to memory.

        Parameters
        ----------
        tool_results : List[Any]
            List of tool execution results (merged via ArgsMappingRule.MERGE).
        tool_calls : Optional[List[ToolCall]]
            List of tool calls (optional, for building ToolResult messages).
        """
        # TODO: Implement tool results collection logic
        # 1. Convert tool results to ToolResult messages
        # 2. Call self._memory_manager.push_messages([tool_result_messages])
        # 3. Route to check_compression: self.ferry_to("check_compression")

        # Placeholder implementation
        # tool_result_messages = self._convert_results_to_messages(tool_calls, tool_results)
        # self._memory_manager.push_messages(tool_result_messages)
        # self.ferry_to("check_compression")
        pass

    def _match_tool_calls_and_tool_specs(
        self,
        tool_calls: List[ToolCall],
        tool_specs: List[ToolSpec],
    ) -> List[Tuple[ToolCall, ToolSpec]]:
        """
        Match tool calls from LLM with available tool specifications.

        Parameters
        ----------
        tool_calls : List[ToolCall]
            List of tool calls from LLM response.
        tool_specs : List[ToolSpec]
            List of available tool specifications.

        Returns
        -------
        List[Tuple[ToolCall, ToolSpec]]
            List of matched (tool_call, tool_spec) pairs.
        """
        # TODO: Implement tool matching logic
        # Match tool_call.name with tool_spec.tool_name
        # Return list of (tool_call, tool_spec) tuples
        matched = []
        for tool_call in tool_calls:
            for tool_spec in tool_specs:
                if tool_spec.tool_name == tool_call.name:
                    matched.append((tool_call, tool_spec))
                    break
        return matched

    @override
    def dump_to_dict(self) -> Dict[str, Any]:
        state_dict = super().dump_to_dict()
        state_dict["llm"] = self._llm
        state_dict["tool_specs"] = self._tool_specs
        state_dict["memory_manager"] = self._memory_manager
        return state_dict

    @override
    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_from_dict(state_dict)
        self._llm = state_dict["llm"]
        self._tool_specs = state_dict["tool_specs"]
        self._memory_manager = state_dict["memory_manager"]

