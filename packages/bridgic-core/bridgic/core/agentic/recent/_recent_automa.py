import uuid
import asyncio
from typing import Optional, List, Any, Dict, Tuple, Union, Callable
from typing_extensions import override
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel, Field

from bridgic.core.automa import GraphAutoma, Automa, worker, RunningOptions
from bridgic.core.automa.args import ArgsMappingRule
from bridgic.core.model import BaseLlm
from bridgic.core.model.types import Message, Tool, Role, ToolCall
from bridgic.core.model.protocols import StructuredOutput, PydanticModel, ToolSelection
from bridgic.core.agentic.tool_specs import ToolSpec, FunctionToolSpec, AutomaToolSpec
from bridgic.core.agentic.recent._recent_memory_manager import ReCentMemoryManager, ReCentContext
from bridgic.core.agentic.recent._recent_memory_config import ReCentMemoryConfig
from bridgic.core.agentic.recent._episodic_node import CompressionEpisodicNode
from bridgic.core.prompt import EjinjaPromptTemplate
from bridgic.core.utils._console import printer
from bridgic.core.utils._tool_calling import stringify_tool_result


class GoalStatus(BaseModel):
    """
    Goal achievement status.
    """
    brief_thinking: str = Field(..., description="Brief thinking about the current status and whether the goal is achieved.")
    goal_achieved: bool = Field(..., description="Whether the goal has been achieved.")


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
    async def initialize_task_goal(
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
        if not goal:
            raise ValueError("Goal cannot be empty.")
        self._memory_manager.create_goal(goal, guidance)

        # Log task goal initialization in debug mode.
        top_options = self._get_top_running_options()
        if top_options.debug:
            msg = (
                f"[ReCentAutoma] 🎯 Task Goal\n"
                f"{goal}" + (f"\n\n{guidance}" if guidance else '')
            )
            printer.print(msg)

        self.ferry_to("observe")

    @worker()
    async def observe(self):
        """
        Observe the current state and determine if the goal has been achieved.

        This worker builds context from memory, uses LLM (with `StructuredOutput` protocol) to 
        determine if the goal has been achieved, and routes accordingly.
        """
        # 1. Build context from memory.
        context: ReCentContext = await self._memory_manager.abuild_context()

        # 2. Build LLM message list for goal status evaluation.
        messages: List[Message] = []

        # Add system message for goal evaluation.
        system_template = EjinjaPromptTemplate(
            "You are an AI assistant that evaluates whether a task goal has been achieved."
            "{%- if goal_content %}\n\nTask Goal: {{ goal_content }}{% endif %}"
            "{%- if goal_guidance %}\n\nTask Guidance: {{ goal_guidance }}{% endif %}"
        )
        system_message = system_template.format_message(
            role=Role.SYSTEM,
            goal_content=context["goal_content"],
            goal_guidance=context["goal_guidance"],
        )
        messages.append(system_message)

        # Add memory messages (observation history).
        if context["memory_messages"]:
            messages.extend(context["memory_messages"])

        # Add instruction message.
        instruction_template = EjinjaPromptTemplate(
            "Based on the task goal and observation history, determine if the goal has been completed. "
            "Provide a brief thinking about the current status and whether the goal is achieved."
        )
        instruction_message = instruction_template.format_message(role=Role.USER)
        messages.append(instruction_message)

        # 3. Call LLM with StructuredOutput to get goal status.
        if not isinstance(self._llm, StructuredOutput):
            raise TypeError(f"LLM must support StructuredOutput protocol, but {type(self._llm)} does not.")

        goal_status: GoalStatus = await self._llm.astructured_output(
            messages=messages,
            constraint=PydanticModel(model=GoalStatus),
        )
        observation_template = EjinjaPromptTemplate(
            "Goal Status:\n"
            "- Achieved: {%- if goal_status.goal_achieved %}goal_status.goal_achieved{% else %}No{% endif %}\n"
            "- Thinking: {%- if goal_status.brief_thinking %}{{ goal_status.brief_thinking }}{% else %}(No thinking){% endif %}"
        )
        observation_message = observation_template.format_message(
            role=Role.USER,
            goal_status=goal_status,
        )
        self._memory_manager.push_messages([observation_message])

        # Log observation result in debug mode.
        top_options = self._get_top_running_options()
        if top_options.debug:
            msg = (
                f"[ReCentAutoma] 👀 Observation\n"
                f"    Goal Achieved: {goal_status.goal_achieved}\n"
                f"    Brief Thinking: {goal_status.brief_thinking}"
            )
            printer.print(msg, color="gray")

        # 4. Dynamic routing based on goal status.
        if goal_status.goal_achieved or not self._tool_specs:
            # Goal achieved, route to finalize_answer.
            self.ferry_to("finalize_answer")
        else:
            # Goal not achieved, prepare messages and tools, then route to select_tools.
            tool_select_template = EjinjaPromptTemplate(
                "You are an AI assistant. You are able to select appropriate tool(s) to help "
                "complete the next step of the task (if needed)."
                "{%- if goal_content %}\n\nTask Goal: {{ goal_content }}{% endif %}"
                "{%- if goal_guidance %}\n\nTask Guidance: {{ goal_guidance }}{% endif %}"
            )
            tool_select_message = tool_select_template.format_message(
                role=Role.SYSTEM,
                goal_content=context["goal_content"],
                goal_guidance=context["goal_guidance"],
            )
            tool_select_messages = [tool_select_message] + context["memory_messages"]

            tools = [tool_spec.to_tool() for tool_spec in self._tool_specs]

            # Route to select_tools with messages and tools.
            self.ferry_to("select_tools", messages=tool_select_messages, tools=tools)

    @worker()
    async def select_tools(
        self,
        messages: List[Message],
        tools: List[Tool],
    ) -> Tuple[List[ToolCall], Optional[str]]:
        """
        Select tools using LLM's tool selection capability.

        This method calls the LLM's aselect_tool method to select appropriate tools
        based on the conversation context.

        Parameters
        ----------
        messages : List[Message]
            The conversation history and current context.
        tools : List[Tool]
            Available tools that can be selected for use.

        Returns
        -------
        Tuple[List[ToolCall], Optional[str]]
            A tuple containing:
            - List of selected tool calls with determined parameters
            - Optional response text from the LLM
        """
        # Check if LLM supports ToolSelection protocol.
        if not isinstance(self._llm, ToolSelection):
            raise TypeError(f"LLM must support ToolSelection protocol, but {type(self._llm)} does not.")

        # Call LLM's tool selection method.
        tool_calls, llm_response = await self._llm.aselect_tool(
            messages=messages,
            tools=tools,
        )

        # Log selected tools in debug mode.
        top_options = self._get_top_running_options()
        if top_options.debug:
            tool_info_lines = []

            if tool_calls:
                for i, tool_call in enumerate(tool_calls, 1):
                    tool_info_lines.append(f"    Tool {i}: {tool_call.name}")
                    tool_info_lines.append(f"      id: {tool_call.id}")
                    tool_info_lines.append(f"      arguments: {tool_call.arguments}")
            else:
                tool_info_lines.append("    (No tools selected)")

            if llm_response:
                tool_info_lines.append(f"\n    LLM Response: {llm_response}")

            tool_info_lines_str = "\n".join(tool_info_lines)

            msg = (
                f"[ReCentAutoma] 🔧 Tool Selection\n"
                f"{tool_info_lines_str}"
            )
            printer.print(msg, color="orange")

        # Push Assistant message with tool calls and response to memory.
        if tool_calls or llm_response:
            assistant_message = Message.from_tool_call(tool_calls=tool_calls, text=llm_response)
            self._memory_manager.push_messages([assistant_message])

        # Match tool calls with tool specs.
        if tool_calls and self._tool_specs:
            matched_list = self._match_tool_calls_and_tool_specs(tool_calls, self._tool_specs)

            if matched_list:
                # Create tool workers dynamically.
                tool_worker_keys = []
                matched_tool_calls = []

                for tool_call, tool_spec in matched_list:
                    # Create the tool worker.
                    tool_worker_key = f"tool-<{tool_call.name}>-<{tool_call.id}>"
                    tool_worker_obj = tool_spec.create_worker()

                    # Register the tool worker.
                    self.add_worker(key=tool_worker_key, worker=tool_worker_obj)
                    tool_worker_keys.append(tool_worker_key)
                    matched_tool_calls.append(tool_call)

                    # Execute the tool worker in the next dynamic step (via ferry_to).
                    self.ferry_to(tool_worker_key, **tool_call.arguments)

                # Create collect_results worker dynamically. In this worker, after collecting, the tool results will be push to memory.
                def collect_wrapper(tool_results: List[Any]) -> None:
                    return self._collect_tools_results(matched_tool_calls, tool_results)

                self.add_func_as_worker(
                    key=f"collect_results-<{uuid.uuid4().hex[:8]}>",
                    func=collect_wrapper,
                    dependencies=tool_worker_keys,
                    args_mapping_rule=ArgsMappingRule.MERGE,
                )

                tool_selected = True
            else:
                tool_selected = False
        else:
            tool_selected = False

        if not tool_selected:
            self.ferry_to("finalize_answer")

    @worker()
    async def check_compression(self):
        """
        Check whether memory compression is needed.

        This worker checks if the compression conditions are met and decides which nodes to route to.
        Either compress_memory or observe will inevitably be chosen to execute in the next step.
        """
        should_compress = self._memory_manager.should_trigger_compression()

        # Log compression decision in debug mode.
        top_options = self._get_top_running_options()
        if top_options.debug:
            msg = (
                f"[ReCentAutoma] 🧭 Memory Check\n"
                f"    Compression Needed: {should_compress}"
            )
            printer.print(msg, color="olive")

        if should_compress:
            self.ferry_to("compress_memory")
        else:
            self.ferry_to("observe")

    @worker()
    async def compress_memory(self):
        """
        Compress early memory nodes.

        This worker triggers memory compression by creating a compression node
        that summarizes earlier episodic nodes.
        """
        # Create the compression node. The summary is now complete.
        timestep = await self._memory_manager.acreate_compression()

        # Log compression summary in debug mode.
        top_options = self._get_top_running_options()
        if top_options.debug:
            node: CompressionEpisodicNode = self._memory_manager.get_specified_memory_node(timestep)
            summary = await asyncio.wrap_future(node.summary)
            msg = (
                f"[ReCentAutoma] 🗄 Memory Compression\n"
                f"{summary}"
            )
            printer.print(msg, color="blue")

        self.ferry_to("observe")

    @worker(is_output=True)
    async def finalize_answer(self) -> str:
        """
        Generate the final answer based on memory and goal using LLM.

        This worker is the output node of the automa. It builds context from memory,
        calls LLM to generate a comprehensive final answer based on the goal and
        conversation history.

        Returns
        -------
        str
            The final answer string.
        """
        # 1. Build context from memory.
        context: ReCentContext = await self._memory_manager.abuild_context()

        # 2. Build LLM message list for final answer generation.
        messages: List[Message] = []

        # Add system prompt for final answer generation.
        system_template = EjinjaPromptTemplate(
            "You are a helpful assistant. Based on the task goal and the entire conversation history, "
            "provide a clear, complete, and well-structured final answer that addresses the goal."
            "{%- if goal_content %}\n\nTask Goal: {{ goal_content }}{% endif %}"
            "{%- if goal_guidance %}\n\nTask Guidance: {{ goal_guidance }}{% endif %}"
        )
        system_message = system_template.format_message(
            role=Role.SYSTEM,
            goal_content=context["goal_content"],
            goal_guidance=context["goal_guidance"],
        )
        messages.append(system_message)

        # Add memory messages (observation history).
        if context["memory_messages"]:
            messages.extend(context["memory_messages"])

        # Add instruction to generate final answer.
        instruction_template = EjinjaPromptTemplate(
            "Based on the task goal and observation history above, please provide a comprehensive "
            "and well-structured final answer that addresses the goal."
        )
        instruction_message = instruction_template.format_message(role=Role.USER)
        messages.append(instruction_message)

        # 3. Call LLM to generate final answer.
        response = await self._llm.achat(messages=messages)
        final_answer = response.message.content

        # 4. Push final answer as a Message into the memory
        final_answer_msg = Message.from_text(text=final_answer, role=Role.AI)
        self._memory_manager.push_messages([final_answer_msg])

        return final_answer

    def _collect_tools_results(
        self,
        tool_calls: Optional[List[ToolCall]],
        tool_results: List[Any],
    ) -> None:
        """
        Collect results from tool executions and push to memory.

        This method is used as a worker function to collect results from multiple tool workers. 
        It converts tool results to ToolResult messages and pushes them to memory.

        Parameters
        ----------
        tool_calls : Optional[List[ToolCall]]
            List of tool calls (optional, for building ToolResult messages).
        tool_results : List[Any]
            List of tool execution results (merged via ArgsMappingRule.MERGE).
        """
        # Convert tool results to ToolResult messages.
        tool_result_messages: List[Message] = []

        for tool_call, tool_result in zip(tool_calls, tool_results):
            # Convert tool result to string
            result_content = str(tool_result)
            # Create ToolResult message
            tool_result_message = Message.from_tool_result(
                tool_id=tool_call.id,
                content=result_content,
            )
            tool_result_messages.append(tool_result_message)

        # Push tool result messages to memory.
        if tool_result_messages:
            self._memory_manager.push_messages(tool_result_messages)

        # Log tool execution results in debug mode.
        top_options = self._get_top_running_options()
        if top_options.debug:
            result_lines = []
            for i, (tool_call, tool_result) in enumerate(zip(tool_calls, tool_results), 1):
                # Format tool result with MCP support
                result_content = stringify_tool_result(tool_result, verbose=top_options.verbose)
                result_content = result_content[:10000] + "..." if len(result_content) > 10000 else result_content
                result_lines.append(f"    Tool {i}: {tool_call.name}")
                result_lines.append(f"      id: {tool_call.id}")
                result_lines.append(f"      result: {result_content}")

            result_lines_str = "\n".join(result_lines)

            msg = (
                f"[ReCentAutoma] 🚩 Tool Results\n"
                f"{result_lines_str}"
            )
            printer.print(msg, color="green")

        # Route to check_compression.
        self.ferry_to("check_compression")

    def _match_tool_calls_and_tool_specs(
        self,
        tool_calls: List[ToolCall],
        tool_specs: List[ToolSpec],
    ) -> List[Tuple[ToolCall, ToolSpec]]:
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
