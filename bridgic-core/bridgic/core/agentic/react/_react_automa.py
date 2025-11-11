import json

from typing import Optional, Dict, Any, List, Union, Callable, cast, Tuple
from typing_extensions import override
from concurrent.futures import ThreadPoolExecutor
from jinja2 import Environment, PackageLoader, Template

from bridgic.core.automa import Automa, GraphAutoma, worker, RunningOptions
from bridgic.core.automa.args import From, ArgsMappingRule, System
from bridgic.core.model.protocols import ToolSelection
from bridgic.core.model.types import Tool, ToolCall
from bridgic.core.automa.interaction import InteractionFeedback
from bridgic.core.agentic.types._chat_message import ChatMessage, SystemMessage, UserTextMessage, AssistantTextMessage, ToolMessage
from bridgic.core.agentic.types._chat_message import FunctionToolCall, Function
from bridgic.core.agentic.tool_specs import ToolSpec, FunctionToolSpec, AutomaToolSpec
from bridgic.core.agentic.workers import ToolSelectionWorker

DEFAULT_MAX_ITERATIONS = 20
DEFAULT_TEMPLATE_FILE = "tools_chat.jinja"

class ReActAutoma(GraphAutoma):
    """
    A react automa is a subclass of graph automa that implements the [ReAct](https://arxiv.org/abs/2210.03629) prompting framework.
    """

    _llm: ToolSelection
    """ The LLM to be used by the react automa. """
    _tools: Optional[List[ToolSpec]]
    """ The candidate tools to be used by the react automa. """
    _system_prompt: Optional[SystemMessage]
    """ The system prompt to be used by the react automa. """
    _max_iterations: int
    """ The maximum number of iterations for the react automa. """
    _prompt_template: str
    """ The template file for the react automa. """
    _jinja_env: Environment
    """ The Jinja environment to be used by the react automa. """
    _jinja_template: Template
    """ The Jinja template to be used by the react automa. """

    def __init__(
        self,
        llm: ToolSelection,
        system_prompt: Optional[Union[str, SystemMessage]] = None,
        tools: Optional[List[Union[Callable, Automa, ToolSpec]]] = None,
        name: Optional[str] = None,
        thread_pool: Optional[ThreadPoolExecutor] = None,
        running_options: Optional[RunningOptions] = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        prompt_template: str = DEFAULT_TEMPLATE_FILE,
    ):
        super().__init__(name=name, thread_pool=thread_pool, running_options=running_options)

        self._llm = llm
        if system_prompt:
            # Validate SystemMessage...
            if isinstance(system_prompt, str):
                system_prompt = SystemMessage(role="system", content=system_prompt)
            elif ("role" not in system_prompt) or (system_prompt["role"] != "system"):
                raise ValueError(f"Invalid `system_prompt` value received: {system_prompt}. It should contain `role`=`system`.")

        self._system_prompt = system_prompt
        if tools:
            self._tools = [self._ensure_tool_spec(tool) for tool in tools]
        else:
            self._tools = None
        self._max_iterations = max_iterations
        self._prompt_template = prompt_template
        self._jinja_env = Environment(loader=PackageLoader("bridgic.core.agentic.react"))
        self._jinja_template = self._jinja_env.get_template(prompt_template)

        self.add_worker(
            key="tool_selector",
            worker=ToolSelectionWorker(tool_selection_llm=llm),
            dependencies=["assemble_context"],
            args_mapping_rule=ArgsMappingRule.UNPACK,
        )

    @override
    def dump_to_dict(self) -> Dict[str, Any]:
        state_dict = super().dump_to_dict()
        state_dict["tools"] = self._tools
        state_dict["system_prompt"] = self._system_prompt
        state_dict["llm"] = self._llm
        state_dict["max_iterations"] = self._max_iterations
        state_dict["prompt_template"] = self._prompt_template
        return state_dict

    @override
    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_from_dict(state_dict)
        self._max_iterations = state_dict["max_iterations"]
        self._prompt_template = state_dict["prompt_template"]
        self._tools = state_dict["tools"]
        self._system_prompt = state_dict["system_prompt"]
        self._llm = state_dict["llm"]
        self._jinja_env = Environment(loader=PackageLoader("bridgic.core.agentic.react"))
        self._jinja_template = self._jinja_env.get_template(self._prompt_template)

    @property
    def max_iterations(self) -> int:
        return self._max_iterations

    @max_iterations.setter
    def max_iterations(self, max_iterations: int) -> None:
        self._max_iterations = max_iterations

    @property
    def prompt_template(self) -> str:
        return self._prompt_template

    @prompt_template.setter
    def prompt_template(self, prompt_template: str) -> None:
        self._prompt_template = prompt_template

    @override
    async def arun(
        self,
        user_msg: Optional[Union[str, UserTextMessage]] = None,
        *,
        chat_history: Optional[List[Union[UserTextMessage, AssistantTextMessage, ToolMessage]]] = None,
        messages: Optional[List[ChatMessage]] = None,
        tools: Optional[List[Union[Callable, Automa, ToolSpec]]] = None,
        feedback_data: Optional[Union[InteractionFeedback, List[InteractionFeedback]]] = None,
    ) -> Any:
        return await super().arun(
            user_msg=user_msg,
            chat_history=chat_history,
            messages=messages,
            tools=tools,
            feedback_data=feedback_data,
        )

    @worker(is_start=True)
    async def validate_and_transform(
        self,
        user_msg: Optional[Union[str, UserTextMessage]] = None,
        *,
        chat_history: Optional[List[Union[UserTextMessage, AssistantTextMessage, ToolMessage]]] = None,
        messages: Optional[List[ChatMessage]] = None,
        tools: Optional[List[Union[Callable, Automa, ToolSpec]]] = None,
    ) -> Dict[str, Any]:
        """
        Validate and transform the input messages and tools to the canonical format.
        """
        
        # Part One: validate and transform the input messages.
        # Unify input messages of various types to the `ChatMessage` format.
        chat_messages: List[ChatMessage] = []
        if messages:
            # If `messages` is provided, use it directly.
            chat_messages = messages
        elif user_msg:
            # Since `messages` is not provided, join the system prompt + `chat_history` + `user_msg`
            # First, append the `system_prompt`
            if self._system_prompt:
                chat_messages.append(self._system_prompt)
            
            # Second, append the `chat_history`
            if chat_history:
                for history_msg in chat_history:
                    # Validate the history messages...
                    role = history_msg["role"]
                    if role == "user" or role == "assistant" or role == "tool":
                        chat_messages.append(history_msg)
                    else:
                        raise ValueError(f"Invalid role: `{role}` received in history message: `{history_msg}`, expected `user`, `assistant`, or `tool`.")
            
            # Third, append the `user_msg`
            if isinstance(user_msg, str):
                chat_messages.append(UserTextMessage(role="user", content=user_msg))
            elif isinstance(user_msg, dict):
                if "role" in user_msg and user_msg["role"] == "user":
                    chat_messages.append(user_msg)
                else:
                    raise ValueError(f"`role` must be `user` in user message: `{user_msg}`.")
        else:
            raise ValueError(f"Either `messages` or `user_msg` must be provided.")

        # Part Two: validate and transform the intput tools.
        # Unify input tools of various types to the `ToolSpec` format.
        if self._tools:
            tool_spec_list = self._tools
        elif tools:
            tool_spec_list = [self._ensure_tool_spec(tool) for tool in tools]
        else:
            # TODO: whether to support empty tool list?
            tool_spec_list = []
    
        return {
            "initial_messages": chat_messages,
            "candidate_tools": tool_spec_list,
        }

    @worker(dependencies=["validate_and_transform"], args_mapping_rule=ArgsMappingRule.UNPACK)
    async def assemble_context(
        self,
        *,
        initial_messages: Optional[List[ChatMessage]] = None,
        candidate_tools: Optional[List[ToolSpec]] = None,
        tool_selection_outputs: Tuple[List[ToolCall], Optional[str]] = From("tool_selector", default=None),
        tool_result_messages: Optional[List[ToolMessage]] = None,
        rtx = System("runtime_context"),
    ) -> Dict[str, Any]:
        # print(f"\n******* ReActAutoma.assemble_context *******\n")
        # print(f"initial_messages: {initial_messages}")
        # print(f"candidate_tools: {candidate_tools}")
        # print(f"tool_selection_outputs: {tool_selection_outputs}")
        # print(f"tool_result_messages: {tool_result_messages}")
    
        local_space = self.get_local_space(rtx)
        # Build messages memory with help of local space.
        messages_memory: List[ChatMessage] = []
        if initial_messages:
            # If `messages` is provided, use it to re-initialize the messages memory.
            messages_memory = initial_messages.copy()
        else:
            messages_memory = local_space.get("messages_memory", [])
        if tool_selection_outputs:
            # Transform tools_calls format:
            tool_calls = tool_selection_outputs[0]
            tool_calls_list = [
                FunctionToolCall(
                    id=tool_call.id,
                    type="function",
                    function=Function(
                        name=tool_call.name,
                        arguments=tool_call.arguments,
                    ),
                ) for tool_call in tool_calls
            ]
            llm_response = tool_selection_outputs[1]
            assistant_message = AssistantTextMessage(
                role="assistant",
                # TOD: name?
                content=llm_response,
                tool_calls=tool_calls_list,
            )
            messages_memory.append(assistant_message)
        if tool_result_messages:
            messages_memory.extend(tool_result_messages)
        local_space["messages_memory"] = messages_memory
        # print("--------------------------------")
        # print(f"messages_memory: {messages_memory}")
        
        # Save & retrieve tools with help of local space.
        if candidate_tools:
            local_space["tools"] = candidate_tools
        else:
            candidate_tools = local_space.get("tools", [])

        # Note: here 'messages' and `tools` are injected into the template as variables.
        raw_prompt = self._jinja_template.render(messages=messages_memory, tools=candidate_tools)
        # print(f"\n ##### raw_prompt ##### \n{raw_prompt}")

        # Note: the jinjia template must conform to the TypedDict `ChatMessage` format (in json).
        llm_messages = cast(List[ChatMessage], json.loads(raw_prompt))
        llm_tools: List[Tool] = [tool.to_tool() for tool in candidate_tools]
        
        return {
            "messages": llm_messages,
            "tools": llm_tools,
        }

    @worker(dependencies=["tool_selector"], args_mapping_rule=ArgsMappingRule.UNPACK)
    async def plan_next_step(
        self,
        tool_calls: List[ToolCall],
        llm_response: Optional[str] = None,
        messages_and_tools: dict = From("validate_and_transform"),
        rtx = System("runtime_context"),
    ) -> None:
        local_space = self.get_local_space(rtx)
        iterations_count = local_space.get("iterations_count", 0)
        iterations_count += 1
        local_space["iterations_count"] = iterations_count
        if iterations_count > self._max_iterations:
            # TODO: how to report this to users?
            self.ferry_to(
                "finally_summarize", 
                final_answer=f"Sorry, I am unable to answer your question after {self._max_iterations} iterations. Please try again later."
            )
            return

        # TODO: maybe hand over the control flow to users?
        # print(f"\n******* ReActAutoma.plan_next_step *******\n")
        # print(f"tool_calls: {tool_calls}")
        # print(f"llm_response: {llm_response}")
        if tool_calls:
            tool_spec_list = messages_and_tools["candidate_tools"]
            matched_list = self._match_tool_calls_and_tool_specs(tool_calls, tool_spec_list)
            if matched_list:
                matched_tool_calls = []
                tool_worker_keys = []
                for tool_call, tool_spec in matched_list:
                    matched_tool_calls.append(tool_call)
                    tool_worker = tool_spec.create_worker()
                    worker_key = f"tool_{tool_call.name}_{tool_call.id}"
                    self.add_worker(
                        key=worker_key,
                        worker=tool_worker,
                    )
                    # TODO: convert tool_call.arguments to the tool parameters types
                    # TODO: validate the arguments against the tool parameters / json schema
                    self.ferry_to(worker_key, **tool_call.arguments)
                    tool_worker_keys.append(worker_key)
                self.add_func_as_worker(
                    key="merge_tools_results",
                    func=self.merge_tools_results,
                    dependencies=tool_worker_keys,
                    args_mapping_rule=ArgsMappingRule.MERGE,
                )
                return matched_tool_calls
            else:
                # TODO
                ...
        else:
            # Got final answer from the LLM.
            self.ferry_to("finally_summarize", final_answer=llm_response)

    async def merge_tools_results(
        self, 
        tool_results: List[Any],
        tool_calls: List[ToolCall] = From("plan_next_step"),
    ) -> List[ToolMessage]:
        """
        Merge the results of the tools.
        """
        # print(f"\n******* ReActAutoma.merge_tools_results *******\n")
        # print(f"tool_results: {tool_results}")
        # print(f"tool_calls: {tool_calls}")
        assert len(tool_results) == len(tool_calls)
        tool_messages = []
        for tool_result, tool_call in zip(tool_results, tool_calls):
            tool_messages.append(ToolMessage(
                role="tool", 
                # Note: Convert the tool result to string, since a tool can return any type of data.
                # TODO: maybe we can use a better way to serialize the tool result?
                content=str(tool_result), 
                tool_call_id=tool_call.id
            ))
            # Remove the tool workers
            self.remove_worker(f"tool_{tool_call.name}_{tool_call.id}")
        # Remove self...
        self.remove_worker("merge_tools_results")
        self.ferry_to("assemble_context", tool_result_messages=tool_messages)
        return tool_messages

    @worker(is_output=True)
    async def finally_summarize(self, final_answer: str) -> str:
        return final_answer

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

    def _match_tool_calls_and_tool_specs(
        self,
        tool_calls: List[ToolCall],
        tool_spec_list: List[ToolSpec],
    ) -> List[Tuple[ToolCall, ToolSpec]]:
        """
        This function is used to match the tool calls and the tool specs based on the tool name.

        Parameters
        ----------
        tool_calls : List[ToolCall]
            The tool calls to match.
        tool_spec_list : List[ToolSpec]
            The tool specs to match.

        Returns
        -------
        List[(ToolCall, ToolSpec)]
            The matched tool calls and tool specs.
        """
        matched_list: List[Tuple[ToolCall, ToolSpec]] = []
        for tool_call in tool_calls:
            for tool_spec in tool_spec_list:
                if tool_call.name == tool_spec.tool_name:
                    matched_list.append((tool_call, tool_spec))
        return matched_list