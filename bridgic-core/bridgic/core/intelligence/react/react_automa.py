from typing import Optional, Dict, Any, List, Union, Callable, cast
from bridgic.core.intelligence.tool_spec import ToolSpec
from typing_extensions import override
from concurrent.futures import ThreadPoolExecutor
from jinja2 import Environment, PackageLoader, Template
import json

from bridgic.core.automa import Automa, GraphAutoma
from bridgic.core.intelligence.base_llm import Message
from bridgic.core.intelligence.protocol import ToolCall, Tool, ToolSelection
from bridgic.core.automa.interaction import InteractionFeedback
from bridgic.core.prompt.chat_message import ChatMessage, SystemMessage, UserTextMessage, AssistantTextMessage, ToolMessage
from bridgic.core.automa import worker, ArgsMappingRule
from bridgic.core.intelligence import FunctionToolSpec, AutomaToolSpec
from bridgic.core.intelligence.worker import ToolSelectionWorker
from bridgic.core.prompt.utils.prompt_utils import transform_chat_message_to_llm_message

DEFAULT_MAX_ITERATIONS = 20
DEFAULT_TEMPLATE_FILE = "tools_chat.jinja"

## Features:
# - [1] flexible and convenient input message types.
# - [2] full support of human interaction, including serialization and deserialization.
# - [3] customize ToolSpec encoding.
# - [4] customize tool running concurrency.
## TODO: customize window composer, different from prompt template?


class ReActStepDecisionMaker:
    def decide_next_step(
        self,
        messages: List[Message],
        tool_calls: Optional[List[ToolCall]] = None,
    ) -> bool:
        pass

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
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        prompt_template: str = DEFAULT_TEMPLATE_FILE,
        timeout: Optional[float] = None,
    ):
        super().__init__(name=name, thread_pool=thread_pool)

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
        self._jinja_env = Environment(loader=PackageLoader("bridgic.core.intelligence.react"))
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
        # TODO: tools
        state_dict["max_iterations"] = self._max_iterations
        state_dict["prompt_template"] = self._prompt_template
        return state_dict

    @override
    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_from_dict(state_dict)
        self._max_iterations = state_dict["max_iterations"]
        self._prompt_template = state_dict["prompt_template"]

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
        interaction_feedback: Optional[InteractionFeedback] = None,
        interaction_feedbacks: Optional[List[InteractionFeedback]] = None,
    ) -> Any:
        return await super().arun(
            user_msg=user_msg,
            chat_history=chat_history,
            messages=messages,
            tools=tools,
            interaction_feedback=interaction_feedback,
            interaction_feedbacks=interaction_feedbacks,
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
            "messages": chat_messages,
            "tools": tool_spec_list,
        }

    @worker(dependencies=["validate_and_transform"], args_mapping_rule=ArgsMappingRule.UNPACK)
    async def assemble_context(
        self,
        *,
        messages: Optional[List[ChatMessage]] = None,
        tools: Optional[List[ToolSpec]] = None,
        # TODO: type of tool_results
        tool_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        # Note: here 'messages' and `tools` are injected into the template as variables.
        raw_prompt = self._jinja_template.render(messages=messages, tools=tools)

        # Transform this raw prompt to `List[Message]` format expected by the LLM.
        messages = cast(List[Message], json.loads(raw_prompt))
        llm_messages = [transform_chat_message_to_llm_message(message) for message in messages]

        llm_tools: List[Tool] = [tool.to_tool() for tool in tools]
        
        return {
            "messages": llm_messages,
            "tools": llm_tools,
        }

    async def plan_next_step(
        self,
        messages: List[Message],
        tool_calls: Optional[List[ToolCall]] = None,
    ) -> None:
        # TODO: call ReActStepDecisionMaker
        ...

    def _ensure_tool_spec(self, tool: Union[Callable, Automa, ToolSpec]) -> ToolSpec:
        if isinstance(tool, Callable):
            return FunctionToolSpec.from_raw(tool)
        elif isinstance(tool, Automa):
            return AutomaToolSpec.from_raw(tool)
        elif isinstance(tool, ToolSpec):
            return tool
        else:
            raise TypeError(f"Invalid tool type: {type(tool)} detected, expected `Callable`, `Automa`, or `ToolSpec`.")