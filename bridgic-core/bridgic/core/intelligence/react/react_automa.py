from typing import Optional, Dict, Any, List, Union, Callable
from bridgic.core.intelligence.tool_spec import ToolSpec
from typing_extensions import override, is_typeddict
from concurrent.futures import ThreadPoolExecutor

from bridgic.core.automa import GraphAutoma
from bridgic.core.intelligence.base_llm import BaseLlm, Message, Role
from bridgic.core.intelligence.protocol import ToolCall
from bridgic.core.automa.interaction import InteractionFeedback
from bridgic.core.prompt.chat_message import ChatMessage, SystemMessage, UserTextMessage, AssistantTextMessage, ToolMessage
from bridgic.core.automa import worker, ArgsMappingRule

DEFAULT_MAX_ITERATIONS = 20
DEFAULT_TEMPLATE_FILE = "bridgic/core/intelligence/react/default_template.txt"

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

    _llm: BaseLlm
    """ The LLM to be used by the react automa. """
    _system_prompt: Optional[Union[str, SystemMessage, Message]]
    """ The system prompt to be used by the react automa. """
    _max_iterations: int
    """ The maximum number of iterations for the react automa. """
    _prompt_template: str
    """ The template file for the react automa. """

    def __init__(
        self,
        llm: BaseLlm,
        system_prompt: Optional[Union[str, SystemMessage, Message]] = None,
        name: Optional[str] = None,
        thread_pool: Optional[ThreadPoolExecutor] = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        prompt_template: str = DEFAULT_TEMPLATE_FILE,
        timeout: Optional[float] = None,
    ):
        super().__init__(name=name, thread_pool=thread_pool)

        self._llm = llm
        if system_prompt:
            # Validate...
            if isinstance(system_prompt, dict):
                # SystemMessage
                if ("role" not in system_prompt) or (system_prompt["role"] != "system"):
                    raise ValueError(f"Invalid `system_prompt` value received: {system_prompt}. It should contain `role`=`system`.")
            elif isinstance(system_prompt, Message):
                if system_prompt.role != Role.SYSTEM:
                    raise ValueError(f"Invalid `system_prompt` value received: {system_prompt}. It should contain `role`=`Role.SYSTEM`.")

        self._system_prompt = system_prompt
        self._max_iterations = max_iterations
        self._prompt_template = prompt_template

    @override
    def dump_to_dict(self) -> Dict[str, Any]:
        state_dict = super().dump_to_dict()
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
        user_msg: Optional[Union[str, UserTextMessage, Message]] = None,
        *,
        chat_history: Optional[List[Union[UserTextMessage, AssistantTextMessage, ToolMessage, Message]]] = None,
        messages: Optional[List[Union[ChatMessage, Message]]] = None,
        candidate_tools: Optional[List[Union[Callable, ToolSpec]]] = None,
        interaction_feedback: Optional[InteractionFeedback] = None,
        interaction_feedbacks: Optional[List[InteractionFeedback]] = None,
    ) -> Any:
        return await super().arun(
            user_msg=user_msg,
            chat_history=chat_history,
            messages=messages,
            candidate_tools=candidate_tools,
            interaction_feedback=interaction_feedback,
            interaction_feedbacks=interaction_feedbacks,
        )

    @worker(is_start=True)
    async def validate_and_transform(
        self,
        user_msg: Optional[Union[str, UserTextMessage, Message]] = None,
        *,
        chat_history: Optional[List[Union[UserTextMessage, AssistantTextMessage, ToolMessage, Message]]] = None,
        messages: Optional[List[Union[ChatMessage, Message]]] = None,
        candidate_tools: Optional[List[Union[Callable, ToolSpec]]] = None,
    ) -> Dict[str, Any]:
        """
        Validate and transform the input messages and tools to the format expected by the LLM.
        """
         
        # Part One: validate and transform the input messages.
        # Unify input messages of various types to the format expected by the LLM.
        llm_messages: List[Message] = []
        if messages:
            # If `messages` is provided, use it directly.
            for message in messages:
                if isinstance(message, dict):
                    llm_messages.append(self._transform_chat_message_to_llm_message(message))
                elif isinstance(message, Message):
                    llm_messages.append(message)
                else:
                    raise TypeError(f"Invalid `messages` type: {type(message)}, expected `ChatMessage` or `Message`.")
        elif user_msg:
            # Since `messages` is not provided, join the system prompt + `chat_history` + `user_msg`
            if self._system_prompt:
                if isinstance(self._system_prompt, str):
                    llm_messages.append(Message.from_text(self._system_prompt, Role.SYSTEM))
                elif isinstance(self._system_prompt, dict):
                    llm_messages.append(self._transform_chat_message_to_llm_message(self._system_prompt))
                elif isinstance(self._system_prompt, Message):
                    llm_messages.append(self._system_prompt)
                else:
                    raise TypeError(f"Invalid `system_prompt` type: {type(self._system_prompt)}, expected `str`, `SystemMessage`, or `Message`.")
            
            if chat_history:
                for history_msg in chat_history:
                    if isinstance(history_msg, dict):
                        # UserTextMessage, AssistantTextMessage, or ToolMessage
                        role = history_msg["role"]
                        if role == "user" or role == "assistant" or role == "tool":
                            llm_messages.append(self._transform_chat_message_to_llm_message(history_msg))
                        else:
                            raise ValueError(f"Invalid role: `{role}` received in message: `{history_msg}`.")
                    elif isinstance(history_msg, Message):
                        if history_msg.role == Role.USER or history_msg.role == Role.AI or history_msg.role == Role.TOOL:
                            llm_messages.append(history_msg)
                        else:
                            raise ValueError(f"Invalid role: `{history_msg.role}` received in message: `{history_msg}`.")
                    else:
                        raise TypeError(f"Invalid `chat_history` message type: {type(history_msg)}, expected `UserTextMessage`, `AssistantTextMessage`, `ToolMessage` or `Message`.")
            
            # Append the `user_msg`
            if isinstance(user_msg, str):
                llm_messages.append(Message.from_text(user_msg, Role.USER))
            elif isinstance(user_msg, dict):
                role = user_msg["role"]
                if role == "user":
                    llm_messages.append(self._transform_chat_message_to_llm_message(user_msg))
                else:
                    raise ValueError(f"`role` must be `user` in user message: `{user_msg}`.")
            elif isinstance(user_msg, Message):
                if user_msg.role == Role.USER:
                    llm_messages.append(user_msg)
                else:
                    raise ValueError(f"`role` must be `user` in user message: `{user_msg}`.")
            else:
                raise TypeError(f"Invalid `user_msg` type: {type(user_msg)}, expected `str`, `UserTextMessage`, or `Message`.")

        # Part Two: validate and transform the intput tools.
        # Unify input tools of various types to the format expected by the LLM.
    
        return {
            "messages": llm_messages,
        }

    @worker(dependencies=["validate_and_transform"], args_mapping_rule=ArgsMappingRule.UNPACK)
    async def assemble_context(
        self,
        *,
        messages: Optional[List[Message]] = None,
        candidate_tools: Optional[List[ToolSpec]] = None,
        # TODO: type of tool_results
        tool_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        print(f"\n***** in assemble_context, messages: {messages} *****")

    async def plan_next_step(
        self,
        messages: List[Message],
        tool_calls: Optional[List[ToolCall]] = None,
    ) -> None:
        # TODO: call ReActStepDecisionMaker
        ...

    def _transform_chat_message_to_llm_message(self, message: ChatMessage) -> Message:
        role = message["role"]
        extras = {}
        if role == "system":
            name = message.get("name", None)
            if name:
                extras["name"] = name
            return Message.from_text(message["content"], Role.SYSTEM, extras)
        elif role == "user":
            name = message.get("name", None)
            if name:
                extras["name"] = name
            return Message.from_text(message["content"], Role.USER, extras)
        elif role == "assistant":
            name = message.get("name", None)
            if name:
                extras["name"] = name
            # TODO: handle tool_calls; choose between content and tool_calls
            return Message.from_text(message["content"], Role.AI, extras)
        elif role == "tool":
            # tool_call_id is required
            extras["tool_call_id"] = message["tool_call_id"]
            return Message.from_text(message["content"], Role.TOOL, extras)
        else:
            raise ValueError(f"Invalid role: `{role}` received in message: `{message}`.")
