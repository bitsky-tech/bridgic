from typing import Optional, Dict, Any, List, Union, Callable
from bridgic.core.intelligence.tool_spec import ToolSpec
from typing_extensions import override, is_typeddict
from concurrent.futures import ThreadPoolExecutor

from bridgic.core.automa import GraphAutoma
from bridgic.core.intelligence.base_llm import BaseLlm, Message
from bridgic.core.intelligence.protocol import ToolCall
from bridgic.core.automa.interaction import InteractionFeedback
from bridgic.core.prompt.chat_message import ChatMessage, UserTextMessage, AssistantTextMessage, ToolMessage
from bridgic.core.automa import worker

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
    _system_prompt: str
    """ The system prompt to be used by the react automa. """
    _max_iterations: int
    """ The maximum number of iterations for the react automa. """
    _prompt_template: str
    """ The template file for the react automa. """

    def __init__(
        self,
        llm: BaseLlm,
        system_prompt: Optional[str] = None,
        name: Optional[str] = None,
        thread_pool: Optional[ThreadPoolExecutor] = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        prompt_template: str = DEFAULT_TEMPLATE_FILE,
        timeout: Optional[float] = None,
    ):
        super().__init__(name=name, thread_pool=thread_pool)

        self._llm = llm
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
        # Validate Part One: validate the input messages.
        if user_msg is not None:
            is_tpd = is_typeddict(user_msg)
            if not isinstance(user_msg, str) and not is_tpd and not isinstance(user_msg, Message):
                raise ValueError(f"Invalid `user_msg` type: {type(user_msg)}, expected `str`, `TypedDict`, or `Message`.")
            if is_tpd:
                self._validate_typedict_message_type(user_msg, UserTextMessage)
         
         
        # Transform Part One: transform the input messages.
        # Unify input messages of various types to the format expected by the LLM.
        llm_messages: List[Message] = []
        if messages is not None:
            for message in messages:
                if isinstance(message, ChatMessage):
                    llm_messages.append(Message.from_text(message["content"], message["role"]))
                elif isinstance(message, Message):
                    llm_messages.append(message)
                else:
                    raise ValueError(f"Invalid `messages` type: {type(message)}, expected `ChatMessage` or `Message`.")

        # Transform Part Two: transform the intput tools.
        # Unify input tools of various types to the format expected by the LLM.

    async def assemble_context(
        self,
        messages: Optional[List[Message]] = None,
        *,
        candidate_tools: Optional[List[ToolSpec]] = None,
        # TODO: type of tool_results
        tool_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        pass

    async def plan_next_step(
        self,
        messages: List[Message],
        tool_calls: Optional[List[ToolCall]] = None,
    ) -> None:
        # TODO: call ReActStepDecisionMaker
        ...

    async def _validate_typedict_message_type(self, msg: Dict, msg_type: object):
        """
        Validate whether the input message `msg` is a valid instance of the type `msg_type`, which is a TypedDict or a union of several TypedDicts.
        If invalid, raise a TypeError.
        """
        ...