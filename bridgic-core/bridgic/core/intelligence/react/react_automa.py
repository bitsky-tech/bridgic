from typing import Optional, Dict, Any, List, Union
from bridgic.core.intelligence.tool_spec import ToolSpec
from typing_extensions import override
from concurrent.futures import ThreadPoolExecutor

from bridgic.core.automa import GraphAutoma
from bridgic.core.intelligence.base_llm import BaseLlm, Message
from bridgic.core.intelligence.protocol import ToolCall
from bridgic.core.automa.interaction import InteractionFeedback

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

    ##### Variables that need to be serialized #####
    _max_iterations: int
    """ The maximum number of iterations for the react automa. """
    _prompt_template: str
    """ The template file for the react automa. """
    _llm: BaseLlm
    """ The LLM to be used by the react automa. """
    _system_prompt: str
    """ The system prompt to be used by the react automa. """

    def __init__(
        self,
        llm: BaseLlm,
        system_prompt: Optional[str] = None,
        name: Optional[str] = None,
        thread_pool: Optional[ThreadPoolExecutor] = None,
        max_iterations: Optional[int] = DEFAULT_MAX_ITERATIONS,
        prompt_template: Optional[str] = DEFAULT_TEMPLATE_FILE,
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

    async def arun(
        self,
        user_msg: Optional[Union[str, Message]] = None,
        *
        chat_history: Optional[List[ChatMessage]] = None,
        messages: Optional[List[Message]] = None,
        interaction_feedback: Optional[InteractionFeedback] = None,
        interaction_feedbacks: Optional[List[InteractionFeedback]] = None,
        **kwargs: Dict[str, Any]
    ) -> Any:
        ...