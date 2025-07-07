from bridgic.automa import Automa
from typing import Optional, Tuple, Any, Dict, Callable
from enum import Enum
from bridgic.types.common_types import ZeroToOne, LLMOutputFormat, PromptTemplate
from bridgic.core import LLM

class PlanningStrategy(Enum):
    ReAct = 1
    LLMCompiler = 2

# Tool selection scope?
class FuzzyGoalAutoma(Automa):
    descriptive_goal: str
    planning_llm: LLM
    planning_strategy: PlanningStrategy
    canonical_planning_prompt: Optional[str]

    def __init__(
        self,
        descriptive_goal: str, # TODO: upgrade str to prompt template | prompt template path, etc
        planning_llm: LLM,
        expected_output_format: LLMOutputFormat = LLMOutputFormat.FreeText,
        expected_output_json_schema: Optional[str] = None,
        name: Optional[str] = None, # TODO: provide a default LLM model?
        planning_strategy: PlanningStrategy = PlanningStrategy.ReAct,
        canonical_planning_prompt: Optional[PromptTemplate] = None, # TODO: upgrade str to prompt template type
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.planning_llm = planning_llm
        self.planning_strategy = planning_strategy
        self.canonical_planning_prompt = canonical_planning_prompt

    def process_async(self, *args: Optional[Tuple[Any]], automa_context: Dict[str, Any] = None, **kwargs: Optional[Dict[str, Any]]) -> Any:
        pass

def descriptive_worker(
    *,
    name: Optional[str] = None,
    auto_convert_to_prompt: bool = True,
    cost: ZeroToOne = 0.0,
    re_use: bool = True,
    canonical_description: Optional[PromptTemplate] = None, # TODO: upgrade str to prompt template type
) -> Callable:
    """
    Either the canonical_description parameter or the decorated function's docstring must be provided.
    """
    def wrapper(func: Callable):
        # TODO:
        return func
    return wrapper
