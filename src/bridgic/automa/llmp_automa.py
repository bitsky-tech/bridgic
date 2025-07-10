from bridgic.automa import GoalOrientedAutoma
from typing import Optional, Tuple, Any, Dict, Callable, List
from enum import Enum
from bridgic.types.common_types import ZeroToOne, LLMOutputFormat, PromptTemplate
from bridgic.core import LLM
from bridgic.automa.worker import Worker
from typing_extensions import override
from abc import ABCMeta
from bridgic.automa.worker_decorator import get_default_worker_args

class LlmpAutomaMeta(ABCMeta):
    def __new__(mcls, name, bases, dct):
        cls = super().__new__(mcls, name, bases, dct)

        def get_default_worker_args_for_llmp() -> Dict[str, Any]:
            default_args_list = get_default_worker_args()
            for default_args in default_args_list:
                if "canonical_description" in default_args:
                    return default_args
            return None
        
        for attr_name, attr_value in dct.items():
            worker_kwargs = getattr(attr_value, "__worker_kwargs__", None)
            if worker_kwargs is not None:
                default_args = get_default_worker_args_for_llmp()
                complete_args = {**default_args, **worker_kwargs}
                # print(f"%%%%%%%%%%%%%%%%% [{cls}.{attr_name}] worker_kwargs = {worker_kwargs}")
                # print(f"%%%%%%%%%%%%%%%%% [{cls}.{attr_name}] default_args = {default_args}")
                # print(f"%%%%%%%%%%%%%%%%% [{cls}.{attr_name}] complete_args = {complete_args}")
                # TODO:
        
        goal_config = getattr(cls, "__goal_config__", None)
        if goal_config is not None:
            print(f"%%%%%%%%%%%%%%%%% [{cls.__name__}] goal_config = {goal_config}")
            # TODO:
        
        # TODO:
        return cls

class PlanningStrategy(Enum):
    ReAct = 1
    LLMCompiler = 2

class LlmpAutoma(GoalOrientedAutoma, metaclass=LlmpAutomaMeta):
    descriptive_goal: str
    planning_llm: LLM
    planning_strategy: PlanningStrategy
    canonical_planning_prompt: Optional[str]

    def __init__(
        self,
        planning_llm: LLM, # TODO: provide a default LLM model?
        expected_output_format: LLMOutputFormat = LLMOutputFormat.FreeText,
        expected_output_json_schema: Optional[str] = None,
        name: Optional[str] = None, 
        planning_strategy: PlanningStrategy = PlanningStrategy.ReAct,
        canonical_planning_prompt: Optional[PromptTemplate] = None, # TODO: upgrade str to prompt template type
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.planning_llm = planning_llm
        self.planning_strategy = planning_strategy
        self.canonical_planning_prompt = canonical_planning_prompt

        self.goal_config = getattr(self, "__goal_config__", None)
        if self.goal_config is not None:
            print(f"%%%%%%%%%%%%%%%%% in init [{self.__class__.__name__}] goal_config = {self.goal_config}")
            # TODO:
       

    def process_async(self, *args: Optional[Tuple[Any]], automa_context: Dict[str, Any] = None, **kwargs: Optional[Dict[str, Any]]) -> Any:
        pass

    @override
    def remove_worker(self, worker_name: str) -> Worker:
        ...
        # TODO: implement this method
    
    @staticmethod
    def get_canonical_description(method: Callable) -> PromptTemplate:
        ...
        # TODO: implement this method
