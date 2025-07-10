from bridgic.automa import GoalOrientedAutoma
from typing import Optional, Tuple, Any, Dict, Callable, List
from enum import Enum
from bridgic.types.common_types import ZeroToOne, LLMOutputFormat, PromptTemplate
from bridgic.core import LLM
from bridgic.automa.worker import Worker
from typing_extensions import override
from abc import ABCMeta
from bridgic.automa.worker_decorator import packup_worker_decorator_rumtime_args, WorkerDecoratorType

class LlmpAutomaMeta(ABCMeta):
    def __new__(mcls, name, bases, dct):
        cls = super().__new__(mcls, name, bases, dct)

        for attr_name, attr_value in dct.items():
            worker_kwargs = getattr(attr_value, "__worker_kwargs__", None)
            if worker_kwargs is not None:
                complete_args = packup_worker_decorator_rumtime_args(WorkerDecoratorType.LlmpAutomaMethod, worker_kwargs)
                # TODO: use complete_args to configure...
        
        goal_config = getattr(cls, "__goal_config__", None)
        if goal_config is not None:
            # TODO:
            pass
        
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
