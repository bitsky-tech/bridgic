from bridgic.core.automa import GoalOrientedAutoma
from typing import Optional, Tuple, Any, Dict, Callable, List
from enum import Enum
from bridgic.core.types.common import ZeroToOne, LLMOutputFormat, PromptTemplate
from bridgic.core.intelligence.base_llm import BaseLlm
from bridgic.core.automa.worker import Worker
from typing_extensions import override
from abc import ABCMeta
from bridgic.core.automa.worker_decorator import packup_worker_decorator_rumtime_args, WorkerDecoratorType

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
    planning_llm: BaseLlm
    planning_strategy: PlanningStrategy
    canonical_planning_prompt: Optional[str]

    def __init__(
        self,
        planning_llm: BaseLlm, # TODO: provide a default LLM model?
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
       

    async def arun(self, *args: Optional[Tuple[Any]], automa_context: Dict[str, Any] = None, **kwargs: Optional[Dict[str, Any]]) -> Any:
        pass

    def add_worker(
        self,
        name: str, # required parameter
        worker: Worker,
        *,
        cost: ZeroToOne = 0.0,
        re_use: bool = True,
        canonical_description: Optional[PromptTemplate] = None,
    ) -> None:
        """
        Add a worker to the Automa and specify the configuration needed for LlmpAutoma planning. This method can be called at any time during execution to dynamically add new workers to the LlmpAutoma.

        Parameters
        ----------
        name : str
            The name of the worker used within the Automa. Must be unique within the scope of the Automa.
        worker : Worker
            The worker instance to be added.
        cost : ZeroToOne
            The cost of executing this worker, represented as a value between 0 and 1.
        re_use : bool
            Whether the worker can be reused. If True, this worker can be used again in the next scheduling step after being executed.
        canonical_description : Optional[PromptTemplate]
            If canonical_description is not None, the framework will use the prompt template specified by this parameter directly as the prompt for the LLM; if canonical_description is None, the framework will automatically construct the prompt based on the decorated method's information (including function name, parameters, return value, docstring, etc.).

        Returns
        -------
        None

        Raises
        ------
        AutomaDeclarationError
            If the worker specified by worker_name already exists in the Automa, this exception will be raised.
        """
        ...
        # TODO: implement this method

    def add_func_as_worker(
        self,
        name: str, # required parameter
        func: Callable,
        *,
        cost: ZeroToOne = 0.0,
        re_use: bool = True,
        canonical_description: Optional[PromptTemplate] = None,
    ) -> None:
        """
        Add a worker to the Automa and specify the configuration needed for LlmpAutoma planning. This method can be called at any time during execution to dynamically add new workers to the LlmpAutoma.

        Parameters
        ----------
        name : str
            The name of the worker used within the Automa. Must be unique within the scope of the Automa.
        func : Callable
            The function to be added as a worker to the automa.
        cost : ZeroToOne
            The cost of executing this worker, represented as a value between 0 and 1.
        re_use : bool
            Whether the worker can be reused. If True, this worker can be used again in the next scheduling step after being executed.
        canonical_description : Optional[PromptTemplate]
            If canonical_description is not None, the framework will use the prompt template specified by this parameter directly as the prompt for the LLM; if canonical_description is None, the framework will automatically construct the prompt based on the decorated method's information (including function name, parameters, return value, docstring, etc.).

        Returns
        -------
        None

        Raises
        ------
        AutomaDeclarationError
            If the worker specified by worker_name already exists in the Automa, this exception will be raised.
        """
        ...
        # TODO: implement this method

    @override
    def remove_worker(self, name: str) -> Worker:
        ...
        # TODO: implement this method
    