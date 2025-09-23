import copy
from abc import ABCMeta
from typing import Optional, Tuple, Any, Dict, Callable, List, Union
from pydantic import BaseModel, Field
from typing_extensions import override

from bridgic.core.automa import GoalOrientedAutoma
from bridgic.core.types.common import ZeroToOne
from bridgic.core.automa.worker import Worker
from bridgic.core.automa.worker_decorator import packup_worker_decorator_rumtime_args, WorkerDecoratorType, StaticOutputEffect, DynamicOutputEffect

class WorkerConfig(BaseModel):
    name: str
    cost: ZeroToOne = Field(ge=0.0, le=1.0)
    re_use: bool
    preconditions: List[str]
    output_effects: Union[StaticOutputEffect, DynamicOutputEffect]
    extra_effects: List[str]

class GoalConfig(BaseModel):
    preconditions: List[str]
    final: bool
    priority: int

class FunctionConfig(BaseModel):
    worker_func: Callable
    worker_config: WorkerConfig
    goal_config: Optional[GoalConfig] = None

class GoapAutomaMeta(ABCMeta):
    def __new__(mcls, name, bases, dct):
        cls = super().__new__(mcls, name, bases, dct)

        # worker_name -> FunctionConfig
        function_configs: Dict[str, FunctionConfig] = {}
        for attr_name, attr_value in dct.items():
            worker_config = None
            goal_config = None
            worker_kwargs = getattr(attr_value, "__worker_kwargs__", None)
            if worker_kwargs is not None:
                complete_args = packup_worker_decorator_rumtime_args(cls, WorkerDecoratorType.GoapAutomaMethod, worker_kwargs)
                # Use complete_args to configure worker_config.
                # Note: We are extracting configuration information here, not runtime information.
                # So try to keep the stored configuration format consistent with the original format.
                # Note: Make sure each worker has a name.
                if not complete_args.get("name"):
                    complete_args["name"] = attr_name
                worker_config = WorkerConfig(**complete_args)
            goal_configs = getattr(attr_value, "__goal_config__", None)
            if goal_configs is not None:
                # Note: Here we are extracting configuration information, not runtime information. The implementation should be similar to worker_config.
                goal_config = GoalConfig(**goal_configs)
            if worker_config is not None:
                # goal_config may be None.
                function_configs[worker_config.name] = FunctionConfig(
                    worker_func=attr_value,
                    worker_config=worker_config,
                    goal_config=goal_config,
                )
        if function_configs:
            # Note: The current __new__ method executes when GoapAutoma or any of its subclasses are defined.
            # We check if function_configs is not empty to ensure configuration extracting only occurs on GoapAutoma subclasses whose functions have @worker decorators
            cls._worker_funcs_config_map = function_configs
        return cls

class GoapWorker(Worker):
    """
    A decorated worker used for GoapAutoma planning and scheduling. Not intended for external use.
    Follows the `Decorator` design pattern:
    - https://web.archive.org/web/20031204182047/http://patterndigest.com/patterns/Decorator.html
    - https://en.wikipedia.org/wiki/Decorator_pattern
    New Behavior Added: In addition to the original Worker functionality, maintains configuration and state variables related to dynamic planning and scheduling.
    """
    def __init__(self, decorated_worker: Worker):
        # TODO: name is not needed inside Worker/Automa class.
        super().__init__(name=f"goap-worker-{decorated_worker.name}")
        self.decorated_worker = decorated_worker

    async def arun(self, *args: Optional[Tuple[Any]], **kwargs: Optional[Dict[str, Any]]) -> Any:
        return self.decorated_worker.arun(*args, **kwargs)

class GoapAutoma(GoalOrientedAutoma, metaclass=GoapAutomaMeta):
    """
    An Automa that supports GOAP (Goal Oriented Action Planning).
    In GoapAutoma, you can specify one or more workers as goals. The system will then automatically plan and determine the execution path needed to achieve these goals.
    """

    class WorkerPlanningConfig(BaseModel):
        cost: ZeroToOne = Field(ge=0.0, le=1.0)
        preconditions: List[str]
        output_effects: Union[StaticOutputEffect, DynamicOutputEffect]
        extra_effects: List[str]
        is_goal: bool

    class GoalPlanningConfig(BaseModel):
        preconditions: List[str]
        final: bool
        priority: int

    def __init__(
        self,
        name: Optional[str] = None
    ):
        # TODO: name is not needed inside Worker/Automa class.
        super().__init__(name=name)

        self._workers_config_map = {} # worker_name -> WorkerConfig
        self._goals_config_map = {} # worker_name -> GoalConfig
        # Note 1: Copy configuration information from the class (_worker_funcs_config_map) to the GoapAutoma instance (self._workers_config_map and self._goals_config_map).
        # Note 2: The copied configuration information is still not runtime data, but will be updated as workers are dynamically added/removed in GoapAutoma. In contrast, the original _worker_funcs_config_map on the class remains unchanged.
        worker_funcs_config_map: Optional[Dict[str, FunctionConfig]] = getattr(self, "_worker_funcs_config_map", None)
        if worker_funcs_config_map:
            for worker_name, func_config in worker_funcs_config_map.items():
                self._workers_config_map[worker_name] = copy.deepcopy(func_config.worker_config)
                if func_config.goal_config:
                    self._goals_config_map[worker_name] = copy.deepcopy(func_config.goal_config)

    async def arun(self, *args: Optional[Tuple[Any]], automa_context: Dict[str, Any] = None, **kwargs: Optional[Dict[str, Any]]) -> Any:
        # Dynamically determine the starting state
        pass
        # TODO: Implement GOAP logic here.

    def add_worker(
        self,
        name: str, # required parameter
        worker: Worker,
        *,
        cost: ZeroToOne = 0.0,
        re_use: bool = False,
        preconditions: List[str] = [],
        output_effects: Union[StaticOutputEffect, DynamicOutputEffect], # required parameter
        extra_effects: List[str] = [],
    ) -> None:
        """
        Add a worker to the Automa and specify the configuration needed for GoapAutoma planning. This method can be called at any time during execution to dynamically add new workers to the GoapAutoma.

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
        preconditions : List[str]
            The preconditions required for executing the added worker, expressed as a list of precondition IDs. The framework will try to automatically extract preconditions from the `worker_obj` worker, and the preconditions specified here will be merged with the extracted ones.
        output_effects : Union[StaticOutputEffect, DynamicOutputEffect]
            The effects produced by executing the added worker, expressed as a list of effect IDs or a DynamicOutputEffect value. This is a required parameter that must be specified explicitly. DynamicOutputEffect is used for the packing and unpacking mechanism in `dynamic workers` scenarios.
        extra_effects : List[str]
            The extra effects produced by executing the added worker, in addition to the output_effects.

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
        re_use: bool = False,
        preconditions: List[str] = [],
        output_effects: Union[StaticOutputEffect, DynamicOutputEffect], # required parameter
        extra_effects: List[str] = [],
    ) -> None:
        """
        Add a function as a worker to the Automa and specify the configuration needed for GoapAutoma planning. This method can be called at any time during execution to dynamically add new workers to the GoapAutoma.

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
        preconditions : List[str]
            The preconditions required for executing the added worker, expressed as a list of precondition IDs. The framework will try to automatically extract preconditions from the `worker_obj` worker, and the preconditions specified here will be merged with the extracted ones.
        output_effects : Union[StaticOutputEffect, DynamicOutputEffect]
            The effects produced by executing the added worker, expressed as a list of effect IDs or a DynamicOutputEffect value. This is a required parameter that must be specified explicitly. DynamicOutputEffect is used for the packing and unpacking mechanism in `dynamic workers` scenarios.
        extra_effects : List[str]
            The extra effects produced by executing the added worker, in addition to the output_effects.

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
    def remove_worker(self, name: str) -> None:
        pass
        # TODO: Implementation.

    def set_goal(
        self,
        worker_name: str,
        *,
        preconditions: List[str] = [],
        final: bool = False,
        priority: int = 0,
    ) -> None:
        """
        Set a worker as a goal in the GoapAutoma. This method can be called at any time during execution to dynamically set a goal.

        Parameters
        ----------
        worker_name : str
            The name of the worker to be set as a goal. This worker must already exist in the Automa.
        preconditions : List[str]
            The preconditions required for achieving the current goal, expressed as a list of precondition IDs. The framework will try to automatically extract preconditions from the worker named `worker_name`, and the preconditions specified here will be merged with the extracted ones.
        final : bool
            Indicates if this goal is a final goal. When set to True, it has two effects: 1) The Automa will stop executing immediately once this goal is achieved, and 2) The return value from the decorated method will be used as the Automa's output (via arun()).
        priority : int
            Specifies the priority of this goal. Goals with higher priority numbers will be planned and achieved first.

        Returns
        -------
        None

        Raises
        ------
        AutomaDeclarationError
            If the worker specified by worker_name does not exist in the Automa, this exception will be raised.
        """
        ...
        # TODO: implement this method

    def cancel_goal(
        self,
        worker_name: str,
    ) -> None:
        """
        Cancel the worker specified by worker_name as a planning goal. This method can be called at any time during execution to dynamically cancel a goal.

        Parameters
        ----------
        worker_name : str
            The name of the worker to be cancelled as a goal. This worker must already exist in the Automa.

        Returns
        -------
        None

        Raises
        ------
        AutomaDeclarationError
            If the worker specified by worker_name does not exist in the Automa, this exception will be raised.
        """
        ...
        # TODO: implement this method
