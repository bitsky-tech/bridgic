from bridgic.automa import GoalOrientedAutoma
from typing import Optional, Tuple, Any, Dict, Callable, List
from bridgic.types.common_types import ZeroToOne
from bridgic.automa.worker import Worker
from typing_extensions import override
from abc import ABCMeta
from bridgic.automa.worker_decorator import packup_worker_decorator_rumtime_args, WorkerDecoratorType

class GoapAutomaMeta(ABCMeta):
    def __new__(mcls, name, bases, dct):
        cls = super().__new__(mcls, name, bases, dct)

        for attr_name, attr_value in dct.items():
            worker_kwargs = getattr(attr_value, "__worker_kwargs__", None)
            if worker_kwargs is not None:
                complete_args = packup_worker_decorator_rumtime_args(WorkerDecoratorType.GoapAutomaMethod, worker_kwargs)
                # TODO: use complete_args to configure...
            goal_configs = getattr(attr_value, "__goal_config__", None)
            if goal_configs is not None:
                # TODO:
                pass
        return cls


class GoapAutoma(GoalOrientedAutoma, metaclass=GoapAutomaMeta):
    """
    An Automa that supports GOP (Goal-Oriented Planning).
    You are allowed to specify multiple precise goals in PreciseGoalAutoma, meaning that the execution of certain workers within the Automa can be designated as goals. The system will automatically plan an execution path based on these goals.
    """
    def __init__(
        self,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

    def process_async(self, *args: Optional[Tuple[Any]], automa_context: Dict[str, Any] = None, **kwargs: Optional[Dict[str, Any]]) -> Any:
        # Dynamically determine the starting state
        pass

    @override
    def remove_worker(self, worker_name: str) -> Worker:
        pass
