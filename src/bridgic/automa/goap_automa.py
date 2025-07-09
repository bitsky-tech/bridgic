from bridgic.automa import GoalOrientedAutoma
from typing import Optional, Tuple, Any, Dict, Callable, List
from bridgic.types.common_types import ZeroToOne
from bridgic.automa.worker import Worker
from typing_extensions import override

class GoapAutoma(GoalOrientedAutoma):
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



def conditional_worker(
    *,
    name: Optional[str] = None,
    cost: ZeroToOne = 0.0,
    re_use: bool = False,
    pre_conditions: List[str] = [],
    output_effects: List[str],
    extra_effects: List[str] = [],
) -> Callable:
    def wrapper(func: Callable):
        # TODO:
        return func
    return wrapper


def precise_goal(
    *,
    pre_conditions: List[str] = [],
    final_goal: bool = False,
) -> Callable:
    def wrapper(func: Callable):
        # TODO:
        return func
    return wrapper

