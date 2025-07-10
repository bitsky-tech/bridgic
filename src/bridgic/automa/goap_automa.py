from bridgic.automa import GoalOrientedAutoma
from typing import Optional, Tuple, Any, Dict, Callable, List
from bridgic.types.common_types import ZeroToOne
from bridgic.automa.worker import Worker
from typing_extensions import override
from abc import ABCMeta
from bridgic.automa.worker_decorator import get_default_worker_args

class GoapAutomaMeta(ABCMeta):
    def __new__(mcls, name, bases, dct):
        cls = super().__new__(mcls, name, bases, dct)

        def get_default_worker_args_for_llmp() -> Dict[str, Any]:
            default_args_list = get_default_worker_args()
            for default_args in default_args_list:
                if "output_effects" in default_args:
                    return default_args
            return None
        
        for attr_name, attr_value in dct.items():
            worker_kwargs = getattr(attr_value, "__worker_kwargs__", None)
            if worker_kwargs is not None:
                default_args = get_default_worker_args_for_llmp()
                complete_args = {**default_args, **worker_kwargs}
                print(f"%%%%%%%%%%%%%%%%% [{cls}.{attr_name}] worker_kwargs = {worker_kwargs}")
                print(f"%%%%%%%%%%%%%%%%% [{cls}.{attr_name}] default_args = {default_args}")
                print(f"%%%%%%%%%%%%%%%%% [{cls}.{attr_name}] complete_args = {complete_args}")
                # TODO:
        # TODO:
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

def precise_goal(
    *,
    pre_conditions: List[str] = [],
    final_goal: bool = False,
) -> Callable:
    def wrapper(func: Callable):
        # TODO:
        return func
    return wrapper
