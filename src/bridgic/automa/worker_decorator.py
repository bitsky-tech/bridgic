import inspect
from enum import Enum
from typing_extensions import get_overloads, overload

from typing import List, Callable, Optional, Dict, Any
from bridgic.consts.args_mapping_rule import ARGS_MAPPING_RULE_AUTO
from bridgic.types.common_types import ZeroToOne, PromptTemplate
from bridgic.utils.inspect_tools import get_default_argmaps_of_overloaded_funcs

# Constant Definitions
class WorkerDecoratorType(Enum):
    GraphAutomaMethod = 1
    GoapAutomaMethod = 2
    LlmpAutomaMethod = 3

@overload
def worker(
    *,
    name: Optional[str] = None,
    dependencies: List[str] = [],
    is_start: bool = False,
    args_mapping_rule: str = ARGS_MAPPING_RULE_AUTO,
) -> Callable:
    """
    A decorator that marks a method as a worker within an GraphAutoma class. The worker's behavior can be customized through the decorator's parameters.

    Parameters
    ----------
    name : Optional[str]
        The name of the worker. If not provided, the name of the decorated callable will be used.
    dependencies : List[str]
        A list of worker names that the decorated callable depends on.
    is_start : bool
        Whether the decorated callable is a start worker. True means it is, while False means it is not.
    args_mapping_rule : str
        The rule of arguments mapping. The options are: "auto", "as_list", "as_dict", "suppressed".
    """
    ...

@overload
def worker(
    *,
    name: Optional[str] = None,
    cost: ZeroToOne = 0.0,
    re_use: bool = False,
    preconditions: List[str] = [],
    output_effects: List[str], # required parameter
    extra_effects: List[str] = [],
) -> Callable:
    """
    A decorator that marks a method as a worker within an GaopAutoma class. 

    Adding a worker into a GaopAutoma only requires specifying the pre_conditions and the effects, and the framework will call it at the appropriate time, automatically. You don't need to care about the explicit execution order of workers.

    This worker's behavior can be customized through the decorator's parameters.

    Parameters
    ----------
    name : Optional[str]
        The name of the worker. If not provided, the name of the decorated callable will be used.
    cost : ZeroToOne
        The cost of executing this worker, represented as a value between 0 and 1.
    re_use : bool
        Whether the worker can be reused. If True, this worker can be used again in the next scheduling step after being executed.
    preconditions : List[str]
        The preconditions required for executing the current worker, expressed as a list of precondition IDs. The framework will automatically extract preconditions from the decorated method, and the preconditions specified here will be merged with the extracted ones.
    output_effects : List[str]
        The effects produced by executing the current worker, expressed as a list of effect IDs. This is a required parameter that must be specified explicitly.
    extra_effects : List[str]
        The extra effects produced by executing the current worker, in addition to the output_effects.
    """
    ...

@overload
def worker(
    *,
    name: Optional[str] = None,
    cost: ZeroToOne = 0.0,
    re_use: bool = True,
    canonical_description: Optional[PromptTemplate] = None,
) -> Callable:
    """
    A decorator that marks a method as a worker within an LlmpAutoma class. 

    LlmpAutoma uses LLM to autonomously generate execution plans. When generating execution plans, it references information from both the method and this decorator.

    This worker's behavior can be customized through the decorator's parameters.

    Parameters
    ----------
    name : Optional[str]
        The name of the worker. If not provided, the name of the decorated callable will be used.
    cost : ZeroToOne
        The cost of executing this worker, represented as a value between 0 and 1.
    re_use : bool
        Whether the worker can be reused. If True, this worker can be used again in the next scheduling step after being executed.
    canonical_description : Optional[PromptTemplate]
        If canonical_description is not None, the framework will use the prompt template specified by this parameter directly as the prompt for the LLM; if canonical_description is None, the framework will automatically construct the prompt based on the decorated method's information (including function name, parameters, return value, docstring, etc.).
    """
    ...

def worker(**kwargs) -> Callable:
    """
    The implementation of the 3 overloaded worker decorators defined above.
    """
    def wrapper(func: Callable):
        setattr(func, "__worker_kwargs__", kwargs)
        return func
    return wrapper

def get_default_worker_args() -> List[Dict[str, Any]]:
    return get_default_argmaps_of_overloaded_funcs(worker)

def get_worker_decorator_default_argmaps(worker_decorator_type: WorkerDecoratorType) -> Dict[str, Any]:
    return get_worker_decorator_default_argmaps.__saved_argmaps[worker_decorator_type]

def _extract_default_argmaps() -> Dict[WorkerDecoratorType, Dict[str, Any]]:
    """
    This ensures that retrieving default argument mappings is independent of the order in which worker decorators are defined.
    """
    args_defaults_list = get_default_argmaps_of_overloaded_funcs(worker)
    argmaps = {}
    for args_default in args_defaults_list:
        if "dependencies" in args_default:
            argmaps[WorkerDecoratorType.GraphAutomaMethod] = args_default
        elif "output_effects" in args_default:
            argmaps[WorkerDecoratorType.GoapAutomaMethod] = args_default
        elif "canonical_description" in args_default:
            argmaps[WorkerDecoratorType.LlmpAutomaMethod] = args_default
    return argmaps

get_worker_decorator_default_argmaps.__saved_argmaps = _extract_default_argmaps()
