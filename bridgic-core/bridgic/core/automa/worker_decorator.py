import inspect
from enum import Enum
from typing_extensions import overload
from typing import List, Callable, Optional, Dict, Any, Union

from bridgic.core.utils.inspect_tools import get_default_paramaps_of_overloaded_funcs
from bridgic.core.types.error import WorkerSignatureError
from bridgic.core.types.common import AutomaType

class ArgsMappingRule(Enum):
    """
    Definitions of arguments mapping rules:
    - AS_IS: The arguments for a worker are passed as is from the return values of its predecessors, preserving the order of the dependencies list as specified when the current worker is added. All types of return values, including list/tuple, dict, and single value, are all passed in as is (no unpacking or merging is performed).
    - UNPACK: The return value of the predecessor worker is unpacked and passed as arguments to the current worker. Only valid when the current worker has exactly one dependency and the return value of the predecessor worker is a list/tuple or dict.
    - MERGE: The return values of the predecessor workers are merged and passed as arguments to the current worker. Only valid when the current worker has multiple (at least two) dependencies.
    - SUPPRESSED: The return values of the predecessor worker(s) are NOT passed to the current worker. The current worker has to access the outbuf mechanism to get the output data of its predecessor workers.

    Please refer to test/bridgic/automa/test_automa_args_mapping.py for more details.
    """
    AS_IS = "as_is"
    UNPACK = "unpack"
    MERGE = "merge"
    SUPPRESSED = "suppressed"

@overload
def worker(
    *,
    key: Optional[str] = None,
    dependencies: List[str] = [],
    is_start: bool = False,
    is_output: bool = False,
    args_mapping_rule: ArgsMappingRule = ArgsMappingRule.AS_IS,
) -> Callable:
    """
    A decorator that marks a method as a worker within a subclass of GraphAutoma. The worker's behavior can be customized through the decorator's parameters.

    Parameters
    ----------
    key : Optional[str]
        The key of the worker. If not provided, the key of the decorated callable will be used.
    dependencies : List[str]
        A list of worker names that the decorated callable depends on.
    is_start : bool
        Whether the decorated callable is a start worker. True means it is, while False means it is not.
    is_output : bool
        Whether the decorated callable is an output worker. True means it is, while False means it is not.
    args_mapping_rule : ArgsMappingRule
        The rule of arguments mapping.
    """
    ...

@overload
def worker(
    *,
    key: Optional[str] = None,
) -> Callable:
    """
    A decorator that marks a method as a worker within a subclass of ConcurrentAutoma or ReActAutoma.

    Parameters
    ----------
    key : Optional[str]
        The key of the worker. If not provided, the key of the decorated callable will be used.
    """
    ...

@overload
def worker(
    *,
    key: Optional[str] = None,
    args_mapping_rule: ArgsMappingRule = ArgsMappingRule.AS_IS,
) -> Callable:
    """
    A decorator that marks a method as a worker within a subclass of SequentialAutoma.

    Parameters
    ----------
    key : Optional[str]
        The key of the worker. If not provided, the key of the decorated callable will be used.
    args_mapping_rule: ArgsMappingRule
        The rule of arguments mapping.
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

def get_worker_decorator_default_paramap(worker_decorator_type: AutomaType) -> Dict[str, Any]:
    return get_worker_decorator_default_paramap.__saved_paramaps[worker_decorator_type]

def _extract_default_paramaps() -> Dict[AutomaType, Dict[str, Any]]:
    """
    This ensures that retrieving default argument mappings is independent of the order in which worker decorators are defined.
    """
    params_defaults_list = get_default_paramaps_of_overloaded_funcs(worker)
    paramaps = {}
    for params_default in params_defaults_list:
        if "dependencies" in params_default:
            paramaps[AutomaType.Graph] = params_default
        elif "args_mapping_rule" in params_default:
            paramaps[AutomaType.Sequential] = params_default
        else:
            paramaps[AutomaType.Concurrent] = params_default
    return paramaps

get_worker_decorator_default_paramap.__saved_paramaps = _extract_default_paramaps()

def packup_worker_decorator_rumtime_args(
        automa_class_type: type,
        worker_decorator_type: AutomaType,
        worker_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
    default_paramap = get_worker_decorator_default_paramap(worker_decorator_type)
    # Validation One: filter extra args
    extra_args = set(worker_kwargs.keys()) - set(default_paramap.keys())
    if extra_args:
        raise WorkerSignatureError(
            f"Unexpected arguments: {extra_args} for worker decorator when it is decorating "
            f"{automa_class_type.__name__} method"
        )
    # Validation Two: validate required parameters
    missing_params = set(default_paramap.keys()) - set(worker_kwargs.keys())
    missing_required_params = [param_name for param_name in missing_params if default_paramap[param_name] is inspect._empty]
    if missing_required_params:
        raise WorkerSignatureError(
            f"Missing required parameters: {missing_required_params} for worker decorator "
            f"when it is decorating {automa_class_type.__name__} method"
        )
    # Packup and return
    return {**default_paramap, **worker_kwargs}
