import inspect
from enum import Enum

from typing import List, Callable, Optional, Dict, Any, Union, Iterable
from typing_extensions import overload, get_overloads

from bridgic.core.utils.inspect_tools import (
    mark_overload,
    get_mark_by_func,
)
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

@mark_overload("__automa_type__", AutomaType.Fragment)
def worker(
    *,
    key: Optional[str] = None,
    dependencies: List[str] = [],
    is_start: bool = False,
    args_mapping_rule: ArgsMappingRule = ArgsMappingRule.AS_IS,
) -> Callable:
    """
    A decorator for designating a method as a worker node in a `GraphFragment` subclass.

    Parameters
    ----------
    key : Optional[str]
        The key of the worker. If not provided, the name of the decorated callable will be used.
    dependencies : List[str]
        A list of worker names that the decorated callable depends on.
    is_start : bool
        Whether the decorated callable is a start worker. True means it is, while False means it is not.
    args_mapping_rule : ArgsMappingRule
        The rule of arguments mapping.
    """
    ...

@mark_overload("__automa_type__", AutomaType.Graph)
def worker(
    *,
    key: Optional[str] = None,
    dependencies: List[str] = [],
    is_start: bool = False,
    is_output: bool = False,
    args_mapping_rule: ArgsMappingRule = ArgsMappingRule.AS_IS,
) -> Callable:
    """
    A decorator for designating a method as a worker node in a `GraphAutoma` subclass.

    Parameters
    ----------
    key : Optional[str]
        The key of the worker. If not provided, the name of the decorated callable will be used.
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

@mark_overload("__automa_type__", [AutomaType.Concurrent, AutomaType.ReAct])
def worker(
    *,
    key: Optional[str] = None,
) -> Callable:
    """
    A decorator for designating a method as a worker node in a `ConcurrentAutoma` or `ReActAutoma` subclass.

    Parameters
    ----------
    key : Optional[str]
        The key of the worker. If not provided, the name of the decorated callable will be used.
    """
    ...

@mark_overload("__automa_type__", AutomaType.Sequential)
def worker(
    *,
    key: Optional[str] = None,
    args_mapping_rule: ArgsMappingRule = ArgsMappingRule.AS_IS,
) -> Callable:
    """
    A decorator for designating a method as a worker node in a `SequentialAutoma` subclass.

    Parameters
    ----------
    key : Optional[str]
        The key of the worker. If not provided, the name of the decorated callable will be used.
    args_mapping_rule: ArgsMappingRule
        The rule of arguments mapping.
    """
    ...

def worker(**kwargs) -> Callable:
    """
    The actual implementation of the different overloaded worker decorators defined above.
    """
    def wrapper(func: Callable):
        setattr(func, "__worker_kwargs__", kwargs)
        return func
    return wrapper

def _get_default_params_of_each_automa_type() -> Dict[AutomaType, Dict[str, Any]]:
    overload_funcs = get_overloads(worker)
    result: Dict[AutomaType, Dict[str, Any]] = {}
    for ov_func in overload_funcs:
        mark = get_mark_by_func(ov_func, "__automa_type__")
        if mark is not None:
            if isinstance(mark, Iterable):
                for t in mark:
                    if isinstance(t, AutomaType):
                        result[t] = ov_func.__kwdefaults__
                    else:
                        raise ValueError(f"Invalid automa type: {t}")
            else:
                if isinstance(mark, AutomaType):
                    result[mark] = ov_func.__kwdefaults__
                else:
                    raise ValueError(f"Invalid automa type: {mark}")
        else:
            raise ValueError(f"No mark found for {ov_func} at line {ov_func.__code__.co_firstlineno}")
    return result

_automa_type_to_default_params = _get_default_params_of_each_automa_type()

def get_worker_decor_default_params(automa_type: AutomaType) -> Dict[str, Any]:
    return _automa_type_to_default_params[automa_type]

def packup_worker_decor_runtime_args(
    automa_class: type,
    automa_type: AutomaType,
    worker_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    default_paramap = get_worker_decor_default_params(automa_type)
    # Validation One: filter extra args
    extra_args = set(worker_kwargs.keys()) - set(default_paramap.keys())
    if extra_args:
        raise WorkerSignatureError(
            f"Unexpected arguments: {extra_args} for worker decorator when it is decorating "
            f"method within {automa_class.__name__}"
        )
    # Validation Two: validate required parameters
    missing_params = set(default_paramap.keys()) - set(worker_kwargs.keys())
    missing_required_params = [param_name for param_name in missing_params if default_paramap[param_name] is inspect._empty]
    if missing_required_params:
        raise WorkerSignatureError(
            f"Missing required parameters: {missing_required_params} for worker decorator "
            f"when it is decorating method within {automa_class.__name__}"
        )
    # Packup and return
    return {**default_paramap, **worker_kwargs}
