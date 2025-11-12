import inspect
from enum import Enum

from typing import List, Callable, Optional, Dict, Any, Union, Iterable
from typing_extensions import overload, get_overloads

from bridgic.core.utils._inspect_tools import (
    mark_overload,
    get_mark_by_func,
)
from bridgic.core.types._error import WorkerSignatureError
from bridgic.core.types._common import AutomaType, ArgsMappingRule
from bridgic.core.automa.worker._worker_callback import WorkerCallbackBuilder

@mark_overload("__automa_type__", AutomaType.Graph)
def worker(
    *,
    key: Optional[str] = None,
    dependencies: List[str] = [],
    is_start: bool = False,
    is_output: bool = False,
    args_mapping_rule: ArgsMappingRule = ArgsMappingRule.AS_IS,
    callback_builders: List[WorkerCallbackBuilder] = [],
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
    callback_builders: List[WorkerCallbackBuilder]
        A list of worker callback builders to be registered.
        Callback instances will be created from builders when the worker is instantiated.
    """
    ...

@mark_overload("__automa_type__", [AutomaType.Concurrent, AutomaType.ReAct])
def worker(
    *,
    key: Optional[str] = None,
    callback_builders: List[WorkerCallbackBuilder] = [],
) -> Callable:
    """
    A decorator for designating a method as a worker node in a `ConcurrentAutoma` or `ReActAutoma` subclass.

    Parameters
    ----------
    key : Optional[str]
        The key of the worker. If not provided, the name of the decorated callable will be used.
    callback_builders: List[WorkerCallbackBuilder]
        A list of worker callback builders to be registered.
        Callback instances will be created from builders when the worker is instantiated.
    """
    ...

@mark_overload("__automa_type__", AutomaType.Sequential)
def worker(
    *,
    key: Optional[str] = None,
    args_mapping_rule: ArgsMappingRule = ArgsMappingRule.AS_IS,
    callback_builders: List[WorkerCallbackBuilder] = [],
) -> Callable:
    """
    A decorator for designating a method as a worker node in a `SequentialAutoma` subclass.

    Parameters
    ----------
    key : Optional[str]
        The key of the worker. If not provided, the name of the decorated callable will be used.
    args_mapping_rule: ArgsMappingRule
        The rule of arguments mapping.
    callback_builders: List[WorkerCallbackBuilder]
        A list of worker callback builders to be registered.
        Callback instances will be created from builders when the worker is instantiated.
    """
    ...

def worker(**kwargs) -> Callable:
    """
    Decorator for marking a method inside an Automa class as a worker.

    To cover the need to declare workers in various Automa classes, this decorator actually 
    accepts a variable kwargs parameter. Through overloading, it further supports specifying 
    the parameters that need to be passed in when registering a worker under a specific Automa, 
    such as `ConcurrentAutoma`, `SequentialAutoma` and so on.

    Parameters
    ----------
    kwargs : Dict[str, Any]
        The keyword arguments for the worker decorator.
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

def get_worker_decorator_default_paramap(automa_type: AutomaType) -> Dict[str, Any]:
    return _automa_type_to_default_params[automa_type]

def packup_worker_decorator_rumtime_args(
    automa_class: type,
    automa_type: AutomaType,
    worker_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    default_paramap = get_worker_decorator_default_paramap(automa_type)
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
