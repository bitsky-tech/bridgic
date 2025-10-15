import inspect
import importlib
import enum

from typing import Callable, List, Dict, Any, Tuple
from typing_extensions import get_overloads, overload
from bridgic.core.utils._collection import deep_hash

_marked_overloads: Dict[str, Dict[str, Any]] = {}

def mark_overload(key: str, value: Any) -> Callable:
    """
    A decorator to mark an overload function with a key and value. It is useful 
    when you need to mark a function as overloaded and add more information to it.
    """
    def wrapper(func: Callable):
        func_key = f"{func.__module__}.{func.__qualname__}"
        params_key = hash_kw_default_params(func)
        func_params_key = f"{func_key}::{params_key}"
        if func_params_key not in _marked_overloads:
            _marked_overloads[func_params_key] = {}
        _marked_overloads[func_params_key][key] = value
        return overload(func)
    return wrapper

def get_mark_by_func(func: Callable, key: str) -> Any:
    """
    Given a callable object and a specified key, get the pre-set mark.
    """
    func_key = f"{func.__module__}.{func.__qualname__}"
    params_key = hash_kw_default_params(func)
    func_params_key = f"{func_key}::{params_key}"
    return _marked_overloads[func_params_key][key]

def get_param_names_by_kind(
        func: Callable, 
        param_kind: enum.IntEnum,
        exclude_default: bool = False,
    ) -> List[str]:
    """
    Get the names of parameters of a function by the kind of the parameter.

    Parameters
    ----------
    func : Callable
        The function to get the parameter names from.
    param_kind : enum.IntEnum
        The kind of the parameter. One of five possible values:
        - inspect.Parameter.POSITIONAL_ONLY
        - inspect.Parameter.POSITIONAL_OR_KEYWORD
        - inspect.Parameter.VAR_POSITIONAL
        - inspect.Parameter.KEYWORD_ONLY
        - inspect.Parameter.VAR_KEYWORD
    exclude_default : bool
        Whether to exclude the default parameters.

    Returns
    -------
    List[str]
        A list of parameter names.
    """
    sig = inspect.signature(func)
    param_names = []
    for name, param in sig.parameters.items():
        if param.kind == param_kind:
            if exclude_default and param.default is not inspect.Parameter.empty:
                continue
            param_names.append(name)
    return param_names

def get_param_names_all_kinds(
        func: Callable, 
        exclude_default: bool = False,
    ) -> Dict[enum.IntEnum, List[Tuple[str, Any]]]:
    """
    Get the names of parameters of a function.

    Parameters
    ----------
    func : Callable
        The function to get the parameter names from.
    exclude_default : bool
        Whether to exclude the default parameters.

    Returns
    -------
    Dict[enum.IntEnum, List[Tuple[str, Any]]]
        A dictionary of parameter names by the kind of the parameter.
        The key is the kind of the parameter, which is one of five possible values:
        - inspect.Parameter.POSITIONAL_ONLY
        - inspect.Parameter.POSITIONAL_OR_KEYWORD
        - inspect.Parameter.VAR_POSITIONAL
        - inspect.Parameter.KEYWORD_ONLY
        - inspect.Parameter.VAR_KEYWORD
    """
    sig = inspect.signature(func)
    param_names_dict = {}
    for name, param in sig.parameters.items():
        if exclude_default and param.default is not inspect.Parameter.empty:
            continue
        if param.kind not in param_names_dict:
            param_names_dict[param.kind] = []
        
        if param.default is inspect.Parameter.empty:
            param_names_dict[param.kind].append((name, inspect._empty))
        else:
            param_names_dict[param.kind].append((name, param.default))
    return param_names_dict

def hash_kw_default_params(func: Callable) -> int:
    hashable = tuple(sorted((k, v) for k, v in func.__kwdefaults__.items()))
    return deep_hash(hashable)

def list_default_params_of_each_overload(func: Callable) -> List[Dict[str, Any]]:
    """
    Returns a list of dictionaries, each containing the default parameter values 
    for one overload of the given function.
    """
    overloaded_funcs = get_overloads(func)
    params_defaults_list = []
    for ov_func in overloaded_funcs:
        params_defaults_list.append(ov_func.__kwdefaults__)
    return params_defaults_list

def load_qualified_class_or_func(full_qualified_name: str):
    parts = full_qualified_name.split('.')
    if len(parts) < 2:
        raise ValueError(f"Invalid qualified name: '{full_qualified_name}'. Two parts needed at least.")
    
    # Try importing the module step by step until it succeeds
    for i in range(len(parts) - 1, 0, -1):
        module_path = '.'.join(parts[:i])
        try:
            module = importlib.import_module(module_path)
            break
        except ImportError:
            continue
    else:
        raise ModuleNotFoundError(f"Import module failed from path: '{full_qualified_name}'")
    
    # The remaining parts are the qualified name of the class
    cls_path_parts = parts[i:]
    
    # Use getattr step by step to access nested classes
    obj = module
    try:
        for attr in cls_path_parts:
            obj = getattr(obj, attr)
    except AttributeError as e:
        raise ImportError(f"Class not found in path: '{full_qualified_name}' due to error: {e}")

    return obj
