import inspect
from typing import Callable, List, Dict, Any
from typing_extensions import get_overloads
import importlib
import enum

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
    ) -> Dict[enum.IntEnum, List[str]]:
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
    Dict[enum.IntEnum, List[str]]
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
        param_names_dict[param.kind].append(name)
    return param_names_dict


def get_default_paramaps_of_overloaded_funcs(func: Callable) -> List[Dict[str, Any]]:
    """
    Returns a list of dictionaries containing default parameter values for each overloaded function.
    """
    overloaded_funcs = get_overloads(func)
    params_defaults_list = []
    for func in overloaded_funcs:
        sig = inspect.signature(func)
        params_default = {}
        for name, param in sig.parameters.items():
            params_default[name] = param.default
        params_defaults_list.append(params_default)

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





