import inspect
from typing import Callable, List, Dict, Any
from typing_extensions import get_overloads

def get_arg_names(func: Callable) -> List[str]:
    sig = inspect.signature(func)
    arg_names = []
    for name, _ in sig.parameters.items():
        arg_names.append(name)
    return arg_names

def get_default_argmaps_of_overloaded_funcs(func: Callable) -> List[Dict[str, Any]]:
    """
    Returns a list of dictionaries containing default argument values for each overloaded function.
    """
    overloaded_funcs = get_overloads(func)
    args_defaults_list = []
    for func in overloaded_funcs:
        sig = inspect.signature(func)
        args_default = {}
        for name, param in sig.parameters.items():
            args_default[name] = param.default
        args_defaults_list.append(args_default)

    return args_defaults_list





