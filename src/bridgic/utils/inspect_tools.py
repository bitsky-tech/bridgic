import inspect
from typing import Callable, List, Dict, Any
from typing_extensions import get_overloads
import importlib

def get_arg_names(func: Callable) -> List[str]:
    sig = inspect.signature(func)
    arg_names = []
    for name, _ in sig.parameters.items():
        arg_names.append(name)
    return arg_names

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


def load_qualified_class(full_qualified_name: str):
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





