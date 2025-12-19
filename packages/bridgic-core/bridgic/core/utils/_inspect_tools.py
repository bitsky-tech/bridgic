import inspect
import importlib
import enum

from typing import Callable, List, Dict, Any, Tuple, Optional, Annotated, get_origin
from types import MethodType
from typing_extensions import get_overloads, overload
from bridgic.core.utils._collection import deep_hash
from docstring_parser import parse as parse_docstring  # type: ignore

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
    # Handle bound methods by using __func__ to get the unbound method
    if isinstance(func, MethodType):
        func = func.__func__
    
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
    # Handle bound methods: check if instance has a custom signature first
    # This allows per-instance signatures set by set_method_signature to take precedence
    # over the class method signature, preventing signature pollution between instances
    if isinstance(func, MethodType):
        instance = func.__self__
        method_name = func.__func__.__name__
        signature_attr = f"__{method_name}_signature__"
        # Check if the instance has a custom signature stored by set_method_signature
        if hasattr(instance, signature_attr):
            sig = getattr(instance, signature_attr)
        else:
            # If instance doesn't have custom signature, check if class method signature was modified
            # If modified, use the original signature to avoid pollution
            func_obj = func.__func__
            original_sig_attr = f"__{method_name}_original_signature__"
            if hasattr(func_obj, original_sig_attr):
                # Use original signature to avoid pollution from other instances
                sig = getattr(func_obj, original_sig_attr)
            else:
                # Fall back to the class method signature (backward compatible)
                sig = inspect.signature(func_obj)
    else:
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

def get_tool_description_from(
    spec_func: Callable,
    tool_name: Optional[str] = None
) -> str:
    """
    Get the tool description from the spec function.

    Parameters
    ----------
    spec_func : Callable
        The function to get the tool description from.
    tool_name : Optional[str]
        The name of the tool. If not provided, the function name will be used.

    Returns
    -------
    str
        The tool description.
    """
    docstring = parse_docstring(spec_func.__doc__)
    tool_description = docstring.description
    if tool_description:
        tool_description = tool_description.strip()

    if not tool_name:
        tool_name = spec_func.__name__

    if not tool_description:
        # No description provided, use the function signature as the description.
        fn_sig = inspect.signature(spec_func)
        filtered_params = []
        ignore_params: List[str] = ["self", "cls"]

        for param_name, param_value in fn_sig.parameters.items():
            if param_name in ignore_params:
                continue

            # Resolve the original type of the parameter.
            param_type = param_value.annotation
            if get_origin(param_type) is Annotated:
                param_type = param_type.__origin__

            # Remove the default value of the parameter.
            default_value = inspect.Parameter.empty

            filtered_params.append(param_value.replace(
                annotation=param_type,
                default=default_value,
            ))

        fn_sig = fn_sig.replace(parameters=filtered_params)
        tool_description = f"{tool_name}{fn_sig}\n"

    return tool_description
