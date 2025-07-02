import inspect
from typing import Callable, List

def get_arg_names(func: Callable) -> List[str]:
    sig = inspect.signature(func)
    arg_names = []
    for name, _ in sig.parameters.items():
        arg_names.append(name)
    return arg_names






