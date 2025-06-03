from typing import Callable
from .bridge_info import _BridgeInfo

def processor(listen: Callable = None, is_start: bool = False, is_end: bool = False) -> Callable:
    def actual_decorator(func: callable):
        def wrapper(*args, **kwargs):
            results = func(*args, **kwargs)
            return results

        bridge_info = _BridgeInfo(func=func, func_wrapper=wrapper, listen=listen, is_start=is_start, is_end=is_end)
        if is_start:
            bridge_info.predecessor_count = 0

        wrapper._bridge_info = bridge_info
        return wrapper

    return actual_decorator

def router(listen: Callable = None):
    # TODO: 实现router的逻辑
    pass