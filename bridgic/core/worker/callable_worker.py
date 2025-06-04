from typing import Callable, abstractmethod
from bridgic.core.utils.inspect_tools import get_first_arg_type
from bridgic.core.worker import Worker
from bridgic.core.worker.data_model import Task
from typing import Any
import inspect

class CallableWorker(Worker):
    _is_async: bool
    _callable: Callable

    def __init__(self, func_or_method: Callable):
        self._is_async = inspect.iscoroutinefunction(func_or_method)
        self._callable = func_or_method

    async def process_async(self, *args, **kwargs) -> Any:
        result_or_coroutine = self._callable(*args, **kwargs)
        if self._is_async:
            result = await result_or_coroutine
        else:
            result = result_or_coroutine
        return result
