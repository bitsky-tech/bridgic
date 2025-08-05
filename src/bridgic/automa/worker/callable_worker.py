from typing import Callable
from bridgic.automa.worker import Worker
from typing import Any
import inspect
from typing_extensions import override

class CallableWorker(Worker):
    _is_async: bool
    _callable: Callable

    def __init__(self, func_or_method: Callable):
        super().__init__(func_or_method)
        self._is_async = inspect.iscoroutinefunction(func_or_method)
        self._callable = func_or_method

    async def process_async(self, *args, **kwargs) -> Any:
        result_or_coroutine = self._callable(*args, **kwargs)
        if self._is_async:
            result = await result_or_coroutine
        else:
            result = result_or_coroutine
        return result

    @property
    def callable(self):
        return self._callable

    @override
    def __str__(self) -> str:
        return f"CallableWorker(callable={self._callable.__name__})"
