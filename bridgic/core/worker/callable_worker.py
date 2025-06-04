from typing import Callable, abstractmethod
from bridgic.core.utils.inspect_tools import get_first_arg_type
from bridgic.core.worker import Worker
from bridgic.core.worker.data_model import Task
from typing import Any
import inspect

class CallableWorker(Worker):
    _is_async: bool

    async def process_default_async(self, *args, **kwargs) -> Any:
        callable = self.get_callable()
        result_or_coroutine = callable(*args, **kwargs)
        if self._is_async:
            result = await result_or_coroutine
        else:
            result = result_or_coroutine
        return result

    @abstractmethod
    def get_callable(self) -> Callable:
        pass