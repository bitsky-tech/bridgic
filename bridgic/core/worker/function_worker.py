from typing import Callable
import inspect
from bridgic.core.utils.inspect_tools import get_first_arg_type
from bridgic.core.worker import Worker
from bridgic.core.worker.data_model import Task
from bridgic.core.worker.callable_worker import CallableWorker

# 
# 把一个function包装成一个Processor。
# 
class FunctionWorker(CallableWorker):
    def __init__(self, function: Callable):
        self._function = function
        self._is_async = inspect.iscoroutinefunction(function)

    def get_callable(self) -> Callable:
        return self._function