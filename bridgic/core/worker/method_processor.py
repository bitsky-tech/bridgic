from typing import Callable
import inspect
from bridgic.core.utils.inspect_tools import get_first_arg_type
from bridgic.core.worker import Worker
from bridgic.core.worker.data_model import DataRecord
from bridgic.core.worker.callable_processor import CallableProcessor

class MethodProcessor(CallableProcessor):
    def __init__(self, bound_method: Callable):
        if not hasattr(bound_method, "__self__"):
            raise AttributeError("MethodProcessor.__init__ requires a bound method")
        self._bound_method = bound_method
        self._is_async = inspect.iscoroutinefunction(bound_method)

    def get_callable(self) -> Callable:
        return self._bound_method