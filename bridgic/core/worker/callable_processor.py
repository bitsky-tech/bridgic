from typing import Callable, abstractmethod
import inspect
from bridgic.core.utils.inspect_tools import get_first_arg_type
from bridgic.core.worker import Worker
from bridgic.core.worker.data_model import DataRecord

class CallableProcessor(Worker):
    async def process_record(self, data: DataRecord) -> DataRecord:
        # Convert DataRecord to function arguments
        args = ()
        kwargs = {}
        sig = inspect.signature(self.get_callable())
        if len(sig.parameters) == 1 and isinstance(get_first_arg_type(sig), DataRecord):
            args = (data,)
        elif hasattr(data, "value"):
            args = (data.value,)
        elif hasattr(data, "args"):
            args = data.args
        elif hasattr(data, "kwargs"):
            kwargs = data.kwargs
        else:
            kwargs = data.model_dump()

        # Convert function return value to DataRecord
        callable = self.get_callable()
        result_or_coroutine = callable(*args, **kwargs)
        if self._is_async:
            result = await result_or_coroutine
        else:
            result = result_or_coroutine

        # Convert function return values to DataRecord
        if isinstance(result, DataRecord):
            pass
        elif isinstance(result, tuple):
            result = DataRecord(args=result)
        elif isinstance(result, dict):
            result = DataRecord(**result)
        else:
            result = DataRecord(value=result)

        return result

    @abstractmethod
    def get_callable(self) -> Callable:
        pass