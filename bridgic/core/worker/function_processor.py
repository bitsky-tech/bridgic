from typing import Callable
import inspect
from bridgic.core.utils.inspect_tools import get_first_arg_type
from bridgic.core.worker import Worker
from bridgic.core.worker.data_model import Task
from bridgic.core.worker.callable_processor import CallableProcessor


# 把一个function包装成一个Processor。
# 函数参数以及返回值，与DataRecord之间需要一个转换协议。如下：
# 
# （一）DataRecord到函数参数的转换协议：
# 1，如果函数就接受一个参数类型DataRecord，则直接使用DataRecord作为参数传入。
# 2，如果DataRecord.value存在，那么就把DataRecord.value作为参数传入。
# 3，如果DataRecord.args或DataRecord.kwargs存在，那么就把DataRecord.args和DataRecord.kwargs进行unpacking后作为参数传入。
# 4，否则，把DataRecord.model_dump()进行unpacking后作为参数传入。
# 
# （二）函数返回值到DataRecord的转换协议：
# 1，如果函数返回值是DataRecord类型，则直接返回。
# 2，如果函数返回值是tuple，则把tuple封装到DataRecord.args中返回。
# 3，如果函数返回值是dict，则把dict中的每个元素都包装进DataRecord后返回。
# 4，否则，把函数返回值包装成DataRecord.value后返回。

class FunctionProcessor(CallableProcessor):
    def __init__(self, function: Callable):
        self._function = function
        self._is_async = inspect.iscoroutinefunction(function)

    def get_callable(self) -> Callable:
        return self._function