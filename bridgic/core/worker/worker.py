from bridgic.core.worker.data_model import ProcessorData, DataRecord, InEvent, DataStream
from typing import Any
from bridgic.typing.event import InEvent, OutEvent

# Worker is the core element of the orchestration layer in the bridgic framework.
# 分为同步和异步两类接口。
class Worker:

    # 动态参数形式：
    # 输入参数的允许类型：
    # 1，单个参数，DataRecord类型
    # 2，单个参数，UserEvent类型
    # 3，单个参数，DataStream类型
    # 4，单个参数，任意类型
    # 5，多个参数，args
    # 6，多个参数，kwargs

    # 返回值的允许类型：
    # 1，单个返回值，DataRecord类型
    # 2，单个返回值，DataStream类型
    # 3，单个返回值，任意类型（除了tuple和dict）
    # 4，多个返回值，tuple
    # 5，多个返回值，dict
    async def process_async(self, *args, **kwargs) -> Any:
        pass

    async def process_record(self, *args, **kwargs) -> Any:
        pass

    async def process_event(self, event: InEvent) -> Any:
        pass

    async def process_stream(self, data: DataStream) -> ProcessorData:
        return data
