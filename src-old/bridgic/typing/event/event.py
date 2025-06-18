from pydantic import BaseModel, ConfigDict
from typing import Any
from datetime import datetime
from typing import Awaitable

class Event(BaseModel):
    event_id: str
    timestamp: datetime
    model_config = ConfigDict(extra='allow')


# 从外部传入的事件，
# 可以驱动节点下一步执行
class InEvent(Event, Awaitable):
    model_config = ConfigDict(extra='allow')

# 从节点内部发出的事件，
# 一般用于触发系统对外传输信息（比如progress，比如展示表单）
class OutEvent(Event):
    model_config = ConfigDict(extra='allow')

class ProgressEvent(OutEvent):
    message: str
    progress: float

class WorkerFinishEvent(Event):
    worker_id: str
    worker_result: Any
    model_config = ConfigDict(extra='allow')

class WorkerStartEvent(Event):
    worker_id: str
    model_config = ConfigDict(extra='allow')
