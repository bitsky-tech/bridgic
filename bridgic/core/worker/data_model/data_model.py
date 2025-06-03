from pydantic import BaseModel, ConfigDict
from typing import AsyncIterable, Generic, Any
from typing import TypeVar

T = TypeVar('T')

# 
# 在AutoMa中的各个Processor之间传递的数据结构
# 遗留问题：TODO 输入或输出为None是否需要支持？
# 
class DataRecord(BaseModel):
    model_config = ConfigDict(extra='allow')

    # TODO: 提供class_method，把基础类型快速封装成DataRecord
    @classmethod
    def create_from_dict(cls, data: dict) -> "DataRecord":
        return cls(**data)
    
    @classmethod
    def create_from_value(cls, value: Any) -> "DataRecord":
        return cls(value=value)
    


# TODO: Generic?
class DataStream(Generic[T], BaseModel, AsyncIterable[T]):
    model_config = ConfigDict(extra='allow')

ProcessorData = DataRecord | InEvent | DataStream

__all__ = ["DataRecord", "InEvent", "DataStream", "ProcessorData"]