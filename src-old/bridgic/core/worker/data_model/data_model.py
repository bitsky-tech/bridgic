from pydantic import BaseModel, ConfigDict
from typing import Any
from typing import TypeVar

T = TypeVar('T')

# 
# Input structure for each Worker
# 
class Task(BaseModel):
    model_config = ConfigDict(extra='allow')

    # 提供class_method，把基础类型快速封装成DataRecord
    @classmethod
    def create_from_dict(cls, data: dict) -> "Task":
        return cls(**data)
        
# 
# Output structure for each Worker
# 
class TaskResult(BaseModel):
    model_config = ConfigDict(extra='allow')

    @classmethod
    def create_from_dict(cls, data: dict) -> "TaskResult":
        return cls(**data)
