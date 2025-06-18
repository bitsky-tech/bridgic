from pydantic import BaseModel, ConfigDict
from typing import Any

class _OutputValueDescriptor(BaseModel):
    initialized: bool = False
    value_type: type = None
    value: Any = None

class ExecutionFlowOutputBuffer(BaseModel):
    model_config = ConfigDict(extra='allow', frozen=True)
