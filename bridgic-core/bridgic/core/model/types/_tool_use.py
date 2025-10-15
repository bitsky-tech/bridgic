from typing import Dict, Any, Optional, TypedDict
from pydantic import BaseModel, Field

class Tool(BaseModel):
    name: str = Field(..., description="Name of the tool.")
    description: str = Field(..., description="Description of the tool.")
    parameters: Dict[str, Any] = Field(..., description="JSON schema object that describes the parameters of the tool.")

class ToolCall(BaseModel):
    id: Optional[str] = Field(..., description="ID of the tool call.")
    name: str = Field(..., description="Name of the tool.")
    arguments: Dict[str, Any] = Field(..., default_factory=dict, description="Real arguments that are used to call the tool.")

class ToolCallDict(TypedDict):
    id: str
    name: str
    arguments: Dict[str, Any]
