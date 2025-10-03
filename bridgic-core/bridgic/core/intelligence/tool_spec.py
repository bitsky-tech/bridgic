from enum import Enum
from typing import Dict, Any, Optional, Type, Callable, Union
from pydantic import BaseModel

from bridgic.core.automa import GraphAutoma

class ToolType(Enum):
    FUNC_TOOL = "func_tool"
    MCP_TOOL = "mcp_tool"
    AUTOMA_AS_TOOL = "automa_as_tool"

class ToolSpec(BaseModel):
    type: ToolType

class FunctionToolSpec(ToolSpec):
    name: str
    description: str
    fn_schema: Optional[Type[BaseModel]]
    fn: Union[Callable, str]

class McpToolSpec(ToolSpec):
    name: str
    description: str
    inputSchema: dict[str, Any]
    mcp_transport: str

class AutomaToolSpec(ToolSpec):
    name: str
    description: str
    automa: str