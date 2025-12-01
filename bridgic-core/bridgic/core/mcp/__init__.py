"""
The MCP module serves as the foundation for MCP (Model Context Protocol) integration in Bridgic, 
focusing exclusively on connection management and lifecycle control. It provides:

- **Connection Definition**: Abstract base classes and concrete implementations for establishing and
  maintaining connections to MCP servers via different transport protocols (stdio, streamable HTTP, etc.)
- **Connection Lifecycle**: Centralized management of multiple MCP server connections through the
  connection manager, ensuring proper resource allocation and cleanup
- **Unified Interface**: A consistent API for interacting with MCP servers, abstracting away the 
  transport-specific details

Higher-level integrations that utilize MCP connections are implemented in other modules, such as:

- **Tool Integration**: The `bridgic.core.agentic.tool_specs` module provides `McpToolSpec` for
  integrating MCP tools into agentic systems, allowing MCP tools to be used as callable tools
  by LLMs.
- **Prompt Integration**: The `bridgic.core.prompt` module provides `McpPromptTemplate` for
  integrating MCP prompts into the prompt template system, enabling dynamic prompt retrieval
  from MCP servers.

By separating connection management from business logic, this module ensures a clean separation
of concerns and allows for flexible, reusable MCP integration patterns across the framework.
"""

from bridgic.core.mcp._mcp_server_connection import (
    McpServerConnection,
    McpServerConnectionStdio,
    McpServerConnectionStreamableHttp,
)
from bridgic.core.mcp._mcp_server_connection_manager import McpServerConnectionManager

from bridgic.core.types._error import McpServerConnectionError

__all__ = [
    "McpServerConnection",
    "McpServerConnectionStdio",
    "McpServerConnectionStreamableHttp",
    "McpServerConnectionManager",
    "McpServerConnectionError",
]