"""
The MCP module provides integration with the Model Context Protocol (MCP) for Bridgic.

This module enables Bridgic to connect to and interact with MCP servers, allowing:

- **Connection Management**: Establish and manage connections to MCP servers via different
  transport protocols (stdio, streamable HTTP, etc.)
- **Tool Integration**: Use MCP tools as callable tools within Bridgic agentic systems
- **Prompt Integration**: Access and use prompts from MCP servers in Bridgic workflows
- **Worker Integration**: Execute MCP tools through Bridgic's worker system

Core Components
---------------
- `McpServerConnection`: Abstract base class for MCP server connections
- `McpServerConnectionStdio`: Connection implementation using stdio transport
- `McpServerConnectionStreamableHttp`: Connection implementation using streamable HTTP
- `McpServerConnectionManager`: Manager for multiple MCP connections with shared event loop
- `McpToolSpec`: Tool specification for MCP tools
- `McpToolWorker`: Worker implementation for executing MCP tools
- `McpPromptTemplate`: Prompt template implementation for MCP prompts

Example
-------
>>> from bridgic.protocols.mcp import McpServerConnectionStdio
>>> from bridgic.core.automa import GraphAutoma
>>>
>>> # Create and connect to an MCP server
>>> connection = McpServerConnectionStdio(
...     name="my-mcp-server",
...     command="npx",
...     args=["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"],
... )
>>> connection.connect()
>>>
>>> # List available tools
>>> tools = connection.list_tools()
>>>
>>> # Use tools in an automa
>>> automa = GraphAutoma()
>>> for tool_spec in tools:
...     automa.add_worker(tool_spec.create_worker())
"""

from bridgic.protocols.mcp._mcp_server_connection import (
    McpServerConnection,
    McpServerConnectionStdio,
    McpServerConnectionStreamableHttp,
)
from bridgic.protocols.mcp._mcp_server_connection_manager import McpServerConnectionManager
from bridgic.protocols.mcp._mcp_tool_spec import McpToolSpec
from bridgic.protocols.mcp._mcp_tool_worker import McpToolWorker
from bridgic.protocols.mcp._mcp_template import McpPromptTemplate

from bridgic.protocols.mcp._error import McpServerConnectionError

__all__ = [
    "McpServerConnection",
    "McpServerConnectionStdio",
    "McpServerConnectionStreamableHttp",
    "McpServerConnectionManager",
    "McpToolSpec",
    "McpToolWorker",
    "McpPromptTemplate",
    "McpServerConnectionError",
]

