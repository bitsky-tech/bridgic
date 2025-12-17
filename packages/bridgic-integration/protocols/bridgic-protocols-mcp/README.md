# Bridgic MCP Integration

This package provides [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) integration for the Bridgic framework.

## Overview

The `bridgic-protocols-mcp` package enables Bridgic to connect to and interact with MCP servers, allowing you to:

- **Connect to MCP Servers**: Establish connections via different transports and connections with lifecycle management
- **Use MCP Tools**: Integrate MCP tools seamlessly into Bridgic agentic workflows
- **Access MCP Prompts**: Retrieve and use prompts from MCP servers

## Installation

```bash
pip install bridgic-protocols-mcp
```

## Quick Start

### Connecting to an MCP Server

```python
from bridgic.protocols.mcp import McpServerConnectionStdio

# Create a connection to a filesystem MCP server
connection = McpServerConnectionStdio(
    name="filesystem-server",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/path/to/directory"],
)

# Establish the connection
connection.connect()

# List available tools
tools = connection.list_tools()
for tool in tools:
    print(f"Tool: {tool.tool_name} - {tool.tool_description}")
```

### Using MCP Tools in an Automa

```python
from bridgic.core.automa import GraphAutoma
from bridgic.protocols.mcp import McpServerConnectionStdio

# Setup connection
connection = McpServerConnectionStdio(
    name="filesystem-server",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
)
connection.connect()

# Create automa and add MCP tools
automa = GraphAutoma()
for tool_spec in connection.list_tools():
    automa.add_worker(tool_spec.create_worker())

# Now the automa can use MCP tools
```

## Features

### Connection Management

The package provides two connection implementations:

- **`McpServerConnectionStdio`**: For stdio-based MCP servers
- **`McpServerConnectionStreamableHttp`**: For HTTP-based MCP servers

Both support:
- Automatic lifecycle management
- Thread-safe operations
- Timeout configuration

### Tool Integration

MCP tools are automatically wrapped in Bridgic's tool system:

- **`McpToolSpec`**: Represents an MCP tool as a Bridgic tool specification
- **`McpToolWorker`**: Executes MCP tool calls

### Prompt Integration

Access prompts from MCP servers:

```python
# List available prompts
prompts = connection.list_prompts()

# Get a specific prompt
prompt_template = prompts[0]
messages = prompt_template.format_messages(arg1="value1", arg2="value2")
```
