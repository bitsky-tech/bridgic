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

MCP tools can be integrated into Bridgic automas as workers in `GraphAutoma`. Furthermore, in `ReActAutoma`, these tools can be used in a more advanced LLM-driven agents to select and invoke them dynamically.

#### Using MCP Tools as Workers in GraphAutoma

MCP tools can be converted to workers and added to a `GraphAutoma`, giving them the same execution status as regular scheduling units. This allows you to orchestrate MCP tool calls alongside other workers in your workflow.

```python
from bridgic.core.automa import GraphAutoma
from bridgic.protocols.mcp import McpServerConnectionStdio

# Setup connection
connection = ...
connection.connect()

# Create automa and add MCP tools as workers. Then they can be orchestrated with other workers, 
# participate in dependency graphs, and be scheduled just like any other worker in the system.
automa = GraphAutoma()
for tool_spec in connection.list_tools():
    # Convert each MCP tool to a worker and add it to the automa
    worker = tool_spec.create_worker()
    automa.add_worker(worker)

```

#### Using MCP Tools with ReActAutoma

MCP tools can also be used directly with `ReActAutoma`, where the LLM can autonomously select and call tools based on the user's request. This enables building intelligent agents that can interact with MCP servers through natural language.

```python
from bridgic.core.agentic import ReActAutoma
from bridgic.protocols.mcp import McpServerConnectionStreamableHttp
from bridgic.llms.openai import OpenAILlm, OpenAIConfiguration
import os

# Setup MCP connection
connection = ...
connection.connect()

# Create ReActAutoma with MCP tools
react_automa = ReActAutoma(
    llm=llm,
    system_prompt="You are a helpful assistant that can help users query information about GitHub repositories.",
    tools=connection.list_tools(),
)
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
