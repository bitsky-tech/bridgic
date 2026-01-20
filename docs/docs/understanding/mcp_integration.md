# MCP Integration

## Deep Dive into Model Context Protocol Integration

This document explores the architectural design, implementation details, and advanced usage patterns of Bridgic's MCP (Model Context Protocol) integration.

## Architecture Overview

### Protocol Abstraction Design

Bridgic's MCP integration follows a layered architecture that abstracts protocol-specific details while maintaining full compliance with the MCP specification:

- **Connection Layer**: Handles transport protocols (stdio, HTTP) and connection lifecycle
- **Protocol Layer**: Implements MCP protocol messages and state management
- **Integration Layer**: Bridges MCP concepts to Bridgic abstractions (tools, prompts, workers)
- **Application Layer**: Provides high-level APIs for use in automas

### Connection Management Architecture

#### Connection Types

Bridgic supports multiple MCP transport protocols:

- **Stdio Transport**: Process-based communication via standard input/output
- **Streamable HTTP Transport**: HTTP-based communication with streaming support
- **Extensible Design**: Architecture allows adding new transport types

#### Connection Lifecycle

MCP connections in Bridgic follow a well-defined lifecycle:

1. **Initialization**: Connection objects are created with configuration
2. **Connection Establishment**: Async connection setup with protocol handshake
3. **Active State**: Tools and prompts are discoverable and callable
4. **Cleanup**: Graceful disconnection and resource cleanup

### Tool Integration Architecture

#### Tool Specification Wrapping

MCP tools are wrapped as `McpToolSpec` instances that implement Bridgic's `ToolSpec` interface:

- **Protocol Compliance**: Full support for MCP tool schemas and parameter validation
- **Async Execution**: Tools are executed asynchronously through the MCP connection
- **Error Handling**: Protocol errors are translated to Bridgic exceptions

#### Worker Creation

MCP tools can be converted to workers for use in automas:

- **McpToolWorker**: Worker implementation that executes MCP tool calls
- **Dependency Integration**: MCP tool workers participate in dependency graphs
- **Concurrent Execution**: Multiple MCP tools can execute concurrently

### Tool Set Builder Pattern

For resource-like tools (e.g., browser, terminal), Bridgic provides `McpToolSetBuilder`:

- **Exclusive Access**: Ensures only one automa instance uses the tool set at a time
- **Dynamic Tool Creation**: Tools are created on-demand when the automa initializes
- **ReCentAutoma Integration**: Seamlessly integrates with ReCentAutoma's dynamic tool selection

## Advanced Patterns

### Multi-Server Orchestration

Bridgic enables orchestrating multiple MCP servers in a single application:

- **Connection Manager**: Centralized management of multiple connections
- **Shared Event Loop**: Efficient resource utilization across connections
- **Tool Namespace Management**: Avoiding tool name conflicts across servers

### Prompt Template Integration

MCP prompts are integrated as first-class prompt templates:

- **Dynamic Prompt Retrieval**: Prompts are fetched from servers at runtime
- **Parameter Binding**: MCP prompt variables are bound through standard template interface
- **Message Formatting**: Prompts are converted to Bridgic message format

### Error Handling and Resilience

MCP integration includes robust error handling:

- **Protocol Error Translation**: MCP errors are mapped to Bridgic exceptions
- **Connection Retry Logic**: Automatic retry for transient connection failures
- **Graceful Degradation**: Applications continue operating when some tools are unavailable

## Implementation Details

### Async Communication Patterns

MCP integration uses async/await throughout:

- **Non-blocking I/O**: All protocol communication is asynchronous
- **Event Loop Integration**: Proper integration with Python's asyncio
- **Concurrent Operations**: Multiple tool calls can proceed in parallel

### Type Safety and Validation

Strong typing ensures correctness:

- **Protocol Schema Validation**: MCP tool schemas are validated at connection time
- **Parameter Type Checking**: Tool parameters are type-checked before execution
- **Return Type Inference**: Tool return types are preserved through the integration layer

### Resource Management

Efficient resource management:

- **Connection Pooling**: Reuse connections across multiple tool calls
- **Memory Management**: Proper cleanup of protocol resources
- **Thread Safety**: Safe concurrent access to connection objects

## Best Practices

### Connection Configuration

- Use connection managers for multiple servers
- Configure appropriate timeouts for your use case
- Handle connection failures gracefully

### Tool Selection Strategy

- Use tool set builders for exclusive resources
- Consider tool availability when designing workflows
- Implement fallback strategies for critical tools

### Performance Optimization

- Reuse connections across automa instances when possible
- Batch tool calls when appropriate
- Monitor connection health and tool response times
