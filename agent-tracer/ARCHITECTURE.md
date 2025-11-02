# Architecture Documentation

## Overview

Agent Tracer is a framework-agnostic tracing library designed to provide observability for AI agents and LLM applications. It supports multiple tracing backends while maintaining a simple, unified API.

## Design Principles

1. **Framework Agnostic**: No dependencies on specific agent frameworks
2. **Backend Flexible**: Support multiple tracing backends simultaneously
3. **Minimal Dependencies**: Core library has minimal deps; tracer-specific deps are optional
4. **Async First**: Built for modern async Python applications
5. **Type Safe**: Full type hints for better IDE support and type checking
6. **Privacy Focused**: Automatic sensitive data masking
7. **Developer Friendly**: Simple, intuitive API with clear error messages

## Core Components

### 1. TracingService

The main API entry point for users.

**Responsibilities:**
- Lifecycle management (start/end traces)
- Context management via ContextVars
- Background worker for async trace processing
- Tracer initialization and coordination
- Sensitive data masking

**Key Methods:**
- `start_trace()`: Initialize a new trace
- `trace_step()`: Context manager for tracing individual steps
- `set_outputs()`: Record outputs from steps
- `add_log()`: Add log entries
- `end_trace()`: Finalize and flush traces
- `get_langchain_callbacks()`: Get LangChain-compatible callbacks

### 2. BaseTracer (Abstract)

Abstract base class defining the interface all tracers must implement.

**Abstract Methods:**
- `__init__()`: Initialize with trace metadata
- `add_trace()`: Add a child trace/span
- `end_trace()`: End a child trace/span
- `end()`: End the root trace
- `get_langchain_callback()`: Get framework callback (optional)

**Properties:**
- `ready`: Boolean indicating if tracer is properly configured

### 3. Tracer Implementations

Each tracer backend has its own implementation:

- **LangFuseTracer**: LangFuse with hierarchical spans
- **LangWatchTracer**: LangWatch with component tracking
- **OpikTracer**: Opik with thread tracking
- **ConsoleTracer**: Local console output for debugging

### 4. Context Management

Uses Python's `contextvars` for managing trace state:

**TraceContext:**
- Stores trace metadata (ID, name, project, user, session)
- Maintains tracer instances
- Manages background worker and queue
- Aggregates inputs/outputs across steps

**StepTraceContext:**
- Stores step metadata
- Collects inputs/outputs for the step
- Manages logs for the step

### 5. Data Schemas

**Log:**
- Structured log entries with name, message, type
- Automatic serialization of complex objects

**TracingConfig:**
- Configuration for tracing behavior
- Sensitive data masking settings
- Enable/disable tracing globally

## Data Flow

```
User Code
    ↓
TracingService.start_trace()
    ↓
Initialize Tracers (LangFuse, LangWatch, Opik, etc.)
    ↓
Start Background Worker
    ↓
User Code: tracer.trace_step()
    ↓
Create StepTraceContext
    ↓
Queue: tracer.add_trace() for all tracers
    ↓
User Code: tracer.set_outputs()
    ↓
Queue: tracer.end_trace() for all tracers
    ↓
TracingService.end_trace()
    ↓
Stop Background Worker
    ↓
Call end() on all tracers
    ↓
Flush to backends
```

## Async Architecture

### Background Worker Pattern

Traces are processed asynchronously to minimize impact on the main application:

1. User calls tracing methods (synchronous API)
2. Operations are queued to an `asyncio.Queue`
3. Background worker processes queue items
4. Actual tracer calls happen in background
5. Main application continues without blocking

### Context Variables

Thread-safe context management using `contextvars`:

- `trace_context_var`: Current trace context
- `step_context_var`: Current step context

This allows proper context isolation in concurrent scenarios.

## Tracer Initialization

Tracers are initialized lazily based on environment variables:

```python
def _initialize_langfuse_tracer(self, trace_context):
    try:
        from agent_tracer.tracers.langfuse import LangFuseTracer
        trace_context.tracers["langfuse"] = LangFuseTracer(...)
    except ImportError:
        # Tracer not installed, skip silently
        pass
    except Exception as e:
        # Configuration error, log and skip
        logger.debug(f"Error: {e}")
```

This approach:
- Fails gracefully if tracer deps not installed
- Only initializes configured tracers
- Allows multiple tracers simultaneously
- No configuration needed if env vars not set

## Error Handling

### Graceful Degradation

- Missing tracer dependencies → Skip that tracer
- Invalid configuration → Log warning, continue
- Backend API errors → Catch, log, continue
- User code errors → Propagate normally

### Error Reporting

Errors in traces are captured and sent to backends:

```python
try:
    async with tracer.trace_step("risky", inputs):
        result = risky_operation()
except Exception as e:
    # Error automatically captured and sent to tracers
    await tracer.end_trace(error=e)
    raise
```

## Sensitive Data Protection

Automatic masking of sensitive fields:

```python
def _cleanup_inputs(self, inputs):
    sensitive_keywords = ["api_key", "password", "token", ...]
    return mask_sensitive_data(inputs, sensitive_keywords)
```

Applied to:
- All inputs before sending to tracers
- Configurable keyword list
- Recursive masking in nested structures

## Extension Points

### Adding a New Tracer

1. Create new tracer class inheriting from `BaseTracer`
2. Implement all abstract methods
3. Add initialization method to `TracingService`
4. Update `pyproject.toml` with optional dependencies
5. Add tests and documentation

### Custom Serialization

Override serialization for custom types:

```python
def serialize_value(value):
    if isinstance(value, MyCustomType):
        return value.to_dict()
    return default_serialize(value)
```

## Performance Considerations

### Overhead

- Background worker: Minimal main thread impact
- Context vars: O(1) lookup
- Queue operations: Async, non-blocking
- Serialization: Lazy, only when needed

### Optimization Strategies

1. **Batch Processing**: Worker processes multiple items
2. **Lazy Initialization**: Tracers created only when needed
3. **Selective Tracing**: Can disable per-tracer or globally
4. **Efficient Serialization**: Reuse serialized data

### Resource Management

- Worker cleanup on trace end
- Proper exception handling
- Context cleanup via context managers
- No memory leaks from unclosed traces

## Testing Strategy

### Unit Tests

- Test each tracer in isolation
- Mock backend APIs
- Test error handling paths
- Verify serialization logic

### Integration Tests

- Test with real backends (if keys available)
- Test multi-tracer scenarios
- Test concurrent traces
- Test error recovery

### Example Tests

```python
@pytest.mark.asyncio
async def test_basic_flow():
    tracer = TracingService()
    await tracer.start_trace(...)
    async with tracer.trace_step(...):
        tracer.set_outputs(...)
    await tracer.end_trace(...)
```

## Security Considerations

1. **API Key Protection**: Never log API keys
2. **Data Masking**: Automatic sensitive data filtering
3. **Network Security**: Use HTTPS for backend communication
4. **Input Validation**: Validate all inputs before sending
5. **Error Messages**: Don't expose sensitive info in errors

## Future Enhancements

### Planned Features

- Sampling strategies for high-volume scenarios
- Custom span attributes
- Metrics collection (duration, count, etc.)
- Trace linking across services
- Batch export for offline analysis
- CLI tools for trace management
- Web UI for visualization

### Extensibility

- Plugin system for custom tracers
- Custom serializers registry
- Hook system for pre/post processing
- Configuration profiles

## Dependencies

### Core Dependencies

- `pydantic>=2.0.0`: Data validation and serialization
- `typing-extensions>=4.5.0`: Enhanced type hints

### Optional Dependencies

Each tracer has its own optional dependencies:

- LangFuse: `langfuse`
- LangWatch: `langwatch`, `nanoid`
- Opik: `opik`

## Compatibility

- **Python**: 3.10+
- **Async**: Full async/await support
- **Type Checking**: mypy compatible
- **Frameworks**: Works with any Python framework

## References

- [OpenTelemetry Specification](https://opentelemetry.io/docs/specs/otel/)
- [LangFuse Documentation](https://langfuse.com/docs)
- [Python AsyncIO](https://docs.python.org/3/library/asyncio.html)
- [Context Variables](https://docs.python.org/3/library/contextvars.html)

