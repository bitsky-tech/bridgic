# Callback Integrations

## Overview

Bridgic's callback integration module (`bridgic.callbacks`) provides a flexible mechanism for instrumenting worker execution with cross-cutting concerns such as tracing, monitoring, logging, and validation. Callbacks are registered through `WorkerCallbackBuilder` and can be configured at different scopes to achieve fine-grained control over when and where callbacks are applied.

## Design Philosophy

### Scope-Based Configuration

Bridgic supports three levels of callback configuration, each with different scopes of effect:

| Level | Registration Method | Scope of Effect |
|:-----:|:-------------------:|:---------------|
| **Worker-level** | `@worker` decorator | Applies only to that specific worker |
| **Automa-level** | `RunningOptions` | Applies to all workers within a specific Automa instance |
| **Global-level** | `GlobalSetting` | Applies to all workers across all Automa instances |

Callbacks from different levels are merged together, with the order: Global → Automa → Worker. This allows you to set up global defaults while still being able to override or extend behavior at more specific levels.

### WorkerCallbackBuilder Pattern

`WorkerCallbackBuilder` is a factory pattern that creates `WorkerCallback` instances. The builder provides two key features:

1. **Lazy Instantiation**: Callback instances are created only when workers are instantiated, not when builders are registered.
2. **Shared Instance Mode**: When `is_shared=True` (default), all workers within the same scope share the same callback instance, enabling stateful callbacks (e.g., maintaining connections to external services). When `is_shared=False`, each worker gets its own independent callback instance.

### Callback Lifecycle

Each callback implements three hook methods that are invoked during worker execution:

- `on_worker_start()`: Called before worker execution, useful for input validation, logging, or monitoring
- `on_worker_end()`: Called after successful worker execution, useful for result monitoring, event publishing, or logging
- `on_worker_error()`: Called when worker execution raises an exception, useful for error handling, logging, or error suppression

### Callback Method Parameters

All callback methods receive a consistent set of parameters that provide context about the worker execution.

#### Common Parameters (All Methods)

| Parameter | Type | Description |
|:---------:|:----:|:------------|
| `key` | `str` | The unique identifier of the worker being executed. This is typically the method name or a custom key specified in the `@worker` decorator. |
| `is_top_level` | `bool` | Indicates whether the worker is the top-level automa instance. When `True`, the `parent` parameter will be the automa itself (i.e., `parent is self`). |
| `parent` | `Optional[GraphAutoma]` | The parent automa instance that contains this worker. For top-level automa, `parent` is the automa itself. Use this to access the automa's context, post events, or interact with the automa's state.<br><br>**Note**: Some callbacks require this parameter to function correctly and may only work at the Automa or Global configuration level. |
| `arguments` | `Dict[str, Any]` | A dictionary containing the execution arguments passed to the worker. The dictionary has two keys:<br>- `"args"`: A tuple of positional arguments<br>- `"kwargs"`: A dictionary of keyword arguments |

#### Method-Specific Parameters

**`on_worker_start()`** receives only the common parameters listed above.

**`on_worker_end()`** additionally receives:

| Parameter | Type | Description |
|:---------:|:----:|:------------|
| `result` | `Any` | The return value from the worker's execution. This is `None` if the worker doesn't return a value. |

**`on_worker_error()`** additionally receives:

| Parameter | Type | Description |
|:---------:|:----:|:------------|
| `error` | `Exception` | The exception that was raised during worker execution. The type annotation of this parameter is critical for exception matching (see below). |

#### Exception Matching in `on_worker_error()`

The framework uses the type annotation of the `error` parameter to determine which exceptions your callback will handle:

| Type Annotation | Matches |
|:---------------:|:--------|
| `error: ValueError` | `ValueError` and all its subclasses (e.g., `UnicodeDecodeError`) |
| `error: Exception` | All exceptions (since all exceptions inherit from `Exception`) |
| `error: Union[Type1, Type2, ...]` | Multiple exception types specified in the `Union` |

#### Return Value of `on_worker_error()`

| Return Value | Behavior |
|:------------:|:---------|
| `True` | Suppress the exception. The framework will proceed as if there was no error, with the worker result set to `None`. |
| `False` | Observe the error. The framework will re-raise the exception after all matching callbacks are called. |

**Note**: Interaction exceptions (`_InteractionEventException` and `InteractionException`) cannot be suppressed, even if your callback returns `True`.

## Configuration Scenarios

### Scenario 1: Worker-Level Configuration

Apply callbacks to a specific worker by registering builders in the `@worker` decorator. This is useful when you need fine-grained control over individual workers.

**Note**: Not all callbacks support worker-level configuration. Some callbacks (such as `OpikTraceCallback`) require access to the automa context and can only be configured at the Automa or Global level. Check the callback's documentation for supported configuration levels.

```python
from typing import Any, Dict, Optional
from bridgic.core.automa import GraphAutoma, worker
from bridgic.core.automa.worker import WorkerCallback, WorkerCallbackBuilder

# Define a simple logging callback
class LoggingCallback(WorkerCallback):
    async def on_worker_start(
        self,
        key: str,
        is_top_level: bool = False,
        parent: Optional[GraphAutoma] = None,
        arguments: Dict[str, Any] = None,
    ) -> None:
        print(f"Starting worker: {key}")
    
    async def on_worker_end(
        self,
        key: str,
        is_top_level: bool = False,
        parent: Optional[GraphAutoma] = None,
        arguments: Dict[str, Any] = None,
        result: Any = None,
    ) -> None:
        print(f"Completed worker: {key}, result: {result}")

class MyAutoma(GraphAutoma):
    @worker(
        is_start=True,
        callback_builders=[WorkerCallbackBuilder(LoggingCallback)]
    )
    async def step1(self):
        return "hello"
```

### Scenario 2: Automa-Level Configuration

Apply callbacks to all workers within a specific Automa instance by configuring `RunningOptions`. This is useful when you want consistent instrumentation across all workers in an automa without modifying each worker individually.

```python
from bridgic.core.automa import GraphAutoma, RunningOptions, worker
from bridgic.core.automa.worker import WorkerCallbackBuilder

class MyAutoma(GraphAutoma):
    @worker(is_start=True)
    async def step1(self):
        return "hello"

# Configure callback for this automa instance
builder = WorkerCallbackBuilder(LoggingCallback)
running_options = RunningOptions(callback_builders=[builder])
automa = MyAutoma(running_options=running_options)
```

### Scenario 3: Global-Level Configuration

Apply callbacks to all workers across all Automa instances by configuring `GlobalSetting`. This is useful for application-wide instrumentation, such as global tracing or monitoring.

```python
from bridgic.core.automa import GraphAutoma, worker
from bridgic.core.automa.worker import WorkerCallbackBuilder
from bridgic.core.config import GlobalSetting

# Configure global callback
GlobalSetting.set(callback_builders=[
    WorkerCallbackBuilder(LoggingCallback)
])

class MyAutoma(GraphAutoma):
    @worker(is_start=True)
    async def step1(self):
        return "hello"

# All automa instances automatically use the global callback
automa = MyAutoma()
```

## Available Callback Integrations

| Category | Package | Description |
|:--------:|:-------:|:------------|
| Tracing | `bridgic.callbacks.trace.opik` | Opik tracing integration for distributed tracing and observability |
