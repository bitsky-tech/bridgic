# Worker Callback

## Introduction

The Worker Callback mechanism will invoke pre-defined hook methods at key points during worker execution, allowing you to add cross-cutting concerns such as logging, validation, monitoring, and error handling without modifying your business logic. This tutorial will guide you through understanding and using the Worker Callback mechanism.

`WorkerCallback` is the base class of all callback instances that are invokded around a worker. You can implement the following three methods to subclass `WorkerCallback`:

| Method              | Description                                                                              |
|:-------------------:|:----------------------------------------------------------------------------------------:|
| `on_worker_start()` | Called before worker execution. Use for input validation, logging, or monitoring         |
| `on_worker_end()`   | Called after successful execution. Use for result logging, event publishing, or metrics  |
| `on_worker_error()` | Called when an exception is raised. Use for error handling, logging, or suppression      |


## Creating a Custom Callback

### Step 1: Define a New Class

To create a custom callback, simply subclass `WorkerCallback` and implement the methods above. You don't need to implement all three methods, but only implement the ones that you need. The base `WorkerCallback` class provides default implementations that do nothing. Here we define a `LoggingCallback` class that implements all of the three hook methods.

```python
from typing import Any, Dict, Optional
from bridgic.core.automa import GraphAutoma
from bridgic.core.automa.worker import WorkerCallback

class LoggingCallback(WorkerCallback):
    """Log worker lifecycle events."""

    def __init__(self, tag: str = None):
        self._tag = tag or ""

    async def on_worker_start(
        self,
        key: str,
        is_top_level: bool = False,
        parent: Optional[GraphAutoma] = None,
        arguments: Dict[str, Any] = None,
    ) -> None:
        print(self._tag + f"[START] {key} args={arguments}")

    async def on_worker_end(
        self,
        key: str,
        is_top_level: bool = False,
        parent: Optional[GraphAutoma] = None,
        arguments: Dict[str, Any] = None,
        result: Any = None,
    ) -> None:
        print(self._tag + f"[END] {key} result={result}")

    async def on_worker_error(
        self,
        key: str,
        is_top_level: bool = False,
        parent: Optional[GraphAutoma] = None,
        arguments: Dict[str, Any] = None,
        error: Exception = None,
    ) -> bool:
        print(self._tag + f"[ERROR] {key} -> {error}")
        return False  # Returning False means don't suppress the exception.
```

### Step 2: Decide the Building Mode

Callbacks are instantiated through `WorkerCallbackBuilder`, which delays construction until the worker is created and lets you control its sharing mode.

- **Shared instance mode** (`is_shared=True`, default): all workers within the declaration scope reuse the same callback instance. This ideal for stateful integrations (e.g., a tracing client).
- **Independent instance mode** (`is_shared=False`): every worker receives its own callback instance. This is useful when you need isolated state or thread safety.

```python
from bridgic.core.automa.worker import WorkerCallbackBuilder

shared_builder = WorkerCallbackBuilder(LoggingCallback)
isolated_builder = WorkerCallbackBuilder(LoggingCallback, is_shared=False)
```

### Step 3: Determine the Scope

Choose where the builder should take effect. Bridgic merges builders from the widest scope to the narrowest one:

| Level            | Registration Method         | Scope of Effect                                    |
|:----------------:|:---------------------------:|:---------------------------------------------------|
| **Global-level** | `GlobalSetting`             | Applies to all workers across all Automa instances |
| **Automa-level** | `RunningOptions`            | Applies to every worker inside one Automa instance |
| **Worker-level** | `@worker` or `add_worker()` | Applies only to the targeted worker                |

**Global-Level Configuration**

```python
from bridgic.core.automa import GraphAutoma, worker
from bridgic.core.automa.worker import WorkerCallbackBuilder
from bridgic.core.config import GlobalSetting

GlobalSetting.set(callback_builders=[WorkerCallbackBuilder(LoggingCallback)])

class MyAutoma(GraphAutoma):
    @worker(is_start=True)
    async def step1(self, x: int) -> int:
        return x + 1

automa = MyAutoma()  # Will log for all workers inside.
```

**Automa-Level Configuration**

```python
from bridgic.core.automa import GraphAutoma, RunningOptions, worker
from bridgic.core.automa.worker import WorkerCallbackBuilder

class MyAutoma(GraphAutoma):
    @worker(is_start=True)
    async def step1(self, x: int) -> int:
        return x + 1

running_options = RunningOptions(callback_builders=[WorkerCallbackBuilder(LoggingCallback)])
automa = MyAutoma(running_options=running_options)  # Will log for all workers inside.
```

**Worker-Level Configuration**

```python
from bridgic.core.automa import GraphAutoma, worker
from bridgic.core.automa.worker import WorkerCallbackBuilder

class MyAutoma(GraphAutoma):
    @worker(
        is_start=True,
        callback_builders=[WorkerCallbackBuilder(LoggingCallback)],
    )
    async def step1(self, x: int) -> int:
        return x + 1
```

## Callback Propagation

Nested automa inherit callbacks from their parent scope (by reading their initialized running options). This keeps instrumentation consistent across multi-level graphs.

## Exception Matching and Suppression

### Exception Type Matching

The `on_worker_error()` method allows for fine-grained and flexible error handling by inspecting the type annotation of its `error` parameter. You can indicate exactly which exception types you want your handler to respond to, by annotating the `error` parameter with a specific exception type. At runtime, the Bridgic framework will automatically match and invoke your callback only for those exceptions that match the annotation.

Below is a simple example comparison table:

| Type annotation of `error` | The matched exception type(s) to trigger `on_worker_error`   |
|:--------------------------:|:-------------------------------------------------------------|
| `Exception`                | All exceptions                                               |
| `ValueError`               | `ValueError` and its subclasses (e.g., `UnicodeDecodeError`) |
| `Union[Type1, Type2, ...]` | `Type1` and `Type2` will be considered to be matched         |

### Exception Suppression

The return value of `on_worker_error` will determine whether to suppress the captured exception:

| Value   | Behavior                                                                   |
|:-------:|:---------------------------------------------------------------------------|
| `True`  | Means to suppress the exception; the worker result becomes `None`.         |
| `False` | Means to observe only; the framework re-raises after all callbacks finish. |

**Interaction exceptions** (`_InteractionEventException`, `InteractionException`) can never be suppressed to ensure human-in-the-loop flows stay intact.

Examples:

```python
class ValueErrorHandler(WorkerCallback):
    async def on_worker_error(..., error: ValueError = None) -> bool:
        log.warning("ValueError in %s: %s", key, error)
        return True  # swallow ValueError

class MultipleErrorHandler(WorkerCallback):
    async def on_worker_error(..., error: Union[KeyError, TypeError] = None) -> bool:
        log.error("Recoverable issue: %s", error)
        return False  # propagate
```

The framework calls every matching callback, so you can compose specialized handlers with broader “catch-all” callbacks.

## Dynamically Added Workers Support

`GraphAutoma.add_worker()` and related APIs allow you to modify the topology at runtime. When a new worker is added, Bridgic automatically builds its callback list using the current global builders, the builders from its ancestors' running options, and the builders passed with the `add_worker()` call. As a result:

- Dynamically added workers receive the same instrumentation guarantees as statically declared ones.
- Nested automa inserted later as a new worker inherits callbacks from its ancestors scopes.

This design keeps long-running agentic systems observable even as they grow or reconfigure themselves during execution.


## Best Practices

1. **Keep callbacks lightweight**: Callback methods are called for each worker that they are responsible for, so keep them fast and avoid blocking operations.

2. **Use appropriate scope**: Use global-level for application-wide concerns, automa-level for specific instance, and worker-level for fine-grained control.

3. **Handle exceptions carefully**: Be thoughtful about which exceptions to suppress. Suppressing exceptions can hide bugs and make debugging difficult.

4. **Use shared instances wisely**: Shared instances are great for maintaining connections or state, but be aware of thread-safety concerns.

5. **Leverage the parent parameter**: The `parent` parameter gives you access to the automa's context, allowing you to post events, request feedback, or interact with the automa's state.

## Next Steps

- Explore [Callback Integrations](../../../extras/callbacks/index.md) for ready-to-use callback implementations
- Learn about [About Observability](../observability/index.md) to see how callbacks enable system transparency
