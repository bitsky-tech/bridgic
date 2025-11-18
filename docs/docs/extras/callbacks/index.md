# Callback Integrations

## Overview

Bridgic's callback integration module (`bridgic.callbacks`) provides a range of ready-to-use implementations based on the `WorkerCallback` framework, offering concrete solutions for common cross-cutting concerns such as tracing, monitoring and more.

These integrations are designed to work seamlessly with the `WorkerCallbackBuilder` pattern, enabling flexible configuration across different scopes, from individual workers to entire Automa instances to global settings, which allows you to precisely control where and how callbacks are applied.

### Main Methods

Each callback implements three hook methods that are invoked during worker execution:

| Method            |  Description                                                                |
|:-----------------:|:----------------------------------------------------------------------------|
| `on_worker_start` | Called before worker execution. Useful for input validation, monitoring     |
|  `on_worker_end`  | Called after worker execution. Useful for event publishing, monitoring      |
| `on_worker_error` | Called when worker execution raises an exception. Useful for error handling |

### Building Modes

`WorkerCallbackBuilder` is a factory that creates `WorkerCallback` instances, providing two mode for building callback instances:

1. **Shared Instance Mode** (default, `is_shared=True`): All workers within the same scope share the same callback instance, enabling stateful callbacks that maintain state across workers (e.g., maintaining connections to external services).
2. **Independent Instance Mode** (`is_shared=False`): Each worker gets its own independent callback instance out of the same `WorkerCallback` subclass, useful when callback instances need isolated state or when thread-safety requires separate instances.

### Configuration Scopes

Bridgic supports three levels of callback configuration, each with different scopes of effect:

| Level            | Registration Method           | Scope of Effect                                    |
|:----------------:|:-----------------------------:|:---------------------------------------------------|
| **Global-level** | `GlobalSetting`               | Applies to all workers across all Automa instances |
| **Automa-level** | `RunningOptions`              | Applies to all workers within one Automa instance  |
| **Worker-level** | `@worker` or `add_worker()`   | Applies only to that specific worker               |

Callbacks from different levels are merged together, with the order: Global → Automa → Worker. This allows you to set up global defaults while still being able to override or extend behavior at more specific levels.


## Available Integrations

| Category | Package                        | Description               |
|:--------:|:------------------------------:|:--------------------------|
| Trace    | `bridgic.callbacks.trace.opik` | Opik tracing integration. |
