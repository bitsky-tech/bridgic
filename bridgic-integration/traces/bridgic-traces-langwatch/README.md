LangWatch Observability Integration
===================================

This package integrates LangWatch tracing with the Bridgic framework, providing worker granularity tracing implementation.

Installation
-----

```shell
# Install the LangWatch tracing package
pip install bridgic-traces-langwatch
```

Prerequisites
-----

LangWatch provides a hosted version of the platform, or you can run the platform locally.

- To use the hosted version, you need to [create a LangWatch account](https://langwatch.ai/) and grab your API key from the dashboard.
- To run LangWatch locally, see the [self-hosting guide](https://docs.langwatch.ai/self-hosting/overview) for more information.

Configuration
-----

### Using Environment Variables

Set the following environment variables:

```bash
export LANGWATCH_API_KEY="your-api-key-here"
export LANGWATCH_ENDPOINT="https://app.langwatch.ai"  # Optional, defaults to https://app.langwatch.ai
```

Usage
-----

The `LangWatchTraceCallback` can be configured in two ways:

### Method 1: Per-Automa Scope with RunningOptions

Apply the callback only to a single automa by configuring it through `RunningOptions`. In this mode, every worker instantiated by that automa receives its own callback instance, while other automa remain unaffected.

```python
from bridgic.core.automa import GraphAutoma, RunningOptions, worker
from bridgic.core.automa.worker import WorkerCallbackBuilder
from bridgic.traces.langwatch import LangWatchTraceCallback
import asyncio

class MyAutoma(GraphAutoma):
    @worker(is_start=True)
    async def step1(self):
        await asyncio.sleep(1)
        return "hello"

    @worker(dependencies=["step1"], is_output=True)
    async def step2(self, step1: str):
        await asyncio.sleep(1)
        return f"{step1} world"

async def main():
    builder = WorkerCallbackBuilder(
        LangWatchTraceCallback,
        init_kwargs={"base_attributes": {"app": "demo"}}
    )
    running_options = RunningOptions(callback_builders=[builder])
    automa = MyAutoma(running_options=running_options)
    result = await automa.arun()
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

### Method 2: Global Scope with GlobalSetting

You can register the callback at the global level through `GlobalSetting` to make it effective for every automa in the runtime. Each worker, regardless of which automa creates it, is instrumented with the same callback configuration.

```python
from bridgic.core.automa import GraphAutoma, worker
from bridgic.core.automa.worker import WorkerCallbackBuilder
from bridgic.core.config import GlobalSetting
from bridgic.traces.langwatch import LangWatchTraceCallback
import asyncio

# Configure global callback
GlobalSetting.set(callback_builders=[WorkerCallbackBuilder(
    LangWatchTraceCallback,
    init_kwargs={"base_attributes": {"app": "demo"}}
)])

class MyAutoma(GraphAutoma):
    @worker(is_start=True)
    async def step1(self):
        await asyncio.sleep(1)
        return "hello"

    @worker(dependencies=["step1"], is_output=True)
    async def step2(self, step1: str):
        await asyncio.sleep(1)
        return f"{step1} world"

async def main():
    automa = MyAutoma()
    result = await automa.arun()
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

From the perspective of tracking worker execution, the following approach is equivalent to the above one:

```python
from bridgic.traces.langwatch import start_langwatch_trace

start_langwatch_trace(base_attributes={"app": "demo"})
```

However, `start_langwatch_trace` is a higher-level function that encapsulates the functionality of the first approach. As the framework may add tracking for more important phases in the future, `start_langwatch_trace` will provide a unified interface for all tracking capabilities, making it the recommended approach for most use cases. When parameters are omitted, the helper reads the values from the `LANGWATCH_API_KEY` and `LANGWATCH_ENDPOINT` environment variables.
