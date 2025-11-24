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
        return "hello"

    @worker(dependencies=["step1"], is_output=True)
    async def step2(self, step1: str):
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

asyncio.run(main())
```

### Method 2: Global Scope with GlobalSetting

You can choose between two options:

1. Register the callback at the global level through `GlobalSetting` to make it effective for every automa in the runtime. Each worker, regardless of which automa creates it, is instrumented with the same callback configuration.
2. Use the `start_langwatch_trace` helper to register LangWatch tracing for every worker application-wide. This is the simplest and most recommended approach.

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

class DataAnalysisAutoma(GraphAutoma):
    @worker(is_start=True)
    async def collect_data(self, topic: str) -> dict:
        """Collect data for the given topic."""
        # Simulate data collection
        return {
            "topic": topic,
            "data_points": ["point1", "point2", "point3"],
            "timestamp": "2024-01-01"
        }

    @worker(dependencies=["collect_data"])
    async def analyze_trends(self, data: dict) -> dict:
        """Analyze trends in the collected data."""
        # Simulate trend analysis
        return {
            "trends": ["trend1", "trend2"],
            "confidence": 0.85,
            "source_data": data
        }

    @worker(dependencies=["analyze_trends"], is_output=True)
    async def generate_report(self, analysis: dict) -> str:
        """Generate a final report."""
        return f"Report: Found {len(analysis['trends'])} trends with {analysis['confidence']} confidence."

async def main():
    automa = DataAnalysisAutoma()
    result = await automa.arun(topic="market analysis")
    print(result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

Alternatively, configure tracing in a single call with `start_langwatch_trace`. When parameters are omitted, the helper reads the values from the `LANGWATCH_API_KEY` and `LANGWATCH_ENDPOINT` environment variables.

```python
from bridgic.traces.langwatch import start_langwatch_trace

start_langwatch_trace(base_attributes={"app": "demo"})
```

Once your Bridgic application has finished running, traces will be automatically sent to LangWatch. You can view them in the LangWatch dashboard to explore rich visual insights and detailed traces of your workflow execution.


Parameters
----------

The `start_langwatch_trace` helper and `LangWatchTraceCallback` class accept the following parameters:

- `api_key` (Optional[str]): The API key for the LangWatch tracing service. If `None`, the `LANGWATCH_API_KEY` environment variable will be used.
- `endpoint_url` (Optional[str]): The URL of the LangWatch tracing service. If `None`, the `LANGWATCH_ENDPOINT` environment variable will be used. If that is also not provided, the default value will be `https://app.langwatch.ai`.
- `base_attributes` (Optional[BaseAttributes]): The base attributes to use for the LangWatch tracing client. These attributes will be included in all traces and spans.

Features
--------

- **Worker-level tracing**: Each worker execution is traced as a separate LangWatch span
- **Nested automa support**: Properly handles nested automa instances with hierarchical tracing
- **Error tracking**: Captures and reports errors during worker execution
- **Execution metadata**: Tracks execution duration, start/end times, and other metadata
- **Concurrent execution**: Supports tracing multiple concurrent automa executions