# LangWatch Integration


## Overview

[LangWatch](https://langwatch.ai/) is a comprehensive observability platform designed for LLM applications. The `bridgic-traces-langwatch` package enables seamless integration of LangWatch into your Bridgic-based agentic workflows.

This integration is primarily supported by `LangWatchTraceCallback`, a [WorkerCallback](../core_mechanism/worker_callback.ipynb) implementation that automatically instruments the worker execution with LangWatch tracing, which provides comprehensive observability by:

* **Worker Execution Traces Tracking**: Record the execution of each worker as a span in LangWatch, allowing to visualize start/end time, duration.
* **Worker Execution Data Reporting**: Capture the input, output and other necessary information and then log to the LangWatch platform.
* **Hierarchical Trace Structure**: Organize execution traces in a hierarchy that reflects the nesting between automa layers, making it straightforward to see how top-level automa is composed of the execution of multiple nested workers.


## Prerequisites

LangWatch provides a hosted version of the platform, or you can run the platform locally.

- To use the hosted version, you need to [create a LangWatch account](https://langwatch.ai/) and grab your API Keyfrom the dashboard.
- To run LangWatch locally, see the [self-hosting guide](https://docs.langwatch.ai/self-hosting/overview) for more information.

<div style="text-align: center;">
<img src="../../../imgs/langwatch-api-key.png" alt="LangWatch api key" width="auto">
</div>


## Using LangWatch in Bridgic

### Step 1: Install package

```shell
# Install the LangWatch tracing package
pip install bridgic-traces-langwatch
```

### Step 2: Configure LangWatch

Using Environment Variables

Set the following environment variables:

```bash
export LANGWATCH_API_KEY="your-api-key-here"
export LANGWATCH_ENDPOINT="https://app.langwatch.ai"  # Optional, defaults to https://app.langwatch.ai
```

### Step 3: Register the callback

You can enable LangWatch tracing globally with a single helper call (recommended), or wire the callback manually when you need custom behavior. Bridgic also lets you scope tracing to a single automa via `RunningOptions`.

#### Method 1: Application-wide registration (helper or manual)

Choose whichever snippet fits your setup—they produce the same effect.

=== "start_langwatch_trace"

    ```python
    from bridgic.traces.langwatch import start_langwatch_trace

    start_langwatch_trace(
        api_key=None,            # defaults to LANGWATCH_API_KEY env var
        endpoint_url=None,       # defaults to LANGWATCH_ENDPOINT or https://app.langwatch.ai
        base_attributes=None     # optional: shared attributes applied to every trace
    )
    ```

=== "GlobalSetting"

    ```python
    from bridgic.core.automa.worker import WorkerCallbackBuilder
    from bridgic.core.config import GlobalSetting
    from bridgic.traces.langwatch import LangWatchTraceCallback

    GlobalSetting.set(callback_builders=[WorkerCallbackBuilder(
        LangWatchTraceCallback,
        init_kwargs={
            "api_key": None,
            "endpoint_url": None,
            "base_attributes": None
        }
    )])
    ```

```python
from bridgic.core.automa import GraphAutoma, worker

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

async def automa_arun():
    # Call start_langwatch_trace(...) or configure GlobalSetting(...) once at startup
    from bridgic.traces.langwatch import start_langwatch_trace

    start_langwatch_trace(
        api_key=None,
        endpoint_url=None,
        base_attributes=None
    )

    automa = DataAnalysisAutoma()
    result = await automa.arun(topic="market analysis")
    print(result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(automa_arun())
```

#### Method 2: Per-automa scope with `RunningOptions`

When only a specific automa needs LangWatch tracing, configure the callback through `RunningOptions`.

```python
from bridgic.core.automa import GraphAutoma, RunningOptions, worker
from bridgic.core.automa.worker import WorkerCallbackBuilder
from bridgic.traces.langwatch import LangWatchTraceCallback

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

async def automa_arun():
    builder = WorkerCallbackBuilder(LangWatchTraceCallback, init_kwargs={
        "api_key": None,
        "endpoint_url": None,
        "base_attributes": None
    })
    running_options = RunningOptions(callback_builders=[builder])
    automa = DataAnalysisAutoma(running_options=running_options)
    result = await automa.arun(topic="market analysis")
    print(result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(automa_arun())
```

Once your Bridgic application has finished running, traces will be automatically sent to LangWatch. You can view them in the LangWatch dashboard to explore rich visual insights and detailed traces of your workflow.


<div style="text-align: center;">
<img src="../../../imgs/bridgic-integration-langwatch-demo.png" alt="bridgic integration langwatch demo" width="auto">
</div>

## Parameters

The `start_langwatch_trace` function accept the following parameters:

- `api_key` (Optional[str]): The API key for the LangWatch tracing service. If `None`, the `LANGWATCH_API_KEY` environment variable will be used.
- `endpoint_url` (Optional[str]): The URL of the LangWatch tracing service. If `None`, the `LANGWATCH_ENDPOINT` environment variable will be used. If that is also not provided, the default value will be `https://app.langwatch.ai`.
- `base_attributes` (Optional[BaseAttributes]): The base attributes to use for the LangWatch tracing client. These attributes will be included in all traces and spans.

## Features

- **Worker-level tracing**: Each worker execution is traced as a separate LangWatch span
- **Nested automa support**: Properly handles nested automa instances with hierarchical tracing
- **Error tracking**: Captures and reports errors during worker execution
- **Execution metadata**: Tracks execution duration, start/end times, and other metadata
- **Concurrent execution**: Supports tracing multiple concurrent automa executions
