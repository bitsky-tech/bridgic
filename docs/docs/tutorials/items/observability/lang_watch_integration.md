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

Just build your application using normal Bridgic-style orchestration and register the callback at the scope that you need. The `start_langwatch_trace` function provides a convenient way to register LangWatch tracing for all workers application wide. The below example shows how to use it:

```python
from bridgic.core.automa import GraphAutoma, worker
from bridgic.traces.langwatch import start_langwatch_trace

# Register LangWatch trace callback
# If api_key and endpoint_url are not provided, they will be read from environment variables
start_langwatch_trace(
    api_key=None,  # Optional: defaults to LANGWATCH_API_KEY env var
    endpoint_url=None,  # Optional: defaults to LANGWATCH_ENDPOINT env var or https://app.langwatch.ai
    base_attributes=None  # Optional: base attributes for LangWatch tracing
)

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
    automa = DataAnalysisAutoma()
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
