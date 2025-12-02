# Opik Integration


## Overview

[Comet Opik](https://www.comet.com/docs/opik/) is a comprehensive observability platform designed for agentic systems. The `bridgic-traces-opik` package enables seamless integration of Opik into your Bridgic-based agentic workflows.

This integration is primarily supported by `OpikTraceCallback`, a [WorkerCallback](../core_mechanism/worker_callback.ipynb) implementation that automatically instruments the worker execution with Opik tracing, which provides comprehensive observability by:

* **Worker Execution Traces Tracking**: Record the execution of each worker as a span in Opik, allowing to visualize start/end time, duration.
* **Worker Execution Data Reporting**: Capture the input, output and other necessary information and then log to the opik platform.
* **Hierarchical Trace Structure**: Organize execution traces in a hierarchy that reflects the nesting between automa layers, making it straightforward to see how top-level automa is composed of the execution of multiple nested workers.


## Prerequisites

Comet provides a hosted version of the Opik platform, or you can run the platform locally.

- To use the hosted version, you need to [create a Comet account](https://www.comet.com/signup) and [grab your API Key](https://www.comet.com/account-settings/apiKeys).
- To run the Opik platform locally, see the [installation guide](https://www.comet.com/docs/opik/self-host/overview/) for more information.


## Using Opik in Bridgic

### Step 1: Install package

```shell
# Automatically install the Opik package
pip install bridgic-traces-opik
```

### Step 2: Register the callback

You can register Opik tracing at the scope that best fits your application. `start_opik_trace` is the fastest path (a single line that configures global tracing via `GlobalSetting`). When you want to customize the same global setup or target only a specific automa, Bridgic exposes direct hooks for both use cases.

#### Method 1: Application-wide registration (helper or manual)

Pick one of the three options below—they produce the exact same runtime behavior:

=== "start_opik_trace for opik cloud"

    ```python
    from bridgic.traces.opik import start_opik_trace
    start_opik_trace(project_name="bridgic-integration-demo", api_key="your-api-key")
    ```

=== "start_opik_trace for self-hosted"

    ```python
    from bridgic.traces.opik import start_opik_trace
    start_opik_trace(project_name="bridgic-integration-demo", use_local=True)
    ```

=== "GlobalSetting"

    ```python
    from bridgic.core.automa.worker import WorkerCallbackBuilder
    from bridgic.core.config import GlobalSetting
    from bridgic.traces.opik import OpikTraceCallback

    GlobalSetting.set(callback_builders=[WorkerCallbackBuilder(
        OpikTraceCallback,
        init_kwargs={"project_name": "bridgic-integration-demo", "api_key": "your-api-key"}
    )])
    ```

```python
from bridgic.core.automa import GraphAutoma, worker
from bridgic.traces.opik import start_opik_trace

class DataAnalysisAutoma(GraphAutoma):
    @worker(is_start=True)
    async def collect_data(self, topic: str) -> dict:
        """Collect data for the given topic."""
        # Simulate data collection
        await asyncio.sleep(1)
        return {
            "topic": topic,
            "data_points": ["point1", "point2", "point3"],
            "timestamp": "2024-01-01"
        }

    @worker(dependencies=["collect_data"])
    async def analyze_trends(self, data: dict) -> dict:
        """Analyze trends in the collected data."""
        # Simulate trend analysis
        await asyncio.sleep(1)
        return {
            "trends": ["trend1", "trend2"],
            "confidence": 0.85,
            "source_data": data
        }

    @worker(dependencies=["analyze_trends"], is_output=True)
    async def generate_report(self, analysis: dict) -> str:
        """Generate a final report."""
        await asyncio.sleep(1)
        return f"Report: Found {len(analysis['trends'])} trends with {analysis['confidence']} confidence."

async def automa_arun():
    # Call either start_opik_trace(...) or GlobalSetting.set(...) once at startup
    start_opik_trace(project_name="bridgic-integration-demo", api_key="your-api-key")
    automa = DataAnalysisAutoma()
    result = await automa.arun(topic="market analysis")
    print(result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(automa_arun())
```

#### Method 2: Per-automa scope with `RunningOptions`

When only a specific automa needs tracing, configure the callback through `RunningOptions`. Each automa gets its own callback instance, leaving other automa untouched.

```python
from bridgic.core.automa import GraphAutoma, RunningOptions, worker
from bridgic.core.automa.worker import WorkerCallbackBuilder
from bridgic.traces.opik import OpikTraceCallback

class DataAnalysisAutoma(GraphAutoma):
    @worker(is_start=True)
    async def collect_data(self, topic: str) -> dict:
        """Collect data for the given topic."""
        # Simulate data collection
        await asyncio.sleep(1)
        return {
            "topic": topic,
            "data_points": ["point1", "point2", "point3"],
            "timestamp": "2024-01-01"
        }

    @worker(dependencies=["collect_data"])
    async def analyze_trends(self, data: dict) -> dict:
        """Analyze trends in the collected data."""
        # Simulate trend analysis
        await asyncio.sleep(1)
        return {
            "trends": ["trend1", "trend2"],
            "confidence": 0.85,
            "source_data": data
        }

    @worker(dependencies=["analyze_trends"], is_output=True)
    async def generate_report(self, analysis: dict) -> str:
        """Generate a final report."""
        await asyncio.sleep(1)
        return f"Report: Found {len(analysis['trends'])} trends with {analysis['confidence']} confidence."

async def automa_arun():
    builder = WorkerCallbackBuilder(OpikTraceCallback, init_kwargs={"project_name": "bridgic-integration-demo", "api_key": "your-api-key"})
    running_options = RunningOptions(callback_builders=[builder])
    automa = DataAnalysisAutoma(running_options=running_options)
    result = await automa.arun(topic="market analysis")
    print(result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(automa_arun())
```

Once your Bridgic application has finished running, your terminal might display the following message:

```shell
$ python bridgic-demo/demo.py 
OPIK: Started logging traces to the "bridgic-integration-demo" project at http://localhost:5173/api/v1/session/redirect/projects/?trace_id=019a9709-e437-7b30-861e-76006b75e969&path=aHR0cDovL2xvY2FsaG9zdDo1MTczL2FwaS8=
Report: Found 2 trends with 0.85 confidence.
```

You can dive into the Opik app to explore rich visual insights and detailed traces of your workflow.

<div style="text-align: center;">
<img src="../../../imgs/bridgic-integration-opik-demo.png" alt="bridgic integration opik demo" width="auto">
</div>
