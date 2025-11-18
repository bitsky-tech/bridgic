# Opik Integration


## Overview

[Comet Opik](https://www.comet.com/docs/opik/) is a comprehensive observability platform designed for agentic systems. The `bridgic-callbacks-trace-opik` package enables seamless integration of Opik into your Bridgic-based agentic workflows.

This integration is primarily supported by `OpikTraceCallback`, a [WorkerCallback](../core_mechanism/worker_callback.ipynb) implementation that automatically instruments the worker execution with Opik tracing, which provides comprehensive observability by:

* **Worker Execution Traces Tracking**: Record the execution of each worker as a span in Opik, allowing to visualize start/end time, duration.
* **Worker Execution Data Reporting**: Capture the input, output and other necessary information and then log to the opik platform.
* **Hierarchical Trace Structure**: Organize execution traces in a hierarchy that reflects the nesting between automa layers, making it straightforward to see how top-level automa is composed of the execution of multiple nested workers.


## Prerequisites

Comet provides a hosted version of the Opik platform, or you can run the platform locally.

- To use the hosted version, you need to [create a Comet account](https://www.comet.com/signup) and grab your API Key.
- To run the Opik platform locally, see the [installation guide](https://www.comet.com/docs/opik/self-host/overview/) for more information.


## Using Opik in Bridgic

### Step 1: Install package

```shell
# Automatically install the Opik package
pip install bridgic-callbacks-trace-opik
```

### Step 2: Configure Opik

```python
# Configure Opik to use the cloud service, if you want to use the local service, set use_local=True.
python -c "import opik; opik.configure(use_local=False)"
```

### Step 3: Register the callback

Just build your application using normal Bridgic-style orchestration and register the callback at the scope that you need. The below example shows how to register `OpikTraceCallback` to apply opik tracing for all workers application wide.

```python
from bridgic.core.config import GlobalSetting
from bridgic.core.automa import GraphAutoma, RunningOptions, worker
from bridgic.core.automa.worker import WorkerCallbackBuilder
from bridgic.callbacks.trace.opik import OpikTraceCallback

GlobalSetting.set(callback_builders=[
    WorkerCallbackBuilder(
        OpikTraceCallback, 
        init_kwargs={"project_name": "bridgic-integration-demo"}
    )
])

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

automa = DataAnalysisAutoma()
await automa.arun(topic="market analysis")
```

Once your Bridgic application has finished running,the trace URL will be generated in the terminal, dive into the Opik app to explore rich visual insights and detailed traces of your workflow.

<div style="text-align: center;">
<img src="../../../imgs/bridgic-integration-demo.png" alt="bridgic integration demo" width="auto">
</div>
