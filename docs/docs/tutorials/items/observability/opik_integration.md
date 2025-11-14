# Opik Integration

> Learn how to use Comet Opik to trace and monitor your Bridgic applications with comprehensive worker-level tracing and observability.

## Opik Overview

[Comet Opik](https://www.comet.com/docs/opik/) is a comprehensive observability platform for LLM applications, RAG systems, and agentic workflows. The `OpikTraceCallback` integrates Opik tracing with the Bridgic framework, providing worker-level tracing for debugging and monitoring your Bridgic applications.

The `OpikTraceCallback` is a [Bridgic callback](../../../../extras/callbacks/) that instruments worker execution with Opik tracing. It automatically tracks:

* **Worker Execution Traces**: Creates traces for top-level automa execution and spans for each worker execution
* **Execution Metadata**: Records worker inputs, outputs, execution duration, and dependencies
* **Error Tracking**: Captures exceptions and error information for debugging
* **Nested Automa Support**: Properly handles nested automa instances with hierarchical tracing

**Note**: The `OpikTraceCallback` focuses on tracing worker execution. For LLM-specific tracing, evaluation metrics, and CI/CD testing features, please refer to the [Opik documentation](https://www.comet.com/docs/opik/) for additional integrations and capabilities.

## Setup

Comet provides a hosted version of the Opik platform, or you can run the platform locally.

To use the hosted version, simply [create a free Comet account](https://www.comet.com/signup?utm_medium=github&utm_source=bridgic_docs) and grab your API Key.

To run the Opik platform locally, see the [installation guide](https://www.comet.com/docs/opik/self-host/overview/) for more information.

For this guide we will use a simple Bridgic example to demonstrate the integration.

### Step 1: Install required packages

```shell
pip install bridgic bridgic-callbacks-trace-opik opik --upgrade
```

### Step 2: Configure Opik

```python
import opik

opik.configure(use_local=False)
```

### Step 3: Using Bridgic with Opik

The first step is to create your automa. We will use a simple example:

```python
from bridgic.core.automa import GraphAutoma, RunningOptions, worker
from bridgic.core.automa.worker import WorkerCallbackBuilder
from bridgic.callbacks.trace.opik import OpikTraceCallback

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
```

Now we can configure Opik's callback and run our automa:

```python
# Configure Opik callback at Automa level
builder = WorkerCallbackBuilder(
    OpikTraceCallback, 
    init_kwargs={"project_name": "bridgic-integration-demo"}
)
running_options = RunningOptions(callback_builders=[builder])

automa = DataAnalysisAutoma(running_options=running_options)
result = await automa.arun(topic="market analysis")
print(result)
```

Alternatively, you can configure Opik at the global level:

```python
from bridgic.core.config import GlobalSetting

# Configure Opik callback globally
GlobalSetting.set(callback_builders=[
    WorkerCallbackBuilder(
        OpikTraceCallback, 
        init_kwargs={"project_name": "bridgic-integration-demo"}
    )
])

# All automa instances will automatically use Opik tracing
automa = DataAnalysisAutoma()
result = await automa.arun(topic="market analysis")
print(result)
```

After running your Bridgic application, visit the Opik app to view:

* Worker execution traces and spans with their metadata
* Worker execution flow and dependencies
* Performance metrics like execution duration
* Error tracking and debugging information

## Configuration Scope

The `OpikTraceCallback` is a Bridgic callback that requires access to the automa context. As such, it can only be configured at the **Automa level** (via `RunningOptions`) or **Global level** (via `GlobalSetting`). It does not support worker-level configuration.

For more information about Bridgic's callback system and configuration scopes, see the [Callback Integrations documentation](../../../../extras/callbacks/).

### Automa-Level Configuration

Configure Opik tracing for a specific automa instance:

```python
from bridgic.core.automa import GraphAutoma, RunningOptions, worker
from bridgic.core.automa.worker import WorkerCallbackBuilder
from bridgic.callbacks.trace.opik import OpikTraceCallback

class MyAutoma(GraphAutoma):
    @worker(is_start=True)
    async def step1(self):
        return "hello"

# Configure callback for this automa instance
builder = WorkerCallbackBuilder(
    OpikTraceCallback, 
    init_kwargs={"project_name": "my-project"}
)
running_options = RunningOptions(callback_builders=[builder])
automa = MyAutoma(running_options=running_options)
```

### Global-Level Configuration

Configure Opik tracing for all automa instances:

```python
from bridgic.core.automa import GraphAutoma, worker
from bridgic.core.automa.worker import WorkerCallbackBuilder
from bridgic.core.config import GlobalSetting
from bridgic.callbacks.trace.opik import OpikTraceCallback

# Configure global callback
GlobalSetting.set(callback_builders=[
    WorkerCallbackBuilder(
        OpikTraceCallback, 
        init_kwargs={"project_name": "my-project"}
    )
])

class MyAutoma(GraphAutoma):
    @worker(is_start=True)
    async def step1(self):
        return "hello"

# All automa instances automatically use the global callback
automa = MyAutoma()
```

## What Gets Traced

The `OpikTraceCallback` automatically tracks the following during worker execution:

* **Top-level Automa Execution**: Creates a trace for the entire automa execution, including:
  * Automa inputs (arguments passed to `arun()`)
  * Automa outputs (final result)
  * Execution duration
  * Execution status (completed or failed)

* **Worker Execution**: Creates spans for each worker execution, including:
  * Worker inputs (serialized arguments)
  * Worker outputs (serialized results)
  * Execution duration
  * Dependencies between workers
  * Worker context information (key, dependencies, etc.)

* **Nested Automa**: Handles nested automa instances as workers, maintaining proper trace hierarchy with parent-child relationships

* **Error Tracking**: Captures exceptions and error information (error type, error message) for debugging when worker execution fails

## Resources

* [🦉 Opik Documentation](https://www.comet.com/docs/opik/)
* [📚 Bridgic Callbacks Documentation](../../../../extras/callbacks/)
