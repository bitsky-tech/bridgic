# Complete Usage Guide

This guide covers all aspects of using agent-tracer in your projects.

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Basic Usage](#basic-usage)
4. [Advanced Patterns](#advanced-patterns)
5. [Framework Integration](#framework-integration)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

## Installation

### Standard Installation

```bash
# Minimal installation (core only)
pip install agent-tracer

# With specific tracer
pip install agent-tracer[langsmith]

# With multiple tracers
pip install agent-tracer[langsmith,langfuse]

# With all tracers (recommended for development)
pip install agent-tracer[all]

# With development tools
pip install agent-tracer[dev,all]
```

### Requirements

- Python 3.10 or higher
- pip or poetry for package management

## Configuration

### Environment Variables

Set environment variables for the tracers you want to use:

#### LangSmith

```bash
export LANGCHAIN_API_KEY="lsv2_pt_..."
export LANGCHAIN_PROJECT="my-project"  # Optional, defaults to "AgentTracer"
export LANGCHAIN_TRACING_V2="true"     # Auto-set by tracer
```

#### LangFuse

```bash
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_HOST="https://cloud.langfuse.com"
```

#### LangWatch

```bash
export LANGWATCH_API_KEY="lw_..."
```

#### Arize Phoenix

```bash
# For Arize Cloud
export ARIZE_API_KEY="your-api-key"
export ARIZE_SPACE_ID="your-space-id"
export ARIZE_COLLECTOR_ENDPOINT="https://otlp.arize.com"  # Optional

# For Phoenix OSS (local instance)
export PHOENIX_COLLECTOR_ENDPOINT="http://localhost:6006"
export PHOENIX_API_KEY=""  # Optional for local

# Batch mode (optional, default: false)
export ARIZE_PHOENIX_BATCH="true"
```

#### Opik

```bash
export OPIK_API_KEY="your-api-key"
export OPIK_WORKSPACE="your-workspace"  # Optional
export OPIK_URL_OVERRIDE="https://..."  # Optional, for self-hosted
```

#### Traceloop

```bash
export TRACELOOP_API_KEY="your-api-key"
export TRACELOOP_BASE_URL="https://api.traceloop.com"  # Optional
```

### Programmatic Configuration

```python
from agent_tracer import TracingService, TracingConfig

# Configure tracing behavior
config = TracingConfig(
    deactivate_tracing=False,  # Enable/disable all tracing
    mask_sensitive_data=True,  # Auto-mask API keys, passwords
    sensitive_keywords=["api_key", "password", "token", "secret"]
)

tracer = TracingService(config=config)
```

## Basic Usage

### Simple Workflow

```python
import asyncio
from agent_tracer import TracingService

async def simple_workflow():
    tracer = TracingService()
    
    # Start the trace
    await tracer.start_trace(
        trace_name="Simple Workflow",
        project_name="Demo"
    )
    
    try:
        # Trace a single step
        async with tracer.trace_step(
            step_name="processing",
            inputs={"data": "input data"}
        ):
            # Your processing logic
            result = {"processed": "output data"}
            
            # Record outputs
            tracer.set_outputs("processing", result)
        
        # End successfully
        await tracer.end_trace(outputs={"status": "complete"})
        
    except Exception as e:
        # End with error
        await tracer.end_trace(error=e)
        raise

asyncio.run(simple_workflow())
```

### Multiple Steps

```python
async def multi_step_workflow():
    tracer = TracingService()
    await tracer.start_trace(trace_name="Multi-Step Workflow")
    
    # Step 1
    async with tracer.trace_step("step1", {"input": "A"}):
        result1 = "A-processed"
        tracer.set_outputs("step1", {"output": result1})
    
    # Step 2 (uses output from step 1)
    async with tracer.trace_step("step2", {"input": result1}):
        result2 = f"{result1}-further-processed"
        tracer.set_outputs("step2", {"output": result2})
    
    # Step 3
    async with tracer.trace_step("step3", {"input": result2}):
        final_result = f"{result2}-final"
        tracer.set_outputs("step3", {"output": final_result})
    
    await tracer.end_trace(outputs={"final": final_result})
```

### Adding Logs

```python
async with tracer.trace_step("analysis", {"data": data}):
    # Add informational log
    tracer.add_log("analysis", {
        "name": "start",
        "message": "Analysis started",
        "type": "info"
    })
    
    result = analyze(data)
    
    # Add metric log
    tracer.add_log("analysis", {
        "name": "metrics",
        "message": {"items_processed": len(data), "duration_ms": 150},
        "type": "metric"
    })
    
    # Add debug log
    tracer.add_log("analysis", {
        "name": "debug",
        "message": f"Intermediate result: {result}",
        "type": "debug"
    })
    
    tracer.set_outputs("analysis", {"result": result})
```

## Advanced Patterns

### Nested Tracing (Hierarchical)

```python
async def hierarchical_workflow():
    tracer = TracingService()
    await tracer.start_trace(trace_name="Parent Workflow")
    
    # Parent step
    async with tracer.trace_step("parent", {"task": "coordinate"}):
        
        # Child step 1
        async with tracer.trace_step("child1", {"subtask": "fetch"}):
            data = fetch_data()
            tracer.set_outputs("child1", {"data": data})
        
        # Child step 2
        async with tracer.trace_step("child2", {"subtask": "process"}):
            result = process_data(data)
            tracer.set_outputs("child2", {"result": result})
        
        tracer.set_outputs("parent", {"complete": True})
    
    await tracer.end_trace(outputs={"status": "success"})
```

### Custom Trace Types

```python
# Different trace types for better categorization
async with tracer.trace_step(
    "llm_call",
    inputs={"prompt": "..."},
    trace_type="llm"  # Options: tool, chain, agent, llm, etc.
):
    response = await llm.ainvoke(prompt)
    tracer.set_outputs("llm_call", {"response": response})
```

### Custom Metadata

```python
async with tracer.trace_step(
    "processing",
    inputs={"data": data},
    metadata={
        "version": "1.0",
        "model": "gpt-4",
        "temperature": 0.7
    }
):
    result = process(data)
    tracer.set_outputs("processing", {"result": result})
```

### Error Handling

```python
async def error_handling_example():
    tracer = TracingService()
    await tracer.start_trace(trace_name="Error Example")
    
    try:
        async with tracer.trace_step("risky_operation", {"input": data}):
            result = risky_function(data)  # May raise exception
            tracer.set_outputs("risky_operation", {"result": result})
    
    except ValueError as e:
        # Error is automatically captured in trace
        tracer.add_log("risky_operation", {
            "name": "error",
            "message": f"ValueError occurred: {e}",
            "type": "error"
        })
        await tracer.end_trace(error=e)
        # Handle error or re-raise
        raise
    
    except Exception as e:
        # Catch-all for unexpected errors
        await tracer.end_trace(error=e)
        raise
    
    else:
        # Success path
        await tracer.end_trace(outputs={"status": "success"})
```

### User and Session Tracking

```python
await tracer.start_trace(
    trace_name="User Session",
    project_name="MyApp",
    user_id="user-123",
    session_id="session-abc"
)
```

## Framework Integration

### LangChain

```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from agent_tracer import TracingService

async def langchain_example():
    tracer = TracingService()
    await tracer.start_trace(trace_name="LangChain Agent")
    
    # Get LangChain callbacks
    callbacks = tracer.get_langchain_callbacks()
    
    # Create and run agent with callbacks
    llm = ChatOpenAI()
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    
    result = agent_executor.invoke(
        {"input": query},
        config={"callbacks": callbacks}
    )
    
    await tracer.end_trace(outputs={"result": result})
```

### CrewAI

```python
from crewai import Agent, Task, Crew
from agent_tracer import TracingService

async def crewai_example():
    tracer = TracingService()
    await tracer.start_trace(trace_name="Crew Execution")
    
    # Define crew
    crew = Crew(agents=[...], tasks=[...])
    
    # Trace crew execution
    async with tracer.trace_step("crew_kickoff", {"tasks": len(tasks)}):
        result = crew.kickoff()
        tracer.set_outputs("crew_kickoff", {"result": result})
    
    await tracer.end_trace(outputs={"final": result})
```

### Custom Framework

```python
class MyAgentFramework:
    def __init__(self):
        self.tracer = TracingService()
    
    async def execute(self, task: str):
        await self.tracer.start_trace(
            trace_name=f"MyFramework: {task}",
            project_name="CustomFramework"
        )
        
        try:
            async with self.tracer.trace_step("plan", {"task": task}):
                plan = await self.plan(task)
                self.tracer.set_outputs("plan", {"plan": plan})
            
            async with self.tracer.trace_step("execute", {"plan": plan}):
                result = await self.execute_plan(plan)
                self.tracer.set_outputs("execute", {"result": result})
            
            await self.tracer.end_trace(outputs={"result": result})
            return result
        
        except Exception as e:
            await self.tracer.end_trace(error=e)
            raise
```

## Best Practices

### 1. Always Use Try-Except

```python
try:
    await tracer.start_trace(...)
    # Your code
    await tracer.end_trace(...)
except Exception as e:
    await tracer.end_trace(error=e)
    raise
```

### 2. Use Descriptive Names

```python
# Good
await tracer.start_trace(trace_name="User Query Analysis Pipeline")
async with tracer.trace_step("sentiment_analysis", inputs):
    ...

# Avoid
await tracer.start_trace(trace_name="trace1")
async with tracer.trace_step("step1", inputs):
    ...
```

### 3. Include Relevant Metadata

```python
async with tracer.trace_step(
    "llm_call",
    inputs={"prompt": prompt},
    metadata={
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 1000,
        "version": "v1.0"
    }
):
    ...
```

### 4. Log Important Events

```python
async with tracer.trace_step("processing", inputs):
    tracer.add_log("processing", {
        "name": "start",
        "message": "Processing started",
        "type": "info"
    })
    
    # ... processing ...
    
    tracer.add_log("processing", {
        "name": "checkpoint",
        "message": f"Processed {count} items",
        "type": "metric"
    })
```

### 5. Use Context Managers

```python
# Good - automatic cleanup
async with tracer.trace_step("step", inputs):
    result = process()
    tracer.set_outputs("step", {"result": result})

# Avoid - manual management error-prone
# (though sometimes necessary)
```

## Troubleshooting

### No Traces Appearing

**Check environment variables:**
```bash
echo $LANGCHAIN_API_KEY
echo $LANGFUSE_SECRET_KEY
# etc.
```

**Enable debug logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Verify tracer initialization:**
```python
tracer = TracingService()
await tracer.start_trace(...)
# Check logs for "Error initializing..." messages
```

### Performance Issues

**Disable tracing in production:**
```python
config = TracingConfig(deactivate_tracing=True)
tracer = TracingService(config=config)
```

**Use sampling (future feature):**
```python
# Coming soon: sample only 10% of traces
```

### Import Errors

```bash
# Install missing dependencies
pip install agent-tracer[langsmith]
pip install agent-tracer[all]
```

### API Key Issues

```bash
# Verify keys are valid
# Check for extra spaces or newlines
export LANGCHAIN_API_KEY="$(echo $LANGCHAIN_API_KEY | tr -d '[:space:]')"
```

## Tips & Tricks

### Conditional Tracing

```python
import os

# Only trace in development
enable_tracing = os.getenv("ENVIRONMENT") == "development"
config = TracingConfig(deactivate_tracing=not enable_tracing)
tracer = TracingService(config=config)
```

### Reusing Tracer Instance

```python
# Create once, use everywhere
class AgentService:
    def __init__(self):
        self.tracer = TracingService()
    
    async def operation1(self):
        await self.tracer.start_trace(...)
        ...
    
    async def operation2(self):
        await self.tracer.start_trace(...)
        ...
```

### Custom Sensitive Keywords

```python
config = TracingConfig(
    sensitive_keywords=["api_key", "password", "ssn", "credit_card"]
)
tracer = TracingService(config=config)
```

## Next Steps

- Read [ARCHITECTURE.md](ARCHITECTURE.md) for design details
- Check [examples/](examples/) for more patterns
- See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute
- Join community discussions for support

